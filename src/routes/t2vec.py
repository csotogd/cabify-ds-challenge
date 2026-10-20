"""Reduced t2vec: spatial cells, GRU and a decoder with a sampled spatial loss.

An explicit adaptation of Li et al. (ICDE 2018), not an exact reproduction of their
hyperparameters or results. It uses no human labels during pretraining.
"""
import copy
import json
import time
from collections import Counter
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from pyproj import Transformer
from .data import ROOT, SEED

PAD, BOS, EOS = 0,1,2
CELL_M = 250
MAX_POINTS = 64

def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)

def thin(route,n=MAX_POINTS):
    # Equiespaciado en longitud, sin normalizar cada ruta por separado.
    r=np.asarray(route)
    step=np.linalg.norm(np.diff(r,axis=0)*[111320,111320*np.cos(np.radians(r[:,0].mean()))],axis=1)
    cum=np.r_[0,np.cumsum(step)]
    if cum[-1]==0:return r[:1]
    at=np.linspace(0,cum[-1],min(n,len(r)))
    return np.column_stack([np.interp(at,cum,r[:,k]) for k in [0,1]])

def ecef(route):
    lat,lon=np.radians(route).T
    return 6371000*np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])

class SpatialVocabulary:
    def __init__(self): self.transformers={}
    def grid(self,route):
        # UTM da metros locales. La zona forma parte del identificador de celda.
        lon=float(np.median(route[:,1])); lat=float(np.median(route[:,0]))
        zone=int(np.clip(np.floor((lon+180)/6)+1,1,60)); north=lat>=0
        crs=32600+zone if north else 32700+zone
        if crs not in self.transformers:
            self.transformers[crs]=Transformer.from_crs(4326,crs,always_xy=True)
        t=self.transformers[crs]
        x,y=t.transform(route[:,1],route[:,0])
        return [(crs,int(a//CELL_M),int(b//CELL_M)) for a,b in zip(x,y)]
    def fit(self,route_list):
        sums={};counts=Counter()
        for r in route_list:
            r=thin(r)
            for key,xyz in zip(self.grid(r),ecef(r)):
                sums[key]=sums.get(key,np.zeros(3))+xyz;counts[key]+=1
        keys=sorted(sums)
        self.lookup={k:i+3 for i,k in enumerate(keys)}
        self.centroids=np.asarray([sums[k]/counts[k] for k in keys])
        self.tree=cKDTree(self.centroids)
        self.size=len(keys)+3
        ds,idx=self.tree.query(self.centroids,k=min(8,len(keys)))
        self.neighbors=np.zeros((self.size,8),dtype=np.int64)
        self.weights=np.zeros((self.size,8),dtype=np.float32)
        self.neighbors[3:,:idx.shape[1]]=idx+3
        weights=np.exp(-ds/250)
        weights/=weights.sum(axis=1,keepdims=True)
        self.weights[3:,:idx.shape[1]]=weights
        for token in [PAD,BOS,EOS]:
            self.neighbors[token,0]=token;self.weights[token,0]=1
        return self
    def encode(self,route,augment=False,rng=None):
        r=thin(route)
        if augment and len(r)>3:
            keep=rng.random(len(r))>rng.uniform(.05,.25);keep[[0,-1]]=True
            r=r[keep].copy()
            noise=rng.normal(0,15,size=r.shape)
            r+=noise/[111320,111320*np.cos(np.radians(r[:,0].mean()))]
        keys=self.grid(r); xyz=ecef(r)
        out=[];missing=0
        for key,point in zip(keys,xyz):
            token=self.lookup.get(key)
            if token is None:
                token=int(self.tree.query(point)[1])+3;missing+=1
            if not out or token!=out[-1]:out.append(token)
        return out or [EOS],missing/len(keys)

class EncoderDecoder(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,32,padding_idx=PAD)
        self.encoder=nn.GRU(32,32,batch_first=True,bidirectional=True)
        self.decoder=nn.GRU(32,64,batch_first=True)
        self.output=nn.Embedding(vocab_size,64)
        self.output_bias=nn.Embedding(vocab_size,1)
    def encode(self,tokens,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(tokens),lengths.cpu(),
                                                batch_first=True,enforce_sorted=False)
        _,h=self.encoder(packed)
        return torch.cat([h[0],h[1]],dim=1)
    def forward(self,src,lengths,decoder_in):
        z=self.encode(src,lengths)
        h,_=self.decoder(self.embedding(decoder_in),z.unsqueeze(0))
        return h

def padded(sequences):
    out=np.zeros((len(sequences),max(map(len,sequences))),dtype=np.int64)
    for i,s in enumerate(sequences):out[i,:len(s)]=s
    return torch.tensor(out),torch.tensor([len(s) for s in sequences])

def spatial_loss(model,h,target,vocab,negative_generator):
    mask=target!=PAD;state=h[mask];true=target[mask]
    pos=torch.from_numpy(vocab.neighbors)[true]
    weights=torch.from_numpy(vocab.weights)[true]
    neg=torch.randint(3,vocab.size,(len(true),64),generator=negative_generator)
    neg_mask=(neg[:,:,None]==pos[:,None,:]).any(-1)
    logits_pos=(state[:,None,:]*model.output(pos)).sum(-1)+model.output_bias(pos).squeeze(-1)
    logits_neg=(state[:,None,:]*model.output(neg)).sum(-1)+model.output_bias(neg).squeeze(-1)
    logits_pos=logits_pos.masked_fill(weights==0,-1e9)
    logits_neg=logits_neg.masked_fill(neg_mask,-1e9)
    normalizer=torch.logsumexp(torch.cat([logits_pos,logits_neg],dim=1),dim=1)
    return (normalizer-(weights*logits_pos).sum(dim=1)).mean()

def train_encoder(df,routes,epochs=12):
    set_seed(SEED)
    ids=list(df.index[df.split=='train'])
    rng=np.random.default_rng(SEED)
    rng.shuffle(ids)
    # Pretraining validation lives inside train; grouped by journey.
    n_inner=max(1,round(.1*len(ids)))
    fit_ids,inner_ids=ids[n_inner:],ids[:n_inner]
    train_routes=[r for j in fit_ids for r in routes[j]]
    inner_routes=[r for j in inner_ids for r in routes[j]]
    # Even the vocabulary is learned only on the fitting subset.
    vocab=SpatialVocabulary().fit(train_routes)
    model=EncoderDecoder(vocab.size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    targets=[vocab.encode(r)[0]+[EOS] for r in train_routes]
    history=[];best=np.inf;best_state=None;stale=0;start=time.perf_counter()
    print(f't2vec: {len(train_routes)} rutas, vocabulario {vocab.size}, vector 64',flush=True)
    def batches(route_list,target_list,training,epoch):
        rr=np.random.default_rng(SEED+epoch if training else SEED+100)
        order=rr.permutation(len(route_list)) if training else np.arange(len(route_list))
        losses=[];ng=torch.Generator().manual_seed(SEED+epoch if training else SEED+101)
        for begin in range(0,len(order),32):
            inds=order[begin:begin+32]
            src=[vocab.encode(route_list[i],augment=training,rng=rr)[0] for i in inds]
            tgt=[target_list[i] for i in inds]
            src,lens=padded(src);y,_=padded(tgt)
            dec,_=padded([[BOS]+t[:-1] for t in tgt])
            with torch.set_grad_enabled(training):
                h=model(src,lens,dec);loss=spatial_loss(model,h,y,vocab,ng)
                if training:
                    optimizer.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5)
                    optimizer.step()
            losses.append((float(loss.detach()),len(inds)))
        return float(np.average([x[0] for x in losses],weights=[x[1] for x in losses]))
    inner_targets=[vocab.encode(r)[0]+[EOS] for r in inner_routes]
    for epoch in range(epochs):
        model.train();loss=batches(train_routes,targets,True,epoch)
        model.eval();valid=batches(inner_routes,inner_targets,False,epoch)
        history.append(dict(epoch=epoch+1,train_loss=loss,inner_validation_loss=valid))
        print(f't2vec epoch {epoch+1}: {loss:.3f} / {valid:.3f}',flush=True)
        if valid<best-1e-4:
            best=valid;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=4:break
    model.load_state_dict(best_state);model.eval()
    embeddings={};oov={}
    with torch.no_grad():
        for part in range(0,len(df),64):
            batch_ids=list(df.index[part:part+64])
            seq=[];missing=[]
            for jid in batch_ids:
                for route in routes[jid]:
                    tokens,rate=vocab.encode(route);seq.append(tokens);missing.append(rate)
            src,lens=padded(seq)
            z=model.encode(src,lens).numpy().reshape(len(batch_ids),2,64)
            for jid,vec,miss in zip(batch_ids,z,np.array(missing).reshape(-1,2)):
                embeddings[jid]=vec;oov[jid]=float(miss.mean())
    np.savez_compressed(ROOT/'data/cache/t2vec_embeddings.npz',
                        ids=np.array(list(embeddings)),vectors=np.array(list(embeddings.values())))
    torch.save(dict(model=model.state_dict(),lookup=vocab.lookup,centroids=vocab.centroids,
                    neighbors=vocab.neighbors,weights=vocab.weights),ROOT/'models/routes/t2vec.pt')
    manifest=dict(history=history,vocab_size=vocab.size,cell_m=CELL_M,embedding_dim=64,
                  pretrained_on_journeys=fit_ids,inner_validation_journeys=inner_ids,
                  seconds=time.perf_counter()-start,oov_by_split={s:float(np.mean([oov[j] for j in df.index[df.split==s]])) for s in ['train','val','test']},
                  oov_policy='An unseen cell is mapped to the nearest training centroid; the vocabulary is not expanded.',
                  adaptation='Bidirectional GRU, 32 per direction; GRU decoder 64; 250 m UTM cells; K=8 neighbours; 64 negatives per token; spatial loss with a sampled normaliser. No separate cell pretraining. Not an exact reproduction of the paper.',
                  source='https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf')
    (ROOT/'results/routes/t2vec_manifest.json').write_text(json.dumps(manifest,indent=2))
    return embeddings

def pair_vectors(embeddings,ids):
    z=np.asarray([embeddings[j] for j in ids]);a,b=z[:,0],z[:,1]
    return np.concatenate([a,b,np.abs(a-b),a*b],axis=1)
