"""Corrected full-vocabulary spatial loss; new experiment, original artifacts intact."""
import copy,json,time
import numpy as np
import torch
from .data import ROOT,SEED,load_data
from .t2vec import SpatialVocabulary,EncoderDecoder,set_seed,padded,EOS,BOS,PAD,pair_vectors
from .train import train_head,choose_threshold,metrics,soft_logloss

OUT=ROOT/'results/routes_t2vec_corrected'
MODELS=ROOT/'models/routes_t2vec_corrected'


def full_spatial_loss(model,h,target,vocab):
    mask=target!=PAD;state=h[mask];true=target[mask]
    logits=state@model.output.weight.T+model.output_bias.weight.flatten()
    logits[:,[PAD,BOS]]=-1e9
    neighbors=torch.as_tensor(vocab.neighbors)[true]
    weights=torch.as_tensor(vocab.weights)[true]
    positive=logits.gather(1,neighbors)
    return (torch.logsumexp(logits,dim=1)-(weights*positive).sum(dim=1)).mean()


def main():
    OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True);set_seed(SEED)
    df,routes=load_data()
    original=json.loads((ROOT/'results/routes/t2vec_manifest.json').read_text())
    fit_ids=original['pretrained_on_journeys'];inner_ids=original['inner_validation_journeys']
    fit_routes=[r for j in fit_ids for r in routes[j]];inner_routes=[r for j in inner_ids for r in routes[j]]
    protocol=dict(reason='EOS omitted from negatives in original loss; use exact full-vocabulary normalizer.',
        selection='12 epochs maximum, checkpoint chosen by corrupted inner-validation spatial loss, patience 4',
        architecture='Unchanged 64D encoder, 250m cells, 8 spatial neighbors; still reduced adaptation',
        classifier='Same two classifier configurations, validation selection and threshold as original',
        data='Identical fit and inner validation journeys, no test fitting',exploratory_after_test_inspection=True,
        original_artifacts_untouched=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    vocab=SpatialVocabulary().fit(fit_routes);model=EncoderDecoder(vocab.size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    def epoch(route_list,training,epoch_id):
        rng=np.random.default_rng(SEED+epoch_id if training else SEED+100)
        order=rng.permutation(len(route_list)) if training else np.arange(len(route_list))
        losses=[]
        for start in range(0,len(order),32):
            rr=[route_list[i] for i in order[start:start+32]]
            src,lens=padded([vocab.encode(r,augment=True,rng=rng)[0] for r in rr])
            tgt=[vocab.encode(r)[0]+[EOS] for r in rr]
            y,_=padded(tgt);dec,_=padded([[BOS]+t[:-1] for t in tgt])
            with torch.set_grad_enabled(training):
                h=model(src,lens,dec);loss=full_spatial_loss(model,h,y,vocab)
                if training:
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append((float(loss.detach()),len(rr)))
        return float(np.average([x[0] for x in losses],weights=[x[1] for x in losses]))
    start=time.monotonic();history=[];best=np.inf;best_state=None;stale=0
    for k in range(12):
        model.train();tr=epoch(fit_routes,True,k)
        model.eval();va=epoch(inner_routes,False,k)
        history.append(dict(epoch=k+1,train_loss=tr,inner_validation_loss=va))
        print('t2vec corregido',k+1,tr,va,'segundos',round(time.monotonic()-start),flush=True)
        if va<best-1e-4:best=va;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=4:break
    model.load_state_dict(best_state);model.eval()
    torch.save(dict(model=model.state_dict(),lookup=vocab.lookup,centroids=vocab.centroids,
        neighbors=vocab.neighbors,weights=vocab.weights),MODELS/'t2vec.pt')
    (OUT/'encoder_manifest.json').write_text(json.dumps(dict(history=history,seconds=time.monotonic()-start,
        pretrained_on_journeys=fit_ids,inner_validation_journeys=inner_ids,vocab_size=vocab.size,
        loss='Full spatial cross-entropy over cells plus EOS. PAD and BOS excluded.'),indent=2))
    embeddings={}
    with torch.no_grad():
        all_ids=list(df.index)
        for start in range(0,len(df),64):
            ids=all_ids[start:start+64]
            seq=[vocab.encode(r)[0] for j in ids for r in routes[j]]
            src,lens=padded(seq);z=model.encode(src,lens).numpy().reshape(len(ids),2,64)
            embeddings.update(zip(ids,z))
    pair=pair_vectors(embeddings,df.index)
    p,detail=train_head(pair,df,save_path=MODELS/'head.pt')
    hv=(df.split=='val')&df.target_hard.notna()
    threshold=choose_threshold(df.loc[hv,'target_hard'].values,p[hv])
    (OUT/'frozen_selection.json').write_text(json.dumps(dict(threshold=threshold,val_soft_logloss=detail['val_soft_logloss']),indent=2))
    ht=(df.split=='test')&df.target_hard.notna();st=(df.split=='test')&df.target_soft.notna()
    result=metrics(df.loc[ht,'target_hard'].values,p[ht],threshold)
    result.update(validation=metrics(df.loc[hv,'target_hard'].values,p[hv],threshold),threshold=threshold,
        val_soft_logloss=detail['val_soft_logloss'],test_soft_logloss=soft_logloss(df.loc[st,'target_soft'].values,p[st]),
        exploratory=True)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    (OUT/'training_details.json').write_text(json.dumps(detail,indent=2))
    (OUT/'predictions.json').write_text(json.dumps([dict(journey_id=j,split=df.loc[j,'split'],probability=float(p[i])) for i,j in enumerate(df.index)],indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
