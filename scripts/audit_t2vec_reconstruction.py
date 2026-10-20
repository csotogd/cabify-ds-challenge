"""Post-hoc diagnostic of frozen encoder/decoder on its inner validation routes."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import torch
from scipy.spatial import cKDTree
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rutas.data import load_data,SEED
from rutas.t2vec import SpatialVocabulary,EncoderDecoder,thin,ecef,padded,BOS,EOS,PAD,set_seed

@njit
def discrete_frechet(distances):
    n,m=distances.shape
    dp=np.full((n,m),np.inf)
    for i in range(n):
        for j in range(m):
            if i==0 and j==0:dp[i,j]=distances[i,j]
            elif i==0:dp[i,j]=max(dp[i,j-1],distances[i,j])
            elif j==0:dp[i,j]=max(dp[i-1,j],distances[i,j])
            else:dp[i,j]=max(min(dp[i-1,j],dp[i-1,j-1],dp[i,j-1]),distances[i,j])
    return dp[-1,-1]

def spatial_distance(a,b):
    if not len(a) or not len(b):return None
    # Normalize centroid means back to the sphere before computing arc distances.
    a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    d=2*6371000*np.arcsin(np.clip(np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)/2,0,1))
    return float(discrete_frechet(d))

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,u in enumerate(a,1):
        new=[i]
        for j,v in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(u!=v)))
        row=new
    return row[-1]/max(len(a),len(b),1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corrected',action='store_true');args=parser.parse_args()
    set_seed(SEED);df,routes=load_data()
    manifest=json.loads((ROOT/'resultados/rutas/t2vec_manifest.json').read_text())
    model_path=ROOT/('modelos/rutas_t2vec_corregido/t2vec.pt' if args.corrected else 'modelos/rutas/t2vec.pt')
    state=torch.load(model_path,weights_only=False,map_location='cpu')
    vocab=SpatialVocabulary();vocab.lookup=state['lookup'];vocab.centroids=state['centroids']
    vocab.tree=cKDTree(vocab.centroids);vocab.size=len(vocab.lookup)+3
    model=EncoderDecoder(vocab.size);model.load_state_dict(state['model']);model.eval()
    ids=manifest['inner_validation_journeys']
    assert set(ids).isdisjoint(manifest['pretrained_on_journeys'])
    assert all(df.loc[j,'split']=='train' for j in ids)
    out=ROOT/('resultados/rutas_reconstruccion_corregida' if args.corrected else 'resultados/rutas_reconstruccion');out.mkdir(exist_ok=True)
    protocol=dict(scope='Post-hoc diagnostic; 143 inner-validation journeys used previously for encoder checkpoint selection, NOT a fresh test',
        scenarios=['clean','20% independent point dropout + 15m Gaussian noise'],
        decoding='Greedy full-vocabulary scores; PAD/BOS excluded; EOS termination; maximum128 steps',
        metrics=['teacher-forced exact token accuracy including EOS','free normalized edit distance excluding EOS',
                 'spherical discrete Frechet to original thinned GPS','EOS/empty/max-length frequencies'],
        baselines=['retained input points','oracle clean quantized target centroids'],
        selection='No fitting, tuning or model selection from this diagnostic',seed=SEED+311)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def tokens(points):
        result=[];oov=0
        for key,xyz in zip(vocab.grid(points),ecef(points)):
            value=vocab.lookup.get(key)
            if value is None:value=int(vocab.tree.query(xyz)[1])+3;oov+=1
            if not result or value!=result[-1]:result.append(value)
        return result,oov/len(points)
    samples=[];rng=np.random.default_rng(SEED+311)
    for jid in ids:
        for kind,route in zip(['estimated','real'],routes[jid]):
            clean=thin(route);target,oov=tokens(clean)
            for scenario in ['clean','corrupted']:
                source=clean.copy()
                if scenario=='corrupted':
                    keep=rng.random(len(source))>=.2;keep[[0,-1]]=True
                    source=source[keep].copy()
                    source+=rng.normal(0,15,size=source.shape)/[111320,111320*np.cos(np.radians(source[:,0].mean()))]
                src,_=tokens(source)
                samples.append(dict(journey_id=jid,kind=kind,scenario=scenario,clean=clean,source=source,
                    target=target,src=src,oov=oov))
    rows=[]
    with torch.no_grad():
        for begin in range(0,len(samples),32):
            batch=samples[begin:begin+32];src,lengths=padded([s['src'] for s in batch])
            targets=[s['target']+[EOS] for s in batch]
            dec,_=padded([[BOS]+s[:-1] for s in targets]);y,_=padded(targets)
            h=model(src,lengths,dec)
            logits=h@model.output.weight.T+model.output_bias.weight.flatten()
            logits[:,:,[PAD,BOS]]=-torch.inf;tf=logits.argmax(-1)
            hidden=model.encode(src,lengths).unsqueeze(0)
            next_token=torch.full((len(batch),1),BOS,dtype=torch.long)
            generated=[[] for _ in batch];finished=np.zeros(len(batch),bool)
            for step in range(128):
                h,hidden=model.decoder(model.embedding(next_token),hidden)
                logits=h[:,-1]@model.output.weight.T+model.output_bias.weight.flatten()
                logits[:,[PAD,BOS]]=-torch.inf
                next_token=logits.argmax(-1).unsqueeze(1)
                for i,tok in enumerate(next_token[:,0].tolist()):
                    if not finished[i]:
                        if tok==EOS:finished[i]=True
                        else:generated[i].append(tok)
                if finished.all():break
            for i,s in enumerate(batch):
                target=s['target'];pred=generated[i];original=ecef(s['clean'])
                row={k:s[k] for k in ['journey_id','kind','scenario','oov']}
                row.update(n_target=len(target),n_pred=len(pred),ended=bool(finished[i]),empty=not pred,
                    teacher_forced_token_accuracy=float((tf[i,:len(target)+1]==y[i,:len(target)+1]).float().mean()),
                    free_normalized_edit=edit_distance(target,pred),
                    input_frechet_m=spatial_distance(original,ecef(s['source'])),
                    oracle_quantized_frechet_m=spatial_distance(original,vocab.centroids[np.array(target)-3]),
                    decoded_frechet_m=spatial_distance(original,vocab.centroids[np.array(pred,dtype=int)-3]),
                    target_tokens=target,predicted_tokens=pred)
                rows.append(row)
            print('Reconstrucción',min(begin+32,len(samples)),'/',len(samples),flush=True)
    summary={}
    for scenario in ['clean','corrupted']:
        sub=[r for r in rows if r['scenario']==scenario]
        summary[scenario]=dict(n_routes=len(sub),n_journeys=len(ids),
            eos_fraction=float(np.mean([r['ended'] for r in sub])),empty_fraction=float(np.mean([r['empty'] for r in sub])))
        for key in ['teacher_forced_token_accuracy','free_normalized_edit','input_frechet_m','oracle_quantized_frechet_m','decoded_frechet_m','oov']:
            v=[r[key] for r in sub if r[key] is not None]
            summary[scenario][key]=dict(n=len(v),median=float(np.median(v)) if v else None,mean=float(np.mean(v)) if v else None)
    # An implementation defect, checked directly: EOS cannot be a negative in
    # the original sampled objective, so false early-stop scores are unpenalized.
    if args.corrected:
        from rutas.t2vec_corrected import full_spatial_loss
        def spatial_loss(model,h,y,vocab,generator):return full_spatial_loss(model,h,y,vocab)
    else:
        from rutas.t2vec import spatial_loss
    vocab.neighbors=state['neighbors'];vocab.weights=state['weights']
    src,lens=padded([samples[0]['src']]);target=samples[0]['target']
    dec,_=padded([[BOS]+target[:-1]]);y,_=padded([target])
    with torch.no_grad():
        h=model(src,lens,dec)
        before=float(spatial_loss(model,h,y,vocab,torch.Generator().manual_seed(1)))
        model.output_bias.weight[EOS]+=100
        after=float(spatial_loss(model,h,y,vocab,torch.Generator().manual_seed(1)))
    assert after>before+1 if args.corrected else abs(before-after)<1e-6
    summary['loss_audit']=dict(non_end_loss_before=before,non_end_loss_after_adding_100_to_eos_score=after,
        defect=None if args.corrected else 'EOS excluded from all negative draws: sampled loss does not penalize early EOS on non-ending steps.',
        interpretation=('Early EOS is now penalized by the complete normalizer; reconstruction quality assessed separately.' if args.corrected else 'Demonstrated defect of this adaptation, not proof that fixing it alone restores reconstruction.'),
        weights_modified_on_disk=False)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    (out/'predictions.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
