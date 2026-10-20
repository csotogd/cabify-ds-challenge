"""Selection happens on validation only; test is opened after the models are frozen."""
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from .data import ROOT, SEED, load_data
from .features import build_features, BASE_FEATURES
from .t2vec import pair_vectors, set_seed

def soft_logloss(y,p):
    p=np.clip(p,1e-6,1-1e-6)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))

def choose_threshold(y,p):
    candidates=np.linspace(.1,.9,81)
    scores=[balanced_accuracy_score(y,p>=t) for t in candidates]
    best=max(scores)
    eligible=[t for t,s in zip(candidates,scores) if abs(s-best)<1e-12]
    return float(min(eligible,key=lambda t:abs(t-.5)))

def metrics(y,p,threshold):
    pred=(p>=threshold).astype(int)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                f1=float(f1_score(y,pred)),roc_auc=float(roc_auc_score(y,p)),
                average_precision=float(average_precision_score(y,p)),
                brier=float(np.mean((y-p)**2)),logloss=soft_logloss(y,p),
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def train_xgb(x,df,name,columns):
    x=np.ascontiguousarray(x,dtype=np.float32)
    train=(df.split=='train')&df.target_soft.notna()
    val=(df.split=='val')&df.target_soft.notna()
    configs=[dict(max_depth=2,min_child_weight=8,reg_lambda=8),
             dict(max_depth=3,min_child_weight=8,reg_lambda=8),
             dict(max_depth=4,min_child_weight=12,reg_lambda=12)]
    best=np.inf;chosen=None;trials=[];start=time.perf_counter()
    for cfg in configs:
        model=XGBRegressor(objective='binary:logistic',eval_metric='logloss',
                           n_estimators=600,learning_rate=.035,subsample=.85,colsample_bytree=.9,
                           early_stopping_rounds=35,random_state=SEED,n_jobs=1,tree_method='hist',**cfg)
        model.fit(x[train],df.loc[train,'target_soft'].to_numpy(dtype=np.float32),
                  eval_set=[(x[val],df.loc[val,'target_soft'].to_numpy(dtype=np.float32))],verbose=False)
        score=soft_logloss(df.loc[val,'target_soft'].values,model.predict(x[val]))
        trials.append(dict(config=cfg,val_soft_logloss=score,best_iteration=int(model.best_iteration)))
        if score<best:best=score;chosen=model
    chosen.save_model(ROOT/f'models/routes/{name}.ubj')
    return chosen.predict(x),dict(trials=trials,val_soft_logloss=best,seconds=time.perf_counter()-start,
                                  feature_importance=dict(zip(columns,map(float,chosen.feature_importances_))))

class PairClassifier(nn.Module):
    def __init__(self,n,hidden):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Dropout(.25),nn.Linear(hidden,1))
    def forward(self,x):return self.layers(x).squeeze(-1)

def train_head(x,df,save_path=None):
    set_seed(SEED)
    train=((df.split=='train')&df.target_soft.notna()).values
    val=((df.split=='val')&df.target_soft.notna()).values
    scaler=StandardScaler().fit(x[train])
    xx=torch.tensor(np.clip(scaler.transform(x),-10,10),dtype=torch.float32)
    yy=torch.tensor(df.target_soft.fillna(.5).values,dtype=torch.float32)
    it=np.flatnonzero(train);iv=np.flatnonzero(val)
    best=np.inf;best_model=None;trials=[];start=time.perf_counter()
    for hidden,lr in [(32,.001),(64,.001)]:
        set_seed(SEED)
        model=PairClassifier(x.shape[1],hidden)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.02)
        rng=np.random.default_rng(SEED)
        history=[];best_trial=np.inf;state=None;stale=0
        for epoch in range(100):
            model.train();losses=[]
            order=rng.permutation(it)
            for start_batch in range(0,len(order),64):
                ids=order[start_batch:start_batch+64]
                loss=nn.functional.binary_cross_entropy_with_logits(model(xx[ids]),yy[ids])
                opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                vl=float(nn.functional.binary_cross_entropy_with_logits(model(xx[iv]),yy[iv]))
            history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=vl))
            if vl<best_trial-1e-4:best_trial=vl;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if stale>=12:break
        trials.append(dict(hidden=hidden,learning_rate=lr,val_soft_logloss=best_trial,history=history))
        if best_trial<best:
            best=best_trial;best_model=model;best_model.load_state_dict(state)
    best_model.eval()
    with torch.no_grad():pred=torch.sigmoid(best_model(xx)).numpy()
    torch.save(dict(state=best_model.state_dict(),scaler_mean=scaler.mean_,scaler_scale=scaler.scale_),
               save_path or ROOT/'models/routes/t2vec_head.pt')
    return pred,dict(trials=trials,val_soft_logloss=best,seconds=time.perf_counter()-start)

def run():
    df,_=load_data();features=build_features().loc[df.index]
    emb_file=ROOT/'data/cache/t2vec_embeddings.npz'
    if not emb_file.exists():raise RuntimeError('Ejecuta primero el preentrenamiento t2vec.')
    encoder_manifest=json.loads((ROOT/'results/routes/t2vec_manifest.json').read_text())
    encoder_ids=set(encoder_manifest['pretrained_on_journeys'])|set(encoder_manifest['inner_validation_journeys'])
    assert encoder_ids==set(df.index[df.split=='train']), 'The encoder does not correspond to this split.'
    emb=np.load(emb_file);embeddings=dict(zip(emb['ids'],emb['vectors']))
    pair=pair_vectors(embeddings,df.index)
    protocol=dict(primary_selection='log loss against the vote fraction, on validation',
                  threshold_selection='maximum balanced accuracy on validation, ties broken towards 0.5',
                  target='proportion of They differ votes among votes expressing a decision',
                  unknown_policy='I do not know is read neither as a class nor as a 0.5 vote; with no votes there is no supervised loss.',
                  annotation_policy='One vote per annotator and journey; contradictory decisions from the same person count as an abstention.',
                  tie_policy='0.5 is used in probabilistic training; excluded from binary metrics.',
                  metric_population='Journeys with a defined majority; coverage and probabilistic metrics are reported separately.',
                  primary_test_metric='balanced_accuracy',no_retraining_on_train_plus_val=True,
                  planned_models=['geometria_base','vector_interpretable','t2vec_decoder_clasificador','hibrido_vector_t2vec'],
                  bootstrap='2000 remuestreos emparejados por viaje del test; IC percentil 95%; no incluye incertidumbre del entrenamiento',seed=SEED)
    (ROOT/'results/routes/protocol.json').write_text(json.dumps(protocol,indent=2))
    outputs={};details={}
    for name,cols in [('geometria_base',BASE_FEATURES),('vector_interpretable',list(features))]:
        print('Entrenando',name,flush=True)
        outputs[name],details[name]=train_xgb(features[cols].to_numpy(),df,name,cols)
    print('Entrenando cabeza supervisada t2vec',flush=True)
    outputs['t2vec_decoder_clasificador'],details['t2vec_decoder_clasificador']=train_head(pair,df)
    print('Entrenando vector interpretable + t2vec',flush=True)
    outputs['hibrido_vector_t2vec'],details['hibrido_vector_t2vec']=train_xgb(
        np.column_stack([features.to_numpy(),pair]),df,'hibrido_vector_t2vec',
        list(features)+[f'z_pair_{k}' for k in range(pair.shape[1])])
    valid=(df.split=='val')&df.target_hard.notna()
    thresholds={name:choose_threshold(df.loc[valid,'target_hard'].values,p[valid]) for name,p in outputs.items()}
    selected=min(details,key=lambda n:details[n]['val_soft_logloss'])
    frozen=dict(selected=selected,thresholds=thresholds,validation_results={n:d['val_soft_logloss'] for n,d in details.items()})
    (ROOT/'results/routes/frozen_selection.json').write_text(json.dumps(frozen,indent=2))
    # From here the test set is opened. No architecture, weights or threshold are refitted.
    print('Frozen selection:',selected,'; test is evaluated once.',flush=True)
    test=(df.split=='test')&df.target_hard.notna();y=df.loc[test,'target_hard'].values
    test_soft=(df.split=='test')&df.target_soft.notna()
    all_metrics={}
    rng=np.random.default_rng(SEED+17)
    draws=rng.integers(0,len(y),(2000,len(y)))
    boot={}
    for name,p in outputs.items():
        pt=p[test];threshold=thresholds[name]
        m=metrics(y,pt,threshold)
        ba=[]
        pred=pt>=threshold
        for idx in draws:
            yi=y[idx];pr=pred[idx]
            if len(np.unique(yi))<2:continue
            ba.append(.5*((pr[yi==1]==1).mean()+(pr[yi==0]==0).mean()))
        boot[name]=np.array(ba)
        m.update(threshold=threshold,balanced_accuracy_ci95=np.quantile(ba,[.025,.975]).tolist(),
                 soft_logloss_all_decisive_votes=soft_logloss(df.loc[test_soft,'target_soft'].values,p[test_soft]),
                 selected_on_validation=name==selected,
                 validation=metrics(df.loc[valid,'target_hard'].values,p[valid],threshold))
        all_metrics[name]=m
    comparisons={name:dict(delta_balanced_accuracy=float(all_metrics[name]['balanced_accuracy']-all_metrics['geometria_base']['balanced_accuracy']),
                          paired_ci95=np.quantile(boot[name]-boot['geometria_base'],[.025,.975]).tolist())
                 for name in outputs if name!='geometria_base'}
    predictions=[]
    for i,jid in enumerate(df.index):
        row=df.loc[jid]
        predictions.append(dict(journey_id=jid,split=row.split,status=row.status,region=row.region,
                                target_soft=None if pd.isna(row.target_soft) else float(row.target_soft),
                                target_hard=None if pd.isna(row.target_hard) else int(row.target_hard),
                                n_votes=int(row.n_votes),probabilities={n:float(p[i]) for n,p in outputs.items()}))
    for filename,value in [('metrics.json',all_metrics),('training_details.json',details),
                           ('comparisons.json',comparisons),('predictions.json',predictions)]:
        (ROOT/'results/routes'/filename).write_text(json.dumps(value,indent=2))
    print(json.dumps(all_metrics,indent=2),flush=True)

if __name__=='__main__':run()
