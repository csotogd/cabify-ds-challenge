"""Freeze validation selection before exploratory test evaluation. Preserve originals."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from routes.data import load_data,SEED
from routes.features import build_features
from routes.graph_extension import build_graph_features,REPORT
from routes.train import soft_logloss,choose_threshold,metrics


def main():
    df,_=load_data();geo=build_features().loc[df.index];graph=build_graph_features().loc[df.index]
    assert np.isfinite(graph.fillna(0).to_numpy()).all()
    train=(df.split=='train')&df.target_soft.notna();val=(df.split=='val')&df.target_soft.notna()
    hardval=(df.split=='val')&df.target_hard.notna()
    outputs={};details={};thresholds={}
    models=ROOT/'models/routes_grafo';models.mkdir(exist_ok=True)
    for name,features in [('grafo_xgb',graph),('geometria_grafo_xgb',pd.concat([geo,graph],axis=1))]:
        x=np.ascontiguousarray(features,dtype=np.float32);best=np.inf;chosen=None;trials=[]
        for config in [dict(max_depth=2,min_child_weight=8,reg_lambda=8),
                       dict(max_depth=3,min_child_weight=8,reg_lambda=8),
                       dict(max_depth=4,min_child_weight=12,reg_lambda=12)]:
            model=XGBRegressor(objective='binary:logistic',eval_metric='logloss',n_estimators=600,
                learning_rate=.035,subsample=.85,colsample_bytree=.9,early_stopping_rounds=35,
                random_state=SEED,n_jobs=1,tree_method='hist',**config)
            model.fit(x[train],df.loc[train,'target_soft'].to_numpy(dtype=np.float32),
                eval_set=[(x[val],df.loc[val,'target_soft'].to_numpy(dtype=np.float32))],verbose=False)
            score=soft_logloss(df.loc[val,'target_soft'].values,model.predict(x[val]))
            trials.append(dict(config=config,val_soft_logloss=score,best_iteration=int(model.best_iteration)))
            if score<best:best=score;chosen=model
        chosen.save_model(models/f'{name}.ubj')
        outputs[name]=chosen.predict(x)
        thresholds[name]=choose_threshold(df.loc[hardval,'target_hard'].values,outputs[name][hardval])
        details[name]=dict(val_soft_logloss=best,trials=trials,n_features=len(features.columns),
            feature_importance=dict(zip(features.columns,map(float,chosen.feature_importances_))))
    selected=min(details,key=lambda n:details[n]['val_soft_logloss'])
    (REPORT/'frozen_selection.json').write_text(json.dumps(dict(selected_new_variant=selected,
        validation_results={k:v['val_soft_logloss'] for k,v in details.items()},thresholds=thresholds,
        original_selection_unchanged='geometria_base',test_is_exploratory=True),indent=2))
    print('Congelado en validación:',selected,flush=True)
    # Only now access test labels/predictions for evaluation of new models.
    test=(df.split=='test')&df.target_hard.notna();softtest=(df.split=='test')&df.target_soft.notna()
    original=pd.DataFrame(json.loads((ROOT/'results/routes/predictions.json').read_text())).set_index('journey_id').loc[df.index]
    base=original.probabilities.map(lambda p:p['geometria_base']).to_numpy()
    base_threshold=json.loads((ROOT/'results/routes/frozen_selection.json').read_text())['thresholds']['geometria_base']
    y=df.loc[test,'target_hard'].to_numpy();rng=np.random.default_rng(SEED+17)
    draws=rng.integers(0,len(y),(2000,len(y)))
    def bootstrap(p,t):
        pred=p[test]>=t
        return np.array([.5*((pred[ii][y[ii]==1]==1).mean()+(pred[ii][y[ii]==0]==0).mean()) for ii in draws])
    base_boot=bootstrap(base,base_threshold);result={}
    for name,p in outputs.items():
        threshold=thresholds[name];b=bootstrap(p,threshold)
        m=metrics(y,p[test],threshold)
        m.update(validation=metrics(df.loc[hardval,'target_hard'].values,p[hardval],threshold),
            val_soft_logloss=details[name]['val_soft_logloss'],test_soft_logloss=soft_logloss(df.loc[softtest,'target_soft'].values,p[softtest]),
            threshold=threshold,balanced_accuracy_ci95=np.quantile(b,[.025,.975]).tolist(),
            delta_base_ci95=np.quantile(b-base_boot,[.025,.975]).tolist(),
            delta_base=float(m['balanced_accuracy']-metrics(y,base[test],base_threshold)['balanced_accuracy']))
        result[name]=m
    preds=[dict(journey_id=jid,split=df.loc[jid,'split'],probabilities={n:float(p[i]) for n,p in outputs.items()}) for i,jid in enumerate(df.index)]
    for filename,value in [('metrics.json',result),('training_details.json',details),('predictions.json',preds)]:
        (REPORT/filename).write_text(json.dumps(value,indent=2))
    hashes=json.loads((REPORT/'original_artifact_hashes.json').read_text())
    for relative,sha in hashes.items():
        if relative=='results/routes/eda.json':continue  # Explicitly requested descriptive extension.
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha,relative+' changed'
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
