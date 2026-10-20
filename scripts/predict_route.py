"""Predice una pareja nueva con la referencia elegida en validación.

Uso: python scripts/predict_route.py pareja.json
El JSON contiene estimated_route y real_route, ambas listas de [lat, lon].
"""
import json
import sys
from pathlib import Path
import numpy as np
from xgboost import XGBRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rutas.data import ROOT
from rutas.features import project_pair,features_xy,BASE_FEATURES

def predict_pair(estimated,real):
    a,b=np.asarray(estimated,dtype=float),np.asarray(real,dtype=float)
    for p in [a,b]:
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
            raise ValueError('Cada ruta debe contener al menos dos coordenadas válidas [lat, lon].')
        if (np.abs(p[:,0])>90).any() or (np.abs(p[:,1])>180).any():
            raise ValueError('Latitud o longitud fuera de rango.')
    f=features_xy(*project_pair(a,b))
    model=XGBRegressor(n_jobs=1)
    model.load_model(ROOT/'modelos/rutas/geometria_base.ubj')
    x=np.asarray([[f[k] for k in BASE_FEATURES]],dtype=np.float32)
    p=float(model.predict(x)[0])
    frozen=json.loads((ROOT/'resultados/rutas/frozen_selection.json').read_text())
    threshold=frozen['thresholds']['geometria_base']
    return dict(model='geometria_base',probability_different=p,threshold=threshold,
                prediction='They differ' if p>=threshold else 'Both are the same',
                features={k:f[k] for k in BASE_FEATURES},
                interpretation='Predicción del juicio humano; no duración ni precio reales.')

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit(__doc__)
    pair=json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(predict_pair(pair['estimated_route'],pair['real_route']),indent=2))
