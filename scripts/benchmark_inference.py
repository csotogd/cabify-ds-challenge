"""Coste de calcular una decisión para una pareja nueva.

El enunciado justifica automatizar porque Cabify sirve millones de trayectos. Esa frase
plantea una pregunta que las métricas de acierto no responden: ¿cabe el modelo elegido en
el volumen real? Medimos el camino completo de producción para una pareja: proyectar a
metros, calcular las ocho variables y puntuar con XGBoost.

Medición de un solo proceso en la máquina de desarrollo, sin paralelizar ni vectorizar por
lotes. Es una cota superior razonable, no una cifra de producción: un despliegue real
repartiría el trabajo y reutilizaría cachés.
"""
import json
import platform
import time

import numpy as np
import shapely
from xgboost import XGBRegressor

from routes.data import ROOT, load_data
from routes.features import project_pair, clean_line, sample_line, BASE_FEATURES

N_PAIRS = 300
WARMUP = 5

def base_features(a, b):
    """Las ocho variables del modelo elegido, sin calcular las 42 completas."""
    est, real = clean_line(a), clean_line(b)
    le, lr = est.length, real.length
    out = {'length_est': le, 'length_real': lr,
           'relative_length_delta': (lr-le)/max((le+lr)/2, 1.),
           'start_gap_m': float(np.linalg.norm(a[0]-b[0])),
           'end_gap_m': float(np.linalg.norm(a[-1]-b[-1]))}
    for name, line, other in [('est_to_real', est, real), ('real_to_est', real, est)]:
        n = int(np.clip(np.ceil(line.length/20), 16, 1600))
        ds = shapely.distance(shapely.points(sample_line(line, n, midpoints=True)), other)
        out[f'mean_{name}_m'] = float(ds.mean())
    out['hausdorff_discrete_m'] = float(shapely.hausdorff_distance(est, real, densify=.25))
    return out

def measure():
    df, routes = load_data()
    ids = list(routes)[:N_PAIRS]
    model = XGBRegressor(n_jobs=1); model.load_model(ROOT/'models/routes/geometria_base.ubj')
    for jid in ids[:WARMUP]: base_features(*project_pair(*routes[jid]))

    start = time.perf_counter()
    projected = [project_pair(*routes[jid]) for jid in ids]
    projection = (time.perf_counter()-start)/len(ids)

    start = time.perf_counter()
    rows = [base_features(a, b) for a, b in projected]
    measurement = (time.perf_counter()-start)/len(ids)

    x = np.ascontiguousarray([[r[k] for k in BASE_FEATURES] for r in rows], dtype=np.float32)
    for _ in range(3): model.predict(x)
    start = time.perf_counter()
    for _ in range(20): model.predict(x)
    scoring = (time.perf_counter()-start)/(20*len(ids))

    total = projection+measurement+scoring
    points = float(np.mean([len(routes[j][0])+len(routes[j][1]) for j in ids]))
    return {'pairs_measured': len(ids), 'mean_points_per_pair': points,
            'machine': f'{platform.processor() or platform.machine()} · {platform.system()}',
            'ms_projection': 1000*projection, 'ms_features': 1000*measurement,
            'ms_scoring': 1000*scoring, 'ms_total': 1000*total,
            'cpu_hours_per_million': 1e6*total/3600,
            'cores_continuous_per_million_day': 1e6*total/3600/24,
            'note': 'Un proceso, sin lotes ni paralelismo. Cota superior.'}

if __name__ == '__main__':
    result = measure()
    out = ROOT/'results/routes'; out.mkdir(parents=True, exist_ok=True)
    (out/'inference_cost.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
