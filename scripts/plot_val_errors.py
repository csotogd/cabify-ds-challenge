"""Rutas de validación donde la predicción del modelo elegido no coincide con la mayoría humana.

Inspección descriptiva posterior. No ajusta modelos, umbrales ni particiones. Validación
intervino en elegir modelo, iteración y umbral: estos casos no son una medida de error limpia.
"""
import json
import sys
from pathlib import Path

import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from rutas.data import ROOT, load_data
from rutas.features import project_pair, clean_line, sample_line

R = ROOT/'resultados/rutas'
FIG = ROOT/'figuras/rutas'; FIG.mkdir(parents=True, exist_ok=True)
INK, TEAL, ORANGE, RED, MUTED, BG = '#193441','#117C83','#E87524','#B74F4F','#637781','#FAF9F5'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,
 'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
 'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
 'axes.titlelocation':'left','axes.titlesize':11,'figure.dpi':110,'savefig.dpi':150})

model = sys.argv[1] if len(sys.argv) > 1 else None
frozen = json.loads((R/'frozen_selection.json').read_text())
model = model or frozen['selected']
threshold = frozen['thresholds'][model]
names = {'geometria_base':'Geometría básica','vector_interpretable':'Vector interpretable',
         't2vec_decoder_clasificador':'t2vec + clasificador','hibrido_vector_t2vec':'Vector + t2vec'}

predictions = json.loads((R/'predictions.json').read_text())
df, routes = load_data()

rows = []
for r in predictions:
    if r['split'] != 'val' or r['target_hard'] is None:
        continue
    p = r['probabilities'][model]
    pred = int(p > threshold)
    if pred == r['target_hard']:
        continue
    meta = df.loc[r['journey_id']]
    different = int(round(meta.n_votes*meta.target_soft))
    rows.append(dict(journey_id=r['journey_id'], p=p, pred=pred, truth=r['target_hard'],
                     kind='FP' if r['target_hard'] == 0 else 'FN', region=r['region'],
                     n_votes=int(meta.n_votes), same=int(meta.n_votes)-different,
                     different=different, unknown=int(meta.n_unknown),
                     conflicted=int(meta.n_conflicted_annotators)))
# Falsas alarmas primero, y dentro de cada clase por distancia al umbral.
rows.sort(key=lambda r: (r['kind'] != 'FP', -abs(r['p']-threshold)))
assert rows, 'Sin discrepancias en validación con este modelo y umbral.'

def separation(a, b):
    """Mayor separación y proporción a más de 50 m, muestreando 400 posiciones por ruta."""
    la, lb = clean_line(a), clean_line(b)
    sa, sb = sample_line(la, 400, midpoints=True), sample_line(lb, 400, midpoints=True)
    da = shapely.distance(shapely.points(sa), lb)
    db = shapely.distance(shapely.points(sb), la)
    return float(max(da.max(), db.max())), float((da > 50).mean()), float((db > 50).mean())

cols = 4
grid = (len(rows)+cols-1)//cols
fig, axs = plt.subplots(grid, cols, figsize=(4.1*cols, 3.9*grid))
report = []
for ax, row in zip(axs.flat, rows):
    a, b = project_pair(*routes[row['journey_id']])
    ax.plot(a[:,0]/1000, a[:,1]/1000, color=INK, lw=2.6, alpha=.85, label='Estimada')
    ax.plot(b[:,0]/1000, b[:,1]/1000, color=ORANGE, lw=1.6, ls='--', label='Real')
    for xy, color in [(a, INK), (b, ORANGE)]:
        ax.scatter(xy[0,0]/1000, xy[0,1]/1000, marker='o', s=34, facecolors=BG, edgecolors=color, zorder=4)
        ax.scatter(xy[-1,0]/1000, xy[-1,1]/1000, marker='s', s=28, facecolors=BG, edgecolors=color, zorder=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.12); ax.tick_params(labelsize=8)
    ax.set_xlabel('Este local · km', fontsize=8); ax.set_ylabel('Norte local · km', fontsize=8)
    gap, fa, fb = separation(a, b)
    human = 'iguales' if row['truth'] == 0 else 'diferentes'
    predicted = 'iguales' if row['pred'] == 0 else 'diferentes'
    votes = f"{row['same']} ig. / {row['different']} dif."
    if row['unknown']: votes += f" / {row['unknown']} no sé"
    if row['conflicted']: votes += f" / {row['conflicted']} contr."
    ax.set_title(f"{row['kind']} · Humano: {human} | Modelo: {predicted}\n"
                 f"p(dif.)={row['p']:.3f} · {row['region']}", fontsize=10,
                 color=RED if row['kind'] == 'FP' else INK)
    ax.text(.02, .02, f"ID {row['journey_id'][:8]} · {votes}\nSeparación máx. {gap:,.0f} m".replace(',','.'),
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(facecolor=BG, alpha=.9, edgecolor='none'))
    report.append({**row, 'max_separation_m': round(gap, 1),
                   'fraction_est_over_50m': round(fa, 4), 'fraction_real_over_50m': round(fb, 4)})
for ax in list(axs.flat)[len(rows):]:
    ax.axis('off')

handles, labels = axs.flat[0].get_legend_handles_labels()
n_fp = sum(r['kind'] == 'FP' for r in rows)
fig.legend(handles, labels, loc='upper right', fontsize=10, ncol=2, frameon=False)
fig.suptitle(f'Validación · {len(rows)} discrepancias entre modelo y mayoría humana\n'
             f"{names.get(model, model)} · umbral {str(round(threshold,4)).replace('.',',')} · "
             f'{n_fp} falsas alarmas y {len(rows)-n_fp} diferencias no detectadas, de 443 casos con mayoría',
             x=.02, ha='left', fontsize=14, weight='bold', y=.998)
fig.text(.02, .004, 'Escalas propias por viaje · sin mapa base · validación intervino en la selección: '
                    'no es una medida de error independiente', fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, .018, 1, .978))
out = FIG/f'12_errores_validacion_{model}.png'
fig.savefig(out, bbox_inches='tight'); plt.close(fig)

path = R/f'val_errors_{model}.json'
path.write_text(json.dumps({'model': model, 'threshold': threshold, 'split': 'val',
    'purpose': 'Inspección descriptiva; no modifica modelos, umbrales ni particiones.',
    'n_decisive': 443, 'n_mismatches': len(rows), 'n_false_positives': n_fp,
    'figure': str(out.relative_to(ROOT)), 'cases': report}, indent=2, ensure_ascii=False))
print(f'{out}\n{path}\n{len(rows)} discrepancias · {n_fp} FP · {len(rows)-n_fp} FN')
