"""Construye el relato del experimento; los entrenamientos viven en src/routes."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Did they follow the same route?
## From two GPS traces to a decision we can defend

**Part 2 · Model Prototyping · Cabify Data Science Challenge**

> ### In six lines
>
> 1. **Simple geometry is enough.** Eight measurements and an XGBoost give 94.3 % balanced accuracy on test. A 42-variable vector, a t2vec adaptation and a hybrid do not improve on that in any defensible way: their paired confidence intervals include zero.
> 2. **The limit is not the model, it is the label.** 83 % of journeys are judged by a single person, and the panel is eight annotators with different criteria: one marks «they differ» 65 % of the time and another 40 %. The model agrees 98 % with the lenient annotator and 88 % with the strict one. It has learned a particular threshold, not a truth.
> 3. **Translated into money, the median deviation is 11 cents** on an average fare of €4.63. That sounds small until you compare it with the margin: Cabify closed 2025 with 3.3 % EBITDA, so **almost half of all journeys deviate by more than that journey leaves in profit**.
> 4. **The model catches 90.2 % of the cases but recovers 97.4 % of the money:** it fails on the cheap trips. Accuracy understates what it does.
> 5. **The error can be turned into a lever.** With conformal prediction, sending 6 % of journeys to human review brings the automated error down from 6.3 % to 1.9 %; 13 % brings it to 0.5 %. The business picks the point.
> 6. **Recommendation:** deploy basic geometry with conformal prediction, review by economic impact rather than by a fixed rate, and spend the next euro on a second annotation — not on a bigger model. The detail is in [section 11](#recomendacion).

We have two traces of the same journey: the one that was predicted and the one that happened. The label we are given is perceptual: whether a person considers them similar when looking at a map. But the brief motivates the problem from **pricing**, which is set before the trip using the estimated route. Those two questions are not the same, and part of the value of this work lies in measuring how far apart they are: we do that in [section 9](#dinero), after solving the problem as posed.

Two questions guide the walkthrough. The technical one: **do we need to learn a complex representation, or are a few well-chosen geometric measurements enough?** And the business one, which appears in the second half: **what does it cost when the routing engine is wrong, and how much human review is needed to decide the rest automatically?**

We will compare an eight-variable baseline, an interpretable 42-measurement vector, a reduced t2vec adaptation with a reconstruction decoder and a classifier, and a combination of vectors. We add exploratory extensions using the road network and a t2vec version corrected after auditing its reconstruction, both carried out after inspecting the original test set.

The path of the analysis is: understand the votes → build representations → compare on validation → check on test → go back to the routes to understand hits and misses. **The best model will be the one chosen by the rule fixed on validation**, not whichever comes first when sorting the test results.

| Method we try | What the classifier receives | Why try it | What it may lose or cost |
|---|---|---|---|
| Basic geometry + XGBoost | 8 measurements: both lengths, relative difference, endpoints, mean separations and discrete maximum | Check how much a small, understandable baseline already solves | Summarises each pair; may dilute the weight of a localised detour |
| Interpretable vector + XGBoost | 42 measurements, including the previous 8: far-apart proportions, segments, IoU, turns and ordering | Explicitly represent the differences we proposed looking at on a map | More variables do not necessarily add new information; they can overfit |
| Reduced t2vec + neural classifier | Two 64-dimensional embeddings, plus their absolute difference and product: 256 values | Learn patterns without labels before learning the human judgement | The later audit found failed reconstruction and a defect in the sampled loss; it is not a valid benchmark of the original t2vec |
| Interpretable vector + t2vec + XGBoost | The 42 measurements and the 256 values above: 298 in total | Check whether what was learned complements what we measured by hand | More complexity; it could add noise without improving stably |
| Reduced t2vec with corrected loss + classifier | The same architecture and vector comparison, retrained with a fully normalised objective | Fix the handling of the end-of-sequence token and evaluate reconstruction and classification separately | It still needs data and adequate resolution; documented as a new experiment |
| Road graph and travel times + XGBoost, alone and with geometry | Street variables, connectivity, estimated times and map-match quality | Separate geometric closeness from connectivity and cost of driving | Current map, approximations and uncertain matches; exploratory comparison made after the original test |

**These are not just difference vectors.** The baseline already keeps the individual lengths; t2vec keeps both full embeddings in addition to their comparison. We have not run an ablation of «concatenation only» against «differences only»: answering that would require testing it on validation.

> The results that follow come from the real challenge file. They are not simulations. The estimates for the photography experiment belong to the previous notebook.

The brief mentions 5,000 selected journeys. The linked file contains **2,386 distinct journeys**. The 3,000/1,000/1,000 request is read as training/validation/test; we keep its 60/20/20 proportion without duplicating journeys: **1,432 / 477 / 477**.
""")

code("""
from pathlib import Path
import sys, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, Markdown

ROOT = Path.cwd()
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/cabify-mpl')
from routes.data import load_data
from routes.features import build_features, project_pair, clean_line, sample_line, BASE_FEATURES
from routes.road_graph import load_graph

# Antes de reproducir las figuras: python scripts/run_routes.py
# Este notebook lee los resultados congelados; no vuelve a seleccionar con test.
R = ROOT/'results/routes'
def read(name):return json.loads((R/name).read_text())
audit=read('audit.json'); metrics=read('metrics.json'); frozen=read('frozen_selection.json')
details=read('training_details.json'); encoder=read('t2vec_manifest.json')
comparisons=read('comparisons.json'); graph_status=read('road_graph_status.json')
df,routes=load_data()
features=build_features()
predictions=pd.DataFrame(read('predictions.json')).set_index('journey_id')

INK,TEAL,ORANGE,RED,MUTED,BG='#193441','#117C83','#E87524','#B74F4F','#637781','#FAF9F5'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,
 'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':INK,
 'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
 'axes.titlelocation':'left','axes.titlesize':13,'figure.dpi':110,'savefig.dpi':150})
FIG=ROOT/'figures/routes';FIG.mkdir(parents=True,exist_ok=True)
def finish(fig,name,note='Dataset del challenge · resultados reproducibles · semilla 20261014'):
    fig.text(.02,.015,note,fontsize=9,color=MUTED)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    plt.show();plt.close(fig)
names={'geometria_base':'Geometría básica','vector_interpretable':'Vector interpretable',
       't2vec_decoder_clasificador':'t2vec + clasificador','hibrido_vector_t2vec':'Vector + t2vec'}
""")

md(r"""
## 1. Before modelling: how many examples do we actually have?

A row is an **annotation**, not a new journey. On top of that, some people appear several times for the same journey. Counting those rows as independent votes would over-weight that person.

We apply a rule that predates training:

1. An annotator counts at most once per journey.
2. Their identical repeated answers collapse into one.
3. If they combine «I don't know» with a single decision, we keep the decision.
4. If the same person says both «the same» and «they differ», their vote is undetermined.
5. Across different people, we keep the disagreement instead of inventing certainty.

We find **123 additional records for the same annotator-journey pair**, with seven contradictory annotator-journey pairs. This does not mean the remaining votes are objective truth.
""")

code("""
counts=df.status.value_counts()
fig,axs=plt.subplots(1,2,figsize=(12,4.6))
labels=['Filas de anotación','Viajes únicos','Con mayoría definida']
values=[audit['n_records'],len(df),int(df.target_hard.notna().sum())]
axs[0].barh(labels[::-1],values[::-1],color=[TEAL,ORANGE,INK])
for y,v in enumerate(values[::-1]):axs[0].text(v+35,y,f'{v:,}'.replace(',','.'),va='center',weight='bold')
axs[0].set_xlim(0,3500);axs[0].set_title('Las filas no son ejemplos independientes')
status_order=['same','different','tie','unknown']
axs[1].bar(['Iguales','Diferentes','Empate','Sin voto válido'],
           [counts.get(k,0) for k in status_order],color=[TEAL,ORANGE,MUTED,RED])
for i,k in enumerate(status_order):axs[1].text(i,counts.get(k,0)+15,str(counts.get(k,0)),ha='center')
axs[1].set_title('Qué etiqueta podemos defender');axs[1].tick_params(axis='x',labelsize=9)
finish(fig,'01_muestra_y_etiquetas')
display(pd.crosstab(df.split,df.status).rename(columns={'same':'Iguales','different':'Diferentes','tie':'Empate','unknown':'Sin voto válido'}))
""")

md(r"""
### 1.1 Are there no nulls, or are there no data problems?

Those are different questions. First we check the original file: missing fields, null JSON values and empty contents. Then we look inside the coordinate lists. A field called «route» can exist and still not contain a usable trajectory.

Each point must have two finite numbers in the order **[latitude, longitude]**, with latitude between −90 and 90 and longitude between −180 and 180. The route also needs at least two distinct positions. Passing these checks guarantees neither GPS accuracy nor that the trace follows a road.

**Scope of this addition:** this EDA was incorporated after the models had been evaluated; we place it here to make the notebook easier to read. Structural quality is audited across the whole file. The distributions we connect to modelling decisions are studied on training data; the comparison across splits is descriptive. No models, labels, filters or thresholds are changed on the basis of this analysis.
""")

code("""
import hashlib
from pyproj import Geod
raw_records=json.loads((ROOT/'data/raw/challenge_dataset.json').read_text())
raw_fields=['journey_id','annotator','annotation','estimated_route','real_route']
missing_report=[]
for field in raw_fields:
    present=[row[field] for row in raw_records if field in row]
    missing_report.append({'Campo':field,'Filas revisadas':len(raw_records),
        'Campo ausente':sum(field not in row for row in raw_records),
        'Valor null':sum(value is None for value in present),
        'Vacío (texto o lista)':sum(isinstance(value,(str,list,dict)) and len(value)==0 for value in present)})
missing_report=pd.DataFrame(missing_report).set_index('Campo')
display(missing_report)

# Una geometría por viaje: las anotaciones repetidas no inflan este recuento.
unique_raw={}
for row in raw_records:
    unique_raw.setdefault(row['journey_id'],row)
geod=Geod(ellps='WGS84')
route_rows=[];quality={}
for field,kind in [('estimated_route','Estimada'),('real_route','Real')]:
    checks={'Rutas revisadas':len(unique_raw),'Ruta ausente, nula o vacía':0,
            'Puntos con estructura inválida':0,'Componentes de coordenada null':0,
            'Puntos no numéricos o no finitos':0,'Puntos fuera de rango':0,
            'Rutas con menos de 2 posiciones distintas':0,'Puntos (0, 0)':0}
    for jid,row in unique_raw.items():
        value=row.get(field)
        if not isinstance(value,list) or not value:
            checks['Ruta ausente, nula o vacía']+=1
            continue
        valid=[]
        for point in value:
            if not isinstance(point,list) or len(point)!=2:
                checks['Puntos con estructura inválida']+=1
                continue
            nulls=sum(x is None for x in point)
            checks['Componentes de coordenada null']+=nulls
            if nulls:continue
            if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in point):
                checks['Puntos no numéricos o no finitos']+=1
                continue
            lat,lon=point
            if abs(lat)>90 or abs(lon)>180:
                checks['Puntos fuera de rango']+=1
                continue
            checks['Puntos (0, 0)']+=int(lat==0 and lon==0)
            valid.append(point)
        coords=np.asarray(valid,float)
        if len(coords)<2 or len(np.unique(coords,axis=0))<2:
            checks['Rutas con menos de 2 posiciones distintas']+=1
            continue
        # Nunca calculamos sobre una ruta reparada silenciosamente.
        if len(valid)!=len(value):continue
        _,_,steps=geod.inv(coords[:-1,1],coords[:-1,0],coords[1:,1],coords[1:,0])
        adjacent=(np.diff(coords,axis=0)==0).all(axis=1)
        feature_length=features.loc[jid,'length_est' if kind=='Estimada' else 'length_real']
        route_rows.append(dict(journey_id=jid,split=df.loc[jid,'split'],kind=kind,
            n_points=len(coords),n_repeated_adjacent=int(adjacent.sum()),n_steps=len(steps),
            length_km=float(feature_length/1000),geodesic_length_km=float(steps.sum()/1000),
            max_step_m=float(steps.max()),median_step_m=float(np.median(steps)),
            points_per_km=float(len(coords)/(feature_length/1000)),
            geometry_hash=hashlib.sha256(coords.tobytes()).hexdigest()))
    quality[kind]=checks
route_eda=pd.DataFrame(route_rows)
quality=pd.DataFrame(quality)
display(quality)
assert len(route_eda)==2*len(df), 'Hay rutas inválidas: revisar antes de modelar, no imputar coordenadas automáticamente.'
feature_checks=pd.Series({
    'Viajes sin objetivo probabilístico (sin votos válidos)':int(df.target_soft.isna().sum()),
    'Viajes sin mayoría binaria (sin votos o empate)':int(df.target_hard.isna().sum()),
    'Valores NaN en las 42 variables geométricas':int(features.isna().to_numpy().sum()),
    'Valores infinitos en las 42 variables geométricas':int(np.isinf(features.to_numpy()).sum()),
    'Variables constantes en entrenamiento':int((features.loc[df.index[df.split=='train']].nunique()<=1).sum())
},name='Recuento')
display(feature_checks.to_frame())
train_ids=df.index[df.split=='train']
train_routes=route_eda[route_eda.split=='train'].copy()
train_features=features.loc[train_ids].join(df[['status','n_unique_annotators','n_votes']])
""")

md(r"""
**Reading:** we find no missing fields, nulls, empty routes or invalid coordinates under these rules. Nor are there missing or infinite values in the computed features. So this file **needs no numerical imputation**.

There is, however, missing supervised information: **139 journeys with no valid vote** and **31 ties**. In total, 170 journeys have no defined binary majority. «I don't know» is a valid human answer, not a null and not a «the same» vote. We do not fill it with the most frequent class. Later on we define how to use the votes we do have.

Variables the schema never contains are missing too: **per-point timestamps, observed duration and a route validated against the road network**. The absence of nulls does not create that information.
""")

md(r"""
### 1.2 How large are these trips?

Let us not confuse the **length of the trip** with the **number of coordinates**. We compute the first by summing the distances between consecutive points of the polyline:

$$L=\sum_{i=1}^{n-1}\|x_{i+1}-x_i\|.$$

The $x_i$ are projected to metres in a local system shared by the pair. This is the length of the trace as represented, not a guarantee of kilometres actually driven. A stretch with no observations is joined by a straight line and may hide curves or a different path altogether.

In the tables and charts that follow, each training journey counts once. We show the **median** and percentiles because a mean can hide the tail of long trips. P95 means that 95 % of routes do not exceed that value.
""")

code("""
length_summary=train_routes.groupby('kind').length_km.agg(
    Viajes='count',Mínimo='min',P25=lambda s:s.quantile(.25),Mediana='median',
    Media='mean',P75=lambda s:s.quantile(.75),P95=lambda s:s.quantile(.95),Máximo='max')
display(length_summary.style.format({c:'{:.2f}' for c in length_summary.columns if c!='Viajes'})
        .set_caption('Longitud en km · solo entrenamiento'))

fig,axs=plt.subplots(1,2,figsize=(12,4.8))
bins=np.geomspace(train_routes.length_km.min()*.95,train_routes.length_km.max()*1.05,26)
for kind,color in [('Estimada',INK),('Real',ORANGE)]:
    values=train_routes.loc[train_routes.kind==kind,'length_km']
    axs[0].hist(values,bins=bins,weights=np.full(len(values),100/len(values)),histtype='step',lw=2,
                color=color,label=f'{kind} · mediana {values.median():.2f} km')
axs[0].set_xscale('log')
axs[0].set_xticks([.5,1,2,5,10,20,50],['0,5','1','2','5','10','20','50'])
axs[0].set(xlabel='Longitud · km (escala logarítmica)',ylabel='Rutas por intervalo · %')
axs[0].set_title('Un mismo problema en escalas diferentes');axs[0].legend(fontsize=9)
status_colors={'same':TEAL,'different':ORANGE,'tie':MUTED,'unknown':MUTED}
for status,label in [('same','Iguales'),('different','Diferentes'),('tie','Empate / sin voto válido'),('unknown',None)]:
    subset=train_features[train_features.status==status]
    axs[1].scatter(subset.length_est/1000,subset.length_real/1000,s=13,alpha=.4,
                   color=status_colors[status],label=label,edgecolors='none',rasterized=True)
lims=[min(train_routes.length_km.min()*.8,.5),train_routes.length_km.max()*1.2]
axs[1].plot(lims,lims,color=MUTED,ls='--',lw=1,label='Misma longitud')
axs[1].set(xscale='log',yscale='log',xlim=lims,ylim=lims,
           xlabel='Longitud estimada · km',ylabel='Longitud real · km')
axs[1].set_title('Mismos kilómetros no implican misma ruta')
axs[1].legend(fontsize=8);axs[1].grid(alpha=.12)
finish(fig,'eda_01_longitudes','Entrenamiento: 1.432 viajes · las escalas logarítmicas son solo para visualizar, no transformaciones nuevas del modelo')

ratio=train_features.length_real/train_features.length_est
display(Markdown(f'**Lectura:** la ruta real tiene mediana **{train_features.length_real.median()/1000:.2f} km** '
    f'y P95 **{train_features.length_real.quantile(.95)/1000:.2f} km**. La razón longitud real/estimada '
    f'va de **{ratio.min():.2f}** a **{ratio.max():.2f}**: existen rutas mucho más cortas o más largas que lo previsto. '
    'La diagonal mezcla etiquetas: recorrer los mismos kilómetros no basta para decidir si se siguieron las mismas calles.'))

extreme_ids=pd.Index([
    train_features.length_est.idxmax(),train_features.length_real.idxmax(),ratio.idxmin(),ratio.idxmax()
]).unique()
extremes=train_features.loc[extreme_ids,['length_est','length_real','status']].copy()
extremes[['length_est','length_real']]/=1000
extremes['Real / estimada']=ratio.loc[extreme_ids]
extremes=extremes.rename(columns={'length_est':'Estimada · km','length_real':'Real · km','status':'Etiqueta disponible'})
extremes['Etiqueta disponible']=extremes['Etiqueta disponible'].map(
    {'same':'Iguales','different':'Diferentes','tie':'Empate','unknown':'Sin voto válido'})
display(extremes.style.format({'Estimada · km':'{:.2f}','Real · km':'{:.2f}','Real / estimada':'{:.2f}'})
        .set_caption('Casos extremos de entrenamiento: máximas longitudes y razones mínima/máxima'))
""")

md(r"""
Extremes deserve inspection, not automatic removal. A large detour may be precisely the example we need to recognise. We apply no percentile trimming and delete no long journeys: doing so would first require telling a data error apart from a valid but uncommon route.
""")

md(r"""
### 1.3 Do more points mean better information?

Two 10 km routes can contain 100 and 1,000 points. That does not make the second one ten times longer, nor does it justify giving it ten times the weight. On top of that, some consecutive points repeat exactly the same coordinate: they describe a zero-length step.

We measure three different things: **number of points**, **consecutive repetitions** and **largest distance between two consecutive observations**. The last one flags stretches represented with little information. The 1 km reference shown below is only an inspection marker; it is not a validated rule for detecting errors and no filter is applied with it.

**This is not just a theoretical example:** in 2,376 of the 2,386 journeys — **99.6 %** — the two traces have a different number of points. The median is 249.5 points for the estimated route and 198 for the real one, across the whole file. That they almost never match exactly does not mean every difference is large: this is why we also look at the magnitude of the ratio between the two counts.
""")

code("""
point_counts=df[['n_estimated','n_real']].copy()
point_counts['ratio']=point_counts.n_real/point_counts.n_estimated
different_counts=int((point_counts.n_estimated!=point_counts.n_real).sum())
assert different_counts==2376
fig,axs=plt.subplots(1,2,figsize=(12,4.6))
axs[0].scatter(point_counts.n_estimated,point_counts.n_real,s=12,alpha=.25,color=TEAL,rasterized=True)
lims=[min(point_counts.n_estimated.min(),point_counts.n_real.min())*.8,
      max(point_counts.n_estimated.max(),point_counts.n_real.max())*1.15]
axs[0].plot(lims,lims,color=RED,ls='--',label='Mismo número de puntos')
axs[0].set(xscale='log',yscale='log',xlim=lims,ylim=lims,
    xlabel='Puntos de la ruta estimada · escala log',ylabel='Puntos de la ruta real · escala log')
axs[0].set_title('Cada punto del gráfico es un viaje');axs[0].legend(fontsize=9)
bins=np.geomspace(point_counts.ratio.min(),point_counts.ratio.max(),36)
axs[1].hist(point_counts.ratio,bins=bins,color=ORANGE,alpha=.85)
axs[1].axvline(1,color=RED,ls='--');axs[1].set_xscale('log')
axs[1].set(xlabel='Número de puntos real / estimada · escala log',ylabel='Viajes')
axs[1].set_title('Distinto recuento no siempre es gran diferencia')
finish(fig,'eda_02a_puntos_por_pareja',f'Archivo completo · {different_counts:,} de {len(df):,} parejas con distinto recuento · sin usar etiquetas')

sampling_example='7f29deb0-5c8c-11ec-8622-eff6f2c898b6'
sample_a,sample_b=project_pair(*routes[sampling_example])
assert len(sample_a)==323 and len(sample_b)==97
fig,axs=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
both=np.concatenate([sample_a,sample_b])/1000
low=both.min(axis=0);high=both.max(axis=0);padding=np.maximum((high-low)*.07,.05)
for ax,line,other,label,color in zip(axs,[sample_a,sample_b],[sample_b,sample_a],['Estimada','Real'],[INK,ORANGE]):
    ax.plot(other[:,0]/1000,other[:,1]/1000,color=MUTED,alpha=.3,lw=3,label='Otro trazado, como referencia')
    ax.plot(line[:,0]/1000,line[:,1]/1000,color=color,lw=1,alpha=.7)
    ax.scatter(line[:,0]/1000,line[:,1]/1000,s=13,color=color,alpha=.75,label='Puntos del archivo')
    ax.set_title(f'{label}: {len(line)} puntos')
    ax.set(xlim=(low[0]-padding[0],high[0]+padding[0]),ylim=(low[1]-padding[1],high[1]+padding[1]),
           xlabel='Este local · km',ylabel='Norte local · km')
    ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8)
finish(fig,'eda_02b_ejemplo_323_97','Ejemplo descriptivo elegido por recuentos próximos a 300 y 100 · mismos ejes · no elegido por el resultado del modelo')
display(Markdown(f'El viaje `{sampling_example}` tiene **323 puntos estimados y 97 reales**. '
    f'Las polilíneas miden **{features.loc[sampling_example,"length_est"]/1000:.2f} km** y '
    f'**{features.loc[sampling_example,"length_real"]/1000:.2f} km**, respectivamente. '
    'El recuento, la longitud y la similitud son propiedades distintas. Los marcadores superpuestos pueden ocultar '
    'coordenadas repetidas. Este ejemplo ilustra el muestreo, sin afirmar que ambas rutas sean idénticas.'))

sampling_rows=[]
for kind,group in route_eda.groupby('kind',sort=False):
    train=group[group.split=='train']
    sampling_rows.append({'Ruta':kind,'Puntos por ruta · mediana train':train.n_points.median(),
        'Puntos por km · mediana train':train.points_per_km.median(),
        'Pasos de longitud cero · archivo completo':int(group.n_repeated_adjacent.sum()),
        '% pasos de longitud cero · archivo completo':100*group.n_repeated_adjacent.sum()/group.n_steps.sum(),
        'Rutas con algún paso >1 km · archivo completo':int((group.max_step_m>1000).sum())})
sampling_summary=pd.DataFrame(sampling_rows).set_index('Ruta')
display(sampling_summary.style.format({'Puntos por ruta · mediana train':'{:.1f}',
    'Puntos por km · mediana train':'{:.1f}','% pasos de longitud cero · archivo completo':'{:.2f} %'}))

fig,axs=plt.subplots(1,2,figsize=(12,4.6))
def ecdf(ax,values,color,label):
    values=np.sort(np.asarray(values))
    ax.plot(values,np.arange(1,len(values)+1)/len(values)*100,color=color,label=label,lw=2)
for kind,color in [('Estimada',INK),('Real',ORANGE)]:
    subset=train_routes[train_routes.kind==kind]
    ecdf(axs[0],subset.points_per_km,color,kind)
    ecdf(axs[1],subset.max_step_m,color,kind)
for ax in axs:
    ax.set_xscale('log');ax.set_ylabel('Rutas acumuladas · %');ax.set_ylim(0,101);ax.legend();ax.grid(alpha=.15)
axs[0].set_xlabel('Puntos registrados por km · escala log')
axs[0].set_title('La densidad de puntos no es uniforme')
axs[1].set_xlabel('Mayor paso entre puntos de una ruta · m')
axs[1].set_title('Una ruta densa también puede tener huecos')
axs[1].axvline(1000,color=RED,ls='--',lw=1)
axs[1].text(1100,18,'1 km: referencia\\nde inspección',color=RED,fontsize=9)
finish(fig,'eda_02_muestreo','Solo entrenamiento · una curva acumulada al 80 % indica el valor que no supera el 80 % de las rutas')

# Ilustramos el mayor paso entre observaciones de una ruta real de train.
worst=train_routes[train_routes.kind=='Real'].sort_values('max_step_m',ascending=False).iloc[0]
eda_jid=worst.journey_id
eda_a,eda_b=project_pair(*routes[eda_jid])
raw_b=routes[eda_jid][1]
_,_,eda_steps=geod.inv(raw_b[:-1,1],raw_b[:-1,0],raw_b[1:,1],raw_b[1:,0])
largest=int(np.argmax(eda_steps))
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
axs[0].plot(eda_a[:,0]/1000,eda_a[:,1]/1000,color=INK,lw=2,label='Estimada')
axs[0].plot(eda_b[:,0]/1000,eda_b[:,1]/1000,color=ORANGE,lw=1.3,label='Real: puntos unidos')
axs[0].scatter(eda_b[:,0]/1000,eda_b[:,1]/1000,s=4,color=ORANGE,alpha=.35)
segment=eda_b[largest:largest+2]/1000
axs[0].plot(segment[:,0],segment[:,1],color=RED,lw=3,label=f'Paso sin puntos intermedios: {eda_steps[largest]/1000:.2f} km')
axs[0].scatter(segment[:,0],segment[:,1],facecolors=BG,edgecolors=RED,s=45,zorder=5)
axs[0].set_aspect('equal',adjustable='datalim')
axs[0].set(xlabel='Este local · km',ylabel='Norte local · km')
axs[0].set_title('La línea dibujada no es una observación')
axs[0].legend(fontsize=8)
axs[1].plot(np.arange(1,len(eda_steps)+1),eda_steps/1000,color=ORANGE,lw=1)
axs[1].scatter(largest+1,eda_steps[largest]/1000,color=RED,s=40,zorder=4)
axs[1].set(xlabel='Índice del paso entre puntos · no tiempo',ylabel='Distancia entre observaciones · km')
axs[1].set_title(f'{int(worst.n_points)} puntos, pero un paso de {worst.max_step_m/1000:.1f} km')
finish(fig,'eda_03_hueco_trazado',f'Entrenamiento · ID {eda_jid} · elección: mayor paso de una ruta real · sin timestamps')

display(Markdown('**Lo que implica:** no sabemos si este hueco procede de simplificar el trazado, perder registros '
    'o de otro mecanismo. Unir extremos y remuestrear crea posiciones interpoladas, no recupera las calles recorridas. '
    'Por eso una longitud, IoU o distancia geométrica puede depender de la resolución del archivo, además de la conducción.'))
""")

md(r"""
In the geometry as implemented we remove **consecutive** repetitions before forming the lines. We keep later revisits to the same place, which may represent a real loop. Means and proportions are computed over positions equally spaced by length, not over the raw points. This reduces the artificial weight of densely recorded areas, but **it does not reconstruct lost information**.

In t2vec we also resample by length, with a maximum of 64 positions, and collapse identical consecutive cells. That cap controls sequence size, although it can lose detail on long routes. Without timestamps we cannot read the repetitions as minutes spent stationary, nor compute speed.
""")

md(r"""
### 1.4 Are we representing the same problem across all splits?

The sample mixes geographic areas and degrees of human certainty. We check both before interpreting any global figure.

The area is not an original field: it is an **approximation** based on where the real route starts and how close that is to a set of predefined city centres, with a maximum radius of 65 km. It is not a validated municipality. We pick the five most frequent areas in training and group the rest so the comparison stays readable.
""")

code("""
split_names={'train':'Entrenamiento','val':'Validación','test':'Test'}
top_regions=df.loc[train_ids,'region'].value_counts().head(5).index.tolist()
geo_labels=df.region.where(df.region.isin(top_regions),'Resto')
geo_counts=pd.crosstab(df.split,geo_labels).reindex(index=['train','val','test'],columns=top_regions+['Resto'],fill_value=0)
geo_pct=geo_counts.div(geo_counts.sum(axis=1),axis=0)*100
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
left=np.zeros(3)
palette=[TEAL,ORANGE,INK,'#8AA7A0','#BA9575','#CBD3D2']
for region,color in zip(geo_pct.columns,palette):
    values=geo_pct[region].to_numpy()
    axs[0].barh(np.arange(3),values,left=left,color=color,label=region,height=.62)
    for i,v in enumerate(values):
        if v>=10:axs[0].text(left[i]+v/2,i,f'{v:.0f} %',ha='center',va='center',
                             color='white' if color in palette[:3] else INK,fontsize=9)
    left+=values
axs[0].set_yticks(np.arange(3),[split_names[s] for s in geo_pct.index])
axs[0].invert_yaxis();axs[0].set_xlim(0,100);axs[0].set_xlabel('Viajes de cada conjunto · %')
axs[0].set_title('Parecida mezcla no garantiza rutas nuevas')
axs[0].legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18))
voters=df.loc[train_ids,'n_unique_annotators'].value_counts().sort_index()
axs[1].bar(voters.index.astype(str),voters.values,color=[ORANGE]+[TEAL]*(len(voters)-1))
for i,v in enumerate(voters.values):axs[1].text(i,v+12,str(v),ha='center')
axs[1].set_ylim(0,voters.max()*1.16)
axs[1].set(xlabel='Personas distintas por viaje · incluye abstenciones',ylabel='Viajes de entrenamiento')
axs[1].set_title('La mayoría suele ser una sola persona')
finish(fig,'eda_04_mezcla_y_anotadores','Zonas aproximadas · orden y categorías elegidos en entrenamiento · comparación entre conjuntos solo descriptiva')

split_rows=[]
for split in ['train','val','test']:
    ids=df.index[df.split==split];part=df.loc[ids];f=features.loc[ids]
    real=route_eda[(route_eda.split==split)&(route_eda.kind=='Real')]
    split_rows.append({'Conjunto':split_names[split],'Viajes':len(part),
        'Longitud real mediana · km':f.length_real.median()/1000,
        'Longitud real P95 · km':f.length_real.quantile(.95)/1000,
        '% con mayoría definida':100*part.target_hard.notna().mean(),
        '% diferentes entre mayorías':100*part.target_hard.dropna().mean(),
        '% rutas reales con paso >1 km':100*(real.max_step_m>1000).mean()})
split_summary=pd.DataFrame(split_rows).set_index('Conjunto')
display(split_summary.style.format({c:'{:.2f}' for c in split_summary.columns if c!='Viajes'}))

duplicate_geometry={kind:int(group.geometry_hash.duplicated().sum()) for kind,group in route_eda.groupby('kind')}
display(Markdown(f'**Comprobación adicional:** {duplicate_geometry["Estimada"]} rutas estimadas y '
    f'{duplicate_geometry["Real"]} rutas reales son copias exactas adicionales de la secuencia de otro viaje. '
    'La ausencia de copias exactas no descarta compartir calles o corredores. Esta partición aleatoria mide rendimiento '
    'dentro de una mezcla geográfica conocida; para evaluar una ciudad nueva habría que separarla expresamente.'))

quartiles=pd.qcut(train_features.length_real/1000,4)
length_label=pd.crosstab(quartiles,train_features.status).reindex(columns=['same','different','tie','unknown'],fill_value=0)
length_label['Con mayoría']=length_label['same']+length_label['different']
length_label['% diferentes entre mayorías']=100*length_label['different']/length_label['Con mayoría']
length_label['Sin mayoría']=length_label['tie']+length_label['unknown']
display(length_label[['Con mayoría','% diferentes entre mayorías','Sin mayoría']].style
        .format({'% diferentes entre mayorías':'{:.1f} %'})
        .set_caption('Entrenamiento · cuatro grupos por longitud real (km) · asociación descriptiva, no causal'))
""")

md(r"""
The class proportions are similar because the split was stratified by the initial state of the labels. Frequent areas appear in all three splits, but that does not remove dependence between trips that share streets. Besides, most journeys have only one annotator: a 0 or 1 label does not imply unanimity across a broad panel.

Grouping by length helps detect whether the label is associated with trip size. It does not turn that association into a universal rule or into a criterion for dropping observations. We use it as context when interpreting the metrics.
""")

md(r"""
### 1.5 How many people hold up each label?

The previous subsection counted **people who looked at** each journey. Here we count **decided votes**: abstentions
(«I don't know») and annotators who contradict themselves do not add up. This is the number that actually holds up each label.

It matters because the word «majority» suggests a panel and here there almost never is one. A journey with a single vote has a
label every bit as binary as one with three, but **we cannot know whether a second person would have agreed**.
That difference shows up in no aggregate metric; we recover it in section 8.4.
""")

code("""
votes_total=df.n_votes.value_counts().sort_index()
votes_status=pd.crosstab(df.n_votes,df.status).reindex(columns=['same','different','tie','unknown'],fill_value=0)
fig,axs=plt.subplots(1,2,figsize=(12,4.6))
colors=[RED]+[ORANGE]+[TEAL]*(len(votes_total)-2)
axs[0].bar(votes_total.index.astype(str),votes_total.values,color=colors)
for i,v in enumerate(votes_total.values):
    axs[0].text(i,v+28,f'{v:,}'.replace(',','.'),ha='center',weight='bold',fontsize=10)
    axs[0].text(i,v*.5 if v>200 else v+150,f'{v/len(df):.0%}',ha='center',fontsize=9,
                color='white' if v>200 else MUTED)
axs[0].set_ylim(0,votes_total.max()*1.18)
axs[0].set(xlabel='Votos decididos por viaje · sin «no sé» ni contradicciones',ylabel='Viajes')
axs[0].set_title('Tres de cada cuatro etiquetas las sostiene una persona')
rows=votes_status.loc[[1,2,3]] if 3 in votes_status.index else votes_status
share=rows.div(rows.sum(axis=1),axis=0)*100
left=np.zeros(len(rows))
for col,color,label in [('same',TEAL,'Mayoría: iguales'),('different',ORANGE,'Mayoría: diferentes'),
                        ('tie',MUTED,'Empate, sin mayoría')]:
    values=share[col].to_numpy()
    axs[1].barh(np.arange(len(rows)),values,left=left,color=color,label=label,height=.6)
    for i,v in enumerate(values):
        if v>=8:axs[1].text(left[i]+v/2,i,f'{v:.0f} %',ha='center',va='center',color='white',fontsize=9)
    left+=values
axs[1].set_yticks(np.arange(len(rows)),[f'{k} voto' if k==1 else f'{k} votos' for k in rows.index])
axs[1].invert_yaxis();axs[1].set_xlim(0,100)
axs[1].set_xlabel('Viajes con ese número de votos · %')
axs[1].set_title('El desacuerdo solo es observable con dos o más')
axs[1].legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18))
finish(fig,'eda_05_votos_por_ruta','Archivo completo · votos ya deduplicados por anotador y viaje')

ties=int((df.status=='tie').sum());multi=int((df.n_votes>=2).sum())
def miles(n):return f'{n:,}'.replace(',','.')
display(Markdown(
    f'**Lo que se puede y no se puede medir.** {miles(int(votes_total.get(1,0)))} viajes tienen un solo voto: en ellos '
    f'el acuerdo entre personas es **inobservable**. Entre los {miles(multi)} con dos o más, {ties} terminan en empate '
    f'({ties/multi:.1%}), y {int(df.n_conflicted_annotators.sum())} anotadores se contradicen sobre un mismo viaje. '
    f'Hay además {miles(int(df.n_unknown.sum()))} abstenciones repartidas en {miles(int((df.n_unknown>0).sum()))} viajes. '
    'El enunciado pide expresamente una valoración «según su propia percepción»: no existe un criterio fijo de '
    'cuántos metros de desvío convierten dos rutas en diferentes. El desacuerdo es una propiedad del problema, '
    'no un defecto de los datos.'))
""")

md(r"""
#### And who votes? The panel is eight people, very unevenly matched

If a label is almost always held up by one person, it is worth knowing **which person**. The panel has eight annotators,
but they do not share the work evenly: the most active one made 44 times as many annotations as the least active.

Two things interest us. **How much each one weighs** — especially on journeys where nobody else voted, because there their criterion
is the label with no counterweight — and **how alike they are** when they overlap on the same journey.
""")

code("""
from itertools import combinations
from collections import defaultdict, Counter
by_journey=defaultdict(lambda: defaultdict(set))
for row in raw_records: by_journey[row['journey_id']][row['annotator']].add(row['annotation'])
decided=lambda labels: labels-{"I don't know"}

annotator_rows=[]
for ann in sorted({row['annotator'] for row in raw_records}):
    mine=[row for row in raw_records if row['annotator']==ann]
    journeys={row['journey_id'] for row in mine}
    sole=sum(1 for j in journeys if set(by_journey[j])=={ann})
    conflicted=sum(1 for j in journeys if len(decided(by_journey[j][ann]))>1)
    counts=Counter(row['annotation'] for row in mine)
    annotator_rows.append({'Anotador':ann,'Anotaciones':len(mine),'Viajes':len(journeys),
        'Único anotador':sole,'Acompañado':len(journeys)-sole,
        '% «diferentes»':100*counts['They differ']/len(mine),
        '% «no sé»':100*counts["I don't know"]/len(mine),
        'Se contradice':conflicted})
panel=pd.DataFrame(annotator_rows).set_index('Anotador')

agreement=defaultdict(lambda:[0,0])
for jid,anns in by_journey.items():
    votes={a:int('They differ' in labels) for a,labels in anns.items() if len(decided(labels))==1}
    for x,y in combinations(sorted(votes),2):
        agreement[(x,y)][0]+=1
        agreement[(x,y)][1]+=int(votes[x]==votes[y])
MIN_PAIRS=15
people=sorted(panel.index)
matrix=np.full((len(people),len(people)),np.nan)
# El acuerdo bruto premia a los pares que casi siempre dicen lo mismo. Kappa descuenta la coincidencia por azar.
kappa_rows=[]
for (x,y),(n,ok) in agreement.items():
    if n<MIN_PAIRS: continue
    matrix[people.index(x),people.index(y)]=matrix[people.index(y),people.index(x)]=100*ok/n
    votes=[(int('They differ' in by_journey[j][x]),int('They differ' in by_journey[j][y]))
           for j,anns in by_journey.items()
           if len(decided(anns.get(x,set())))==1 and len(decided(anns.get(y,set())))==1]
    v=np.array(votes);po=(v[:,0]==v[:,1]).mean();p1,p2=v[:,0].mean(),v[:,1].mean()
    pe=p1*p2+(1-p1)*(1-p2)
    kappa_rows.append({'Par':f'{x} y {y}','Viajes compartidos':n,'Acuerdo':100*po,
                       'Kappa de Cohen':(po-pe)/(1-pe)})
kappa=pd.DataFrame(kappa_rows).sort_values('Kappa de Cohen').set_index('Par')

fig,axs=plt.subplots(1,2,figsize=(13,5))
order=panel.sort_values('Anotaciones').index
axs[0].barh([str(a) for a in order],panel.loc[order,'Único anotador'],color=ORANGE,label='Único anotador del viaje')
axs[0].barh([str(a) for a in order],panel.loc[order,'Acompañado'],left=panel.loc[order,'Único anotador'],
            color=TEAL,label='Alguien más también votó')
for i,a in enumerate(order):
    axs[0].text(panel.loc[a,'Viajes']+14,i,f"{panel.loc[a,'Viajes']}",va='center',fontsize=9,weight='bold')
axs[0].set_xlim(0,panel.Viajes.max()*1.14)
axs[0].set(xlabel='Viajes anotados',ylabel='Anotador')
axs[0].set_title('Una persona define en solitario el 26 % del archivo')
axs[0].legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.16))
im=axs[1].imshow(matrix,cmap='BuGn',vmin=70,vmax=100)
axs[1].set_xticks(range(len(people)),people);axs[1].set_yticks(range(len(people)),people)
for i in range(len(people)):
    for j in range(len(people)):
        if i==j: axs[1].text(j,i,'—',ha='center',va='center',color=MUTED)
        elif np.isnan(matrix[i,j]): axs[1].text(j,i,'·',ha='center',va='center',color=MUTED,fontsize=14)
        else: axs[1].text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=9,
                          color='white' if matrix[i,j]>92 else INK)
axs[1].set(xlabel='Anotador',ylabel='Anotador')
axs[1].set_title(f'Acuerdo cuando coinciden · % · «·» = menos de {MIN_PAIRS} viajes')
fig.colorbar(im,ax=axs[1],fraction=.046,label='Acuerdo · %')
finish(fig,'eda_06_panel_anotadores',
       'Archivo completo · el acuerdo se calcula solo sobre viajes que ambos votaron con decisión')

display(panel[['Anotaciones','Viajes','Único anotador','% «diferentes»','% «no sé»','Se contradice']]
        .style.format({'% «diferentes»':'{:.1f} %','% «no sé»':'{:.1f} %'}))
display(Markdown('**Acuerdo descontando el azar.** Dos personas que digan «diferentes» casi siempre coincidirán '
 'mucho sin compartir criterio. Kappa corrige eso: 1 es acuerdo perfecto, 0 es el que se esperaría al azar.'))
display(kappa.style.format({'Acuerdo':'{:.1f} %','Kappa de Cohen':'{:.3f}'}))
""")

md(r"""
**The distribution is not balanced.** Annotator 2 signs 918 annotations — 30 % of the file — across 856 journeys, and on
**612 of them is the only person who voted**: their criterion, uncontrasted, fixes 26 % of the dataset's labels.
Adding the three most active (2, 7 and 1) reaches 1,335 of the 1,897 single-annotator journeys. At the other end,
annotator 5 made 21 annotations.

**And personal criteria differ.** Annotator 1 marks «they differ» 65 % of the time; annotator 7, 39.7 %. That is
25 points of difference between two people who saw nearly 500 and more than 600 journeys respectively. It is not a small
sample nor a coincidence: it is a different tolerance threshold.

That gap is confirmed when we cross them. Among the pairs with enough shared journeys, **1 and 7 are precisely the ones who
agree least: 82.6 % over 46 journeys**, while other pairs reach 94-100 %. Cells marked with «·»
have fewer than 15 journeys in common and are not interpreted; with 4 or 8 comparisons, a 100 % means nothing.

**The consequence for the model.** The target is not «do these routes look alike?» but «would this particular person say
they look alike?». In 80 % of cases that person is a single one, and with high probability it is annotator 2 or 7.
The model learns, in part, **their** tolerance threshold. If the panel changes tomorrow, the target changes, and the accuracy
measured here does not transfer automatically.

**Raw agreement overstates the case**, because two people who say «they differ» most of the time will coincide a lot
without sharing a criterion. Cohen's kappa discounts that expected coincidence and ranks the pairs
differently: 1 and 7 drop to **0.66**, and 1 and 8 to **0.63**, against several pairs above 0.87. These are values
usually described as «substantial» agreement, not «almost perfect»: there is a shared criterion, but with real slack.

This reinforces what §1.5 and §8.4 already said from another angle: the dominant limitation of this prototype is not the
model's capacity, but **how many people hold up each label and with what criterion**. §8.6 closes the
question from the other side: which annotator the model we trained resembles most.
""")

md(r"""
### 1.6 So, do we need to normalise?

**Yes to making distances and sampling comparable; no to dividing everything by length.** They are different operations:

| What we want to solve | Treatment and reason | Prototype status |
|---|---|---|
| Latitude and longitude are in degrees | Project both routes to a shared local system in metres. One degree of longitude is not the same distance at every latitude. | Implemented for the geometric features |
| Routes have different point densities | Resample by length and remove consecutive duplicates when building the polyline. That way a densely sampled area does not weigh more just for having more points. | Implemented; it does not fill gaps with real information |
| A detour can be large in metres or relative to the trip | Keep measurements in metres **and** versions divided by a shared length scale. | The extended vector includes both; the baseline includes lengths, absolute distances and relative length difference |
| Variables with very different units enter a neural network | Standardise using the mean and standard deviation computed on training only. Apply those same parameters to validation and test. | Implemented in the neural head over embeddings; afterwards clipped to [−10, 10] |
| Variables with different units enter XGBoost | Its trees compare each variable against its own thresholds; they do not need z-scores to put metres and proportions on a common scale. | Geometric and hybrid inputs are not standardised |
| Proportions and IoU are already dimensionless | Keep their values between 0 and 1; do not divide by kilometres again. | Implemented |
| There is a tail of long trips | Use logarithmic axes to see it. Transforming variables or trimming extremes would be a separate decision, to be validated. | The log scale in the EDA is purely visual; no inputs were changed and no extremes removed |

The common scale we use is

$$s=\max\left(\frac{L_e+L_r}{2},1\text{ m}\right),\qquad
d_{\mathrm{rel}}=\frac{d}{s},\qquad
\Delta L_{\mathrm{rel}}=\frac{L_r-L_e}{s}.$$

The 1-metre floor protects the division in degenerate cases; none of the valid routes in this file comes close to it. The value $s$ is computed from the two routes of the journey itself, without learning from other journeys or their labels.

For example, a mean separation of **100 m** is **10 %** of a **1 km** scale, and **0.5 %** of a **20 km** scale. These are complementary readings: the second avoids treating a proportionally small detour the same as a large one; the first preserves the fact that 100 metres can be a different street in either case. **If we kept only the relative version, we could hide a difference a person considers important.**

We do not centre, rotate or rescale each route separately to unit size: that could make trips that genuinely pass through different places look identical. The shared projection preserves the separation between the two.

For the neural head, by contrast, standardising a component $x_k$ means

$$z_k=\frac{x_k-\mu_{k,\mathrm{train}}}{\sigma_{k,\mathrm{train}}}.$$

This is a rescaling of the network's inputs, not a normalisation of kilometres. Means and standard deviations are not recomputed on validation or test, and constant components are handled without dividing by zero. The vocabulary and the encoder are also fitted exclusively within the training set.

**What we take from the EDA:** no null imputation is needed; units and sampling density do need respecting, absolute and relative scales both need keeping, and we should remember that trace gaps and the scarcity of votes limit what we can learn. Any further cleaning or change of representation would be evaluated as a new experiment, without silently replacing the results that follow.
""")

code("""
# Informe ligero para reproducir las cifras sin extraerlas de las figuras.
eda_report={
 'scope':'Calidad: archivo completo. Distribuciones y relación con etiquetas: train. Comparación de splits: descriptiva.',
 'added_after_model_evaluation':True,
 'raw_rows':len(raw_records),'unique_journeys':len(df),
 'missing_fields':missing_report.to_dict(orient='index'),
 'coordinate_quality':quality.to_dict(),
 'derived_quality':feature_checks.to_dict(),
 'train_length_km':length_summary.to_dict(orient='index'),
 'sampling':sampling_summary.to_dict(orient='index'),
 'paired_point_counts':{'different':different_counts,'total':len(df),
     'estimated_median':float(point_counts.n_estimated.median()),'real_median':float(point_counts.n_real.median()),
     'example_journey_id':sampling_example,'example_estimated_points':323,'example_real_points':97},
 'split_summary':split_summary.to_dict(orient='index'),
 'exact_duplicate_routes':duplicate_geometry,
 'largest_real_train_step':{'journey_id':eda_jid,'step_m':float(worst.max_step_m),'n_points':int(worst.n_points)},
 'transformations_changed':False}
(R/'eda.json').write_text(json.dumps(eda_report,indent=2,ensure_ascii=False,default=int))
""")

md(r"""
## 2. The target represents human judgement, uncertainty included

For a journey $j$, we use

$$p_j=\frac{\text{valid annotators saying «they differ»}}{\text{valid annotators who make a decision}}.$$

If two people say «they differ» and one says «the same», the target is $2/3$. We do not claim this is the physical probability that the routes are different: it is an estimate of this panel's judgement. With a single vote there is uncertainty too, even though $p_j$ is 0 or 1.

We train with cross-entropy against $p_j$, with **equal weight per journey**. Ties contribute a target of 0.5. Journeys with no valid votes contribute no supervised loss; their training routes can still take part in unlabelled pretraining.

To evaluate binary classification we use the defined majority, $y_j=1[p_j>0.5]$, excluding ties and cases with no valid vote. We also measure the probabilistic loss against the votes, ties included.

> The test set contains 477 journeys, but only 443 have a defined majority. Binary performance refers to **92.9 %** of the test set. We do not know the accuracy on the other 34 journeys.

All annotations for a journey stay together. There are no exactly duplicated geometry pairs across identifiers. The split was frozen before evaluating models and kept when cleaning the votes. This is a random test within the mix of cities; **it does not demonstrate generalisation to a new city**.
""")

code("""
fig,axs=plt.subplots(1,3,figsize=(12,4.3))
for ax,status,title in zip(axs,['same','different','unknown'],['Mayoría: iguales','Mayoría: diferentes','Sin decisión válida']):
    ids=df.index[(df.split=='train')&(df.status==status)]
    # Caso mediano en longitud: selección determinista, no por acierto del modelo.
    jid=features.loc[ids,'length_real'].sort_values().index[len(ids)//2]
    a,b=project_pair(*routes[jid])
    ax.plot(a[:,0]/1000,a[:,1]/1000,color=INK,lw=2.5,label='Estimada')
    ax.plot(b[:,0]/1000,b[:,1]/1000,color=ORANGE,lw=1.7,label='Real')
    ax.set_title(title);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Este local · km')
    ax.set_ylabel('Norte local · km');ax.legend(fontsize=9)
finish(fig,'02_ejemplos_reales','Ejemplos del entrenamiento · coordenadas proyectadas localmente · sin mapa base')
""")

md(r"""
## 3. The first representation: measure before compressing

Coordinates are projected to metres in a local system shared by **both** routes. We do not centre, rotate or scale each one separately: that could turn different streets into apparently identical shapes.

For a position $x$ on route $A$ we compute $d(x,B)=\min_{y\in B}\|x-y\|$. The nearest position may lie **in the middle of a segment**, even if no GPS point was recorded there.

We repeat this in both directions. Means and proportions are approximated using positions equally spaced by length (every 20 metres, minimum 16 and maximum 1,600). That way each metre carries roughly the same weight. The precision of this approximation is limited on very long routes or with very short detours.

**Baseline:** eight variables — lengths, relative difference, endpoint separation, directional mean distances and discrete Hausdorff — plus XGBoost.

**The extended vector:** 42 variables. It adds 95th percentiles, relative versions, proportions and continuous stretches further than 20/50/100 metres away, corridor IoU, straightness, turns and discrete Fréchet over 64 points per route.

Three examples clarify what each measurement adds. **The mean** describes the usual separation; **the maximum of the minimum distances in both directions — Hausdorff —** can detect a significant detour even if brief, but is sensitive to a single anomalous point. **A proportion of 0.20 beyond 50 metres** means roughly a fifth of that route's length lies far from the other, even if the mean is small.

For **IoU** we draw a corridor around each route and divide the intersection area by the union area. Near 1 there is a lot of overlap; near 0, little. Width matters: 100-metre corridors can overlap even when the cars drive down different streets. This measurement does not consult the road network.

$$\Delta L_{\mathrm{rel}}=\frac{L_{\mathrm{real}}-L_{\mathrm{estimated}}}{(L_{\mathrm{real}}+L_{\mathrm{estimated}})/2}.$$

We keep absolute and relative values. IoU is already normalised; we do not divide it by length again.

**Where does XGBoost learn?** These measurements are computed without learning from the labels. Afterwards, the model combines decision trees learned from the votes: a 50-metre separation can be read differently depending on length, endpoints and the other variables. We do not hand-impose a single rule of «if it exceeds 50 metres, they are different».

> Sharing all the points does not imply traversing them in the same order. Hausdorff and IoU can ignore that difference; Fréchet adds ordering information. Their discrete versions are approximations, not exact measurements of a continuous curve.
""")

code("""
from routes.features import features_xy
# Comprobación conceptual: dos líneas idénticas pero recorridas al revés.
a=np.array([[0.,0.],[1000.,0.]])
demo=features_xy(a,a[::-1])
display(pd.DataFrame([{'Hausdorff (m)':demo['hausdorff_discrete_m'],
                      'IoU corredor 50 m':demo['buffer_iou_50m'],
                      'Fréchet discreto (m)':demo['frechet_64_m']}]))

from shapely import distance,points
ids=df.index[(df.split=='train')&(df.status=='different')]
jid=(features.loc[ids,'mean_real_to_est_m']-100).abs().sort_values().index[0]
a,b=project_pair(*routes[jid]);la,lb=clean_line(a),clean_line(b)
positions=sample_line(lb,400,midpoints=True)
distances=distance(points(positions),la)
fig,axs=plt.subplots(1,2,figsize=(12,4.2))
axs[0].plot(a[:,0],a[:,1],color=INK,lw=2,label='Estimada')
axs[0].plot(b[:,0],b[:,1],color=ORANGE,lw=1.5,label='Real')
axs[0].set_aspect('equal',adjustable='datalim');axs[0].legend();axs[0].set_title('Una pareja de rutas de entrenamiento')
axs[0].set_xlabel('Este local · m');axs[0].set_ylabel('Norte local · m')
axs[1].plot((np.arange(400)+.5)/400*100,distances,color=ORANGE)
axs[1].axhline(50,color=TEAL,ls='--',label='50 m')
axs[1].fill_between((np.arange(400)+.5)/400*100,distances,where=distances>50,color=ORANGE,alpha=.2)
axs[1].set_xlabel('Porcentaje recorrido de la ruta real');axs[1].set_ylabel('Distancia a la estimada · m')
axs[1].set_title('Del trazado a un perfil interpretable');axs[1].legend()
finish(fig,'03_perfil_de_desviacion')
""")

from graph_notebook_sections import setup_cells, result_cells
cells.extend(setup_cells())

md(r"""
## 5. Learning a vector: what t2vec means in this prototype

### 5.1 The starting problem: a list of points has no fixed size

One route can have 100 points and another 300. **This happens in the dataset:** 2,376 of the 2,386 journeys have a different number of points between their two traces. For instance, journey `7f29deb0-5c8c-11ec-8622-eff6f2c898b6` contains 323 estimated points and 97 real ones. More points does not mean more kilometres or a different route: it can simply mean denser sampling.

We want a function $f_\theta$ that turns any admissible sequence into a vector of the same size:

$$T=(p_1,\ldots,p_n)\quad\longmapsto\quad z=f_\theta(T)\in\mathbb R^d.$$

Here $n$ can change, while $d$ is fixed when designing the network. The parameter $\theta$ stands for the weights we learn. Having the same size lets us compare vectors and feed a classifier. **It does not guarantee those vectors preserve the differences the business cares about:** that has to be checked.

There are three distinct levels: the GPS coordinate; the map cell we assign that coordinate to; and the learned vector for the whole trajectory. There is also a learned vector per cell, which is the encoder's numerical input. That local vector should not be confused with the final summary of the trip.

**What the encoder does as it reads:** it starts with a numerical memory, reads A and updates it; then reads C and combines what it remembers with C; then does the same with D. When it finishes it returns that memory. Our GRU reads in both directions and concatenates its summaries. It does not assign one component per point: the whole sequence influences a fixed-size vector. A sequence of 3 cells and one of 30 both produce 64 numbers. The 64-point preprocessing cap and the 64 vector components are two separate decisions that happen to coincide numerically here.

It is worth laying out the three components before going further:

| Component | Input | Output | What it is used for |
|---|---|---|---|
| Encoder | One route, expressed as a sequence of cells | A vector of 64 numbers | Summarise the route |
| Reconstruction decoder | That vector and the context of previous cells | Predictions of successive cells, including END | Provide a signal to train the encoder |
| Pair classifier | The two vectors and their comparisons | Probability of «they differ» | Solve the challenge task |

**The decoder is not the classifier.** It is trained in the first phase and takes no part in the final label prediction.

### 5.2 The original idea: learning without asking for human labels

[Li et al. (2018)](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf) propose representing trajectories with an encoder-decoder and an objective that accounts for the spatial proximity of cells. Their original task is trajectory similarity and search; here we add a classification based on human annotations.

Consider a sequence of cells **A → B → C → D**. We generate a degraded version, say **A → C → D**, and perturb some positions. The encoder summarises that version; the decoder tries to reconstruct the original sequence. We know the answer because we kept the trajectory before altering it. This is **self-supervised** learning: the training signal comes from the data, with no human votes.

In our case points are removed: no `[MASK]` symbol is left marking where they are missing. The objective is to reconstruct the full original cell sequence, not just guess the removed point. If the decoder predicts **A → C → C → D**, we compare those predictions with **A → B → C → D** and adjust the weights of both networks to reduce the loss. Nor do we ask it to interpolate exact GPS coordinates: we predict the cells of the discretised representation.

The task forces it to preserve useful patterns despite noise and irregular sampling. It is no guarantee of recovering streets we never observed: if several routes explain the surviving points, the model can get it wrong. Proximity between cells informs about geography, not about bridges, one-way streets or road connections.

### 5.3 What «penalising by distance» means

Suppose the next point corresponds to cell B. Predicting an adjacent cell and predicting one several kilometres away should not be treated exactly the same. In our implementation the target is spread across eight nearby cells:

$$q(c\mid B)=\frac{\exp[-d(c,B)/250]}{\sum_{u\in N_8(B)}\exp[-d(u,B)/250]},\qquad c\in N_8(B).$$

Distances are in metres and $N_8(B)$ includes the cell itself. The correct cell gets the largest weight; nearby ones get less. To avoid evaluating every cell at each step, our prototype adds 64 sampled negative cells and computes a loss over that subset. **This sampled normaliser is neither the full softmax nor an exact reproduction of the paper's NCE approximation.** We do not read its value as an exact probability over the whole route.

The decoder uses *teacher forcing*: when predicting the next cell during training it receives the correct previous cell. This makes reconstruction easier to learn, but free-running reconstruction can accumulate errors. Our end goal is to exploit the encoder, not to evaluate a trajectory generator.

### 5.4 What was implemented: a small version, with explicit decisions

We implement a **reduced adaptation**, not an exact reproduction of their architecture nor a measurement of the paper's performance:

1. We discretise positions into 250-metre UTM cells, identified also by their zone.
2. The vocabulary is learned from a fraction of the training set only. The internal validation set for pretraining contains different journeys.
3. The bidirectional GRU encoder compresses each route into 64 numbers.
4. A GRU decoder tries to reconstruct the original sequence from a version with points removed and small GPS perturbations.
5. The loss favours the true cells and their spatial neighbours. It uses eight neighbours and 64 negatives per position, with a sampled normaliser.
6. We freeze the encoder and train a small classifier on the human votes.

The model has smaller dimensions, a different grid and projection, 12 epochs as a maximum budget, and does not include the paper's separate cell pretraining. It uses teacher forcing in the decoder. The sampled reconstruction loss serves as a diagnostic, not as an exact log likelihood over the full vocabulary.

**There are two different outputs:** the pretraining decoder reconstructs trajectories; the supervised head produces the probability of «they differ».

We do not use the estimated and real routes as a positive pair during pretraining: they might be precisely the ones that differ.

The vocabulary and encoder weights are learned from **2,578 trajectories across 1,289 training journeys**; another 143 training journeys serve as internal validation for this phase. Each trace is capped at 64 points before discretising. The encoder uses 32-component cell embeddings and a bidirectional GRU with 32 units per direction: the result is 64 components per route. The paper includes a cell pretraining step that is omitted here. The [original repository](https://github.com/boathit/t2vec) makes it possible to tell the published implementation apart from this adaptation.

Those 64 numbers **are not 64 measurements with known names**. The model decides how to spread information across them. That is why we cannot point at one component and claim it represents «the maximum detour». Our geometric vector does have that interpretability, because we defined it explicitly.

### 5.5 From one route's vector to a decision about two routes

For each pair we build $[z_e,z_r,|z_e-z_r|,z_e\odot z_r]$: 256 values. The head is a network with one hidden layer, dropout and a sigmoid output. We also try combining those 256 values with the 42 interpretable measurements and using XGBoost.

The first 128 values keep **both full vectors**. The next 64 make per-component differences explicit; the last 64 are component-wise products. Those products alone are not equivalent to a calibrated similarity. They are additional inputs the classifier can use or ignore.

In this second phase we do use the human votes. For a journey with target $p_j$, the classifier predicts $\hat p_j=\sigma(h_\phi(x_j))$ and we minimise

$$\mathcal L_j=-p_j\log\hat p_j-(1-p_j)\log(1-\hat p_j).$$

The encoder stays frozen: we learn $\phi$, the head's weights, without modifying $\theta$. The reconstruction decoder no longer takes part. So there are two successive objectives: **representing a trajectory** and **learning how people judge a pair**. Solving the first well does not automatically solve the second.

### 5.6 What the experiment allows us to conclude

Our variant gets worse metrics than the geometric measurements. This evaluates that specific configuration. Corpus size, the 250-metre resolution, the training budget and unseen cells are possible explanations, not demonstrated causes. Separating them would require controlled ablations. Nor have we compared «concatenation only» against «differences only»; our input contains both.
""")

code("""
fig,ax=plt.subplots(figsize=(12,5.6));ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
def diagram_box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,alpha=.1,edgecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color=INK)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':MUTED,'lw':1.6})
ax.text(.15,5.65,'1. Aprender a representar una ruta · sin votos humanos',fontsize=13,weight='bold')
diagram_box(.2,3.9,2.1,1.1,'Ruta alterada\\nA → C → D',TEAL)
diagram_box(3,3.9,2.1,1.1,'Encoder\\n64 números',TEAL)
diagram_box(5.8,3.9,2.1,1.1,'Decoder\\nreconstrucción',TEAL)
diagram_box(8.6,3.9,3.0,1.1,'Objetivo original\\nA → B → C → D',TEAL)
for x1,x2 in [(2.3,3),(5.1,5.8),(7.9,8.6)]:arrow(x1,4.45,x2,4.45)
ax.text(.2,3.25,'Se ajustan encoder y decoder. Las celdas cercanas reciben un tratamiento espacial.',fontsize=10,color=MUTED)
ax.text(.15,2.55,'2. Aprender a juzgar una pareja · con votos humanos',fontsize=13,weight='bold')
diagram_box(.2,.8,2.1,1.1,'Ruta estimada\\nRuta real',ORANGE)
diagram_box(3,.8,2.1,1.1,'Mismo encoder\\ncongelado',ORANGE)
diagram_box(5.8,.8,2.1,1.1,'Dos vectores +\\ncomparaciones',ORANGE)
diagram_box(8.6,.8,3.0,1.1,'Clasificador\\nP(diferentes)',ORANGE)
for x1,x2 in [(2.3,3),(5.1,5.8),(7.9,8.6)]:arrow(x1,1.35,x2,1.35)
ax.text(.2,.2,'Se ajusta el clasificador. El decoder ya no se utiliza para predecir la etiqueta.',fontsize=10,color=MUTED)
finish(fig,'05a_t2vec_dos_etapas','Esquema de nuestra adaptación · dos objetivos distintos · no es una reproducción exacta del artículo')
""")

code("""
fig,axs=plt.subplots(1,2,figsize=(12,4.3))
h=pd.DataFrame(encoder['history'])
axs[0].plot(h.epoch,h.train_loss,color=TEAL,label='Entrenamiento')
axs[0].plot(h.epoch,h.inner_validation_loss,color=ORANGE,label='Validación interna')
axs[0].set_xlabel('Época');axs[0].set_ylabel('Pérdida espacial muestreada')
axs[0].set_title('Baja la pérdida; falta comprobar la reconstrucción');axs[0].legend()
rates=encoder['oov_by_split']
axs[1].bar(['Entrenamiento','Validación','Test'],[100*rates[s] for s in ['train','val','test']],color=[TEAL,ORANGE,INK])
for i,s in enumerate(['train','val','test']):axs[1].text(i,100*rates[s]+.6,f'{100*rates[s]:.1f} %',ha='center')
axs[1].set_ylim(0,32);axs[1].set_ylabel('Puntos en celdas no vistas · %')
axs[1].set_title('Reconstruir no garantiza comparar bien')
finish(fig,'05_t2vec_aprendizaje')
""")

md(r"""
The per-route average of points falling in unseen cells reaches roughly 24 % on validation and 22 % on test. They are assigned to the nearest training centroid without extending the vocabulary. This allows inference but can lose spatial detail. 2.5 % of training includes the internal subset excluded from fitting the vocabulary.

Besides, a 250-metre cell can hide separations between nearby streets. This is a **plausible** explanation, not a demonstrated cause, for the embedding performing worse. Identifying it would require comparing resolutions with new validation experiments.
""")

md(r"""
### 5.7 The check that was missing: does it actually reconstruct?

This question prompted an additional audit of the already-saved model. We use the **286 trajectories from the 143 internal validation journeys**: they did not fit the weights, but they did take part in selecting the checkpoint. They are not a new test set. We use no human labels and modify no weights.

We evaluate two situations: clean input, and input with 20 % of points independently removed plus Gaussian noise of 15 metres. Endpoints are kept when removing points, although they also receive noise. We generate cells by taking the highest score over the full vocabulary, without giving the decoder the correct previous cell; it stops on predicting END or after 128 steps.

We measure four different things:

1. **Per-token accuracy with teacher forcing:** percentage of correct positions when the decoder receives the true previous token. It includes END; it is a demanding measure because neighbouring cells also receive weight in the loss.
2. **Empty free reconstruction:** how often it predicts END without emitting a single cell. This cannot be hidden by excluding those routes from a spatial average.
3. **Normalised edit distance:** insertions, deletions and substitutions needed to turn the predicted sequence into the original, divided by the greater length. It runs from 0 — a match — to 1 — maximum discrepancy on this scale. We exclude END.
4. **Spatial discrete Fréchet:** compares the point ordering of the resampled original route against the reconstructed centroids, with distances over an Earth sphere in metres. If the prediction is empty, the distance is left missing and that failure is reported separately.

As references we keep the available input points and the target sequence represented by vocabulary centroids. The second is a diagnostic of quantisation and unknown cells, **not a reconstruction any model could obtain without knowing the answer**.
""")

code("""
recon_dir=ROOT/'results/routes_reconstruction'
recon_metrics=json.loads((recon_dir/'metrics.json').read_text())
recon_predictions=pd.DataFrame(json.loads((recon_dir/'predictions.json').read_text()))
recon_rows=[]
for scenario,label in [('clean','Sin alteración'),('corrupted','20 % de eliminación + ruido')]:
    m=recon_metrics[scenario]
    recon_rows.append({'Entrada':label,'Rutas':m['n_routes'],'Salida vacía':m['empty_fraction'],
        'Exactitud tokens con ayuda':m['teacher_forced_token_accuracy']['mean'],
        'Edición libre media':m['free_normalized_edit']['mean'],
        'Rutas no vacías':m['decoded_frechet_m']['n'],
        'Fréchet mediano no vacías · km':(m['decoded_frechet_m']['median'] if m['decoded_frechet_m']['median'] is not None else np.nan)/1000})
display(pd.DataFrame(recon_rows).set_index('Entrada').style.format({
    'Salida vacía':'{:.1%}','Exactitud tokens con ayuda':'{:.1%}','Edición libre media':'{:.4f}',
    'Fréchet mediano no vacías · km':'{:.1f}'}))
corrupted=recon_predictions[recon_predictions.scenario=='corrupted']
nonempty=corrupted[~corrupted['empty']]
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
empty_n=int(corrupted['empty'].sum());nonempty_n=len(nonempty)
axs[0].bar(['Termina sin celdas','Emite alguna celda'],[empty_n,nonempty_n],color=[RED,ORANGE])
for i,n in enumerate([empty_n,nonempty_n]):axs[0].text(i,n+4,f'{n}/{len(corrupted)}',ha='center')
axs[0].set_ylim(0,len(corrupted));axs[0].set_ylabel('Trayectorias de validación interna')
axs[0].set_title('El decoder puede terminar antes de empezar')
fields=['input_frechet_m','oracle_quantized_frechet_m','decoded_frechet_m']
values=[nonempty[k].median()/1000 for k in fields]
axs[1].bar(['Puntos\\ndisponibles','Objetivo\\ncuantizado','Decoder\\nsin ayuda'],values,color=[TEAL,INK,RED])
axs[1].set_yscale('log');axs[1].set_ylabel('Fréchet mediano · km · escala log')
axs[1].set_title(f'En las mismas {len(nonempty)} salidas no vacías')
for i,v in enumerate(values):axs[1].text(i,v*1.2,f'{v:.2f} km',ha='center',fontsize=9)
axs[1].set_ylim(min(values)*.4,max(values)*4)
finish(fig,'05b_auditoria_reconstruccion','Modelo original congelado · diagnóstico posterior · las 199 salidas vacías se muestran aparte, no se ocultan')
known=corrupted.assign(geography=np.where(corrupted.oov==0,'Todas las celdas conocidas','Alguna celda no vista'))
display(known.groupby('geography').agg(Rutas=('journey_id','size'),
    Fraccion_vacia=('empty','mean'),Edicion_media=('free_normalized_edit','mean')))
""")

md(r"""
**The result forces a correction of the earlier interpretation:** the decoder in this adaptation does not reconstruct properly. With altered input, 199 of 286 outputs are empty — 69.6 % — and the remaining 87 show enormous spatial error. Free reconstruction fails with clean input too. The falling training loss did not demonstrate that generation worked.

We also found a **concrete defect in the implemented loss**: negatives are sampled only among cells and never include END. So predicting END with a very high score at a step that should contain a cell is not penalised by that sampled objective. We verified it by raising END's score by 100: the loss of a stretch that does not terminate stays exactly the same. This does not happen with a loss normalised over all possible outputs.

That defect helps explain the premature ending, but fixing it does not by itself guarantee a good representation: the normaliser approximation, data scarcity, resolution and unseen cells all remain. **The saved supervised metrics describe this failed attempt; they are not valid evidence against the t2vec of the paper.** The geometry and the road-network experiment do not depend on this decoder.

This finding requires correcting the objective before interpreting the quality of the method. We run that experiment separately below, keeping the original splits and results intact.
""")

md(r"""
### 5.8 We correct the objective and repeat the check

The new version normalises over **all cells and END**, excluding only the auxiliary start and padding symbols. If the decoder assigns too much score to END when it should predict a cell, the denominator grows and the loss penalises it. We checked this with a gradient test and with the same artificial score increase used to detect the defect.

$$\mathcal L_t=\log\sum_{c\in\mathcal V\cup\{\mathrm{END}\}}e^{s_t(c)}-\sum_c q_t(c)s_t(c).$$

$s_t(c)$ is the output score for each cell and $q_t$ spreads the target across spatial neighbours; for END all the weight falls on END. The normaliser is no longer sampled. This costs more compute, but removes the omission we found.

We retrain from the same seed, with the same architecture, vocabulary, journeys and maximum budget of 12 epochs. We pick the checkpoint using altered internal-validation inputs, with reproducible noise. This also makes the selection task more consistent with the training task; we therefore do not attribute all subsequent differences solely to including END. We then train the same family of classifiers, selecting their size and threshold on supervised validation.

Weights and results are saved to new folders. The audit uses the same routes and alterations as before. **It is still a small adaptation of t2vec and an exploratory experiment run after inspecting the test set**, even though the loss is now correctly normalised.
""")

code("""
corrected_recon=json.loads((ROOT/'results/routes_reconstruction_corrected/metrics.json').read_text())
corrected_model=json.loads((ROOT/'results/routes_t2vec_corrected/metrics.json').read_text())
comparison_rows=[]
for label,report in [('Original: pérdida defectuosa',recon_metrics),('Pérdida corregida',corrected_recon)]:
    m=report['corrupted']
    comparison_rows.append({'Versión':label,'Salida vacía':m['empty_fraction'],
        'Termina antes del límite':m['eos_fraction'],'Edición libre media':m['free_normalized_edit']['mean'],
        'Exactitud tokens con ayuda':m['teacher_forced_token_accuracy']['mean'],
        'Rutas no vacías':m['decoded_frechet_m']['n'],
        'Fréchet mediano no vacías · km':(m['decoded_frechet_m']['median'] if m['decoded_frechet_m']['median'] is not None else np.nan)/1000})
display(pd.DataFrame(comparison_rows).set_index('Versión').style.format({
    'Salida vacía':'{:.1%}','Termina antes del límite':'{:.1%}','Edición libre media':'{:.4f}',
    'Exactitud tokens con ayuda':'{:.1%}','Fréchet mediano no vacías · km':'{:.2f}'}))
fig,axs=plt.subplots(1,2,figsize=(12,4.2))
for ax,key,title in zip(axs,['empty_fraction','free_normalized_edit'],['¿Evita terminar inmediatamente?','¿Recupera la secuencia de celdas?']):
    vals=[]
    for report in [recon_metrics,corrected_recon]:
        value=report['corrupted'][key]
        vals.append(value['mean'] if isinstance(value,dict) else value)
    ax.bar(['Original','Corregido'],vals,color=[RED,TEAL]);ax.set_ylim(0,1.1)
    for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.3f}',ha='center')
    ax.set_title(title);ax.set_ylabel('Fracción vacía' if key=='empty_fraction' else 'Distancia de edición (0–1)')
finish(fig,'05c_reconstruccion_corregida','Mismas 286 trayectorias y alteraciones · corregir un error de implementación no garantiza resolver la tarea')
display(pd.DataFrame([
    {'Modelo':'Original','LL val':details['t2vec_decoder_clasificador']['val_soft_logloss'],
     'BA val':metrics['t2vec_decoder_clasificador']['validation']['balanced_accuracy'],
     'BA test':metrics['t2vec_decoder_clasificador']['balanced_accuracy']},
    {'Modelo':'Corregido','LL val':corrected_model['val_soft_logloss'],
     'BA val':corrected_model['validation']['balanced_accuracy'],'BA test':corrected_model['balanced_accuracy']}
]).set_index('Modelo').style.format('{:.4f}'))
display(Markdown('Estas métricas de clasificación responden a otra pregunta: si el vector permite reproducir los votos humanos. '
    'Hay que leerlas junto a la reconstrucción, las celdas desconocidas y la referencia geométrica. '
    'Un objetivo de entrenamiento correcto es necesario para interpretar el experimento, pero no asegura que '
    'la representación aprendida sea suficiente para esta tarea.'))
""")

md(r"""
**What changed and what did not:** with altered inputs, the corrected version cuts empty outputs from 69.6 % to 4.9 %, but its mean edit distance stays close to 1 and the routes it generates remain spatially wrong. Terminating immediately less often is not the same as reconstructing well. Accuracy with teacher forcing is still low, so the problem is not explained solely by error accumulation during free generation either.

Classification reaches 77.1 % balanced accuracy on test against 71.8 % for the original attempt, but stays far from the 94.3 % of the geometric baseline. This is an exploratory comparison on the same test set we had already inspected. **We would not pick this version for the current prototype.** Before going further we would need more pretraining and experiments on architecture, resolution and coverage, checking reconstruction from the start and with new data to confirm the final result.
""")

md(r"""
## 6. The protocol is decided before looking at the test set

This section describes the original four-model experiment. The later road-network extension keeps its fitting rules, but its exploratory evaluation is explicitly identified in section 7.1.

**Model selection:** lowest cross-entropy against the votes on validation. This rule rewards probabilities consistent with human disagreement.

**Threshold selection:** highest balanced accuracy on validation, scanning 0.10–0.90 in steps of 0.01; ties favour the threshold closest to 0.5. We assume neither that 0.5 is optimal nor that the errors have known business costs.

Balanced accuracy is the average of the proportion of «the same» and «they differ» correctly recognised. It stops a majority class from dominating the figure.

The positive class is **«they differ»**. It is worth separating three questions:

| Question | Metric and reading |
|---|---|
| How many decisions do we get right at the fixed threshold? | **Accuracy:** fraction correct. **Balanced accuracy:** mean recall over both classes. Higher is better. |
| How does the «they differ» alert behave? | **Precision:** of the alerts raised, how many match the human majority. **Recall:** of the genuinely different journeys, how many we catch. **F1:** their harmonic mean. Higher is better. |
| Do we rank cases well without fixing a single threshold? | **ROC-AUC:** class separation across thresholds. **AP:** a summary of precision and recall for «they differ»; it depends on prevalence. Higher is better. |
| Are the probabilities reasonable? | **Brier:** squared error against the binary majority. **Binary log loss:** penalises confident errors especially. **Vote log loss:** uses the proportion of votes, ties included. Lower is better. |

To see the difference between the two losses, let $q_j$ be the model's probability and $p_j$ the fraction of «they differ» votes:

$$\mathrm{LL}_{\text{votes}}=-\frac{1}{N}\sum_j[p_j\log q_j+(1-p_j)\log(1-q_j)].$$

In binary log loss we replace $p_j$ with the majority $y_j\in\{0,1\}$ and exclude ties. A 2-to-1 vote becomes 1 in that second evaluation, but stays at $2/3$ in the first. **We select using the vote loss; we do not switch criteria once we see which metric favours which model.**

Three predefined configurations of depth and regularisation are tried per XGBoost, with early stopping on validation. For the neural head we try 32 and 64 hidden units. The models use the same split and targets. Vocabulary, encoder and scaling are never fitted using test data.

Once models, epochs and thresholds are chosen, the decisions are saved and test is evaluated. **We do not retrain on validation and we do not change the winner after seeing test.** The four variants are planned comparisons; later changes would require keeping the distinction between exploration and a fresh confirmation.
""")

code("""
def metric_table(split):
    columns={}
    for key in names:
        m=metrics[key]['validation'] if split=='val' else metrics[key]
        tn,fp,fn,tp=np.array(m['confusion_matrix']).ravel()
        soft=frozen['validation_results'][key] if split=='val' else m['soft_logloss_all_decisive_votes']
        columns[names[key]]={
            'Accuracy ↑':f"{m['accuracy']:.2%}",
            'Balanced accuracy ↑':f"{m['balanced_accuracy']:.2%}",
            'Precisión · diferentes ↑':f'{tp/(tp+fp):.2%}',
            'Recall · diferentes ↑':f'{tp/(tp+fn):.2%}',
            'F1 · diferentes ↑':f"{m['f1']:.2%}",
            'ROC-AUC ↑':f"{m['roc_auc']:.4f}",
            'Average precision (AP) ↑':f"{m['average_precision']:.4f}",
            'Brier binario ↓':f"{m['brier']:.4f}",
            'Log loss binaria ↓':f"{m['logloss']:.4f}",
            'Log loss de votos ↓ · criterio de selección en val':f'{soft:.4f}',
            'Umbral elegido en validación':f"{metrics[key]['threshold']:.2f}",
            'Viajes con mayoría / con votos válidos':f"{m['n']} / {int(((df.split==split)&df.target_soft.notna()).sum())}",
            'TN / FP / FN / TP':f'{tn} / {fp} / {fn} / {tp}'}
    return pd.DataFrame(columns)

display(Markdown('### Validación: donde se tomaron las decisiones'))
display(metric_table('val'))
display(Markdown('### Test: evaluación con modelos y umbrales congelados'))
display(metric_table('test'))
display(Markdown('↑ Mayor es mejor; ↓ menor es mejor. TN/TP son aciertos en iguales/diferentes; FP/FN son falsas alarmas/diferencias no detectadas. La columna geométrica es la elegida en validación. Las métricas de validación han participado en el ajuste y la selección; no son una estimación independiente del rendimiento final.'))
keys=list(names);yy=np.arange(len(keys))
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
vals=[frozen['validation_results'][k] for k in keys]
axs[0].barh(yy,vals,color=[TEAL if k==frozen['selected'] else MUTED for k in keys])
axs[0].set_yticks(yy,[names[k] for k in keys]);axs[0].invert_yaxis()
axs[0].set_xlabel('Log loss frente a votos · menor es mejor')
axs[0].set_title('La elección se hace en validación')
means=np.array([metrics[k]['balanced_accuracy'] for k in keys])*100
ci=np.array([metrics[k]['balanced_accuracy_ci95'] for k in keys])*100
axs[1].errorbar(means,yy,xerr=[means-ci[:,0],ci[:,1]-means],fmt='o',color=ORANGE,capsize=4)
axs[1].set_yticks(yy,[names[k] for k in keys]);axs[1].invert_yaxis()
axs[1].set_xlim(60,100);axs[1].set_xlabel('Acierto equilibrado en test · %')
axs[1].set_title('Test: estimación e incertidumbre')
finish(fig,'06_validacion_y_test','Test: 443 viajes con mayoría definida · IC bootstrap percentil 95 % · una partición y una semilla')
""")

md(r"""
## 7. The key moment: more complex does not mean better demonstrated

The validation rule selects **basic geometry + XGBoost**. On test it gets 94.3 % balanced accuracy.

The extended vector reaches 94.8 %, the t2vec adaptation with a classifier 71.8 %, and the combination 95.5 %. The hybrid has the highest point estimate on test, but **we do not pick it retrospectively for that reason**.

To compare, we resample the same test journeys 2,000 times and compute the difference between models in each sample. Using the same journeys, the interval reflects which errors they share.

The improvements of the extended vector and the hybrid over the baseline have intervals that include zero. That does not prove equality: with this sample, the evidence does not let us claim the improvement is stable.

The intervals are conditional on this training run, this split and this geographic mix. They do not capture changes of seed, of city, dependence between corridors, or the uncertainty of the human panel. The secondary comparisons are exploratory and not corrected for multiplicity.
""")

code("""
keys=list(comparisons);means=np.array([comparisons[k]['delta_balanced_accuracy'] for k in keys])*100
ci=np.array([comparisons[k]['paired_ci95'] for k in keys])*100
fig,ax=plt.subplots(figsize=(10,3.8))
ax.axvline(0,color=MUTED,ls='--')
ax.errorbar(means,np.arange(len(keys)),xerr=[means-ci[:,0],ci[:,1]-means],fmt='o',capsize=5,color=TEAL)
ax.set_yticks(np.arange(len(keys)),[names[k] for k in keys]);ax.invert_yaxis()
ax.set_xlabel('Diferencia frente a geometría básica · puntos porcentuales')
ax.set_title('¿La mejora persiste al cambiar los viajes del test?')
finish(fig,'07_diferencias_pareadas')
""")

cells.extend(result_cells())

md(r"""
## 8. Back to the routes: what does getting it right, and getting it wrong, mean?

A high AUC indicates good ranking, but it does not settle which threshold suits the business. At the threshold chosen on validation, the baseline marks as similar 25 journeys the majority considered different, and marks as different three journeys the majority considered similar.

These figures depend on the chosen human target. The probabilities are not objective certainty. The calibration curve is shown as a diagnostic: it has not been recalibrated using test.

From here on, every route we inspect is from **test** and every prediction comes from **basic geometry + XGBoost**, at threshold **0.75**. The earlier training examples were there to explain the measurements; these are here to audit the final result.

| | The model says «the same» | The model says «they differ» |
|---|---|---|
| The majority says «the same» | **185 correct — TN** | **3 false alarms — FP** |
| The majority says «they differ» | **25 missed differences — FN** | **230 correct — TP** |

«Error» means disagreement with the available majority, which is often **a single person**. It does not mean we have demonstrated the trip is objectively the same or different. Telling a model error apart from a debatable label would require additional review.
""")

code("""
selected=frozen['selected'];m=metrics[selected]
fig,axs=plt.subplots(1,2,figsize=(12,4.3))
cm=np.array(m['confusion_matrix'])
axs[0].imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2):axs[0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=24,color='white' if cm[i,j]>100 else INK)
axs[0].set_xticks([0,1],['Iguales','Diferentes']);axs[0].set_yticks([0,1],['Iguales','Diferentes'])
axs[0].set_xlabel('Predicción');axs[0].set_ylabel('Mayoría humana');axs[0].set_title('Errores de la referencia elegida')
test=predictions[(predictions.split=='test')&predictions.target_hard.notna()].copy()
p=np.array([r[selected] for r in test.probabilities]);y=test.target_hard.to_numpy(float)
edges=np.linspace(0,1,6)
xs=[];ys=[];ns=[]
for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
    mask=(p>=lo)&((p<=hi) if i==len(edges)-2 else (p<hi))
    if mask.any():xs.append(p[mask].mean());ys.append(y[mask].mean());ns.append(mask.sum())
axs[1].plot([0,1],[0,1],ls='--',color=MUTED);axs[1].plot(xs,ys,'o-',color=TEAL)
for a,b,n in zip(xs,ys,ns):
    offset=(-18,-18) if a>.9 else (5,9)
    axs[1].annotate(f'n={n}',(a,b),xytext=offset,textcoords='offset points',fontsize=9)
axs[1].set_xlabel('Probabilidad media predicha');axs[1].set_ylabel('Fracción de mayorías «diferentes»')
axs[1].set_title('Calibración descriptiva · intervalos de 0,2')
finish(fig,'08_errores_y_calibracion')
""")

code("""
importance=pd.Series(details[selected]['feature_importance']).sort_values()
fig,ax=plt.subplots(figsize=(10,4.3))
feature_labels={'mean_real_to_est_m':'Distancia media: real → estimada',
 'mean_est_to_real_m':'Distancia media: estimada → real','length_real':'Longitud real',
 'length_est':'Longitud estimada','hausdorff_discrete_m':'Separación máxima · Hausdorff discreto',
 'relative_length_delta':'Diferencia relativa de longitud','start_gap_m':'Separación de orígenes',
 'end_gap_m':'Separación de destinos'}
ax.barh([feature_labels[k] for k in importance.index],importance.values,color=TEAL)
ax.set_xlabel('Importancia gain normalizada · no causal')
ax.set_title('Qué utiliza el modelo de ocho variables')
finish(fig,'09_variables_referencia')
""")

md(r"""
### 8.1 First, recognising the two kinds of correct call

A metrics curve does not let you see whether the model recognises two almost-overlapping trips or an obvious deviation. We show **two correct calls of each class**: the one in the central position when sorting by score, and the one closest to the threshold. That way we see a central case and a tighter decision, without hand-picking the most striking pictures.

The dark line is the estimated route; the dashed orange one is the real route. Axes are in kilometres, with the same horizontal and vertical scale within each panel. Panels adapt to each journey: they do not all span the same area. Circles mark starts and squares mark ends.

When reading «0.74», remember: it is a probabilistic score for «they differ», but the chosen rule requires **at least 0.75** to emit that label. It is not a percentage of differing metres.
""")

code("""
test['p']=p
test['pred']=(p>=m['threshold']).astype(int)
assert selected=='geometria_base'
assert set(test.index).issubset(set(df.index[df.split=='test']))
assert np.array_equal(cm,np.array([[((y==i)&(test.pred.to_numpy()==j)).sum() for j in range(2)] for i in range(2)]))

# Selección descriptiva, reproducible y posterior a congelar el modelo.
case_rows=[]
for kind,label in [('TN',0),('TP',1)]:
    group=test[(test.target_hard==label)&(test.pred==label)].sort_values(['p'],kind='stable')
    central=group.index[len(group)//2]
    boundary=(group.p-m['threshold']).abs().sort_values(kind='stable').index[0]
    for suffix,jid,reason in [('1',central,'Puntuación central entre los aciertos de su clase'),
                              ('2',boundary,'Acierto de su clase más próximo al umbral')]:
        case_rows.append(dict(case=kind+suffix,kind=kind,journey_id=jid,selection=reason))
fp=test[(test.target_hard==0)&(test.pred==1)].sort_values('p',ascending=False,kind='stable')
fn=test[(test.target_hard==1)&(test.pred==0)].sort_values('p',kind='stable')
for k,jid in enumerate(fp.index,1):
    case_rows.append(dict(case=f'FP{k}',kind='FP',journey_id=jid,selection='Todas las falsas alarmas, ordenadas por puntuación'))
for k,rank in enumerate([0,len(fn)//2,len(fn)-1],1):
    case_rows.append(dict(case=f'FN{k}',kind='FN',journey_id=fn.index[rank],
        selection=['Menor puntuación entre los falsos negativos','Puntuación central entre los falsos negativos',
                   'Falso negativo más próximo al umbral'][k-1]))
cases=pd.DataFrame(case_rows).set_index('case')
assert cases.journey_id.is_unique and len(cases)==10
assert len(fp)==3 and len(fn)==25

def vote_counts(jid):
    row=df.loc[jid]
    different=int(round(row.n_votes*row.target_soft))
    return int(row.n_votes)-different,different,int(row.n_unknown),int(row.n_conflicted_annotators)

def vote_text(jid):
    same,different,unknown,conflict=vote_counts(jid)
    text=f'{same} iguales / {different} diferentes'
    if unknown:text+=f' / {unknown} no sé'
    if conflict:text+=f' / {conflict} contradictorios'
    return text

def plot_routes(ax,a,b,ends=True,legend=True):
    ax.plot(a[:,0]/1000,a[:,1]/1000,color=INK,lw=2.8,alpha=.85,label='Estimada')
    ax.plot(b[:,0]/1000,b[:,1]/1000,color=ORANGE,lw=1.7,ls='--',label='Real')
    if ends:
        for xy,color in [(a,INK),(b,ORANGE)]:
            ax.scatter(xy[0,0]/1000,xy[0,1]/1000,marker='o',s=43,facecolors=BG,edgecolors=color,zorder=4)
            ax.scatter(xy[-1,0]/1000,xy[-1,1]/1000,marker='s',s=35,facecolors=BG,edgecolors=color,zorder=4)
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_xlabel('Este local · km');ax.set_ylabel('Norte local · km')
    if legend:ax.legend(fontsize=9,loc='best')
    ax.grid(alpha=.12)

fig,axs=plt.subplots(2,2,figsize=(12,9))
for ax,(case,meta) in zip(axs.flat,cases[cases.kind.isin(['TN','TP'])].iterrows()):
    jid=meta.journey_id;row=test.loc[jid]
    a,b=project_pair(*routes[jid])
    plot_routes(ax,a,b)
    human='iguales' if row.target_hard==0 else 'diferentes'
    reason='central' if case.endswith('1') else 'cerca del umbral'
    ax.set_title(f'{case} · Ambos dicen {human} · {reason}\\np(diferentes)={row.p:.4f} · umbral 0,75',fontsize=11)
    ax.text(.02,.02,f'ID {jid[:8]} · {vote_text(jid)}',transform=ax.transAxes,fontsize=8,
            bbox=dict(facecolor=BG,alpha=.9,edgecolor='none'))
finish(fig,'10_aciertos_test','Cuatro aciertos de test · selección por puntuación central / cercanía al umbral · escalas propias por viaje')

ledger=[]
for case,meta in cases.iterrows():
    jid=meta.journey_id;row=test.loc[jid];f=features.loc[jid]
    ledger.append({'Caso':case,'journey_id':jid,'Votos de personas distintas':vote_text(jid),
       'p(diferentes)':row.p,'Long. estimada · km':f.length_est/1000,'Long. real · km':f.length_real/1000,
       'Media real → estimada · m':f.mean_real_to_est_m,'Media estimada → real · m':f.mean_est_to_real_m,
       'Hausdorff discreto · m':f.hausdorff_discrete_m,
       'Real a >50 m de estimada':f.fraction_real_to_est_over_50m})
ledger=pd.DataFrame(ledger).set_index('Caso')
display(Markdown('**Registro de los diez casos.** Se conservan sus identificadores completos para localizarlos en el archivo; los votos ya están deduplicados. La proporción a más de 50 m ayuda a leer los dibujos, pero no es una entrada del modelo de ocho variables.'))
display(ledger[['journey_id','Votos de personas distintas','p(diferentes)']].style.format({'p(diferentes)':'{:.4f}'}))
display(Markdown('**Medidas geométricas de los mismos casos.** Los identificadores TN, TP, FP y FN conectan estas medidas con las figuras.'))
display(ledger.drop(columns=['journey_id','Votos de personas distintas','p(diferentes)']).style.format({'Long. estimada · km':'{:.2f}','Long. real · km':'{:.2f}',
    'Media real → estimada · m':'{:.1f}','Media estimada → real · m':'{:.1f}',
    'Hausdorff discreto · m':'{:.1f}','Real a >50 m de estimada':'{:.1%}'}))
""")

md(r"""
### 8.2 The errors: separating trace, decision and label

We inspect **the three false positives** and **three of the 25 false negatives**: the lowest-scoring one, the central one and the one closest to the threshold. They are not a random sample of errors and cannot be used to estimate how often each explanation applies. The prediction log keeps all 28 errors.

Each figure answers four questions:

1. **Full view:** do the routes look alike overall?
2. **Zoom:** is there a stretch the trip's scale was hiding? We centre the window on the largest separation found when sampling 800 positions per route.
3. **Distance profile:** is the difference brief or persistent? We measure from each position to the opposite polyline. The percentage refers to the length of **each route**, not to time or to synchronised positions.
4. **Model contributions:** which variables push this prediction towards «the same» or «they differ»?

For the last panel we use [XGBoost's TreeSHAP contributions](https://xgboost.readthedocs.io/en/stable/prediction.html). They decompose the trained model's score before the sigmoid:

$$\log\frac{q}{1-q}=\phi_0+\sum_{k=1}^{8}\phi_k,\qquad
q=\frac{1}{1+\exp[-(\phi_0+\sum_k\phi_k)]}.$$

$\phi_0$ is the trees' implicit baseline; each $\phi_k$ is a contribution in **log-odds**, not in percentage points. Orange raises the «they differ» score and green lowers it. We use the trained model's internal baseline, without adding test as a background set. We keep exactly the trees of the iteration chosen on validation and check that the sum reconstructs the frozen prediction.

This explains **how the model scores this input**, not why a person drove that way or why the annotator voted that way. The variables are related to each other; their contributions are not causal effects.
""")

code("""
from xgboost import XGBRegressor,DMatrix
import hashlib
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib.patches import Rectangle

case_ids=cases.journey_id.tolist()
model_path=ROOT/f'models/routes/{selected}.ubj'
inspection_path=R/'inspection.json'
if model_path.exists():
    fitted=XGBRegressor(n_jobs=1);fitted.load_model(model_path)
    x=np.ascontiguousarray(features.loc[case_ids,BASE_FEATURES],dtype=np.float32)
    contributions=fitted.get_booster().predict(DMatrix(x),pred_contribs=True,
                                               iteration_range=(0,fitted.best_iteration+1))
    model_digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
else:
    # Permite reproducir las figuras con las contribuciones exportadas.
    saved=json.loads(inspection_path.read_text())
    assert saved['selected']==selected and saved['threshold']==m['threshold']
    model_digest=saved['model_sha256']
    lookup={r['journey_id']:r for r in saved['cases']}
    contributions=np.array([[lookup[j]['contributions'][k] for k in BASE_FEATURES]+[lookup[j]['reference']]
                            for j in case_ids])
contributions=pd.DataFrame(contributions,index=case_ids,columns=BASE_FEATURES+['reference'])
reconstructed=1/(1+np.exp(-contributions.sum(axis=1).to_numpy()))
assert np.allclose(reconstructed,test.loc[case_ids,'p'].to_numpy(),atol=1e-6,rtol=1e-5)

inspection={'selected':selected,'threshold':m['threshold'],'model_sha256':model_digest,
 'purpose':'Inspección posterior de test, sin ajuste de modelos ni umbral.',
 'selection_policy':{'correct':'Puntuación central y más próxima al umbral por clase.',
                     'fp':'Todos los falsos positivos, por puntuación descendente.',
                     'fn':'Mínima, central y máxima puntuación entre los falsos negativos.'},
 'confusion_matrix':cm.tolist(),'cases':[]}
for case,meta in cases.iterrows():
    jid=meta.journey_id
    same,different,unknown,conflict=vote_counts(jid)
    inspection['cases'].append(dict(case=case,kind=meta.kind,journey_id=jid,split='test',
        selection=meta.selection,probability=float(test.loc[jid,'p']),
        prediction=int(test.loc[jid,'pred']),human_majority=int(test.loc[jid,'target_hard']),
        votes=dict(same=same,different=different,unknown=unknown,conflicted=conflict),
        features={k:float(features.loc[jid,k]) for k in BASE_FEATURES},
        contributions={k:float(contributions.loc[jid,k]) for k in BASE_FEATURES},
        reference=float(contributions.loc[jid,'reference'])))
inspection_path.write_text(json.dumps(inspection,indent=2,ensure_ascii=False))

short_labels={'length_est':'Longitud estimada','length_real':'Longitud real',
 'relative_length_delta':'Diferencia relativa de longitud','start_gap_m':'Separación de orígenes',
 'end_gap_m':'Separación de destinos','mean_est_to_real_m':'Media estimada → real',
 'mean_real_to_est_m':'Media real → estimada','hausdorff_discrete_m':'Hausdorff discreto'}

def error_figure(case,jid):
    row=test.loc[jid];a,b=project_pair(*routes[jid])
    la,lb=clean_line(a),clean_line(b)
    sa,sb=sample_line(la,800,midpoints=True),sample_line(lb,800,midpoints=True)
    da,db=distance(points(sa),lb),distance(points(sb),la)
    if da.max()>=db.max():
        furthest=sa[np.argmax(da)];nearest=np.array(nearest_points(Point(furthest),lb)[1].coords[0])
    else:
        furthest=sb[np.argmax(db)];nearest=np.array(nearest_points(Point(furthest),la)[1].coords[0])
    center=(furthest+nearest)/2
    radius=max(100,np.linalg.norm(furthest-nearest)*.7)
    fig,axs=plt.subplots(2,2,figsize=(12,9.2))
    full,zoom,profile,shap_ax=axs.flat
    plot_routes(full,a,b)
    full.add_patch(Rectangle((center-radius)/1000,2*radius/1000,2*radius/1000,
                            fill=False,ec=TEAL,lw=1.3,ls=':'))
    full.set_title('1 · El viaje completo',fontsize=12)
    plot_routes(zoom,a,b,ends=False)
    zoom.set_xlim((center[0]-radius)/1000,(center[0]+radius)/1000)
    zoom.set_ylim((center[1]-radius)/1000,(center[1]+radius)/1000)
    zoom.set_aspect('equal',adjustable='box')
    zoom.plot([furthest[0]/1000,nearest[0]/1000],[furthest[1]/1000,nearest[1]/1000],
              color=TEAL,ls=':',lw=2)
    zoom.scatter(*furthest/1000,color=TEAL,s=28,zorder=5)
    zoom.set_title('2 · Un detalle que la escala puede ocultar',fontsize=12)
    xx=(np.arange(800)+.5)/800*100
    profile.plot(xx,da,color=INK,lw=1.4,label='Estimada → real')
    profile.plot(xx,db,color=ORANGE,lw=1.4,ls='--',label='Real → estimada')
    profile.axhline(50,color=TEAL,ls=':',lw=1,label='50 m · referencia visual')
    profile.set(xlabel='Porcentaje de longitud de cada ruta',ylabel='Distancia a la otra ruta · m')
    profile.set_title('3 · ¿Breve o persistente?',fontsize=12)
    profile.legend(fontsize=8);profile.grid(alpha=.12)
    s=contributions.loc[jid,BASE_FEATURES].sort_values()
    shap_ax.barh([short_labels[k] for k in s.index],s.values,
                  color=[ORANGE if v>=0 else TEAL for v in s.values])
    shap_ax.axvline(0,color=MUTED,lw=.8)
    shap_ax.tick_params(axis='y',labelsize=9)
    shap_ax.set_xlabel('Contribución log-odds · ← iguales | diferentes →',fontsize=10)
    shap_ax.set_title('4 · Cómo llega el modelo a esta puntuación',fontsize=12)
    human='iguales' if row.target_hard==0 else 'diferentes'
    predicted='iguales' if row.pred==0 else 'diferentes'
    fig.suptitle(f'{case} · Humano: {human} | Modelo: {predicted}\\n'
                 f'p(diferentes)={row.p:.4f} · umbral 0,7500 · {vote_text(jid)}',
                 x=.02,ha='left',fontsize=14,weight='bold',y=.995)
    finish(fig,f'11_error_test_{case.lower()}',
           f'Test · ID {jid} · ampliación geométrica sin mapa base · contribuciones descriptivas, no causales')

for case,meta in cases[~cases.kind.isin(['TN','TP'])].iterrows():
    jid=meta.journey_id;row=test.loc[jid];f=features.loc[jid]
    heading='Falsa alarma' if meta.kind=='FP' else 'Diferencia no detectada'
    display(Markdown(f'### {case} · {heading}\\n\\n**Selección:** {meta.selection}. '
                     f'ID completo: **{jid}**. Votos válidos y abstenciones: **{vote_text(jid)}**.'))
    error_figure(case,jid)
    stats=(f'**Lo que medimos:** ruta estimada {f.length_est/1000:.2f} km y real {f.length_real/1000:.2f} km; '
           f'medias de separación real→estimada **{f.mean_real_to_est_m:.1f} m** y estimada→real '
           f'**{f.mean_est_to_real_m:.1f} m**; máximo discreto **{f.hausdorff_discrete_m:.0f} m**. '
           f'El **{f.fraction_real_to_est_over_50m:.1%}** de la longitud real está a más de 50 m de la estimada.')
    display(Markdown(stats))
    s=contributions.loc[jid,BASE_FEATURES]
    explain='; '.join(f'{short_labels[k]} {v:+.2f}' for k,v in s.reindex(s.abs().sort_values(ascending=False).index[:3]).items())
    display(Markdown(f'**Lo que hace el modelo:** sus tres mayores contribuciones en valor absoluto son '
                     f'{explain} (log-odds). Con referencia {contributions.loc[jid,"reference"]:.3f}, '
                     f'la suma total es {contributions.loc[jid].sum():.3f}; '
                     f'la sigmoide recupera **{row.p:.4f}**, frente al umbral **{m["threshold"]:.2f}**.'))
    if meta.kind=='FP':
        interpretation=('**Lectura:** hay separación geométrica, pero el único voto válido dice «iguales». '
            'El modelo cruza el umbral; el anotador pudo tolerar ese cambio al juzgar el viaje completo. '
            'Es una hipótesis para revisar con más personas, no una prueba de que la etiqueta esté mal. '
            'Tampoco sabemos si el trazado refleja el recorrido real con precisión: no tenemos una validación independiente del GPS.')
    elif case=='FN1':
        interpretation=('**Lectura:** la separación máxima supera medio kilómetro, pero las medias quedan en unas decenas de metros '
            'dentro de un recorrido de unos 15 km. Las contribuciones de las medias y las longitudes reducen la puntuación. '
            'Aquí se ve cómo un resumen global puede dar poco peso a una diferencia localizada. '
            'No basta con bajar ligeramente el umbral: la puntuación es muy baja. El único anotador válido la consideró diferente; '
            'habría que revisar la importancia del tramo y la consistencia de ese juicio.')
    elif case=='FN2':
        interpretation=('**Lectura:** las medias empujan hacia «diferentes», pero las longitudes empujan hacia «iguales». '
            'La puntuación queda cerca de 0,5 y no alcanza 0,75. Hay un voto «diferentes» y dos «no sé»: '
            'no es un consenso de tres personas. Este caso muestra incertidumbre en la etiqueta además del compromiso del modelo.')
    else:
        interpretation=('**Lectura:** 0,749593 queda justo por debajo de 0,750000. Este error está en la frontera de decisión: '
            'redondear ambos números a 0,75 ocultaría lo ocurrido. Mover el umbral lo cambiaría de clase, '
            'pero también alteraría otros aciertos y errores. No lo ajustamos para corregir este viaje de test; '
            'el umbral debe responder al coste de errores y validarse en datos independientes.')
    display(Markdown(interpretation))
""")

md(r"""
### 8.3 What the inspection taught us

The same «error» label can hide different problems. A localised detour can be diluted in the averages of a long trip; two geometric signals can push in opposite directions; and a score almost identical to the threshold can flip the decision by ten-thousandths.

The human reference matters too. All three false alarms have a single decided vote per journey. The central false negative has one decision and two abstentions. **More model complexity will not, on its own, settle which similarity criterion we want to reproduce.**

The inspection proposes hypotheses, it does not confirm them. To move forward we would ask for additional annotations with clear instructions, analyse the cost of both error types and compare representations on a separate validation set. After exploring these examples, this test set should no longer be used both as ground for new decisions and as independent confirmation of those improvements.
""")

md(r"""
### 8.4 Where does it fail? Almost always where a single person holds up the label

The inspection above suggested something worth measuring rather than hinting at: **a single decided vote** kept
appearing among the errors. We now split performance by how many people hold up the label, on validation
and on test, with the model and threshold already frozen. It changes nothing about the experiment: it redistributes the same correct calls.

There are two readings, and it is worth not stopping at the first.
""")

code("""
from scipy.stats import hypergeom
vote_rows=[];concentration={}
for split,label in [('val','Validación'),('test','Test')]:
    part=predictions[(predictions.split==split)&predictions.target_hard.notna()]
    prob=np.array([r[selected] for r in part.probabilities])
    pred=(prob>m['threshold']).astype(int);truth=part.target_hard.to_numpy(int)
    nv=part.n_votes.to_numpy(int)
    for k in sorted(set(nv)):
        g=nv==k;yy,pp=truth[g],pred[g]
        tpr=(pp[yy==1]==1).mean() if (yy==1).any() else np.nan
        tnr=(pp[yy==0]==0).mean() if (yy==0).any() else np.nan
        vote_rows.append({'Conjunto':label,'Votos':int(k),'Viajes':int(g.sum()),
            'Acierto':float(100*(yy==pp).mean()),'Errores':int((yy!=pp).sum()),
            'Acierto equilibrado':float(100*np.nanmean([tpr,tnr])),
            '% del conjunto':float(100*g.mean())})
    # ¿Se concentran los errores en un voto más de lo que explica su frecuencia?
    wrong=truth!=pred;N=len(truth);K=int((nv==1).sum());n=int(wrong.sum());x=int((nv[wrong]==1).sum())
    concentration[label]=dict(N=int(N),K=K,n=n,x=x,expected=float(n*K/N),
                              p_value=float(hypergeom.sf(x-1,N,K,n)))
votes_eval=pd.DataFrame(vote_rows)

fig,axs=plt.subplots(1,2,figsize=(12,4.6))
width=.36
for i,(label,color) in enumerate([('Validación',TEAL),('Test',ORANGE)]):
    part=votes_eval[votes_eval.Conjunto==label]
    xpos=np.arange(len(part))+(i-.5)*width
    axs[0].bar(xpos,part['Acierto'],width,color=color,label=label)
    for xv,acc,nn in zip(xpos,part['Acierto'],part['Viajes']):
        axs[0].text(xv,acc+.45,f'{acc:.1f} %',ha='center',fontsize=9,weight='bold')
        axs[0].text(xv,90.25,f'n={nn}',ha='center',fontsize=8,color=MUTED)
axs[0].set_xticks(np.arange(3),['1 voto','2 votos','3 votos'])
axs[0].set_ylim(90,102.4);axs[0].set_ylabel('Acierto · %')
axs[0].set_title('El acierto sube con cada voto adicional')
axs[0].legend(fontsize=9,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.14));axs[0].grid(axis='y',alpha=.12)
sizes=votes_eval[votes_eval.Conjunto=='Test'].set_index('Votos')['Viajes']
axs[1].bar([f'{k} voto' if k==1 else f'{k} votos' for k in sizes.index],sizes.values,color=ORANGE)
for i,v in enumerate(sizes.values):axs[1].text(i,v+6,str(v),ha='center',weight='bold')
axs[1].set_ylim(0,sizes.max()*1.16);axs[1].set_ylabel('Viajes de test con mayoría')
axs[1].set_title('Pero los grupos grandes son diminutos')
axs[1].text(.5,.62,'Con 72 y 6 viajes, un acierto\\ndel 100 % es compatible con\\nuna tasa de error real no nula',
            transform=axs[1].transAxes,fontsize=9,color=MUTED,ha='center',
            bbox=dict(facecolor=BG,edgecolor=MUTED,alpha=.75,boxstyle='round,pad=.5'))
finish(fig,'12_rendimiento_por_votos',
       'Modelo y umbral congelados en validación · reparto descriptivo de los mismos aciertos')

display(votes_eval.set_index(['Conjunto','Votos']).style.format({
    'Acierto':'{:.1f} %','Acierto equilibrado':'{:.1f} %','% del conjunto':'{:.1f} %'}))
display(Markdown(
    '**¿Es más que la frecuencia de cada grupo?** Los viajes de un voto son mayoría, así que concentran errores '
    'aunque el modelo no distinga entre grupos. Comparamos lo observado con lo esperado si los errores cayeran al azar '
    '(prueba hipergeométrica unilateral, descriptiva y posterior a ver los datos): '
    + ' · '.join(f"**{k}**: {v['x']} de {v['n']} errores en viajes de un voto, "
                 f"esperados {v['expected']:.1f}, p = {v['p_value']:.3f}".replace('.',',',1)
                 for k,v in concentration.items()) + '.'))
(R/'errors_by_votes.json').write_text(json.dumps(
    {'model':selected,'threshold':m['threshold'],
     'purpose':'Reparto descriptivo posterior; no modifica modelos, umbrales ni particiones.',
     'by_votes':vote_rows,'concentration':concentration},indent=2,ensure_ascii=False))
""")

md(r"""
**First reading: the model does better the more people back the label.** On validation it goes from 94.3 %
with one vote to 100 % with two or three; on test, from 92.6 % to 98.6 % and 100 %. The 21 validation errors and 27 of the 28
test errors are on single-vote journeys. The concentration exceeds what group size alone would explain, though only
just: we would expect 17.3 of 21 on validation and 23.1 of 28 on test.

**Second reading, the one that stops us celebrating the first: the groups are not comparable.** A journey with two votes and
a defined majority is, *by construction*, a journey where two people agreed — had they disagreed it would be a
tie and carry no label. The «two or more votes» group is not «the same difficulty, measured better»: it is **the subset
of cases that turned out visually unambiguous**. Put another way, we compared the hard cases with the
easy ones and discovered the hard ones are harder.

The two explanations are indistinguishable with this file:

| Explanation | What it implies | How it would be told apart |
|---|---|---|
| **Label noise** · some of those 21 and 27 «errors» are individual opinions another person would not share | Our real accuracy is underestimated; the ceiling is not 100 % | Re-annotate those cases with several people and measure how many majorities change |
| **Genuine difficulty** · cases that are ambiguous for people are ambiguous for geometry too | The model has room to improve where it is hardest | Re-annotate and check the majority holds despite the disagreement |

That 72 two-vote journeys give 100 % and 98.6 % is not reassuring in itself either: at those sizes, a 95 %
confidence interval for a proportion of 72/72 reaches down to roughly 95 %. We have not demonstrated the
model is infallible on the clear cases; we have observed that it did not fail on a handful.

**The actionable part.** If the aim were to raise measured performance, the shortest path is not a bigger model: it is
**getting a second annotation for the single-vote journeys**, which are 83 % of the decisive cases and in practice all
of the errors. That would turn part of today's «error» into documented human disagreement and would give, for the first
time, a reference against which to compare the model on the cases genuinely in dispute. Before that it is worth
fixing in the instructions what counts as different — a threshold in metres, in time or in detour — because today the
brief leaves the criterion to each annotator's perception.
""")

md(r"""
### 8.5 The ties: are they really confusing cases?

So far we have only evaluated journeys with a majority. The **31 ties** in the file — 7 of them in test — were left
out for having no binary label. They are worth looking at, because they underpin a claim we have repeated without testing:
that human disagreement signals genuine ambiguity rather than annotator carelessness.

If plotting them showed obviously identical or obviously different pairs, the tie would be panel noise and
our consensus rule would be discarding good information. If instead they show intermediate detours — neither overlap
nor clear divergence — then the tie describes the problem.

We add the model's score, which never saw these labels: a second opinion independent of the panel.
""")

code("""
tie_ids=df.index[(df.status=='tie')&(df.split=='test')].tolist()
tie_p=np.array([predictions.loc[j,'probabilities'][selected] for j in tie_ids])
decisive=np.array([r[selected] for r in predictions[(predictions.split=='test')
                   &predictions.target_hard.notna()].probabilities])
band=lambda v:((v>.4)&(v<.9)).mean()
fig,axs=plt.subplots(2,4,figsize=(16,8))
for ax,jid,prob in zip(axs.flat,tie_ids,tie_p):
    a,b=project_pair(*routes[jid])
    plot_routes(ax,a,b,legend=False)
    ax.set_title(f'p(diferentes)={prob:.3f} · modelo dice '
                 f'«{"diferentes" if prob>m["threshold"] else "iguales"}»',fontsize=10)
    ax.text(.02,.02,f'ID {jid[:8]} · 1 igual / 1 diferente',transform=ax.transAxes,fontsize=8,
            bbox=dict(facecolor=BG,alpha=.9,edgecolor='none'))
    ax.tick_params(labelsize=8);ax.set_xlabel('Este local · km',fontsize=8)
    ax.set_ylabel('Norte local · km',fontsize=8)
last=axs.flat[len(tie_ids)]
last.axis('off')
last.text(.0,.5,f'Zona de duda del modelo\\n(p entre 0,4 y 0,9)\\n\\n'
    f'Empates: {band(tie_p):.0%} de 7\\nResto de test: {band(decisive):.0%} de {len(decisive)}\\n\\n'
    'Más empates caen en esa banda,\\npero con siete casos la\\ndiferencia no es concluyente.',
    transform=last.transAxes,fontsize=10,va='center',color=INK,
    bbox=dict(facecolor=BG,edgecolor=MUTED,boxstyle='round,pad=.8'))
for ax in list(axs.flat)[len(tie_ids)+1:]:ax.axis('off')
handles,labels=axs.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper right',ncol=2,fontsize=10,frameon=False)
fig.suptitle('Los 7 empates de test · dos personas, un voto cada clase',
             x=.02,ha='left',fontsize=14,weight='bold',y=.998)
finish(fig,'13_empates_test','Sin etiqueta binaria: excluidos de todas las métricas · el modelo nunca vio estos casos')

# ¿Dónde se separan? Posición del punto de mayor distancia a la otra ruta.
tie_detail=[]
for jid in tie_ids:
    a_,b_=project_pair(*routes[jid]);la,lb=clean_line(a_),clean_line(b_)
    sa,sb=sample_line(la,400,midpoints=True),sample_line(lb,400,midpoints=True)
    da,db=distance(points(sa),lb),distance(points(sb),la)
    d=da if da.max()>=db.max() else db
    tie_detail.append({'Longitud · km':la.length/1000,'Separación máxima · m':float(d.max()),
        'Posición del máximo · % del trayecto':100*(np.argmax(d)+.5)/400,
        '% de ruta a >50 m':100*float((da>50).mean())})
tie_table=pd.DataFrame(tie_detail,index=[j[:8] for j in tie_ids]).assign(
    Zona=[predictions.loc[j,'region'] for j in tie_ids],**{'p(diferentes)':tie_p,
    'Decisión del modelo':['Diferentes' if v>m['threshold'] else 'Iguales' for v in tie_p]})
tie_table.index.name='journey_id'
display(tie_table[['Zona','Longitud · km','Separación máxima · m','Posición del máximo · % del trayecto',
                   '% de ruta a >50 m','p(diferentes)','Decisión del modelo']].style.format(
    {'Longitud · km':'{:.2f}','Separación máxima · m':'{:.0f}','Posición del máximo · % del trayecto':'{:.0f} %',
     '% de ruta a >50 m':'{:.0f} %','p(diferentes)':'{:.3f}'}))
edges_count=int(sum((r['Posición del máximo · % del trayecto']<25) or
                    (r['Posición del máximo · % del trayecto']>75) for r in tie_detail))
display(Markdown(f'En **{edges_count} de los {len(tie_ids)}** empates, la mayor separación cae en el primer o el '
    'último cuarto del trayecto: la discrepancia está en cómo empieza o termina el viaje, no en el camino intermedio.'))
""")

md(r"""
**What the pictures show, and it is more specific than we expected.** None is a perfect overlap
nor a total divergence, but they also share a structure: **in all seven, the central body of the trip
coincides and the discrepancy concentrates at one end** — the largest separation always falls in the first or last
quarter of the trip. Between 6 % and 22 % of the route lies more than 50 m from the other; the rest overlaps.

That suggests the tie does not come from «two different paths», but from **how the trip starts or ends**: the actual
pickup point against the predicted one, the last few metres to the destination, an approach manoeuvre. And there the question
«are these the same route?» has no obvious visual answer: one person may consider the trip the same
with a shifted pickup, and another that the route is already different. The disagreement describes a genuine ambiguity in the
problem, not carelessness, and supports treating the tie as an absent label rather than forcing a 0 or a 1.

It is worth not over-reading seven cases: this is an observation about the test ties, not a law of the dataset. But it is
checkable and it points at a concrete business question — whether what matters for pricing is the trip or also
its endpoints — that the annotation instructions do not resolve today.

**What they do not show.** The model does *not* reproduce that doubt reliably. Three of the seven fall in its
intermediate zone, against 9 % of the rest of test, but on the other four it scores with confidence — 0.091 and 0.261 for pairs
that split the panel down the middle. With seven cases the comparison is not conclusive (one-sided Mann-Whitney,
p = 0.13), and besides, XGBoost's probability is not a measure of uncertainty: it is a point estimate with no
interval. That the model should hesitate where people hesitate is a desirable property that **is not demonstrated here**.

That points at concrete work: quantify the model's uncertainty with a method that offers guarantees
— conformal prediction over the validation set, for instance — and check whether its abstained cases coincide with
the ties and the split votes. It needs more ties than there are here: 31 across the whole file is few for
measuring that overlap with useful precision.
""")

md(r"""
### 8.6 Who does the model agree with?

If 80 % of the labels are signed by a single person, «being right» means *agreeing with that person*. That raises
a question the global metrics do not answer: **is it equally accurate with every annotator?**

We take the validation and test journeys where there was **a single annotator with a decided vote**, group them by who
it was, and measure the frozen model's accuracy in each group. These are different partitions of the same already-inspected test set:
a descriptive breakdown, not a new evaluation.
""")

code("""
judge_rows=[]
for ann in sorted(panel.index):
    ids=[j for j,anns in by_journey.items() if set(anns)=={ann} and len(decided(anns[ann]))==1
         and predictions.loc[j,'split'] in ('val','test') and predictions.loc[j,'target_hard'] is not None]
    if not ids: continue
    prob=np.array([predictions.loc[j,'probabilities'][selected] for j in ids])
    truth=np.array([predictions.loc[j,'target_hard'] for j in ids],dtype=int)
    pred=(prob>m['threshold']).astype(int)
    judge_rows.append({'Anotador':ann,'Viajes':len(ids),'Acierto':100*(truth==pred).mean(),
        '% que etiquetó «diferentes»':100*truth.mean(),
        '% «diferentes» en todo el archivo':float(panel.loc[ann,'% «diferentes»'])})
judges=pd.DataFrame(judge_rows).set_index('Anotador')
enough=judges[judges.Viajes>=40]

fig,axs=plt.subplots(1,2,figsize=(13,4.8))
colors=[ORANGE if v<92 else TEAL for v in enough['Acierto']]
axs[0].bar([str(a) for a in enough.index],enough['Acierto'],color=colors)
for i,(a,row) in enumerate(enough.iterrows()):
    axs[0].text(i,row['Acierto']+.5,f"{row['Acierto']:.1f} %",ha='center',weight='bold',fontsize=10)
    axs[0].text(i,85.4,f"n={int(row['Viajes'])}",ha='center',fontsize=8,color=MUTED)
axs[0].set_ylim(85,101);axs[0].set(xlabel='Anotador',ylabel='Acierto del modelo · %')
axs[0].set_title('El modelo coincide más con unos que con otros')
axs[0].grid(axis='y',alpha=.12)
axs[1].scatter(enough['% «diferentes» en todo el archivo'],enough['Acierto'],
               s=enough['Viajes']*1.1,color=TEAL,alpha=.75,zorder=3)
for a,row in enough.iterrows():
    axs[1].annotate(str(a),(row['% «diferentes» en todo el archivo'],row['Acierto']),
                    xytext=(8,-4),textcoords='offset points',fontsize=10,weight='bold')
axs[1].set(xlabel='Con qué frecuencia esa persona dice «diferentes» · %',ylabel='Acierto del modelo · %')
axs[1].set_title('Cuanto más exigente la persona, peor coincide')
axs[1].grid(alpha=.12)
finish(fig,'14_acuerdo_por_anotador',
       'Validación y test · solo viajes con un único anotador decidido · tamaño del punto ∝ número de viajes')

display(judges.style.format({'Acierto':'{:.1f} %','% que etiquetó «diferentes»':'{:.1f} %',
                             '% «diferentes» en todo el archivo':'{:.1f} %'}))
""")

md(r"""
**The model is not equally good with everyone.** Among the annotators with enough solo journeys, accuracy
runs from **88.2 % with annotator 1** (119 journeys) to **98.0 % with annotator 7** (147 journeys). Almost ten
points of difference on the same model, the same threshold and the same metric.

**And the direction is not accidental.** The right-hand panel orders it: the more often a person says «they differ», the worse
the model agrees with them. Annotator 1 is the strictest on the panel — marking «they differ» 65 % of the time — and
is the one it agrees with least; annotator 7 is the most lenient — 39.7 % — and is the one it agrees with most. Put plainly: the model
has learned **a tolerance threshold closer to annotator 7's than to annotator 1's**, and that is consistent with its
errors being mostly missed differences: 17 of 21 on validation, 25 of 28 on test. It fails towards «the same»,
just like the most lenient annotator.

**Three cautions before taking this too far.** Each group contains different journeys, so part of the
gap may be difficulty rather than criterion. Small groups are omitted from the chart (fewer than 40 journeys) because with
twenty cases a percentage point means nothing. And this is a descriptive breakdown of an already-inspected test set, not
an independent measurement.

**What it does make clear.** There is no such thing as «the» target: there is the target of a particular panel, dominated by a few
people with thresholds that differ from each other. Before putting this into production we would have to decide **what criterion
the company wants** — how many metres of detour matter for pricing — and re-annotate with that instruction, rather than implicitly inheriting
the bar set by whichever annotator looked at the most journeys.
""")

md(r"""
<a id="dinero"></a>

## 9. How much money is at stake?

Everything above optimises a perceptual question: *would these look different to a person?* But the brief
motivates the problem from **pricing**: the fare is set before the trip using the estimated distance and duration.
If the real trip differs, the gap between what was charged and what that trip would have cost is the real error.

Those two questions are not the same, and it is worth measuring how far apart they are before refining the first any further.

**How we estimate the euro.** The base fare and the minimum fare are identical for the estimated and the real route, so
they cancel out when subtracting. Only the per-kilometre and per-minute prices are needed:

$$\Delta\unicode{x20AC} = \unicode{x20AC}/\mathrm{km}\cdot(\mathrm{km}_{real}-\mathrm{km}_{est})
  + \unicode{x20AC}/\mathrm{min}\cdot(\mathrm{min}_{real}-\mathrm{min}_{est}).$$

The fares are Cabify's public ones by city, standard category, consulted on 19 October 2026. The
durations come from the OSRM map match where it is reliable (1,398 of 2,386 journeys) and, for the rest, from distance
divided by the median speed OSRM implies in that same city.

> **This is an order-of-magnitude estimate, not a financial audit.** The fares are from 2026 and the file
> is from December 2021: with Argentine inflation over that period, the historical figure would not be comparable. We assume
> the standard category for every journey, and ignore tolls, surge pricing, waiting time and promotions. Colombia and
> Ecuador do not publish a single value and use declared estimates. The durations do not incorporate real traffic.
""")

code("""
from routes.pricing import build_impact, tariff_table, EBITDA_MARGIN
impact=build_impact()
impact['p']=[predictions.loc[j,'probabilities'][selected] for j in impact.index]
impact['pred']=(impact.p>m['threshold']).astype(int)
display(Markdown('**Tarifas utilizadas.** «Estimada» marca las que no proceden de una cifra única publicada.'))
display(tariff_table()[['Ciudades','currency','per_km_local','per_min_local','per_km_eur','per_min_eur','estimated']]
        .rename(columns={'currency':'Moneda','per_km_local':'Por km (local)','per_min_local':'Por min (local)',
                         'per_km_eur':'Por km (€)','per_min_eur':'Por min (€)','estimated':'Estimada'})
        .style.format({'Por km (local)':'{:.2f}','Por min (local)':'{:.2f}',
                       'Por km (€)':'{:.3f}','Por min (€)':'{:.3f}'}).hide(axis='index'))

decided=impact[impact.target_hard.notna()]
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
bins=np.logspace(-3,1.3,40)
for lab,color,name in [(0,TEAL,'Humano: iguales'),(1,ORANGE,'Humano: diferentes')]:
    axs[0].hist(decided[decided.target_hard==lab].abs_delta_eur.clip(1e-3),bins=bins,
                color=color,alpha=.68,label=name)
axs[0].set_xscale('log');axs[0].axvline(.5,color=INK,ls=':',lw=1.2)
axs[0].text(.52,axs[0].get_ylim()[1]*.82,'0,50 €',fontsize=9,color=INK)
axs[0].set(xlabel='Desviación de precio |Δ€| · escala logarítmica',ylabel='Viajes')
axs[0].set_title('Las etiquetas se solapan mucho en euros')
axs[0].legend(fontsize=9)
share=impact.abs_delta_eur.sort_values(ascending=False).cumsum()/impact.abs_delta_eur.sum()
xs=100*np.arange(1,len(share)+1)/len(share)
axs[1].plot(xs,100*share.values,color=TEAL,lw=2.4)
axs[1].plot([0,100],[0,100],ls='--',color=MUTED,lw=1,label='Si el impacto fuera uniforme')
for q,color in [(10,ORANGE),(25,INK)]:
    y=100*share.values[int(q/100*len(share))-1]
    axs[1].plot([q,q],[0,y],ls=':',color=color,lw=1.4)
    axs[1].plot([0,q],[y,y],ls=':',color=color,lw=1.4)
    axs[1].annotate(f'{q} % de los viajes\\nacumulan el {y:.0f} %',(q,y),xytext=(14,-26),
                    textcoords='offset points',fontsize=9,color=color,weight='bold')
axs[1].set(xlabel='Viajes ordenados por |Δ€| descendente · %',ylabel='Porcentaje del |Δ€| total',
           xlim=(0,100),ylim=(0,100))
axs[1].set_title('El impacto se concentra en una minoría')
axs[1].legend(fontsize=9,loc='lower right');axs[1].grid(alpha=.12)
finish(fig,'15_euros_impacto','Tarifas públicas de 2026 sobre viajes de 2021 · categoría estándar · estimación de orden de magnitud')

from scipy.stats import spearmanr
rho_label=spearmanr(decided.target_hard,decided.abs_delta_eur).statistic
top=decided.nlargest(int(.1*len(decided)),'abs_delta_eur')
display(Markdown(
 f'**Cifras.** Precio medio estimado del viaje: **{impact.charged_eur.mean():.2f} €**. Desviación absoluta mediana: '
 f'**{impact.abs_delta_eur.median():.3f} €**, el {100*(impact.abs_delta_eur/impact.charged_eur).median():.1f} % de la '
 f'tarifa; P95 **{impact.abs_delta_eur.quantile(.95):.2f} €**. La media con signo es '
 f'**{impact.delta_eur.mean():+.4f} €**: las desviaciones se compensan y no hay una fuga sistemática en una dirección. '
 f'La correlación de Spearman entre la etiqueta humana y |Δ€| es **{rho_label:.2f}** — hay señal, pero lejos de ser '
 f'la misma pregunta. Y sin embargo, del 10 % de viajes con mayor impacto, el '
 f'**{100*top.target_hard.mean():.0f} %** está etiquetado «diferentes», frente al '
 f'{100*decided.drop(top.index).target_hard.mean():.0f} % del resto: **el juicio humano acierta justo donde más '
 'dinero hay**, y se vuelve ambiguo en la cola de desviaciones pequeñas.'))
""")

md(r"""
### 9.1 Who absorbs the difference?

The fare is closed before the trip starts. The customer pays what was agreed, whatever happens afterwards. So the deviation
**is not shared: it falls entirely on one side or the other**, and is never refunded.

| If the real trip turns out… | Who absorbs it |
|---|---|
| **longer** than predicted | Cabify and the driver: that kilometre is driven and nobody charges for it |
| **shorter** than predicted | The customer: they paid for a trip that did not happen, and there is no refund |

Looking only at the mean hides exactly this, because the two cases cancel each other out.
""")

code("""
fig,axs=plt.subplots(1,2,figsize=(13,4.9))
edges=np.linspace(-2.5,2.5,61)
over=impact.delta_eur.clip(-2.5,2.5)
axs[0].hist(over[impact.delta_eur<0],bins=edges,color=ORANGE,alpha=.8,label='El cliente pagó de más')
axs[0].hist(over[impact.delta_eur>0],bins=edges,color=TEAL,alpha=.8,label='Cabify y conductor lo absorben')
axs[0].axvline(0,color=INK,lw=1.1)
n_neg=int((impact.delta_eur<0).sum());n_pos=int((impact.delta_eur>0).sum())
axs[0].text(-2.35,axs[0].get_ylim()[1]*.88,f'{n_neg} viajes\\n{abs(impact.delta_eur[impact.delta_eur<0].sum()):.0f} €',
            fontsize=10,color=ORANGE,weight='bold',va='top')
axs[0].text(2.35,axs[0].get_ylim()[1]*.88,f'{n_pos} viajes\\n{impact.delta_eur[impact.delta_eur>0].sum():.0f} €',
            fontsize=10,color=TEAL,weight='bold',ha='right',va='top')
axs[0].set(xlabel='Δ€ · negativo: el cliente pagó de más | positivo: lo absorbe Cabify',ylabel='Viajes')
axs[0].set_title('Se compensan en el agregado, no para cada persona')
axs[0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
ratio=impact.deviation_over_margin.clip(0,20)
axs[1].hist(ratio,bins=np.linspace(0,20,61),color=TEAL,alpha=.85)
axs[1].axvline(1,color=RED,lw=1.8,ls='--')
share=100*(impact.deviation_over_margin>1).mean()
axs[1].text(1.5,axs[1].get_ylim()[1]*.8,
            f'El {share:.0f} % de los viajes\\nse desvía más de lo que\\nese viaje deja de margen',
            fontsize=10,color=RED,weight='bold',va='top')
axs[1].set(xlabel='Desviación como múltiplo del margen de ese viaje',ylabel='Viajes')
axs[1].set_title('Comparado con un margen del 3,3 %, no es calderilla')
finish(fig,'15b_euros_quien_paga',
       'Margen EBITDA del grupo Cabify en 2025 (3,3 %) aplicado a cada viaje · referencia de escala, no contabilidad unitaria')

rel=100*impact.delta_eur/impact.charged_eur
display(Markdown(
 f'**Para Cabify, en el agregado, casi nada.** El neto es {impact.delta_eur.sum():+.0f} € sobre '
 f'{impact.charged_eur.sum():.0f} € facturados: un {100*impact.delta_eur.sum()/impact.charged_eur.sum():+.2f} %, '
 f'el {abs(100*(impact.delta_eur.sum()/impact.charged_eur.sum())/EBITDA_MARGIN):.0f} % del margen. '
 f'**Pero el dinero que se mueve mal no es el neto, es el bruto:** {impact.abs_delta_eur.sum():.0f} €, un '
 f'{100*impact.abs_delta_eur.sum()/impact.charged_eur.sum():.1f} % de la facturación, frente a un margen EBITDA del '
 f'{100*EBITDA_MARGIN:.1f} %. Es **{impact.abs_delta_eur.sum()/impact.charged_eur.sum()/EBITDA_MARGIN:.1f} veces el '
 f'margen** moviéndose en direcciones opuestas y cancelándose.\\n\\n'
 f'**Y para las personas concretas, mucho.** {int((rel<-10).sum())} clientes ({100*(rel<-10).mean():.1f} %) pagaron '
 f'más de un 10 % por encima de lo que valía su trayecto real, y {int((rel<-25).sum())} pagaron más de un 25 % de '
 f'más. En sentido contrario, Cabify y el conductor absorbieron más de un 25 % del precio en '
 f'{int((rel>25).sum())} viajes ({100*(rel>25).mean():.1f} %).'))
""")

md(r"""
### 9.2 Do the model's errors cost money?

The question that matters is not how many mistakes it makes, but **what they are worth**. We break the price deviations down
by cell of the test confusion matrix.
""")

code("""
test_impact=decided[decided.split=='test']
cells=[('TN','Acierto · iguales',(test_impact.pred==0)&(test_impact.target_hard==0),TEAL),
       ('TP','Acierto · diferentes',(test_impact.pred==1)&(test_impact.target_hard==1),TEAL),
       ('FP','Falsa alarma',(test_impact.pred==1)&(test_impact.target_hard==0),RED),
       ('FN','Diferencia no detectada',(test_impact.pred==0)&(test_impact.target_hard==1),ORANGE)]
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
data=[test_impact[mask].abs_delta_eur.values for _,_,mask,_ in cells]
bp=axs[0].boxplot(data,vert=True,widths=.6,patch_artist=True,showfliers=False)
for patch,(_,_,_,color) in zip(bp['boxes'],cells):patch.set_facecolor(color);patch.set_alpha(.65)
for element in ['medians','whiskers','caps']:
    for line in bp[element]:line.set_color(INK)
axs[0].set_xticks(range(1,5),[f'{k}\\nn={int(mask.sum())}' for k,_,mask,_ in cells])
axs[0].set_yscale('log');axs[0].set_ylabel('|Δ€| · escala logarítmica')
axs[0].set_title('Los fallos caen donde hay poco dinero')
axs[0].grid(axis='y',alpha=.12)
totals=[test_impact[mask].abs_delta_eur.sum() for _,_,mask,_ in cells]
total=sum(totals)
axs[1].bar([k for k,_,_,_ in cells],[100*t/total for t in totals],
           color=[color for _,_,_,color in cells])
for i,(t,(k,_,mask,_)) in enumerate(zip(totals,cells)):
    axs[1].text(i,100*t/total+1.2,f'{100*t/total:.1f} %\\n{t:.0f} €',ha='center',fontsize=9,weight='bold')
axs[1].set_ylim(0,max(100*t/total for t in totals)*1.28)
axs[1].set_ylabel('Porcentaje del |Δ€| total de test')
axs[1].set_title('Reparto del dinero, no de los casos')
axs[1].grid(axis='y',alpha=.12)
finish(fig,'16_euros_y_errores','Test · 443 viajes con mayoría · mismo modelo y umbral congelados')

captured=100*sum(t for t,(k,_,_,_) in zip(totals,cells) if k in ('TP','FN'))
missed=test_impact[(test_impact.pred==0)&(test_impact.target_hard==1)].abs_delta_eur.sum()
flagged=test_impact[(test_impact.pred==1)&(test_impact.target_hard==1)].abs_delta_eur.sum()
display(Markdown(
 f'**El modelo recupera el {100*flagged/(flagged+missed):.1f} % del dinero en juego entre los viajes realmente '
 f'distintos**, pese a detectar solo el {100*230/255:.1f} % de los casos. Las 25 diferencias que se le escapan suman '
 f'**{missed:.2f} €** de los **{flagged+missed:.0f} €** de ese grupo. Medido en euros en lugar de en viajes, el '
 'modelo rinde mejor de lo que sugiere su exactitud.'))
""")

md(r"""
**Three conclusions, and one of them corrects a reasonable suspicion.**

**First: per journey, the money is small.** The median deviation is 11 cents, 3.1 % of an average fare
of €4.63. No customer complains about that. What makes it a problem is scale: Cabify serves hundreds of
thousands of trips a day, and an average of €0.31 of absolute deviation per journey is tens of thousands of euros
misallocated daily, even if not lost.

**Second: there is no systematic leak, but that does not make it harmless.** The signed mean is practically
zero — the net is −€9 on €11,054 billed — because the trips that cost more offset those that cost
less. The problem is that **the netting happens in the spreadsheet, not on the street**: the fare is closed
before the trip starts, so each deviation falls entirely on someone and is not refunded. 8.9 % of customers paid
more than 10 % above what their trip was worth, and on 2.6 % of journeys Cabify and the driver absorbed
more than 25 % of the fare.

And the right scale for judging it is not revenue, it is margin. Cabify closed 2025 with EBITDA of €27.2M
on €833.3M, **3.3 %**. On an average fare of €4.63, that is about 15 cents of margin per journey, against
a median deviation of 11. Seen that way: **almost half of all journeys — 48 % — deviate by more than that same
journey leaves in profit**, and on 10 % the deviation is five times the margin. The gross amount moved is equivalent to
2.1 times the company's margin. This is not loose change; it is noise the size of the profit, which today cancels out by
luck rather than by design.

Two warnings about that comparison. The 3.3 % is the **group** margin, not that of a particular trip: it serves
to give scale, not to attribute profit per journey. And the sample is dominated by Latin American cities with
low fares, while Spain contributes almost a third of group revenue, so the average fare in this
file does not represent the business.

**Third, and this is the one that most changes the verdict on the model: it fails on the cheap trips and gets the expensive ones right.**
It was reasonable to fear the opposite. The model inherited the most lenient annotator's criterion (§8.6) and almost
always errs towards «the same»: 25 of its 28 test errors are differences it does not detect. If those cases were the ones with the
largest deviation, the system would be letting through precisely what it was meant to catch.

The data say the opposite. The 25 undetected differences have a median deviation of €0.06, **lower than
that of the correct calls** (€0.19), and add up to €3 of the €113 at stake among the genuinely different journeys. That is:

> **The model catches 90.2 % of the cases, but recovers 97.4 % of the money.**

That seven-point gap is the value accuracy does not see. A report saying «90 % accuracy» describes the
system worse than one saying «it captures 97 of every 100 euros deviated». And it was not predictable from the
classification metrics: it took putting a price on each case to find out. It should not be read as a deliberate virtue of the
model — nobody trained it for this — but as a fortunate property of this file, which would need re-checking
against new data before relying on it.

**What this changes in practice.** If the goal is protecting the price, the metric should not be balanced
accuracy but **money recovered**, and the effort should concentrate on the 10 % of journeys that account for
55 % of the impact. Going from 94 % to 96 % accuracy does not move the needle if the two points gained are in the
cheap tail. This is a design recommendation, not a change we apply here: the model, the threshold and the metrics
of the experiment remain the ones frozen on validation.
""")

md(r"""
## 10. Deciding under uncertainty: conformal prediction

The model returns a probability, and that **is not a measure of uncertainty**. A value of 0.52 does not mean «I am unsure»:
it means the trees averaged out there. When the system decides with a fixed threshold, it fails on 6.3 % of the
test cases and **we do not know which ones**. That is the operational limitation, not the accuracy.

Conformal prediction changes the contract. Instead of a forced label, it produces an interval over the proportion
of «they differ» votes and turns it into a three-valued decision:

| The interval… | Decision |
|---|---|
| lies entirely below 0.5 | the same |
| lies entirely above 0.5 | they differ |
| contains 0.5 | **no decision: send to human review** |

**Why CV+ and not split conformal.** Validation already took part in choosing model, iteration and threshold: using it as
calibration would invalidate the guarantee. Holding out a training fold would leave ~50 cases in the two-or-more
votes group. CV+ uses the 1,348 training journeys via out-of-fold residuals and guarantees coverage ≥ 1 − 2α.

**Why two calibration groups.** The interval width comes from a percentile of past errors. Mixing
single-vote journeys with two-or-more ones produces an average width that is too narrow for the former and
too wide for the latter: global coverage is met by compensation and fails precisely where it matters. We do not
separate three and four votes because with 8 and 2 calibration cases they do not reach the minimum n ≥ 1/α − 1.
""")

code("""
from routes.conformal import run as run_conformal, GROUPS
conformal_path=ROOT/'results/routes_conformal/conformal.json'
if conformal_path.exists():
    stored=json.loads(conformal_path.read_text())
    curve=pd.DataFrame(stored['curve']);residual_q90=stored['residual_q90_by_group']
    calibration_sizes=stored['calibration_sizes']
else:
    curve,_,(residuals,votes_cal)=run_conformal()
    residual_q90={name:float(np.quantile(residuals[belongs(votes_cal)],.9))
                  for name,belongs in GROUPS.items()}
    calibration_sizes={name:int(belongs(votes_cal).sum()) for name,belongs in GROUPS.items()}
baseline_error=100*28/443

fig,axs=plt.subplots(1,2,figsize=(13,4.9))
# alpha=0,05 abstiene en el 84 %: queda fuera de escala para no aplastar la zona útil.
usable=curve[curve.abstention_rate<.5]
dropped=curve[curve.abstention_rate>=.5]
axs[0].plot(100*usable.abstention_rate,100*usable.error_rate_answered,'o-',color=TEAL,lw=2.2,zorder=3)
# Varios alfa pueden caer en el mismo punto: se etiquetan juntos en vez de superponerse.
merged={}
for _,row in usable.iterrows():
    merged.setdefault((round(100*row.abstention_rate,3),round(100*row.error_rate_answered,3)),[]
                      ).append(f"{row.alpha:.2f}".replace('.',','))
for (xv,yv),labels in merged.items():
    axs[0].annotate('α='+' y '.join(labels),(xv,yv),xytext=(9,6),
                    textcoords='offset points',fontsize=9)
axs[0].scatter([0],[baseline_error],color=RED,s=95,zorder=4,marker='D')
axs[0].annotate(f'Umbral fijo actual · {baseline_error:.1f} % de error, sin revisión',(0,baseline_error),
                xytext=(14,-2),textcoords='offset points',fontsize=9.5,color=RED,weight='bold')
axs[0].axhline(2,color=MUTED,ls=':',lw=1.2)
axs[0].set_xlim(-.7,16.5);axs[0].set_ylim(-.45,7.6)
axs[0].text(16.2,2.12,'objetivo de ejemplo: 2 % de error',fontsize=9,color=MUTED,ha='right')
for _,row in dropped.iterrows():
    axs[0].text(16.2,5.3,f"α={row.alpha:.2f} queda fuera de escala:\\n"
                f"{100*row.abstention_rate:.0f} % a revisión y "
                f"{100*row.error_rate_answered:.0f} % de error".replace('.',',',1),
                fontsize=8.5,color=MUTED,ha='right',va='center',
                bbox=dict(facecolor=BG,edgecolor=MUTED,alpha=.9,boxstyle='round,pad=.45'))
axs[0].set(xlabel='Viajes enviados a revisión humana · %',ylabel='Error en las decisiones automáticas · %')
axs[0].set_title('La palanca: cuánta revisión compras, cuánto error te queda')
axs[0].grid(alpha=.12)
width=.36
groups=list(GROUPS)
for i,(name,color) in enumerate(zip(groups,[ORANGE,TEAL])):
    values=[100*row['abstention_by_group'][name] for _,row in curve.iterrows()]
    axs[1].bar(np.arange(len(curve))+(i-.5)*width,values,width,color=color,
               label=f"{name} · calibración n={calibration_sizes[name]}")
axs[1].set_xticks(np.arange(len(curve)),[f"{a:.2f}".replace('.',',') for a in curve.alpha])
axs[1].set(xlabel='α',ylabel='Viajes sin decisión · %')
axs[1].set_title('La duda se concentra donde la etiqueta no está corroborada')
axs[1].legend(fontsize=8,ncol=1,loc='upper right');axs[1].grid(axis='y',alpha=.12)
finish(fig,'17_conformal_palanca',
       'CV+ sobre entrenamiento · calibración Mondrian por número de votos · evaluado en test')

display(curve[['alpha','nominal_coverage','empirical_coverage','abstention_rate',
               'error_rate_answered','n_answered','n_abstained','mean_width']]
        .rename(columns={'alpha':'α','nominal_coverage':'Cobertura garantizada',
            'empirical_coverage':'Cobertura observada','abstention_rate':'A revisión',
            'error_rate_answered':'Error en lo automático','n_answered':'Decididos',
            'n_abstained':'Sin decisión','mean_width':'Ancho medio'})
        .style.format({'α':'{:.2f}','Cobertura garantizada':'{:.0%}','Cobertura observada':'{:.1%}',
            'A revisión':'{:.1%}','Error en lo automático':'{:.2%}','Ancho medio':'{:.3f}'}).hide(axis='index'))
narrowing=100*(1-residual_q90['2 o más votos']/residual_q90['1 voto'])
display(Markdown(
 f'**El precio de la anotación.** El residuo en el percentil 90 es **{residual_q90["1 voto"]:.3f}** para los viajes '
 f'de un voto y **{residual_q90["2 o más votos"]:.3f}** para los de dos o más: un intervalo un '
 f'**{narrowing:.0f} % más estrecho** cuando la etiqueta está corroborada. En α = 0,10 eso se traduce en una tasa de '
 'revisión del 14,5 % frente al 5,1 %.'))
""")

md(r"""
### What this gives operations

**A contract, not a report metric.** Today the 0.75 threshold was chosen by an `argmax` of balanced accuracy and the
result is «we fail 6.3 % of the time and we do not know where». With the curve on the left, the conversation changes direction:
the business sets the error it tolerates and the system answers with the review volume needed.

| If you want… | You send to review | Error on automated calls |
|---|---:|---:|
| today's behaviour | 0 % | 6.3 % |
| under 3 % error | 4.1 % | 2.8 % |
| under 2 % | 6.1 % | 1.9 % |
| under 1 % | 12.9 % | **0.5 %** |

With 13 % of journeys reviewed, the error on automated decisions drops from 6.3 % to 0.5 %: **twelve times lower**.
And the other way round, if operations can only review 5 % of the volume, the table says what error remains. That can be sized
in people and in cost.

**And it puts a price on annotation.** The interval is 30 % narrower when the label has two votes instead of
one, which at α = 0.10 means going from 14.5 % to 5.1 % review. Getting a second annotation for the
single-vote journeys — 83 % of the file — would cut the review volume to about a third. That turns «we should
annotate more» into a proposal with two figures facing each other: what annotating costs against what reviewing costs.
""")

md(r"""
### 10.1 Does this fit the real volume? And if it does, does the review fit?

The brief justifies automating because Cabify serves millions of trips. That is a product claim that
no accuracy metric answers, so we measure it: the full path for a new pair — project to
metres, compute the eight variables, score — over 300 real pairs, in a single process with no batching or parallelism.
""")

code("""
cost_path=R/'inference_cost.json'
if cost_path.exists():
    cost=json.loads(cost_path.read_text())
else:
    sys.path.insert(0,str(ROOT/'scripts'))
    from benchmark_inference import measure
    cost=measure();cost_path.write_text(json.dumps(cost,indent=2,ensure_ascii=False))
steps=[('Proyectar a metros',cost['ms_projection']),('Calcular las 8 variables',cost['ms_features']),
       ('Puntuar con XGBoost',cost['ms_scoring'])]
fig,axs=plt.subplots(1,2,figsize=(13,4.4))
labels=[k for k,_ in steps];values=[v for _,v in steps]
axs[0].barh(labels[::-1],values[::-1],color=[INK,ORANGE,TEAL])
for i,v in enumerate(values[::-1]):
    axs[0].text(v+.12,i,f'{v:.2f} ms' if v>=.01 else f'{v*1000:.1f} µs',va='center',weight='bold',fontsize=10)
axs[0].set_xlim(0,max(values)*1.32);axs[0].set_xlabel('Milisegundos por pareja')
axs[0].set_title(f"El modelo no es el coste · total {cost['ms_total']:.1f} ms")
volumes=np.array([1e5,5e5,1e6,5e6])
reviews={'α=0,20 · 4,1 %':.041,'α=0,15 · 6,1 %':.061,'α=0,10 · 12,9 %':.129}
for (name,rate),color in zip(reviews.items(),[TEAL,ORANGE,RED]):
    axs[1].plot(volumes,volumes*rate,'o-',color=color,lw=2,label=name)
axs[1].axhline(2000,color=INK,ls='--',lw=1.4)
axs[1].text(1.05e5,2300,'~2.000 revisiones/día ≈ 10 personas a jornada completa',fontsize=9,color=INK)
axs[1].set(xscale='log',yscale='log',xlabel='Viajes al día',ylabel='Revisiones humanas al día')
axs[1].set_title('El cuello de botella no es la CPU, son las personas')
axs[1].legend(fontsize=9,title='Tasa de abstención',title_fontsize=9);axs[1].grid(alpha=.15,which='both')
finish(fig,'18_coste_y_escala',
       f"Medición de un proceso sobre {cost['pairs_measured']} parejas · {cost['mean_points_per_pair']:.0f} puntos de media · cota superior")

display(Markdown(
 f"**Cómputo: irrelevante.** {cost['ms_total']:.1f} ms por pareja, de los que el modelo consume "
 f"{1000*cost['ms_scoring']:.1f} µs: el coste está en la geometría, no en XGBoost. Un millón de viajes al día son "
 f"**{cost['cpu_hours_per_million']:.1f} horas de CPU**, o {cost['cores_continuous_per_million_day']:.2f} núcleos "
 'en continuo. Cabe en una máquina pequeña, y el camino obvio de optimización —vectorizar por lotes y cachear la '
 'proyección— ni siquiera hace falta tocarlo todavía.'))
""")

md(r"""
**But the human review does not fit.** And this is the conclusion that forces a redesign of the previous section's lever.

At a million journeys a day, a 12.9 % abstention rate is **129,000 reviews a day**. At 6.1 %, it is
61,000. Even if one person reviewed a case per minute without a break, it would take hundreds of full-time
reviewers. **The table above, read as a global policy, is unworkable at real scale.**

It does not invalidate the method: it invalidates how it is applied. And section 9 already contains the fix. If 10 % of
journeys account for 55 % of the economic impact, there is no reason to review at a uniform uncertainty rate. The
sensible policy combines both signals:

> **Review only where the model is unsure *and* there is money at stake.**

With a fixed budget — say 2,000 reviews a day, around ten people — the abstained cases are ranked by
estimated deviation in euros and reviewed from the top down. What goes unreviewed are doubtful cases with cent-level
impact, where being wrong costs nothing. The rest is decided automatically, and for those the coverage
guarantee still holds.

That turns two analyses that until now ran separately into a single workable policy: **section 10 says what
the model is unsure about, and section 9 says which of those cases deserve a person's time.** Neither
is enough alone: the first generates an unmanageable queue and the second cannot tell a confidently detected deviation
from a doubtful one.

We do not implement it here because it requires the real review budget, the cost per review and the internal
pricing formula, which are business decisions and data we do not have. But the two pieces — the interval and the
euro — are already computed per journey, and combining them is a ranking, not a new model.
""")

md(r"""
### The limits, which are real

**α = 0.05 is not usable.** The abstention rate jumps to 84 %, and in the single-vote group it reaches 100 %: the
system refuses to decide anything. This is not a failure of the method; it is an honest measurement of the fact that with one vote per journey the
label is not determined. No model improvement fixes that.

**Observed coverage exceeds the guarantee** at every level — 92.1 % against a nominal 80 % at α = 0.10.
That is the conservative behaviour expected of CV+, not a sign the method works better than promised.

**The guarantee is approximate.** Conformal prediction requires exchangeability between calibration and evaluation; our split is
random and the trips share streets and corridors. Empirical coverage on test supports it, but does not
demonstrate it for a new city.

**And the interval is over the opinion, not over the money.** It covers the proportion of «they differ» votes, which is the
target we have. Section 9 argues that the quantity that really matters is euros. An interval over
euros — «this trip deviated by between €2 and €7, with 90 % confidence» — would be directly actionable for a
refund decision, and is the natural continuation of this work once the economic label exists.
""")

md(r"""
<a id="recomendacion"></a>

## 11. What I would do next

### The recommendation, in one sentence

> **Deploy basic geometry with conformal prediction at α = 0.15, review by economic impact rather than by a
> fixed rate, and spend the next euro on a second annotation rather than on a bigger model.**

Unpacked: eight variables and an XGBoost, 6 ms per pair, a conformal interval calibrated separately for journeys
with one and with two or more votes. The undecided cases are ranked by estimated deviation in euros and reviewed
until the daily budget runs out; the rest is decided automatically with 1.9 % error. I would not deploy t2vec
or the road-network extension: neither demonstrates an improvement and both multiply the operational cost.

### What is demonstrated and what is not

**Demonstrated with this file and this protocol.** A small geometric representation performs well: 94.3 %
balanced accuracy on test. The 42-variable vector and the t2vec hybrid are competitive but their improvement **is
not demonstrated** — the paired bootstrap intervals include zero — so the rule frozen on
validation picks the small baseline and there is no reason to change it. The t2vec adaptation came out below, and
a later audit found that its reconstruction was failing and that the sampled loss never used END as a
negative; the corrected version improves some indicators and still does not reconstruct. This describes **our
adaptations**, not the published method.

**Measured, and more interesting than the above.** The bottleneck is not in the model. 83 % of the labels
are held up by a single person; the panel is eight annotators whose criteria differ by 25 percentage points; and
the model agrees 98 % with the most lenient annotator and 88 % with the strictest. Translated into euros, the
median deviation is 11 cents, but almost half of all journeys deviate by more than that journey leaves in
margin. And the model recovers 97.4 % of the money while catching 90.2 % of the cases: it errs on the cheap ones.

**Not demonstrated, and worth saying so.** That the model transfers to a new city: the split is random and the
trips share streets. That the favourable asymmetry in euros holds outside this file: nobody was looking for it when
training, it is a fortunate property of this sample. That the conformal guarantee holds exactly: it requires
exchangeability and here that is approximate. And that the euro figures are the real ones: they are public 2026 fares
applied to 2021 trips, at the standard category and without tolls or surge pricing.

### The next three investments, by return

**1 · A second annotation for the single-vote journeys.** They are 83 % of the file and in practice all of the
errors: the 21 on validation and 27 of the 28 on test. And it now has a price: the conformal interval is 30 %
narrower when the label is corroborated, which cuts the review rate from 14.5 % to 5.1 %. That is the
comparison to take to a committee — what annotating costs against what reviewing costs, every day.
Before commissioning it, fix in the instructions what counts as different: today the criterion is each
person's perception, which is why the 31 ties exist.

**2 · Label with euros instead of with opinions.** Section 9 shows that human judgement and economic
impact correlate at only 0.45. With the internal pricing formula — which Cabify has and we do not — the
target stops depending on eight people and becomes a measurable quantity, with a loss weighted by
impact and a threshold the business sets: «we act above €X». The 42 variables already built still apply,
and a conformal interval over euros would be directly actionable for a refund.

**3 · Validate by city before promising geographic coverage.** The random split allows the same
corridor to appear in training and in test. Until a whole city is held out, the test figure describes
performance within a known mix, not an ability to generalise.

### And what I would not do

**I would not invest further in learned representations** without first resolving the label. With 83 % of the votes
uncorroborated, the ceiling is not set by the model: going from 94 % to 96 % does not distinguish a real improvement from having fitted
annotator 2's criterion better.

**I would not deploy the road-network extension.** It contributes an improvement within the noise and in exchange demands Docker, several GB of
cartography, an OSRM service and map matches that fail on 41 % of the pairs. The operational cost is not justified
by the current evidence.

**I would not present balanced accuracy as the system's metric.** It describes the behaviour worse than «it recovers
97 of every 100 euros deviated», and it points effort at the cheap tail.

### Known outstanding work

- Review the contradictory annotations: seven people contradict themselves on the same journey and in all seven cases
  they were the only annotator, so those journeys end up with no label.
- Analyse the sensitivity of the result to the consensus rule; the labels describe one particular panel.
- Compare several seeds and use cross-validation within training to stabilise the selection.
  Hold out a fresh confirmation sample if decisions keep being made after this analysis.
- If t2vec is pursued: smaller cells, more pretraining data, geographic transfer, and
  documenting cost, latency and failures from unseen cells.
- Evaluate GPS noise, incomplete traces, map changes and new routes. More accuracy on this file does not
  guarantee operational robustness.
- Add variables to the vector that describe the trace gaps — largest jump, number of jumps, proportion
  covered — rather than imputing them: filling a gap with a plausible route would push the pair towards «the same» and
  would alter the label the annotators saw.

The original selection stays documented and untouched: the geometric baseline, with the threshold frozen on
validation. Everything added afterwards — inspection, extended EDA, annotator panel, euros, conformal and cost — is
descriptive and modifies no models, thresholds or splits.
""")

md(r"""
## 12. Reproducibility and sources

The extraction, pretraining and classification code is in **src/routes/**. The pipeline for the four original models runs with **python scripts/run_routes.py**, from the repository root, after installing **requirements.txt**. The road-network extension has a separate pipeline, described in the README. The figures are regenerated by running this notebook with its caches available.

Models and raw data stay local, in **models/** and **data/**. The split, the SHA-256 of the original file, the protocol, the hyperparameters tried, the frozen selection, predictions and metrics are all saved in **results/routes/**. The tests check invariance to line sampling, known distances, order sensitivity, split separation and valid probabilities.

The inspection keeps in **results/routes/inspection.json** the ten identifiers, selection rules, votes, predictions and model contributions, along with the SHA-256 of its weights. Each case is verified to belong to test and the contributions are checked to reconstruct the saved probabilities. The notebook modifies no weights, thresholds or metrics of the experiment.

The EDA summary is exported to **results/routes/eda.json**: quality checks, lengths, sampling and a descriptive comparison of splits. Its figures distinguish counts over the whole file from distributions studied on training only.

Reconstruction is audited with **scripts/audit_t2vec_reconstruction.py** and its **--corrected** option. The corrected retraining runs via **python -m routes.t2vec_corrected**, with `src` on the path. The separate reports make it possible to compare the original failure, the corrected objective and its results without modifying the earlier weights.

The road-network extension is reproduced with **scripts/download_geofabrik.py**, **scripts/prepare_road_extract.py**, **scripts/start_local_osrm.py**, **python -m routes.graph_extension** and **scripts/evaluate_graph_extension.py**. It needs Docker and several GB of disk for the sources. The reports are in **results/routes_graph/**; large caches and models are excluded from Git. The engine is pinned by image digest; each download keeps its date and SHA-256. The «latest» URLs can change: the hashes detect that difference, they do not guarantee the provider keeps the same file for future downloads.

- [Challenge brief](https://gist.github.com/jmontess/cd4bacd1b09301edf5f2e99ce07584d3) and [linked file](https://www.dropbox.com/s/rdm11r99pmi13if/challenge_dataset.json?dl=1).
- [Li et al., 2018: t2vec](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf): the precedent for the spatial encoder-decoder. Our adaptation and supervised evaluation are distinct from the original work.
- [Newson and Krumm, 2009](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/): a reference for interpreting the map-matching problem.
- [Koide et al., 2020](https://www.vldb.org/pvldb/vol13/p2188-koide.pdf): sequence comparison over road networks; it guides extensions, it is not an algorithm evaluated here.
- [Google Routes: pass-through points](https://developers.google.com/maps/documentation/routes/pass-through) and [API](https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes): documentation of a possible source of estimated travel times, not used in the models.
- [OpenStreetMap: attribution and licence](https://www.openstreetmap.org/copyright).
- [Geofabrik: OSM extracts](https://download.geofabrik.de/), [OSRM: code and profile](https://github.com/Project-OSRM/osrm-backend) and [map matching service](https://project-osrm.org/docs/v5.24.0/api/#match-service).
- [XGBoost: prediction and TreeSHAP contributions](https://xgboost.readthedocs.io/en/stable/prediction.html): local decomposition of the frozen model and use of the selected iteration.
""")

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'}})
nbf.write(nb,ROOT/'02_route_modelling.ipynb')
print('Notebook generado:',len(cells),'celdas')
