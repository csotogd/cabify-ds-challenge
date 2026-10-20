"""Construye el relato del experimento; los entrenamientos viven en src/rutas."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# ¿Han seguido la misma ruta?
## De dos trazados GPS a una decisión que podamos defender

**Parte 2 · Model Prototyping · Cabify Data Science Challenge**

> ### En seis líneas
>
> 1. **Basta con geometría simple.** Ocho medidas y un XGBoost dan un 94,3 % de acierto equilibrado en test. Un vector de 42 variables, una adaptación de t2vec y un híbrido no mejoran de forma demostrable: sus intervalos de confianza pareados incluyen el cero.
> 2. **El límite no es el modelo, es la etiqueta.** El 83 % de los viajes los juzga una sola persona, y el panel son ocho anotadores con criterios distintos: uno marca «diferentes» el 65 % de las veces y otro el 40 %. El modelo acierta un 98 % con el anotador indulgente y un 88 % con el exigente. Ha aprendido un umbral concreto, no una verdad.
> 3. **Traducido a dinero, la desviación mediana son 11 céntimos** sobre una tarifa media de 4,63 €. Parece poco hasta que se compara con el margen: Cabify cerró 2025 con un 3,3 % de EBITDA, así que **casi la mitad de los viajes se desvía más de lo que ese viaje deja de beneficio**.
> 4. **El modelo detecta el 90,2 % de los casos pero recupera el 97,4 % del dinero:** se equivoca en los viajes baratos. La exactitud infravalora lo que hace.
> 5. **Se puede convertir el error en una palanca.** Con predicción conforme, enviar el 6 % de los viajes a revisión humana baja el error automático del 6,3 % al 1,9 %; el 13 %, al 0,5 %. Negocio elige el punto.
> 6. **Recomendación:** desplegar geometría básica con conformal, revisar por impacto económico y no por tasa fija, y gastar el siguiente euro en una segunda anotación —no en un modelo mayor. El detalle está en la [sección 11](#recomendacion).

Tenemos dos recorridos de un mismo viaje: el previsto y el realizado, y la etiqueta que nos dan es perceptual: si una persona los considera similares al verlos en un mapa. Pero el enunciado motiva el problema desde el **precio**, que se fija antes de viajar con la ruta estimada. Esas dos preguntas no son la misma, y una parte del valor de este trabajo consiste en medir cuánto se separan: lo hacemos en la [sección 9](#dinero), después de haber resuelto el problema tal y como viene planteado.

Dos preguntas guían el recorrido. La técnica: **¿necesitamos aprender una representación compleja o bastan unas medidas geométricas bien elegidas?** Y la de negocio, que aparece en la segunda mitad: **¿cuánto cuesta que el motor de rutas se equivoque, y cuánta revisión humana hace falta para decidir el resto automáticamente?**

Compararemos una referencia de ocho variables, un vector interpretable de 42 medidas, una adaptación reducida de t2vec con decoder de reconstrucción y clasificador, y una combinación de vectores. Añadimos ampliaciones exploratorias con red viaria y una versión de t2vec corregida tras auditar su reconstrucción, realizadas después de inspeccionar el test original.

El recorrido del análisis es: entender los votos → construir representaciones → comparar en validación → comprobar en test → volver a las rutas para entender aciertos y errores. **El mejor modelo será el elegido por la regla fijada en validación**, no el que salga primero al ordenar el test.

| Método que probamos | Qué recibe el clasificador | Por qué probarlo | Qué puede perder o costar |
|---|---|---|---|
| Geometría básica + XGBoost | 8 medidas: ambas longitudes, diferencia relativa, extremos, medias de separación y máximo discreto | Comprobar cuánto resuelve una referencia pequeña y comprensible | Resume cada pareja; puede diluir la importancia de un desvío localizado |
| Vector interpretable + XGBoost | 42 medidas, incluidas las 8 anteriores: proporciones alejadas, tramos, IoU, giros y orden | Representar explícitamente las diferencias que proponíamos mirar en el mapa | Más variables no aportan necesariamente información nueva; pueden sobreajustar |
| t2vec reducido + clasificador neuronal | Dos embeddings de 64 dimensiones, más su diferencia absoluta y producto: 256 valores | Aprender patrones sin etiquetas antes de aprender el juicio humano | La auditoría posterior detectó reconstrucción fallida y un defecto en la pérdida muestreada; no representa un benchmark válido del t2vec original |
| Vector interpretable + t2vec + XGBoost | Las 42 medidas y los 256 valores anteriores: 298 en total | Comprobar si lo aprendido complementa lo que medimos a mano | Mayor complejidad; podría añadir ruido sin mejorar de forma estable |
| t2vec reducido con pérdida corregida + clasificador | La misma arquitectura y comparación de vectores, reentrenadas con un objetivo normalizado completo | Corregir el tratamiento del fin de secuencia y evaluar reconstrucción y clasificación por separado | Sigue requiriendo datos y una resolución adecuada; se documenta como experimento nuevo |
| Grafo viario y tiempos + XGBoost, solo y con geometría | Variables de calles, conectividad, tiempos estimados y calidad del encaje | Distinguir cercanía geométrica de conectividad y coste de circulación | Mapa actual, aproximaciones y encajes inciertos; comparación exploratoria posterior al test original |

**No son solo vectores de diferencias.** La referencia ya conserva las longitudes individuales; t2vec conserva ambos embeddings completos además de su comparación. No hemos hecho una ablación de «solo concatenación» frente a «solo diferencias»: para responder esa pregunta habría que probarla con validación.

> Los resultados que siguen proceden del archivo real del challenge. No son simulaciones. Las estimaciones del experimento de fotografía pertenecen al notebook anterior.

El enunciado menciona 5.000 viajes seleccionados. El archivo enlazado contiene **2.386 viajes distintos**. La petición 3.000/1.000/1.000 se interpreta como entrenamiento/validación/test; conservamos su proporción 60/20/20 sin duplicar viajes: **1.432 / 477 / 477**.
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
from rutas.data import load_data
from rutas.features import build_features, project_pair, clean_line, sample_line, BASE_FEATURES
from rutas.road_graph import load_graph

# Antes de reproducir las figuras: python scripts/run_routes.py
# Este notebook lee los resultados congelados; no vuelve a seleccionar con test.
R = ROOT/'resultados/rutas'
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
FIG=ROOT/'figuras/rutas';FIG.mkdir(parents=True,exist_ok=True)
def finish(fig,name,note='Dataset del challenge · resultados reproducibles · semilla 20261014'):
    fig.text(.02,.015,note,fontsize=9,color=MUTED)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    plt.show();plt.close(fig)
names={'geometria_base':'Geometría básica','vector_interpretable':'Vector interpretable',
       't2vec_decoder_clasificador':'t2vec + clasificador','hibrido_vector_t2vec':'Vector + t2vec'}
""")

md(r"""
## 1. Antes de modelar: ¿cuántos ejemplos tenemos realmente?

Una fila es una **anotación**, no un viaje nuevo. Además, algunas personas figuran varias veces para el mismo viaje. Si contamos esas filas como votos independientes, sobreponderamos a esa persona.

Aplicamos una regla anterior al entrenamiento:

1. Un anotador cuenta como máximo una vez por viaje.
2. Sus respuestas repetidas idénticas se colapsan.
3. Si combina «no sé» con una única decisión, conservamos la decisión.
4. Si la misma persona dice tanto «iguales» como «diferentes», su voto queda indeterminado.
5. Entre personas distintas, conservamos el desacuerdo en vez de inventar una certeza.

Encontramos **123 registros adicionales del mismo anotador-viaje**, con siete parejas anotador-viaje contradictorias. Esto no significa que los demás votos sean verdad objetiva.
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
### 1.1 ¿No hay nulos, o no hay problemas de datos?

Son preguntas distintas. Primero revisamos el archivo original: campos ausentes, valores JSON nulos y contenidos vacíos. Después miramos dentro de las listas de coordenadas. Un campo llamado «ruta» puede existir y, aun así, no contener una trayectoria utilizable.

Cada punto debe tener dos números finitos en el orden **[latitud, longitud]**, con latitud entre −90 y 90 y longitud entre −180 y 180. Además, la ruta necesita al menos dos posiciones distintas. Que pase estas comprobaciones no garantiza precisión del GPS ni que siga una carretera.

**Alcance de esta ampliación:** este EDA se incorpora después de haber evaluado los modelos; lo situamos aquí para facilitar la lectura. La calidad estructural se audita en todo el archivo. Las distribuciones que relacionamos con decisiones de modelado se estudian en entrenamiento; la comparación de conjuntos es descriptiva. No se cambian los modelos, etiquetas, filtros ni umbrales a partir de este análisis.
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
**Lectura:** no encontramos campos ausentes, nulos, rutas vacías ni coordenadas inválidas con estas reglas. Tampoco hay valores ausentes o infinitos en las características calculadas. Por tanto, este archivo **no necesita una imputación numérica**.

Pero sí hay ausencia de información supervisada: **139 viajes sin voto válido** y **31 empates**. En total, 170 viajes carecen de una mayoría binaria definida. «No sé» es una respuesta humana válida, no un null ni un voto «iguales». No la rellenamos con la clase más frecuente. Más adelante definimos cómo aprovechar los votos disponibles.

Faltan también variables que el esquema nunca contiene: **tiempos de cada punto, duración observada y una ruta validada contra la carretera**. La ausencia de nulos no crea esa información.
""")

md(r"""
### 1.2 ¿Qué tamaño tienen los recorridos?

No confundamos la **longitud del recorrido** con el **número de coordenadas**. Calculamos la primera sumando las distancias entre puntos consecutivos de la polilínea:

$$L=\sum_{i=1}^{n-1}\|x_{i+1}-x_i\|.$$

Los $x_i$ están proyectados a metros en un sistema local común a la pareja. Es la longitud del trazado representado, no una garantía de kilómetros realmente conducidos. Un tramo sin observaciones queda unido por una línea y puede ocultar curvas o un recorrido distinto.

En las tablas y gráficas siguientes, cada viaje de entrenamiento pesa una vez. Mostramos la **mediana** y percentiles porque una media puede ocultar la cola de viajes largos. P95 significa que el 95 % de las rutas no supera ese valor.
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
Los extremos merecen inspección, no eliminación automática. Un gran desvío puede ser precisamente el ejemplo que debemos reconocer. No aplicamos recortes por percentiles ni borramos viajes largos: antes habría que distinguir un fallo de datos de una ruta válida poco frecuente.
""")

md(r"""
### 1.3 ¿Tener más puntos significa tener mejor información?

Dos rutas de 10 km pueden contener 100 y 1.000 puntos. Eso no convierte la segunda en diez veces más larga ni justifica que tenga diez veces más peso. Además, algunos puntos consecutivos repiten exactamente la misma coordenada: describen un paso de longitud cero.

Medimos tres cosas distintas: **cantidad de puntos**, **repeticiones consecutivas** y **mayor distancia entre dos observaciones consecutivas**. Esta última detecta tramos representados con poca información. La referencia de 1 km que aparece abajo es solo una marca de inspección; no es una regla validada para detectar errores ni un filtro aplicado.

**No es solo un ejemplo teórico:** en 2.376 de los 2.386 viajes —el **99,6 %**— los dos trazados tienen distinto número de puntos. La mediana es 249,5 puntos para la estimada y 198 para la real, en el archivo completo. Que casi nunca coincidan exactamente no significa que todas las diferencias sean grandes: por eso miramos también la magnitud de la proporción entre ambos recuentos.
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
En la geometría implementada eliminamos repeticiones **consecutivas** antes de formar las líneas. Conservamos revisitas posteriores al mismo lugar, que pueden representar un bucle real. Las medias y proporciones se calculan sobre posiciones equiespaciadas por longitud; no sobre los puntos brutos. Esto reduce el peso artificial de zonas con muchos registros, pero **no reconstruye información perdida**.

En t2vec también remuestreamos por longitud, con un máximo de 64 posiciones, y colapsamos celdas consecutivas iguales. Ese límite controla el tamaño de la secuencia, aunque puede perder detalle en rutas largas. Sin tiempos no podemos interpretar las repeticiones como minutos parado ni calcular velocidad.
""")

md(r"""
### 1.4 ¿Representamos el mismo problema en todos los conjuntos?

La muestra mezcla zonas geográficas y grados de certeza humana. Revisamos ambas cosas antes de interpretar una cifra global.

La zona no viene como campo original: es una **aproximación** basada en el comienzo de la ruta real y su cercanía a centros urbanos predefinidos, con un radio máximo de 65 km. No equivale a un municipio validado. Elegimos las cinco zonas más frecuentes en entrenamiento y agrupamos el resto para que la comparación sea legible.
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
Las proporciones de clases son parecidas porque la partición se estratificó por el estado inicial de las etiquetas. Las zonas frecuentes aparecen en los tres conjuntos, pero eso no elimina dependencia entre recorridos que comparten calles. Además, la mayoría de viajes solo tiene una persona anotadora: una etiqueta 0 o 1 no implica unanimidad de un panel amplio.

Agrupar por longitud ayuda a detectar si la etiqueta se asocia al tamaño del viaje. No convierte esa asociación en una regla universal ni en un criterio para eliminar observaciones. Lo usaremos como contexto para interpretar las métricas.
""")

md(r"""
### 1.5 ¿Cuántas personas sostienen cada etiqueta?

El apartado anterior contaba **personas que miraron** cada viaje. Aquí contamos **votos decididos**: las abstenciones
(«no sé») y los anotadores que se contradicen a sí mismos no suman. Es el número que realmente sostiene cada etiqueta.

Importa porque la palabra «mayoría» sugiere un panel y aquí casi nunca lo hay. Un viaje con un solo voto tiene una
etiqueta tan binaria como otro con tres, pero **no podemos saber si una segunda persona habría coincidido**.
Esa diferencia no aparece en ninguna métrica agregada; la recuperamos en la sección 8.4.
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
#### ¿Y quién vota? El panel son ocho personas, muy desiguales entre sí

Si una etiqueta la sostiene casi siempre una persona, conviene saber **qué persona**. El panel tiene ocho anotadores,
pero no reparten el trabajo: el más activo hizo 44 veces más anotaciones que el menos activo.

Dos cosas nos interesan. **Cuánto pesa cada uno** —sobre todo en viajes donde nadie más votó, porque ahí su criterio
es la etiqueta sin contrapeso— y **cuánto se parecen entre sí** cuando coinciden sobre el mismo viaje.
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
**El reparto no es equilibrado.** El anotador 2 firma 918 anotaciones —el 30 % del archivo— sobre 856 viajes, y en
**612 de ellos es la única persona que votó**: su criterio, sin contraste, fija el 26 % de las etiquetas del dataset.
Sumando los tres más activos (2, 7 y 1) se llega a 1.335 de los 1.897 viajes de anotador único. En el otro extremo,
el anotador 5 hizo 21 anotaciones.

**Y los criterios personales difieren.** El anotador 1 marca «diferentes» el 65 % de las veces; el 7, el 39,7 %. Son
25 puntos de diferencia entre dos personas que vieron casi 500 y más de 600 viajes respectivamente. No es una muestra
pequeña ni una casualidad: es un umbral de tolerancia distinto.

Esa brecha se confirma al cruzarlos. Entre los pares con suficientes viajes compartidos, **1 y 7 son justo los que
menos coinciden: 82,6 % sobre 46 viajes**, mientras que otros pares llegan al 94-100 %. Las casillas marcadas con «·»
tienen menos de 15 viajes en común y no se interpretan; con 4 u 8 comparaciones, un 100 % no significa nada.

**La consecuencia para el modelo.** El objetivo no es «¿se parecen estas rutas?» sino «¿diría esta persona concreta
que se parecen?». En el 80 % de los casos esa persona es una sola, y con una probabilidad alta es el anotador 2 o el 7.
El modelo aprende, en parte, **su** umbral de tolerancia. Si mañana cambia el panel, cambia el objetivo, y la exactitud
medida aquí no se transfiere automáticamente.

**El porcentaje bruto exagera el acuerdo**, porque dos personas que digan «diferentes» la mayor parte del tiempo
coincidirán mucho sin compartir criterio. La kappa de Cohen descuenta esa coincidencia esperable y ordena los pares
de otra forma: 1 y 7 bajan a **0,66**, y 1 y 8 a **0,63**, frente a varios pares por encima de 0,87. Son valores que
se suelen describir como acuerdo «sustancial», no «casi perfecto»: hay criterio compartido, pero con holgura real.

Esto refuerza lo que ya decían §1.5 y §8.4 desde otro ángulo: la limitación dominante de este prototipo no es la
capacidad del modelo, sino **cuántas personas y con qué criterio sostienen cada etiqueta**. La §8.6 cierra la
pregunta por el otro lado: a qué anotador se parece más el modelo que hemos entrenado.
""")

md(r"""
### 1.6 Entonces, ¿hay que normalizar?

**Sí hay que hacer comparables las distancias y el muestreo; no hay que dividirlo todo por la longitud.** Son operaciones distintas:

| Qué queremos resolver | Tratamiento y motivo | Estado del prototipo |
|---|---|---|
| Latitud y longitud están en grados | Proyectar ambas rutas a un sistema local común en metros. Un grado de longitud no representa la misma distancia en todas las latitudes. | Implementado para las características geométricas |
| Las rutas tienen distintas densidades de puntos | Remuestrear por longitud y eliminar duplicados consecutivos al construir la polilínea. Así una zona muy muestreada no pesa más solo por tener más puntos. | Implementado; no rellena huecos con información real |
| Un desvío puede ser grande en metros o relativo al recorrido | Conservar medidas en metros **y** versiones divididas por una escala común de longitud. | El vector ampliado incluye ambas; la referencia incluye longitudes, distancias absolutas y diferencia relativa de longitud |
| Variables con unidades muy diferentes entran en una red neuronal | Estandarizar con media y desviación calculadas solo en entrenamiento. Aplicar esos mismos parámetros a validación y test. | Implementado en la cabeza neuronal sobre embeddings; se limita después a [−10, 10] |
| Variables con unidades diferentes entran en XGBoost | Sus árboles comparan cada variable con umbrales propios; no necesitan z-scores para poner metros y proporciones en una misma escala. | Las entradas geométricas e híbridas no se estandarizan |
| Proporciones e IoU ya son adimensionales | Conservar sus valores entre 0 y 1; no dividir otra vez por kilómetros. | Implementado |
| Hay una cola de recorridos largos | Usar ejes logarítmicos para verla. Transformar variables o recortar extremos sería otra decisión, a validar. | El log del EDA es solo visual; no se han cambiado las entradas ni eliminado extremos |

La escala común que usamos es

$$s=\max\left(\frac{L_e+L_r}{2},1\text{ m}\right),\qquad
d_{\mathrm{rel}}=\frac{d}{s},\qquad
\Delta L_{\mathrm{rel}}=\frac{L_r-L_e}{s}.$$

El mínimo de 1 metro protege la división en casos degenerados; ninguna de las rutas válidas de este archivo se acerca a ese mínimo. El valor $s$ se calcula con las dos rutas del propio viaje, sin aprender de otros viajes ni de sus etiquetas.

Por ejemplo, una separación media de **100 m** equivale al **10 %** de una escala de **1 km**, y al **0,5 %** de una escala de **20 km**. Son lecturas complementarias: la segunda evita tratar igual un desvío proporcionalmente pequeño y uno grande; la primera conserva que 100 metros pueden ser otra calle en ambos casos. **Si guardásemos solo la versión relativa, podríamos ocultar una diferencia que una persona considera importante.**

No centramos, rotamos ni ajustamos a tamaño unidad cada ruta por separado: podríamos hacer coincidir recorridos que realmente pasan por lugares diferentes. La proyección común conserva la separación entre las dos.

Para la cabeza neuronal, en cambio, estandarizar una componente $x_k$ significa

$$z_k=\frac{x_k-\mu_{k,\mathrm{train}}}{\sigma_{k,\mathrm{train}}}.$$

Es un cambio de escala de las entradas de la red, no una normalización de kilómetros. No se recalculan medias y desviaciones en validación o test, y las componentes constantes se manejan sin dividir por cero. El vocabulario y el encoder también se ajustan exclusivamente dentro del conjunto de entrenamiento.

**Qué nos llevamos del EDA:** no hace falta imputar nulos; sí respetar unidades y densidad de muestreo, conservar escalas absolutas y relativas, y recordar que los huecos del trazado y la escasez de votos limitan lo que podemos aprender. Cualquier nueva limpieza o cambio de representación se evaluaría como un experimento nuevo, sin sustituir silenciosamente los resultados que siguen.
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
## 2. El objetivo representa el juicio humano, incluida su incertidumbre

Para un viaje $j$, usamos

$$p_j=\frac{\text{anotadores válidos que dicen «diferentes»}}{\text{anotadores válidos que toman una decisión}}.$$

Si dos personas dicen «diferentes» y una dice «iguales», el objetivo es $2/3$. No afirmamos que esa sea la probabilidad física de que las rutas sean distintas: es una estimación del juicio de este panel. Con un único voto también hay incertidumbre, aunque $p_j$ sea 0 o 1.

Entrenamos con entropía cruzada contra $p_j$, con **igual peso por viaje**. Los empates aportan un objetivo 0,5. Los viajes sin votos válidos no aportan pérdida supervisada; sus rutas de entrenamiento sí pueden participar en el preentrenamiento sin etiquetas.

Para evaluar clasificación binaria usamos la mayoría definida, $y_j=1[p_j>0,5]$, excluyendo empates y casos sin voto válido. También medimos la pérdida probabilística contra los votos incluyendo los empates.

> El test contiene 477 viajes, pero solo 443 tienen mayoría definida. El rendimiento binario se refiere al **92,9 %** del test. No conocemos la exactitud sobre los otros 34 viajes.

Todas las anotaciones de un viaje permanecen juntas. No hay parejas de geometrías exactamente duplicadas entre identificadores. La partición se congeló antes de evaluar modelos y se mantuvo al depurar los votos. Este es un test aleatorio dentro de la mezcla de ciudades; **no demuestra generalización a una ciudad nueva**.
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
## 3. La primera representación: medir antes de comprimir

Las coordenadas se proyectan a metros en un sistema local común a **ambas** rutas. No centramos, rotamos ni escalamos cada una por separado: eso podría convertir calles distintas en formas aparentemente idénticas.

Para una posición $x$ de la ruta $A$ calculamos $d(x,B)=\min_{y\in B}\|x-y\|$. La posición más cercana puede estar **en medio de un segmento**, aunque no haya un punto GPS registrado allí.

Repetimos en ambos sentidos. Las medias y proporciones se aproximan con posiciones equiespaciadas por longitud (20 metros, mínimo 16 y máximo 1.600). Así cada metro tiene aproximadamente el mismo peso. La precisión de esta aproximación queda limitada en rutas muy largas o con desvíos muy breves.

**Referencia:** ocho variables —longitudes, diferencia relativa, separación de extremos, distancias medias direccionales y Hausdorff discreto— más XGBoost.

**Tu vector ampliado:** 42 variables. Añade percentiles 95, versiones relativas, proporciones y tramos continuos alejados a 20/50/100 metros, IoU de corredores, rectitud, giros y Fréchet discreto sobre 64 puntos por ruta.

Tres ejemplos aclaran qué aporta cada medida. **La media** describe la separación habitual; **el máximo de las distancias mínimas en ambos sentidos —Hausdorff—** puede detectar un desvío importante aunque sea breve, pero es sensible a un punto anómalo. **Una proporción de 0,20 a más de 50 metros** significa que aproximadamente un quinto de la longitud de esa ruta queda alejado de la otra, aunque la media sea pequeña.

Para la **IoU** dibujamos un corredor alrededor de cada ruta y dividimos el área de intersección entre el área de unión. Cerca de 1 hay mucho solapamiento; cerca de 0, poco. El ancho importa: corredores de 100 metros pueden solaparse aunque los coches pasen por calles distintas. Esta medida no consulta la red de carreteras.

$$\Delta L_{\mathrm{rel}}=\frac{L_{\mathrm{real}}-L_{\mathrm{estimada}}}{(L_{\mathrm{real}}+L_{\mathrm{estimada}})/2}.$$

Conservamos valores absolutos y relativos. La IoU ya está normalizada; no la dividimos por la longitud.

**¿Dónde aprende XGBoost?** Estas medidas se calculan sin aprender de las etiquetas. Después, el modelo combina árboles de decisión aprendidos con los votos: una separación de 50 metros puede recibir una lectura distinta según la longitud, los extremos y las demás variables. No imponemos a mano una regla única de «si supera 50 metros, son diferentes».

> Compartir todos los puntos no implica recorrerlos en el mismo orden. Hausdorff e IoU pueden ignorar esa diferencia; Fréchet aporta información de orden. Sus versiones discretas son aproximaciones, no mediciones exactas de una curva continua.
""")

code("""
from rutas.features import features_xy
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
## 5. Aprender un vector: qué significa t2vec en este prototipo

### 5.1 El problema de partida: una lista de puntos no tiene tamaño fijo

Una ruta puede tener 100 puntos y otra 300. **Esto sucede en el dataset:** 2.376 de los 2.386 viajes tienen distinto número de puntos entre sus dos trazados. Por ejemplo, el viaje `7f29deb0-5c8c-11ec-8622-eff6f2c898b6` contiene 323 puntos estimados y 97 reales. Más puntos no implica más kilómetros ni otra ruta: también puede significar un muestreo más denso.

Queremos una función $f_\theta$ que convierta cualquier secuencia admitida en un vector del mismo tamaño:

$$T=(p_1,\ldots,p_n)\quad\longmapsto\quad z=f_\theta(T)\in\mathbb R^d.$$

Aquí $n$ puede cambiar, mientras que $d$ queda fijado al diseñar la red. El parámetro $\theta$ representa los pesos que aprendemos. Tener el mismo tamaño permite comparar vectores y alimentar un clasificador. **No garantiza que esos vectores conserven las diferencias que interesan al negocio:** eso se debe comprobar.

Hay tres niveles distintos: la coordenada GPS; la celda del mapa a la que asignamos esa coordenada; y el vector aprendido de toda la trayectoria. También existe un vector aprendido por celda, que es la entrada numérica del encoder. No hay que confundir ese vector local con el resumen final del viaje.

**Qué hace el encoder al leer:** empieza con una memoria numérica, lee A y la actualiza; después lee C y combina lo que recuerda con C; después hace lo mismo con D. Al terminar devuelve esa memoria. Nuestra GRU realiza la lectura en ambos sentidos y concatena sus resúmenes. No asigna una componente a cada punto: toda la secuencia influye en un vector de tamaño fijo. Una secuencia de 3 celdas y otra de 30 producen ambas 64 números. El límite de 64 puntos del preprocesamiento y las 64 componentes del vector son dos decisiones distintas que aquí coinciden numéricamente.

Conviene presentar los tres componentes antes de seguir:

| Componente | Entrada | Salida | Para qué se utiliza |
|---|---|---|---|
| Encoder | Una ruta, expresada como secuencia de celdas | Un vector de 64 números | Resumir la ruta |
| Decoder de reconstrucción | Ese vector y el contexto de celdas anteriores | Predicciones de celdas sucesivas, incluido FIN | Dar una señal para entrenar el encoder |
| Clasificador de parejas | Los dos vectores y sus comparaciones | Probabilidad de «diferentes» | Resolver la tarea del challenge |

**El decoder no es el clasificador.** Se entrena en la primera fase y no interviene en la predicción final de la etiqueta.

### 5.2 La idea original: aprender sin pedir etiquetas humanas

[Li et al. (2018)](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf) proponen representar trayectorias mediante un encoder-decoder y un objetivo que tiene en cuenta la proximidad espacial de las celdas. Su tarea original es similitud y búsqueda de trayectorias; aquí añadimos una clasificación basada en anotaciones humanas.

Pensemos en una secuencia de celdas **A → B → C → D**. Generamos una versión degradada, por ejemplo **A → C → D**, y perturbamos algunas posiciones. El encoder resume esa versión; el decoder intenta reconstruir la secuencia original. Conocemos la respuesta porque conservamos la trayectoria antes de alterarla. Es un aprendizaje **autosupervisado**: la señal de entrenamiento sale de los datos, sin votos de personas.

En nuestro caso se eliminan puntos: no se deja un símbolo `[MASK]` señalando dónde faltan. El objetivo es reconstruir la secuencia completa de celdas original, no solo adivinar el punto eliminado. Si el decoder predice **A → C → C → D**, comparamos esas predicciones con **A → B → C → D** y ajustamos los pesos de ambas redes para reducir la pérdida. Tampoco le pedimos interpolar coordenadas GPS exactas: predecimos las celdas de la representación discretizada.

El reto obliga a conservar patrones útiles pese a ruido y muestreo irregular. No es una garantía de recuperar calles que nunca observamos: si varias rutas explican los puntos conservados, el modelo puede equivocarse. La proximidad entre celdas informa sobre geografía, no sobre puentes, sentidos únicos o conexiones viarias.

### 5.3 Qué significa «penalizar según la distancia»

Imaginemos que el siguiente punto corresponde a la celda B. Predecir una celda adyacente y predecir otra a varios kilómetros no deberían recibir exactamente el mismo tratamiento. En nuestra implementación se reparte el objetivo entre ocho celdas próximas:

$$q(c\mid B)=\frac{\exp[-d(c,B)/250]}{\sum_{u\in N_8(B)}\exp[-d(u,B)/250]},\qquad c\in N_8(B).$$

Las distancias están en metros y $N_8(B)$ incluye la propia celda. La celda correcta recibe el mayor peso; las próximas, menos. Para no evaluar todas las celdas en cada paso, nuestro prototipo añade 64 celdas negativas muestreadas y calcula una pérdida con ese subconjunto. **Este normalizador muestreado no es el softmax completo ni reproduce exactamente la aproximación NCE del artículo.** No interpretamos su valor como una probabilidad exacta de toda la ruta.

El decoder utiliza *teacher forcing*: al predecir la siguiente celda durante el entrenamiento recibe la celda anterior correcta. Facilita aprender la reconstrucción, pero una reconstrucción autónoma puede acumular errores. Nuestro objetivo final es aprovechar el encoder, no evaluar un generador de trayectorias.

### 5.4 Lo implementado: una versión pequeña, con decisiones explícitas

Implementamos una **adaptación reducida**, no una reproducción exacta de su arquitectura ni una medición del rendimiento del paper:

1. Discretizamos posiciones en celdas UTM de 250 metros, identificadas también por su zona.
2. El vocabulario se aprende solo con una fracción del entrenamiento. La validación interna del preentrenamiento contiene viajes distintos.
3. El encoder GRU bidireccional comprime cada ruta en 64 números.
4. Un decoder GRU intenta reconstruir la secuencia original desde una versión con puntos eliminados y pequeñas perturbaciones GPS.
5. La pérdida favorece las celdas verdaderas y sus vecinas espaciales. Utiliza ocho vecinas y 64 negativos por posición, con normalizador muestreado.
6. Congelamos el encoder y entrenamos un clasificador pequeño con los votos humanos.

El modelo tiene dimensiones menores, una cuadrícula y proyección diferentes, 12 épocas como presupuesto máximo y no incluye el preentrenamiento separado de celdas del artículo. Usa teacher forcing en el decoder. La pérdida de reconstrucción muestreada sirve como diagnóstico, no como log likelihood exacta de todo el vocabulario.

**Hay dos salidas diferentes:** el decoder de preentrenamiento reconstruye trayectorias; la cabeza supervisada produce la probabilidad de «diferentes».

No usamos la ruta estimada y la real como pareja positiva durante el preentrenamiento: podrían ser precisamente distintas.

El vocabulario y los pesos del encoder se aprenden con **2.578 trayectorias de 1.289 viajes de entrenamiento**; otros 143 viajes del entrenamiento sirven para validación interna de esta fase. Cada trazado se limita a 64 puntos antes de discretizarlo. El encoder usa embeddings de celda de 32 componentes y una GRU bidireccional de 32 unidades por sentido: el resultado son 64 componentes por ruta. El artículo incluye un preentrenamiento de celdas que aquí se omite. El [repositorio original](https://github.com/boathit/t2vec) permite distinguir la implementación publicada de esta adaptación.

Esos 64 números **no son 64 medidas con nombres conocidos**. El modelo decide cómo distribuir la información entre ellos. Por eso no podemos señalar una componente y afirmar que representa «el desvío máximo». Nuestro vector geométrico sí tiene esa interpretabilidad, porque lo hemos definido explícitamente.

### 5.5 Del vector de una ruta a una decisión sobre dos rutas

Para cada pareja construimos $[z_e,z_r,|z_e-z_r|,z_e\odot z_r]$: 256 valores. La cabeza es una red de una capa oculta, dropout y salida sigmoide. También probamos combinar esos 256 valores con las 42 medidas interpretables y usar XGBoost.

Los primeros 128 valores conservan **los dos vectores completos**. Los siguientes 64 explicitan diferencias por componente; los últimos 64 son productos componente a componente. Estos productos no equivalen por sí solos a una similitud calibrada. Son entradas adicionales que el clasificador puede aprovechar o ignorar.

En esta segunda fase sí usamos los votos humanos. Para un viaje con objetivo $p_j$, el clasificador predice $\hat p_j=\sigma(h_\phi(x_j))$ y minimizamos

$$\mathcal L_j=-p_j\log\hat p_j-(1-p_j)\log(1-\hat p_j).$$

El encoder permanece congelado: aprendemos $\phi$, los pesos de la cabeza, sin modificar $\theta$. El decoder de reconstrucción ya no interviene. Por tanto, hay dos objetivos sucesivos: **representar una trayectoria** y **aprender cómo juzgan las personas una pareja**. Que el primero se resuelva bien no implica resolver automáticamente el segundo.

### 5.6 Qué permite concluir el experimento

Nuestra variante obtiene peores métricas que las medidas geométricas. Esto evalúa esta configuración concreta. El tamaño del corpus, la resolución de 250 metros, el presupuesto de entrenamiento y las celdas no vistas son posibles explicaciones, no causas demostradas. Harían falta ablaciones controladas para separarlas. Tampoco se ha comparado «solo concatenación» con «solo diferencias»; nuestra entrada contiene ambas cosas.
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
La media por ruta de puntos en celdas no vistas alcanza aproximadamente el 24 % en validación y el 22 % en test. Se asignan al centroide de entrenamiento más próximo sin ampliar el vocabulario. Esto permite inferencia, pero puede perder detalle espacial. El 2,5 % del entrenamiento incluye el subconjunto interno excluido del ajuste del vocabulario.

Además, una celda de 250 metros puede ocultar separaciones entre calles próximas. Es una explicación **plausible**, no una causa demostrada, de que el embedding funcione peor. Para identificarla habría que comparar resoluciones con nuevos experimentos de validación.
""")

md(r"""
### 5.7 La comprobación que faltaba: ¿reconstruye de verdad?

La pregunta del usuario motivó una auditoría adicional del modelo ya guardado. Usamos las **286 trayectorias de los 143 viajes de validación interna**: no ajustaron los pesos, pero sí participaron en seleccionar el checkpoint. No son un test nuevo. No usamos etiquetas humanas ni modificamos los pesos.

Evaluamos dos situaciones: entrada limpia y entrada con un 20 % de puntos eliminados de forma independiente más ruido gaussiano de 15 metros. Los extremos se conservan al eliminar puntos, aunque también reciben ruido. Generamos celdas escogiendo el mayor score en todo el vocabulario, sin darle al decoder la celda anterior correcta; se detiene al predecir FIN o tras 128 pasos.

Medimos cuatro cosas distintas:

1. **Exactitud por token con ayuda:** porcentaje de posiciones correctas cuando el decoder recibe el token anterior verdadero. Incluye FIN; es una medida exigente porque celdas vecinas también reciben peso en la pérdida.
2. **Reconstrucción libre vacía:** frecuencia de predecir FIN sin emitir ninguna celda. No puede esconderse excluyendo estas rutas de una media espacial.
3. **Distancia de edición normalizada:** inserciones, borrados y sustituciones para transformar la secuencia predicha en la original, dividido por la mayor longitud. Va de 0 —coincidencia— a 1 —máxima discrepancia según esta escala—. Excluimos FIN.
4. **Fréchet discreto espacial:** compara el orden de puntos de la ruta original remuestreada y los centroides reconstruidos, con distancias sobre una esfera terrestre en metros. Si la predicción está vacía, la distancia se deja ausente y se informa ese fallo por separado.

Como referencias conservamos los puntos de entrada disponibles y la secuencia objetivo representada mediante centroides del vocabulario. La segunda es un diagnóstico de cuantización y celdas desconocidas, **no una reconstrucción que un modelo pueda obtener sin conocer la respuesta**.
""")

code("""
recon_dir=ROOT/'resultados/rutas_reconstruccion'
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
**El resultado obliga a corregir la interpretación anterior:** el decoder de esta adaptación no reconstruye adecuadamente. Con entrada alterada, 199 de 286 salidas están vacías —69,6 %—; las 87 restantes presentan un error espacial enorme. La reconstrucción libre falla también con entrada limpia. La reducción de la pérdida de entrenamiento no demostraba que la generación funcionase.

Además, encontramos un **defecto concreto de la pérdida implementada**: los negativos se muestrean solo entre celdas y nunca incluyen FIN. Así, predecir FIN con un score muy alto en un paso que debería contener una celda no se penaliza en ese objetivo muestreado. Lo verificamos aumentando en 100 el score de FIN: la pérdida de un tramo que no termina permanece exactamente igual. Esto no ocurre en una pérdida normalizada sobre todas las salidas posibles.

Ese defecto ayuda a explicar el final prematuro, pero corregirlo no garantiza por sí solo una buena representación: también quedan la aproximación del normalizador, la escasez de datos, la resolución y las celdas no vistas. **Las métricas supervisadas guardadas describen este ensayo fallido; no son evidencia válida contra el t2vec del artículo.** La geometría y el experimento viario no dependen de este decoder.

Este hallazgo exige corregir el objetivo antes de interpretar la calidad del método. A continuación hacemos ese experimento por separado, manteniendo las particiones y los resultados originales.
""")

md(r"""
### 5.8 Corregimos el objetivo y repetimos la comprobación

La versión nueva normaliza sobre **todas las celdas y FIN**, excluyendo únicamente los símbolos auxiliares de inicio y padding. Si el decoder asigna demasiado score a FIN cuando debería predecir una celda, aumenta el denominador y la pérdida lo penaliza. Lo comprobamos con una prueba de gradientes y con el mismo aumento artificial de score utilizado para detectar el defecto.

$$\mathcal L_t=\log\sum_{c\in\mathcal V\cup\{\mathrm{FIN}\}}e^{s_t(c)}-\sum_c q_t(c)s_t(c).$$

$s_t(c)$ es el score de salida para cada celda y $q_t$ reparte el objetivo entre vecinas espaciales; para FIN todo el peso recae en FIN. El normalizador ya no está muestreado. Esto cuesta más cálculo, pero elimina la omisión que detectamos.

Reentrenamos desde la misma semilla, con la misma arquitectura, vocabulario, viajes y presupuesto máximo de 12 épocas. Elegimos el checkpoint usando entradas alteradas de validación interna, con ruido reproducible. Esto también hace más coherente la tarea de selección con la tarea de entrenamiento; por tanto, no atribuimos todas las diferencias posteriores únicamente a incluir FIN. Después entrenamos la misma familia de clasificadores, seleccionando su tamaño y umbral en validación supervisada.

Los pesos y resultados se guardan en carpetas nuevas. La auditoría utiliza las mismas rutas y alteraciones que antes. **Sigue siendo una adaptación pequeña de t2vec y un experimento exploratorio posterior a inspeccionar el test**, aunque la pérdida ahora esté normalizada correctamente.
""")

code("""
corrected_recon=json.loads((ROOT/'resultados/rutas_reconstruccion_corregida/metrics.json').read_text())
corrected_model=json.loads((ROOT/'resultados/rutas_t2vec_corregido/metrics.json').read_text())
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
**Qué cambió y qué no:** con entradas alteradas, la versión corregida reduce las salidas vacías del 69,6 % al 4,9 %, pero su distancia de edición media sigue cerca de 1 y sus recorridos generados siguen siendo espacialmente incorrectos. Terminar menos veces de inmediato no equivale a reconstruir bien. La exactitud con ayuda sigue siendo baja, así que el problema tampoco se explica únicamente por acumular errores al generar libremente.

La clasificación alcanza 77,1 % de balanced accuracy en test frente a 71,8 % del ensayo original, pero permanece lejos del 94,3 % de la referencia geométrica. Es una comparación exploratoria sobre el mismo test inspeccionado. **No elegiríamos esta versión para el prototipo actual.** Antes de seguir necesitaríamos más preentrenamiento y experimentos de arquitectura, resolución y cobertura, comprobando la reconstrucción desde el principio y con datos nuevos para confirmar el resultado final.
""")

md(r"""
## 6. El protocolo se decide antes de mirar el test

Este apartado describe el experimento original de cuatro modelos. La extensión viaria posterior conserva sus reglas de ajuste, pero su evaluación exploratoria se identifica expresamente en la sección 7.1.

**Selección del modelo:** menor entropía cruzada frente a los votos en validación. Es una regla que premia probabilidades coherentes con el desacuerdo humano.

**Selección del umbral:** máximo acierto equilibrado en validación, explorando 0,10–0,90 en pasos de 0,01; los empates favorecen el umbral más cercano a 0,5. No asumimos que 0,5 sea óptimo ni que los errores tengan costes de negocio conocidos.

El acierto equilibrado es el promedio de la proporción de «iguales» y «diferentes» correctamente reconocidos. Evita que una clase mayoritaria domine la cifra.

La clase positiva es **«diferentes»**. Conviene separar tres preguntas:

| Pregunta | Métrica y lectura |
|---|---|
| ¿Cuántas decisiones acertamos con el umbral fijado? | **Accuracy:** fracción de aciertos. **Balanced accuracy:** media del recall de ambas clases. Más alto es mejor. |
| ¿Cómo se comporta la alerta «diferentes»? | **Precisión:** de las alertas, cuántas coinciden con la mayoría humana. **Recall:** de los viajes diferentes, cuántos detectamos. **F1:** su media armónica. Más alto es mejor. |
| ¿Ordenamos bien los casos sin fijar un único umbral? | **ROC-AUC:** separación de las clases al recorrer umbrales. **AP:** resumen de precisión y recall para «diferentes»; depende de la prevalencia. Más alto es mejor. |
| ¿Son razonables las probabilidades? | **Brier:** error cuadrático frente a la mayoría binaria. **Log loss binaria:** penaliza especialmente errores seguros. **Log loss de votos:** usa la proporción de votos, incluidos empates. Más bajo es mejor. |

Para ver la diferencia entre las dos pérdidas, sea $q_j$ la probabilidad del modelo y $p_j$ la fracción de votos «diferentes»:

$$\mathrm{LL}_{\text{votos}}=-\frac{1}{N}\sum_j[p_j\log q_j+(1-p_j)\log(1-q_j)].$$

En la log loss binaria sustituimos $p_j$ por la mayoría $y_j\in\{0,1\}$ y excluimos empates. Un voto 2 a 1 se convierte en 1 en esa segunda evaluación, pero sigue siendo $2/3$ en la primera. **Elegimos con la pérdida de votos; no cambiamos de criterio al ver qué métrica favorece a cada modelo.**

Se prueban tres configuraciones predefinidas de profundidad y regularización por XGBoost, con parada temprana en validación. Para la cabeza neuronal se prueban 32 y 64 unidades ocultas. Los modelos usan el mismo reparto y objetivos. El vocabulario, encoder y escalado nunca se ajustan con test.

Una vez elegidos modelos, épocas y umbrales, se guardan las decisiones y se evalúa test. **No se reentrena con validación ni se cambia de ganador después de ver test.** Las cuatro variantes son comparaciones planificadas; posteriores cambios exigirían mantener la distinción entre exploración y una nueva confirmación.
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
## 7. El momento clave: más complejo no equivale a mejor demostrado

La regla de validación selecciona **geometría básica + XGBoost**. En test obtiene un 94,3 % de acierto equilibrado.

Tu vector ampliado alcanza el 94,8 %, la adaptación t2vec con clasificador el 71,8 % y la combinación el 95,5 %. El híbrido tiene la cifra puntual más alta en test, pero **no lo elegimos retrospectivamente por esa razón**.

Para comparar, remuestreamos 2.000 veces los mismos viajes de test y calculamos en cada muestra la diferencia entre modelos. Al usar los mismos viajes, el intervalo refleja qué errores comparten.

Las mejoras del vector ampliado y del híbrido respecto a la referencia tienen intervalos que incluyen cero. Eso no prueba igualdad: con esta muestra, la evidencia no permite asegurar que la mejora sea estable.

Los intervalos condicionan en este entrenamiento, esta partición y esta mezcla geográfica. No recogen cambios de semilla, de ciudad, dependencia entre corredores ni la incertidumbre del panel humano. Las comparaciones secundarias son exploratorias y no están corregidas por multiplicidad.
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
## 8. Volvemos a las rutas: ¿qué significa acertar y qué significa fallar?

Una AUC alta indica buena ordenación, pero no resuelve qué umbral conviene al negocio. Con el umbral elegido en validación, la referencia marca como similares 25 viajes que la mayoría consideró diferentes, y marca como diferentes tres viajes que la mayoría consideró similares.

Estas cifras dependen del objetivo humano elegido. Las probabilidades no son certeza objetiva. La curva de calibración se muestra como diagnóstico: no se ha recalibrado usando test.

En adelante, todas las rutas que inspeccionamos son del **test** y todas las predicciones corresponden a **geometría básica + XGBoost**, con umbral **0,75**. Los ejemplos de entrenamiento anteriores servían para explicar las medidas; estos sirven para auditar el resultado final.

| | El modelo dice «iguales» | El modelo dice «diferentes» |
|---|---|---|
| La mayoría dice «iguales» | **185 aciertos — TN** | **3 falsas alarmas — FP** |
| La mayoría dice «diferentes» | **25 diferencias no detectadas — FN** | **230 aciertos — TP** |

«Fallo» significa desacuerdo con la mayoría disponible, que a menudo es **una sola persona**. No significa que hayamos demostrado que el recorrido es objetivamente igual o diferente. Para distinguir un error del modelo de una etiqueta discutible necesitaríamos revisión adicional.
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
### 8.1 Primero, reconocer los dos tipos de acierto

Una curva de métricas no permite ver si el modelo reconoce dos recorridos casi superpuestos o una desviación evidente. Mostramos **dos aciertos de cada clase**: el que ocupa la posición central al ordenar su puntuación y el más próximo al umbral. Así vemos un caso central y otro de decisión más ajustada, sin seleccionar a mano los dibujos más vistosos.

La línea oscura es la ruta estimada; la naranja discontinua es la real. Los ejes están en kilómetros, con la misma escala horizontal y vertical dentro de cada panel. Los paneles se ajustan a cada viaje: no todos tienen la misma extensión. Los círculos indican comienzos y los cuadrados finales.

Al leer «0,74», recuerda: es una puntuación probabilística de «diferentes», pero la regla elegida exige **al menos 0,75** para emitir esa etiqueta. No es un porcentaje de metros diferentes.
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
### 8.2 Los errores: separar trazado, decisión y etiqueta

Inspeccionamos **los tres falsos positivos** y **tres de los 25 falsos negativos**: el de menor puntuación, el central y el más próximo al umbral. No son una muestra aleatoria de errores ni sirven para estimar la frecuencia de cada explicación. El registro de predicciones conserva los 28 errores.

Cada figura responde a cuatro preguntas:

1. **Vista completa:** ¿las rutas se parecen globalmente?
2. **Ampliación:** ¿hay un tramo que la escala del viaje ocultaba? Centramos la ventana en la mayor separación encontrada al muestrear 800 posiciones por ruta.
3. **Perfil de distancias:** ¿es una diferencia breve o persistente? Medimos desde cada posición a la polilínea contraria. El porcentaje se refiere a la longitud de **cada ruta**, no al tiempo ni a posiciones sincronizadas.
4. **Contribuciones del modelo:** ¿qué variables empujan esta predicción hacia «iguales» o «diferentes»?

Para el último panel usamos las [contribuciones TreeSHAP de XGBoost](https://xgboost.readthedocs.io/en/stable/prediction.html). Descomponen la puntuación del modelo ya entrenado antes de la sigmoide:

$$\log\frac{q}{1-q}=\phi_0+\sum_{k=1}^{8}\phi_k,\qquad
q=\frac{1}{1+\exp[-(\phi_0+\sum_k\phi_k)]}.$$

$\phi_0$ es la referencia implícita de los árboles; cada $\phi_k$ es una contribución en **log-odds**, no en puntos porcentuales. Naranja aumenta la puntuación de «diferentes» y verde la reduce. Usamos la referencia interna del modelo entrenado, sin añadir test como fondo de comparación. Conservamos exactamente los árboles de la iteración elegida en validación y comprobamos que la suma reconstruya la predicción congelada.

Esto explica **cómo puntúa el modelo esta entrada**, no por qué condujo así una persona ni por qué votó así el anotador. Las variables están relacionadas entre sí; sus contribuciones no son efectos causales.
""")

code("""
from xgboost import XGBRegressor,DMatrix
import hashlib
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib.patches import Rectangle

case_ids=cases.journey_id.tolist()
model_path=ROOT/f'modelos/rutas/{selected}.ubj'
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
### 8.3 El aprendizaje de la inspección

El mismo rótulo «error» puede esconder problemas distintos. Un desvío localizado puede quedar diluido en las medias de un viaje largo; dos señales geométricas pueden empujar en sentidos opuestos; y una puntuación casi idéntica al umbral puede cambiar la decisión por unas diezmilésimas.

También importa la referencia humana. En las tres falsas alarmas hay un único voto decidido por viaje. En el falso negativo central hay una decisión y dos abstenciones. **Más complejidad del modelo no resolverá por sí sola qué criterio de similitud queremos reproducir.**

La inspección propone hipótesis, no las confirma. Para avanzar pediríamos anotaciones adicionales con instrucciones claras, analizaríamos el coste de ambos errores y compararíamos representaciones con validación separada. Después de explorar estos ejemplos, este test ya no debería usarse como terreno para nuevas decisiones y, a la vez, como confirmación independiente de esas mejoras.
""")

md(r"""
### 8.4 ¿Dónde falla? Casi siempre donde la etiqueta la sostiene una sola persona

La inspección anterior sugería algo que conviene medir en lugar de insinuar: en los errores aparecía una y otra vez
**un único voto decidido**. Separamos ahora el rendimiento según cuántas personas sostienen la etiqueta, en validación
y en test, con el modelo y el umbral ya congelados. No cambia nada del experimento: reparte los mismos aciertos.

La lectura es doble y conviene no quedarse en la primera mitad.
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
**Primera lectura: el modelo acierta más cuando más personas respaldan la etiqueta.** En validación pasa del 94,3 %
con un voto al 100 % con dos o tres; en test, del 92,6 % al 98,6 % y 100 %. Los 21 errores de validación y 27 de los 28
de test están en viajes de un solo voto. La concentración supera lo que explicaría el tamaño del grupo, aunque por
poco: esperaríamos 17,3 de 21 en validación y 23,1 de 28 en test.

**Segunda lectura, la que impide celebrar la primera: los grupos no son comparables.** Un viaje con dos votos y
mayoría definida es, *por construcción*, un viaje donde dos personas coincidieron — si hubieran discrepado sería un
empate y no tendría etiqueta. El grupo «dos o más votos» no es «la misma dificultad, mejor medida»: es **el subconjunto
de casos que resultaron visualmente inequívocos**. Dicho de otro modo, hemos comparado los casos difíciles con los
fáciles y hemos descubierto que los difíciles son más difíciles.

Las dos explicaciones son indistinguibles con este archivo:

| Explicación | Qué implica | Cómo se distinguiría |
|---|---|---|
| **Ruido de etiqueta** · parte de esos 21 y 27 «errores» son opiniones individuales que otra persona no compartiría | Nuestra exactitud real está infraestimada; el techo no es 100 % | Reanotar esos casos con varias personas y medir cuántas mayorías cambian |
| **Dificultad real** · los casos ambiguos para las personas también lo son para la geometría | El modelo tiene margen donde más cuesta | Reanotar y comprobar que la mayoría se mantiene pese al desacuerdo |

Que 72 viajes de dos votos den 100 % y 98,6 % tampoco es tranquilizador por sí mismo: con esos tamaños, un intervalo
de confianza al 95 % para una proporción de 72/72 llega aproximadamente hasta el 95 %. No hemos demostrado que el
modelo sea infalible en los casos claros; hemos observado que no ha fallado en unos pocos.

**Lo accionable.** Si se quisiera subir el rendimiento medido, el camino más corto no es un modelo mayor: es
**conseguir una segunda anotación para los viajes de un voto**, que son el 83 % de los casos decisivos y la totalidad
práctica de los errores. Eso convertiría parte del «error» actual en desacuerdo humano documentado y daría, por primera
vez, una referencia con la que comparar el modelo en los casos que de verdad están en disputa. Antes de eso conviene
fijar en las instrucciones qué cuenta como diferente —un umbral de metros, de tiempo o de desvío— porque hoy el
enunciado deja el criterio a la percepción de cada anotador.
""")

md(r"""
### 8.5 Los empates: ¿de verdad son casos confusos?

Hasta aquí hemos evaluado solo los viajes con mayoría. Los **31 empates** del archivo —7 de ellos en test— se quedaron
fuera por no tener etiqueta binaria. Conviene mirarlos, porque sostienen una afirmación que hemos repetido sin probar:
que el desacuerdo humano señala ambigüedad real y no descuido de los anotadores.

Si al dibujarlos aparecieran parejas obviamente iguales u obviamente distintas, el empate sería ruido del panel y
nuestra regla de consenso estaría descartando información buena. Si aparecen desvíos intermedios —ni superposición
ni divergencia clara— el empate describe el problema.

Añadimos la puntuación del modelo, que no vio estas etiquetas: es una segunda opinión independiente del panel.
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
**Lo que muestran los dibujos, y es más específico de lo que esperábamos.** Ninguno es una superposición perfecta
ni una divergencia total, pero además comparten una estructura: **en los siete, el cuerpo central del trayecto
coincide y la discrepancia se concentra en un extremo** —la mayor separación cae siempre en el primer o el último
cuarto del recorrido—. Entre el 6 % y el 22 % de la ruta queda a más de 50 m de la otra; el resto se superpone.

Eso sugiere que el empate no nace de «dos caminos distintos», sino de **cómo empieza o acaba el viaje**: el punto de
recogida real frente al previsto, los últimos metros hasta el destino, una maniobra de aproximación. Y ahí la pregunta
«¿son la misma ruta?» no tiene una respuesta visual evidente: una persona puede considerar que el trayecto es el mismo
con una recogida desplazada, y otra que la ruta ya es distinta. El desacuerdo describe una ambigüedad genuina del
problema, no un descuido, y respalda tratar el empate como ausencia de etiqueta en lugar de forzar un 0 o un 1.

Conviene no sobreleer siete casos: es una observación sobre los empates de test, no una ley del dataset. Pero es
comprobable y apunta a una pregunta concreta de negocio —si lo que importa para el precio es el trayecto o también
los extremos— que las instrucciones de anotación no resuelven hoy.

**Lo que no muestran.** El modelo *no* reproduce esa duda de forma fiable. Tres de los siete caen en su zona
intermedia, frente al 9 % del resto de test, pero en los otros cuatro puntúa con aplomo —0,091 y 0,261 para parejas
que partieron al panel por la mitad—. Con siete casos la comparación no es concluyente (Mann-Whitney unilateral,
p = 0,13), y además la probabilidad de XGBoost no es una medida de incertidumbre: es una estimación puntual sin
intervalo. Que el modelo dude donde dudan las personas es una propiedad deseable que **aquí no está demostrada**.

Eso señala un trabajo concreto: cuantificar la incertidumbre del modelo con un método que ofrezca garantías
—predicción conforme sobre el conjunto de validación, por ejemplo— y comprobar si sus casos abstenidos coinciden con
los empates y los votos divididos. Requiere más empates de los que hay aquí: 31 en todo el archivo son pocos para
medir ese solapamiento con precisión útil.
""")

md(r"""
### 8.6 ¿Con quién está de acuerdo el modelo?

Si el 80 % de las etiquetas las firma una sola persona, «acertar» significa *coincidir con esa persona*. Cabe entonces
una pregunta que las métricas globales no responden: **¿acierta igual con todos los anotadores?**

Tomamos los viajes de validación y test donde hubo **un único anotador con voto decidido**, los agrupamos por quién
fue, y medimos el acierto del modelo congelado en cada grupo. Son particiones distintas del mismo test ya inspeccionado:
un reparto descriptivo, no una evaluación nueva.
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
**El modelo no es igual de bueno con todo el mundo.** Entre los anotadores con suficientes viajes en solitario, el
acierto va del **88,2 % con el anotador 1** (119 viajes) al **98,0 % con el anotador 7** (147 viajes). Casi diez
puntos de diferencia sobre el mismo modelo, el mismo umbral y la misma métrica.

**Y la dirección no es casual.** El panel derecho lo ordena: cuanto más a menudo dice «diferentes» una persona, peor
coincide el modelo con ella. El anotador 1 es el más exigente del panel —marca «diferentes» el 65 % de las veces— y
es con quien menos coincide; el 7 es el más indulgente —39,7 %— y es con quien más. Dicho de forma directa: el modelo
ha aprendido **un umbral de tolerancia más cercano al del anotador 7 que al del 1**, y eso es coherente con que sus
errores sean sobre todo diferencias no detectadas: 17 de 21 en validación, 25 de 28 en test. Falla hacia «iguales»,
igual que el anotador más indulgente.

**Tres cautelas antes de llevarse esto demasiado lejos.** Cada grupo contiene viajes distintos, así que parte de la
brecha puede ser dificultad y no criterio. Los grupos pequeños se omiten del gráfico (menos de 40 viajes) porque con
veinte casos un punto porcentual no significa nada. Y esto es un reparto descriptivo de un test ya inspeccionado, no
una medición independiente.

**Lo que sí deja claro.** No existe «el» objetivo: existe el objetivo de un panel concreto, dominado por unas pocas
personas con umbrales distintos entre sí. Antes de poner esto en producción habría que decidir **qué criterio quiere
la empresa** —cuántos metros de desvío importan para el precio— y reanotar con esa instrucción, en vez de heredar
implícitamente el listón del anotador que más viajes miró.
""")

md(r"""
<a id="dinero"></a>

## 9. ¿Cuánto dinero hay en juego?

Todo lo anterior optimiza una pregunta perceptual: *¿le parecerían distintas a una persona?* Pero el enunciado
motiva el problema desde el **precio**: la tarifa se fija antes del viaje con la distancia y la duración estimadas.
Si el recorrido real difiere, la diferencia entre lo cobrado y lo que ese trayecto habría costado es el error real.

Esas dos preguntas no son la misma, y conviene medir cuánto se separan antes de seguir afinando la primera.

**Cómo estimamos el euro.** La tarifa base y la mínima son idénticas con la ruta estimada y con la real, así que
se cancelan al restar. Solo hacen falta los precios por kilómetro y por minuto:

$$\Delta\unicode{x20AC} = \unicode{x20AC}/\mathrm{km}\cdot(\mathrm{km}_{real}-\mathrm{km}_{est})
  + \unicode{x20AC}/\mathrm{min}\cdot(\mathrm{min}_{real}-\mathrm{min}_{est}).$$

Las tarifas son las públicas de Cabify por ciudad, categoría estándar, consultadas el 19 de octubre de 2026. Las
duraciones salen del encaje viario de OSRM donde es fiable (1.398 de 2.386 viajes) y, en el resto, de la distancia
dividida por la velocidad mediana que OSRM implica en esa misma ciudad.

> **Esto es una estimación de orden de magnitud, no una auditoría contable.** Las tarifas son de 2026 y el archivo
> es de diciembre de 2021: con la inflación argentina del periodo, la cifra histórica sería incomparable. Suponemos
> categoría estándar para todos los viajes, e ignoramos peajes, tarifa dinámica, esperas y promociones. Colombia y
> Ecuador no publican un valor único y usan estimaciones declaradas. Las duraciones no incorporan tráfico real.
""")

code("""
from rutas.pricing import build_impact, tariff_table, EBITDA_MARGIN
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
### 9.1 ¿Quién se come la diferencia?

El precio se cierra antes de arrancar. El cliente paga lo pactado, pase lo que pase después. Así que la desviación
**no se reparte: cae entera de un lado u otro**, y nunca se devuelve.

| Si el trayecto real resulta… | Quién lo absorbe |
|---|---|
| **más largo** de lo previsto | Cabify y el conductor: ese kilómetro se recorre y nadie lo cobra |
| **más corto** de lo previsto | El cliente: pagó por un trayecto que no se hizo, y no hay reembolso |

Mirar solo la media esconde justo esto, porque los dos casos se cancelan entre sí.
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
### 9.2 ¿Los errores del modelo cuestan dinero?

La pregunta que importa no es cuántos fallos comete, sino **cuánto valen**. Repartimos las desviaciones de precio
por casilla de la matriz de confusión en test.
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
**Tres conclusiones, y una de ellas corrige una sospecha razonable.**

**Primera: por viaje, el dinero es pequeño.** La desviación mediana es de 11 céntimos, el 3,1 % de una tarifa media
de 4,63 €. Ningún cliente reclama por eso. Lo que lo convierte en un problema es la escala: Cabify sirve cientos de
miles de trayectos al día, y una media de 0,31 € de desviación absoluta por viaje son decenas de miles de euros
diarios mal asignados, aunque no perdidos.

**Segunda: no hay fuga sistemática, pero eso no significa que sea inofensivo.** La media con signo es prácticamente
cero —el neto son −9 € sobre 11.054 € facturados— porque los viajes que costaron de más compensan a los que costaron
de menos. El problema es que **la compensación ocurre en la hoja de cálculo, no en la calle**: el precio se cierra
antes de arrancar, así que cada desviación cae entera sobre alguien y no se devuelve. Un 8,9 % de los clientes pagó
más de un 10 % por encima de lo que valía su trayecto, y en un 2,6 % de los viajes Cabify y el conductor absorbieron
más de un 25 % del precio.

Y la escala correcta para juzgarlo no es la facturación, sino el margen. Cabify cerró 2025 con un EBITDA de 27,2 M€
sobre 833,3 M€, un **3,3 %**. Sobre una tarifa media de 4,63 €, eso son unos 15 céntimos de margen por viaje, frente
a una desviación mediana de 11. Visto así: **casi la mitad de los viajes —el 48 %— se desvía más de lo que ese mismo
viaje deja de beneficio**, y en el 10 % la desviación es cinco veces el margen. El bruto que se mueve equivale a
2,1 veces el margen de la compañía. No es calderilla; es ruido del tamaño del beneficio, que hoy se cancela por
suerte y no por diseño.

Dos advertencias sobre esa comparación. El 3,3 % es el margen **del grupo**, no el de un trayecto concreto: sirve
para dar escala, no para imputar beneficio por viaje. Y la muestra está dominada por ciudades latinoamericanas con
tarifas bajas, mientras que España aporta casi un tercio de los ingresos del grupo, así que la tarifa media de este
archivo no representa la del negocio.

**Tercera, y es la que más cambia el juicio sobre el modelo: falla en los viajes baratos y acierta en los caros.**
Era razonable temer lo contrario. El modelo heredó el criterio del anotador más indulgente (§8.6) y se equivoca casi
siempre hacia «iguales»: 25 de sus 28 errores en test son diferencias que no detecta. Si esos casos fueran los de
mayor desviación, el sistema estaría dejando pasar justo lo que debía capturar.

Los datos dicen lo contrario. Las 25 diferencias no detectadas tienen una desviación mediana de 0,06 €, **menor que
la de los aciertos** (0,19 €), y suman 3 € de los 113 € en juego entre los viajes realmente distintos. Es decir:

> **El modelo detecta el 90,2 % de los casos, pero recupera el 97,4 % del dinero.**

Esa brecha de siete puntos es el valor que la exactitud no ve. Un informe que diga «90 % de acierto» describe peor el
sistema que uno que diga «captura 97 de cada 100 euros desviados». Y no era predecible desde las métricas de
clasificación: hizo falta ponerle precio a cada caso para saberlo. Conviene no leerlo como una virtud deliberada del
modelo —nadie lo entrenó para esto—, sino como una propiedad afortunada de este archivo, que habría que volver a
comprobar con datos nuevos antes de confiar en ella.

**Lo que esto cambia en la práctica.** Si el objetivo es proteger el precio, la métrica no debería ser el acierto
equilibrado sino el **dinero recuperado**, y el esfuerzo debería concentrarse en el 10 % de viajes que acumulan el
55 % del impacto. Afinar del 94 % al 96 % de acierto no mueve la aguja si los dos puntos ganados están en la cola
barata. Esto es una recomendación de diseño, no un cambio que apliquemos aquí: el modelo, el umbral y las métricas
del experimento siguen siendo los congelados en validación.
""")

md(r"""
## 10. Decidir con incertidumbre: predicción conforme

El modelo entrega una probabilidad, y eso **no es una medida de incertidumbre**. Que valga 0,52 no significa «dudo»:
significa que los árboles promediaron ahí. Cuando el sistema decide con un umbral fijo, falla en un 6,3 % de los
casos de test y **no sabemos en cuáles**. Esa es la limitación operativa, no la exactitud.

La predicción conforme cambia el contrato. En lugar de una etiqueta forzada, produce un intervalo sobre la proporción
de votos «diferentes» y lo convierte en una decisión de tres valores:

| El intervalo… | Decisión |
|---|---|
| queda entero por debajo de 0,5 | iguales |
| queda entero por encima de 0,5 | diferentes |
| contiene 0,5 | **sin decisión: a revisión humana** |

**Por qué CV+ y no split conformal.** Validación ya intervino en elegir modelo, iteración y umbral: usarla como
calibración invalidaría la garantía. Reservar un fold de entrenamiento dejaría ~50 casos en el grupo de dos o más
votos. CV+ usa los 1.348 viajes de entrenamiento mediante residuos fuera de fold y garantiza cobertura ≥ 1 − 2α.

**Por qué dos grupos de calibración.** El ancho del intervalo sale del percentil de los errores pasados. Mezclar
viajes de un voto con los de dos o más produce un ancho promedio que es demasiado estrecho para los primeros y
demasiado ancho para los segundos: la cobertura global se cumple por compensación y falla justo donde importa. No
separamos tres y cuatro votos porque con 8 y 2 casos de calibración no alcanzan el mínimo n ≥ 1/α − 1.
""")

code("""
from rutas.conformal import run as run_conformal, GROUPS
conformal_path=ROOT/'resultados/rutas_conformal/conformal.json'
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
### Lo que esto le da a operaciones

**Un contrato, no una métrica de informe.** Hoy el umbral 0,75 lo eligió un `argmax` de acierto equilibrado y el
resultado es «fallamos un 6,3 % y no sabemos dónde». Con la curva de la izquierda, la conversación cambia de sentido:
negocio fija el error que tolera y el sistema responde con el volumen de revisión necesario.

| Si quieres… | Envías a revisión | Error en lo automático |
|---|---:|---:|
| el comportamiento de hoy | 0 % | 6,3 % |
| menos del 3 % de error | 4,1 % | 2,8 % |
| menos del 2 % | 6,1 % | 1,9 % |
| menos del 1 % | 12,9 % | **0,5 %** |

Con un 13 % de los viajes revisados, el error de las decisiones automáticas cae de 6,3 % a 0,5 %: **doce veces menos**.
Y al revés, si operaciones solo puede revisar el 5 % del volumen, la tabla dice qué error queda. Eso es dimensionable
en personas y en coste.

**Y pone precio a la anotación.** El intervalo es un 30 % más estrecho cuando la etiqueta tiene dos votos en lugar de
uno, lo que en α = 0,10 significa pasar del 14,5 % al 5,1 % de revisión. Conseguir una segunda anotación para los
viajes de un voto —el 83 % del archivo— reduciría el volumen de revisión a cerca de un tercio. Eso convierte «habría
que anotar más» en una propuesta con dos cifras enfrentadas: lo que cuesta anotar contra lo que cuesta revisar.

""")

md(r"""
### 10.1 ¿Cabe esto en el volumen real? Y si cabe, ¿cabe la revisión?

El enunciado justifica automatizar porque Cabify sirve millones de trayectos. Es una afirmación de producto que
ninguna métrica de acierto responde, así que la medimos: el camino completo para una pareja nueva —proyectar a
metros, calcular las ocho variables, puntuar— sobre 300 parejas reales, en un solo proceso sin lotes ni paralelismo.
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
**Pero la revisión humana no cabe.** Y esta es la conclusión que obliga a rediseñar la palanca de la sección anterior.

A un millón de viajes diarios, una tasa de abstención del 12,9 % son **129.000 revisiones al día**. A 6,1 %, son
61.000. Aunque una persona revisara un caso por minuto sin descanso, harían falta cientos de revisores a jornada
completa. **La tabla anterior, leída como política global, es inviable a escala real.**

No invalida el método: invalida la forma de aplicarlo. Y la sección 9 ya contiene el arreglo. Si el 10 % de los
viajes acumula el 55 % del impacto económico, no hay razón para revisar por tasa de incertidumbre uniforme. La
política sensata combina las dos señales:

> **Revisar solo donde el modelo duda *y* hay dinero en juego.**

Con un presupuesto fijo —pongamos 2.000 revisiones diarias, unas diez personas— se ordenan los casos abstenidos por
desviación estimada en euros y se revisa de arriba abajo. Lo que queda sin revisar son casos dudosos de impacto
céntimo, donde equivocarse no cuesta nada. El resto se decide automáticamente, y para esos sigue valiendo la
garantía de cobertura.

Eso convierte dos análisis que hasta aquí iban por separado en una sola política operable: **la sección 10 dice de
qué no está seguro el modelo, y la sección 9 dice cuáles de esos casos merecen el tiempo de una persona.** Ninguna
basta sola: la primera genera una cola inabordable y la segunda no distingue una desviación detectada con certeza
de una dudosa.

No lo implementamos aquí porque requiere el presupuesto real de revisión, el coste por revisión y la fórmula de
tarificación interna, que son decisiones de negocio y datos que no tenemos. Pero las dos piezas —el intervalo y el
euro— ya están calculadas por viaje, y combinarlas es una ordenación, no un modelo nuevo.
""")

md(r"""
### Los límites, que son reales

**α = 0,05 no es utilizable.** La tasa de abstención se dispara al 84 %, y en el grupo de un voto llega al 100 %: el
sistema se niega a decidir nada. No es un fallo del método; es la medición honesta de que con un voto por viaje la
etiqueta no está determinada. Ninguna mejora del modelo arregla eso.

**La cobertura observada supera la garantizada** en todos los niveles —92,1 % frente a un 80 % nominal en α = 0,10—.
Es el comportamiento conservador esperable de CV+, no una señal de que el método funcione mejor de lo prometido.

**La garantía es aproximada.** Conformal exige intercambiabilidad entre calibración y evaluación; nuestra partición es
aleatoria y los trayectos comparten calles y corredores. La cobertura empírica en test la respalda, pero no la
demuestra para una ciudad nueva.

**Y el intervalo es sobre la opinión, no sobre el dinero.** Cubre la proporción de votos «diferentes», que es el
objetivo que tenemos. La sección 9 argumenta que la cantidad que de verdad importa son los euros. Un intervalo sobre
euros —«este viaje se desvió entre 2 y 7 €, con un 90 % de confianza»— sería directamente accionable para una
decisión de reembolso, y es la continuación natural de este trabajo una vez exista la etiqueta económica.
""")

md(r"""
<a id="recomendacion"></a>

## 11. Qué haría a continuación

### La recomendación, en una frase

> **Desplegar geometría básica con predicción conforme a α = 0,15, revisar por impacto económico en lugar de por
> tasa fija, y gastar el siguiente euro en una segunda anotación y no en un modelo mayor.**

Desglosada: ocho variables y un XGBoost, 6 ms por pareja, intervalo conforme calibrado por separado para viajes
con uno y con dos o más votos. Los casos sin decisión se ordenan por desviación estimada en euros y se revisan
hasta agotar el presupuesto diario; el resto se decide automáticamente con un 1,9 % de error. No desplegaría t2vec
ni la ampliación viaria: ninguna demuestra mejora y ambas multiplican el coste operativo.

### Qué está demostrado y qué no

**Demostrado con este archivo y este protocolo.** Una representación geométrica pequeña rinde alto: 94,3 % de
acierto equilibrado en test. El vector de 42 variables y el híbrido con t2vec son competitivos pero su mejora **no
queda demostrada** —los intervalos bootstrap pareados incluyen el cero—, de modo que la regla congelada en
validación elige la referencia pequeña y no hay motivo para cambiarla. La adaptación de t2vec quedó por debajo, y
una auditoría posterior encontró que su reconstrucción fallaba y que la pérdida muestreada nunca usaba FIN como
negativo; la versión corregida mejora algunos indicadores y sigue sin reconstruir. Esto describe **nuestras
adaptaciones**, no el método publicado.

**Medido, y más interesante que lo anterior.** El cuello de botella no está en el modelo. El 83 % de las etiquetas
las sostiene una sola persona; el panel son ocho anotadores con criterios que difieren en 25 puntos porcentuales; y
el modelo coincide un 98 % con el anotador más indulgente y un 88 % con el más exigente. Traducido a euros, la
desviación mediana es de 11 céntimos, pero casi la mitad de los viajes se desvía más de lo que ese viaje deja de
margen. Y el modelo recupera el 97,4 % del dinero detectando el 90,2 % de los casos: se equivoca en lo barato.

**No demostrado, y conviene decirlo.** Que el modelo transfiera a una ciudad nueva: la partición es aleatoria y los
trayectos comparten calles. Que la asimetría favorable en euros se mantenga fuera de este archivo: nadie la buscó al
entrenar, es una propiedad afortunada de esta muestra. Que la garantía conforme se sostenga exactamente: exige
intercambiabilidad y aquí es aproximada. Y que las cifras en euros sean las reales: son tarifas públicas de 2026
aplicadas a viajes de 2021, con categoría estándar y sin peajes ni tarifa dinámica.

### Las tres inversiones siguientes, por retorno

**1 · Una segunda anotación para los viajes de un voto.** Es el 83 % del archivo y la práctica totalidad de los
errores: los 21 de validación y 27 de los 28 de test. Y ahora tiene precio: el intervalo conforme es un 30 % más
estrecho cuando la etiqueta está corroborada, lo que baja la tasa de revisión del 14,5 % al 5,1 %. Esa es la
comparación que hay que llevar a un comité — lo que cuesta anotar contra lo que cuesta revisar, todos los días.
Antes de encargarla, fijar en las instrucciones qué cuenta como diferente: hoy el criterio es la percepción de cada
persona, y por eso existen los 31 empates.

**2 · Etiquetar con euros en lugar de con opiniones.** La sección 9 muestra que el juicio humano y el impacto
económico solo correlacionan 0,45. Con la fórmula de tarificación interna —que Cabify tiene y nosotros no— el
objetivo deja de depender de ocho personas y pasa a ser una cantidad medible, con una pérdida que pondera por
impacto y un umbral que fija negocio: «actuamos por encima de X €». Las 42 variables ya construidas sirven igual,
y un intervalo conforme sobre euros sería directamente accionable para un reembolso.

**3 · Validar por ciudad antes de prometer cobertura geográfica.** El reparto aleatorio permite que el mismo
corredor aparezca en entrenamiento y en test. Hasta que no se separe una ciudad entera, la cifra de test describe
rendimiento dentro de una mezcla conocida, no capacidad de generalizar.

### Y lo que no haría

**No invertiría más en representaciones aprendidas** sin resolver antes la etiqueta. Con el 83 % de los votos sin
corroborar, el techo no lo pone el modelo: subir del 94 % al 96 % no distingue una mejora real de haber ajustado
mejor el criterio del anotador 2.

**No desplegaría la ampliación viaria.** Aporta una mejora dentro del ruido y a cambio exige Docker, varios GB de
cartografía, un servicio OSRM y encajes que fallan en un 41 % de las parejas. El coste operativo no se justifica
con la evidencia actual.

**No presentaría el acierto equilibrado como la métrica del sistema.** Describe peor el comportamiento que «recupera
97 de cada 100 euros desviados», y orienta el esfuerzo hacia la cola barata.

### Trabajo pendiente ya identificado

- Revisar las anotaciones contradictorias: siete personas se contradicen sobre el mismo viaje y en los siete casos
  eran el único anotador, así que esos viajes quedan sin etiqueta.
- Analizar la sensibilidad del resultado a la regla de consenso; las etiquetas describen un panel concreto.
- Comparar varias semillas y usar validación cruzada dentro de entrenamiento para estabilizar la selección.
  Reservar una muestra de confirmación nueva si se siguen tomando decisiones después de este análisis.
- Si se continúa con t2vec: celdas más pequeñas, más datos de preentrenamiento, transferencia geográfica, y
  documentar coste, latencia y fallos por celdas no vistas.
- Evaluar ruido GPS, trazados incompletos, cambios de mapa y recorridos nuevos. Más exactitud sobre este archivo no
  garantiza robustez operacional.
- Incorporar al vector variables que describan los huecos de traza —mayor salto, número de saltos, proporción
  cubierta— en lugar de imputarlos: rellenar un hueco con una ruta plausible empujaría la pareja hacia «iguales» y
  alteraría la etiqueta que los anotadores vieron.

La selección original permanece documentada y sin tocar: la referencia geométrica, con el umbral congelado en
validación. Todo lo añadido después —inspección, EDA ampliado, panel de anotadores, euros, conformal y coste— es
descriptivo y no modifica modelos, umbrales ni particiones.
""")

md(r"""
## 12. Reproducibilidad y fuentes

El código de extracción, preentrenamiento y clasificación está en **src/rutas/**. El pipeline de los cuatro modelos originales se ejecuta con **python scripts/run_routes.py**, desde la raíz, después de instalar **requirements.txt**. La extensión viaria tiene un pipeline separado, descrito en el README. Las figuras se regeneran ejecutando este notebook con sus cachés disponibles.

Los modelos y los datos brutos permanecen locales en **modelos/** y **data/**. Se guardan la partición, SHA-256 del archivo original, protocolo, hiperparámetros probados, selección congelada, predicciones y métricas en **resultados/rutas/**. Las pruebas comprueban invariancia al muestreo en líneas, distancias conocidas, sensibilidad al orden, separación de conjuntos y probabilidades válidas.

La inspección conserva en **resultados/rutas/inspection.json** los diez identificadores, reglas de selección, votos, predicciones y contribuciones del modelo, junto con el SHA-256 de sus pesos. Se verifica que cada caso pertenezca a test y que las contribuciones reconstruyan las probabilidades guardadas. El notebook no modifica los pesos, umbrales ni métricas del experimento.

El resumen del EDA se exporta a **resultados/rutas/eda.json**: controles de calidad, longitudes, muestreo y comparación descriptiva de conjuntos. Sus figuras distinguen los recuentos del archivo completo de las distribuciones estudiadas solo en entrenamiento.

La reconstrucción se audita con **scripts/audit_t2vec_reconstruction.py** y su opción **--corrected**. El reentrenamiento corregido se ejecuta mediante **python -m rutas.t2vec_corrected**, con `src` en el path. Los informes separados permiten comparar el fallo original, la corrección del objetivo y sus resultados sin modificar los pesos anteriores.

La ampliación viaria se reproduce con **scripts/download_geofabrik.py**, **scripts/prepare_road_extract.py**, **scripts/start_local_osrm.py**, **python -m rutas.graph_extension** y **scripts/evaluate_graph_extension.py**. Requiere Docker y varios GB de disco para las fuentes. Los informes están en **resultados/rutas_grafo/**; las cachés grandes y modelos se excluyen de Git. El motor se fija por digest de imagen; cada descarga conserva fecha y SHA-256. Las URLs «latest» pueden cambiar: los hashes detectan esa diferencia, no garantizan que el proveedor conserve el mismo archivo para futuras descargas.

- [Enunciado](https://gist.github.com/jmontess/cd4bacd1b09301edf5f2e99ce07584d3) y [archivo enlazado](https://www.dropbox.com/s/rdm11r99pmi13if/challenge_dataset.json?dl=1).
- [Li et al., 2018: t2vec](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf): precedente del encoder-decoder espacial. Nuestra adaptación y evaluación supervisada se distinguen del trabajo original.
- [Newson y Krumm, 2009](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/): referencia para interpretar el problema de map matching.
- [Koide et al., 2020](https://www.vldb.org/pvldb/vol13/p2188-koide.pdf): comparación de secuencias sobre redes viarias; orienta extensiones, no es un algoritmo ya evaluado aquí.
- [Google Routes: puntos de paso](https://developers.google.com/maps/documentation/routes/pass-through) y [API](https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes): documentación de una posible fuente de tiempos estimados, no utilizada en los modelos.
- [OpenStreetMap: atribución y licencia](https://www.openstreetmap.org/copyright).
- [Geofabrik: extractos OSM](https://download.geofabrik.de/), [OSRM: código y perfil](https://github.com/Project-OSRM/osrm-backend) y [servicio de map matching](https://project-osrm.org/docs/v5.24.0/api/#match-service).
- [XGBoost: predicción y contribuciones TreeSHAP](https://xgboost.readthedocs.io/en/stable/prediction.html): descomposición local del modelo congelado y uso de la iteración seleccionada.
""")

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'}})
nbf.write(nb,ROOT/'02_modelado_rutas.ipynb')
print('Notebook generado:',len(cells),'celdas')
