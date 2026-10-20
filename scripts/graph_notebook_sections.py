"""Notebook cells for the explicitly exploratory road extension."""
import nbformat as nbf


def setup_cells():
    return [nbf.v4.new_markdown_cell(r'''
## 4. Del dibujo de una ruta a las calles por las que puede circular

### 4.1 Una API falló; la hipótesis seguía siendo contrastable

El primer intento con Overpass solo consiguió una zona pequeña de Madrid, sin viajes completos. Eso era un problema de acceso al mapa, no una prueba contra los grafos. Ahora descargamos extractos completos de ocho países desde [Geofabrik](https://download.geofabrik.de/), con datos de **OpenStreetMap**. Guardamos fecha y huella digital de cada archivo, y recortamos una zona alrededor de los trazados con un margen de 0,08 grados por lado. El margen facilita la circulación alrededor del recorrido, pero no garantiza incluir cualquier desvío posible.

**Un mapa actual no reconstruye un mapa histórico.** Calles, accesos o sentidos pueden haber cambiado desde estos viajes. Tampoco tenemos sus marcas de tiempo ni tráfico observado. Por eso las variables de duración que añadimos son estimaciones bajo un perfil de coche fijo, no el tiempo real que tardó cada vehículo.

### 4.2 Primero necesitamos saber por qué calles pasó cada ruta

Una coordenada ruidosa puede caer entre dos calles. Asignar cada punto a la calle más próxima de forma independiente puede producir saltos imposibles. Usamos el servicio de *map matching* de [OSRM](https://project-osrm.org/docs/v5.24.0/api/#match-service): busca una secuencia viaria compatible con los puntos y devuelve tramos, puntos rechazados y una medida de confianza.

El motor funciona localmente con una versión fija y el perfil `car.lua`, que incorpora sentidos y las restricciones de giro soportadas por ese perfil y presentes en el mapa. **No sustituimos cada viaje por la ruta más rápida entre su origen y destino:** eso borraría precisamente los desvíos que queremos detectar.

Para limitar el cálculo conservamos hasta 300 puntos **observados**, elegidos por distancia acumulada, incluidos los extremos. No inventamos observaciones intermedias. Suponemos una precisión GPS de 15 metros para generar candidatos; el archivo no proporciona esa precisión. Esta elección puede influir en el resultado y merece un análisis de sensibilidad. Sin timestamps, no imponemos velocidades observadas entre puntos.

La confianza de OSRM no es una probabilidad calibrada para este dataset. Llamamos «encaje completo según nuestra regla» a una ruta con un solo subtrazado, al menos el 90 % de puntos seleccionados aceptados y confianza ≥ 0,5. Es una regla diagnóstica explícita, no una garantía de verdad. **Ningún viaje se elimina del benchmark por incumplirla**: conservamos indicadores de calidad y valores ausentes cuando corresponde.

### 4.3 Qué medimos en el grafo y qué significa cada número

Una arista dirigida representa un tramo de calle en un sentido. Así distinguimos compartir calle de recorrerla en sentidos opuestos. Construimos dos representaciones relacionadas:

| Representación | Para qué se usa | Límite relevante |
|---|---|---|
| OSRM, con perfil de coche | Inferir el trazado y estimar longitud y duración de sus tramos | Mapa actual, perfil fijo; sin tráfico ni duración histórica |
| Grafo dirigido de nodos OSM, ponderado por metros | Distancia mínima desde los nodos de un trazado a los del otro | Respeta sentidos y accesos estáticos simplificados; **no modela restricciones de giro**, accesos condicionales ni ferris |

Para una ruta A y otra B, en este segundo grafo definimos

$$d_G(u,B)=\min_{v\in V_B}\operatorname{dist}_G(u,v).$$

La distancia se recorre siguiendo las flechas. Por eso $d_G(u,B)$ y $d_G(v,A)$ no tienen que coincidir. Calculamos las dos direcciones mediante Dijkstra con múltiples destinos, ejecutado sobre el grafo invertido. El destino es **cualquier nodo del otro trazado**; no imponemos que esté en la misma etapa temporal del viaje.

Esta es una aproximación por nodos, no la distancia exacta entre todos los puntos continuos de dos curvas. Promediamos con pesos de longitud: a cada nodo extremo le corresponde la mitad de los metros realmente recorridos en ese segmento según OSRM. Un tramo dividido por varios puntos GPS no vuelve a contar toda su longitud en cada división. En segmentos parciales aproximamos la posición mediante los nodos de la arista completa; eso puede introducir error cerca de extremos o en aristas largas.

Guardamos media, percentil 95, máximo y proporción de longitud por encima de 50, 100 y 500 metros, en ambos sentidos. Para limitar el coste de búsqueda, truncamos las distancias a 10 km y añadimos la fracción que supera ese límite o no es alcanzable. **Con truncamiento, la media y el máximo son cotas inferiores**, no distancias exactas. Un corte del extracto también puede provocar inalcanzabilidad.

El solapamiento de calles utiliza aristas dirigidas únicas y sus longitudes:

$$\operatorname{IoU}_{\mathrm{calles}}(A,B)=
\frac{\sum_e\min(w_A(e),w_B(e))}{\sum_e\max(w_A(e),w_B(e))}.$$

Aquí $w_A(e)$ es la longitud de la arista si aparece en A y cero si no aparece. Una arista repetida cuenta una sola vez. Tocar parte de una arista cuenta como tocarla entera: esta aproximación puede sobreestimar el solapamiento si las rutas usan porciones distintas de una misma arista. Esta IoU tampoco representa el orden global ni cuántas veces se recorre una calle. Complementa las medidas de orden y geometría que ya tenemos.

Finalmente conservamos longitudes y duraciones estimadas de las **porciones emparejadas** y sus indicadores de calidad. Solo calculamos diferencias relativas de longitud y tiempo como totales comparables cuando ambas rutas cumplen la regla de encaje. Por ejemplo:

$$\Delta T_{\mathrm{rel}}=\frac{\hat T_r-\hat T_e}{\max[(\hat T_r+\hat T_e)/2,1\text{ s}]}.$$

La división compara el sobrecoste relativo; también conservamos los valores individuales. La IoU y las proporciones ya están normalizadas. **La etiqueta sigue siendo el juicio humano sobre similitud**, no «tarda más»: el modelo debe aprender si estas variables ayudan a predecir ese juicio.

### 4.4 La cobertura es un resultado que hay que enseñar

Esta ampliación se propuso después de inspeccionar el test original. Se congeló un nuevo protocolo antes de obtener sus resultados y se mantuvieron los mismos viajes. Sus métricas de test son **exploratorias**, no una confirmación independiente de una mejora.
'''),nbf.v4.new_code_cell(r'''
RG=ROOT/'results/routes_graph'
def read_graph(name):return json.loads((RG/name).read_text())
graph_manifest=read_graph('feature_manifest.json')
extract_manifest=read_graph('extract_manifest.json')
node_manifest=read_graph('node_graph.json')
graph_metrics=read_graph('metrics.json')
graph_frozen=read_graph('frozen_selection.json')
graph_details=read_graph('training_details.json')
graph_features=pd.read_json(ROOT/'data/cache/roads/graph_features.json',orient='index').loc[df.index]
graph_predictions=pd.DataFrame(read_graph('predictions.json')).set_index('journey_id').loc[df.index]
display(Markdown(f'**Mapa importado:** {extract_manifest["nodes"]:,} nodos OSM; '
    f'**grafo simplificado transitable:** {node_manifest["nodes"]:,} nodos y {node_manifest["directed_edges"]:,} aristas. '
    f'Construimos **{graph_manifest["n_features"]} variables viarias y de calidad** por viaje.'))
coverage=pd.DataFrame(graph_manifest['coverage']).set_index('split')
display(coverage[['n','both_any_match','both_reliable']].rename(columns={
    'n':'Viajes','both_any_match':'Algún encaje en ambas','both_reliable':'Ambas cumplen la regla'}))
fig,ax=plt.subplots(figsize=(10,4))
x=np.arange(3);n=coverage.n.to_numpy();good=coverage.both_reliable.to_numpy();anymatch=coverage.both_any_match.to_numpy()
ax.bar(x,good,color=TEAL,label='Ambas cumplen la regla')
ax.bar(x,anymatch-good,bottom=good,color=ORANGE,label='Encaje en ambas, con reservas')
ax.bar(x,n-anymatch,bottom=anymatch,color=RED,label='Falta encaje en al menos una')
for i in x:ax.text(i,n[i]+15,f'{good[i]}/{n[i]} completos',ha='center',fontsize=10)
ax.set_xticks(x,['Entrenamiento','Validación','Test']);ax.set_ylabel('Viajes');ax.set_ylim(0,n.max()*1.2)
ax.set_title('Descargar el mapa no garantiza reconstruir cada viaje');ax.legend(fontsize=9,loc='upper right')
finish(fig,'04_cobertura_grafo','Regla diagnóstica prefijada · ningún viaje se elimina de las métricas por calidad del encaje')
''')]


def result_cells():
    return [nbf.v4.new_markdown_cell(r'''
### 7.1 Ampliación exploratoria: ahora sí podemos probar la red viaria

Comparamos dos variantes nuevas: **grafo + XGBoost** y **42 medidas geométricas + grafo + XGBoost**. Usamos las mismas tres configuraciones de XGBoost y la misma regla de selección por log loss de validación que en el experimento original. Los umbrales se eligen en validación por balanced accuracy. No reentrenamos con test ni descartamos viajes difíciles de emparejar.

Antes de evaluar test guardamos cuál de las dos variantes gana en validación. La referencia geométrica original y sus predicciones permanecen intactas. El test ya había orientado la discusión previa: conservar sus etiquetas fuera del ajuste de estos modelos evita una fuga directa, pero **no convierte una comparación posterior a su inspección en un test confirmatorio nuevo**.
'''),nbf.v4.new_code_cell(r'''
new_names={'grafo_xgb':'Grafo','geometria_grafo_xgb':'Geometría + grafo'}
rows=[]
for name in ['geometria_base','vector_interpretable']:
    m=metrics[name]
    rows.append({'Modelo':names[name],'Variables':8 if name=='geometria_base' else 42,
       'LL val ↓':details[name]['val_soft_logloss'],'BA val ↑':m['validation']['balanced_accuracy'],
       'LL test ↓':m['soft_logloss_all_decisive_votes'],'BA test ↑':m['balanced_accuracy'],
       'F1 test ↑':m['f1'],'AUC test ↑':m['roc_auc'],'Umbral':m['threshold']})
for name,m in graph_metrics.items():
    rows.append({'Modelo':new_names[name],'Variables':graph_details[name]['n_features'],
       'LL val ↓':m['val_soft_logloss'],'BA val ↑':m['validation']['balanced_accuracy'],
       'LL test ↓':m['test_soft_logloss'],'BA test ↑':m['balanced_accuracy'],
       'F1 test ↑':m['f1'],'AUC test ↑':m['roc_auc'],'Umbral':m['threshold']})
road_table=pd.DataFrame(rows).set_index('Modelo')
display(road_table.style.format({c:'{:.4f}' for c in road_table.columns if c!='Variables'}))
selected_graph=graph_frozen['selected_new_variant'];gm=graph_metrics[selected_graph]
ci=gm['delta_base_ci95'];delta=100*gm['delta_base']
comparison='incluye cero: no demuestra una mejora estable' if ci[0]<=0<=ci[1] else ('queda por encima de cero' if ci[0]>0 else 'queda por debajo de cero')
display(Markdown(f'La variante nueva elegida en validación es **{new_names[selected_graph]}**. '
    f'En los mismos **{gm["n"]} viajes de test con mayoría**, la diferencia de balanced accuracy frente a la referencia es '
    f'**{delta:+.2f} puntos porcentuales**, con intervalo bootstrap emparejado del 95 % '
    f'**[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]**. El intervalo {comparison}. '
    'Describe variabilidad entre estos viajes; no incluye repetir entrenamiento, selección o ciudades nuevas.'))
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
order=list(road_table.index);yy=np.arange(len(order));colors=[INK,TEAL,ORANGE,RED]
axs[0].barh(yy,road_table['LL val ↓'],color=colors);axs[0].set_yticks(yy,order)
axs[0].invert_yaxis();axs[0].set_title('Primero seleccionamos con validación');axs[0].set_xlabel('Log loss contra votos · menor es mejor')
for i,v in enumerate(road_table['LL val ↓']):axs[0].text(v+.002,i,f'{v:.3f}',va='center',fontsize=9)
axs[0].set_xlim(0,road_table['LL val ↓'].max()*1.22)
for i,(name,m) in enumerate(graph_metrics.items()):
    lo,hi=np.array(m['delta_base_ci95'])*100;d=m['delta_base']*100
    axs[1].plot([lo,hi],[i,i],color=ORANGE if i==0 else RED,lw=4)
    axs[1].scatter([d],[i],s=65,color=INK,zorder=3)
axs[1].axvline(0,color=MUTED,ls='--');axs[1].set_yticks([0,1],[new_names[k] for k in graph_metrics])
axs[1].set_ylim(-.6,1.6);axs[1].set_xlabel('Cambio de balanced accuracy · puntos porcentuales')
axs[1].set_title('Después cuantificamos la diferencia en test')
finish(fig,'07b_comparacion_grafo','Ampliación exploratoria posterior a la inspección del test · IC bootstrap por viaje, no por ciudad')

# Calidad frente a rendimiento, sin cambiar filtros, modelo ni umbral.
quality_rows=[]
test_ids=df.index[(df.split=='test')&df.target_hard.notna()]
for complete,label in [(1,'Ambas cumplen la regla'),(0,'Encaje parcial o dudoso')]:
    ids=test_ids[graph_features.loc[test_ids,'both_reliable'].to_numpy()==complete]
    y=df.loc[ids,'target_hard'].to_numpy();p=graph_predictions.loc[ids,'probabilities'].map(lambda v:v[selected_graph]).to_numpy()
    pred=p>=gm['threshold']
    quality_rows.append({'Calidad':label,'Viajes':len(ids),'Errores':int((pred!=y).sum()),
        'Exactitud':float((pred==y).mean()) if len(ids) else np.nan})
display(pd.DataFrame(quality_rows).set_index('Calidad').style.format({'Exactitud':'{:.1%}'}))
'''),nbf.v4.new_markdown_cell(r'''
#### Qué nos permite ver el mapa que no ve una distancia en línea recta

Mostramos un ejemplo de **entrenamiento**, elegido de forma determinista por calidad del encaje y longitud central, para inspeccionar la transformación. Las líneas finas son las observaciones; las líneas gruesas, la reconstrucción de OSRM. Una buena apariencia visual no valida todos los viajes ni convierte el mapa actual en verdad histórica.
'''),nbf.v4.new_code_cell(r'''
eligible=df.index[(df.split=='train')&(graph_features.both_reliable==1)]
if len(eligible):
    order_ids=features.loc[eligible,'length_real'].sort_values().index
    road_example=order_ids[len(order_ids)//2]
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,kind,label,route,color in zip(axs,['estimated','real'],['Estimada','Real'],routes[road_example],[INK,ORANGE]):
        raw=json.loads((ROOT/'data/cache/roads/matches'/f'{road_example}_{kind}.json').read_text())
        ax.plot(route[:,1],route[:,0],color=MUTED,alpha=.6,lw=3,label='Observaciones GPS')
        for idx,m in enumerate(raw.get('matchings',[])):
            xy=np.asarray(m['geometry']['coordinates'])
            ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5,label='Encaje en calles' if idx==0 else None)
        ax.set_aspect(1/np.cos(np.radians(route[:,0].mean())))
        ax.set_title(label+' · GPS y red viaria');ax.set_xlabel('Longitud');ax.set_ylabel('Latitud');ax.legend(fontsize=9)
    finish(fig,'07c_map_matching_ejemplo','Viaje de entrenamiento · © OpenStreetMap contributors · ODbL · el encaje es una inferencia')
    display(Markdown(f'Viaje `{road_example}` · IoU de aristas dirigidas: '
        f'**{graph_features.loc[road_example,"directed_edge_iou"]:.3f}**. '
        'Las duraciones corresponden a las calles inferidas bajo el perfil fijo del motor.'))
'''),nbf.v4.new_markdown_cell(r'''
**Qué queda por mejorar:** validar manualmente una muestra de encajes; variar la precisión GPS supuesta y la densidad de puntos; usar cartografía de la fecha del viaje; obtener timestamps; medir distancias en una red con restricciones de giro también para las variables de proximidad; y separar el efecto de conectividad, tiempos y calidad mediante ablaciones. La proximidad por nodos debe contrastarse con una aproximación sobre aristas. También necesitamos evaluar una ciudad no vista y un test nuevo si vamos a seguir desarrollando esta línea.

Los indicadores de calidad pueden ayudar al clasificador, pero también capturar diferencias entre ciudades o dispositivos. Por tanto, una mejora del modelo combinado **no demostraría por sí sola que la conectividad es el mecanismo responsable**. El mapa permite hacer el experimento; su utilidad predictiva y su fiabilidad siguen siendo preguntas empíricas.
''')]
