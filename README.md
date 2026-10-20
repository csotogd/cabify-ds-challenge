# Cabify · Data Science Challenge

Prototipos para las dos partes del challenge de Data Science: evaluación del servicio de fotografía de Just Eat y clasificación de pares de rutas de Cabify.

El notebook combina explicaciones, gráficos, cálculos de tamaño de muestra y simulaciones. Presenta un estudio retrospectivo como evidencia complementaria y un experimento prospectivo aleatorizado por ciudades o áreas metropolitanas emparejadas.

La segunda parte compara geometría básica, un vector interpretable de 42 características, una adaptación reducida de t2vec con decoder de reconstrucción y clasificador, y un híbrido de vectores con XGBoost.

## Contenido

- [Notebook ejecutado](01_fotografia_evaluacion_retrospectiva.ipynb): desarrollo completo, código y resultados. La Parte I trata la evaluación retrospectiva; la Parte II desarrolla el experimento prospectivo.
- [Versión HTML](01_fotografia_evaluacion_retrospectiva.html): lectura en un navegador después de descargar o clonar el repositorio.
- [Gráficos](figuras/): figuras generadas por el notebook.
- [Notas de preparación](notas_preparacion_experimento.json): decisiones metodológicas y referencias para la entrevista.
- [Dependencias](requirements.txt): versiones del entorno utilizado.
- [Enunciado original](enunciado_challenge.md): copia del gist del challenge.
- [Notebook de rutas](02_modelado_rutas.ipynb) y [HTML de rutas](02_modelado_rutas.html): resultados reales, gráficos y límites de la evaluación.
- [Implementación de los modelos](src/rutas/) y [resultados reproducibles](resultados/rutas/).

Los ejemplos numéricos y las simulaciones de la primera parte son sintéticos. Las referencias públicas se citan en el notebook y no sustituyen datos internos de Just Eat. La viabilidad real del experimento queda pendiente de validar la disponibilidad de mercados, la variabilidad histórica, la participación y la interferencia entre ciudades. La segunda parte utiliza el archivo real enlazado en el challenge.

## Reproducir los cálculos

Entorno utilizado: Python 3.12. Ejecutar los comandos desde la raíz del repositorio.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 01_fotografia_evaluacion_retrospectiva.ipynb
```

La ejecución actualiza las salidas del notebook y las imágenes de `figuras/`. También se puede abrir el notebook en un editor compatible con Jupyter y ejecutar todas las celdas en orden. Las simulaciones fijan una semilla para facilitar la reproducción.

Para generar una exportación HTML estándar de Jupyter, conservando la versión maquetada incluida:

```sh
jupyter nbconvert --to html --output 01_fotografia_exportacion 01_fotografia_evaluacion_retrospectiva.ipynb
```

## Estado

Primera parte desarrollada: evaluación retrospectiva complementaria y diseño experimental prospectivo.

La sección 21.4 cierra la decisión que da título al encargo. Define el umbral de rentabilidad como el cociente entre el coste semanal del programa en una ciudad y su margen semanal previo, en las mismas unidades que el efecto estimado, y lo compara con dos cosas: el efecto que la aritmética de dilución permite esperar y el número de ciudades necesarias para detectarlo. Con los supuestos de trabajo —32 publicaciones por ciudad, 40.000 € de margen semanal, 16 semanas y sesiones a 250 €— el umbral queda en 1,25 puntos de crecimiento frente a un techo plausible de 1 punto, y detectarlo exigiría unas 1.010 ciudades. El punto de cruce está alrededor de los 200 € por sesión. La conclusión no es cancelar el experimento sino ordenar las preguntas: con el coste real por sesión y el margen de ciudad, esa tabla se recalcula y decide entre lanzar, rediseñar concentrando el servicio donde el margen expuesto es mayor, o recomendar el cierre sin gastar en el ensayo. La sección también reencuadra el contraste: la hipótesis nula pasa a ser que el efecto no supera el umbral de rentabilidad, unilateral, porque el servicio ya está en marcha y un resultado positivo puede ser real y ruinoso a la vez. Todas las cifras son hipotéticas.

Segunda parte: cuatro modelos entrenados y evaluados. El archivo contiene 2.386 viajes, por lo que se usa 60/20/20: 1.432 de entrenamiento, 477 de validación y 477 de test. Se deduplican votos del mismo anotador y viaje. En test, 443 casos tienen mayoría definida y sirven para evaluar clasificación binaria.

| Modelo | Acierto equilibrado en test |
|---|---:|
| Geometría básica + XGBoost | 94,3 % |
| Vector interpretable + XGBoost | 94,8 % |
| t2vec reducido + clasificador | 71,8 % |
| Vector interpretable + t2vec + XGBoost | 95,5 % |

La selección original por log loss en validación elige geometría básica. Las mejoras del vector ampliado y del híbrido frente a la referencia tienen intervalos bootstrap pareados que incluyen cero. La adaptación t2vec no es una reproducción exacta del artículo. Una auditoría posterior encontró reconstrucción libre fallida y un defecto de su pérdida muestreada: FIN nunca participaba como negativo. Se conserva ese ensayo para trazabilidad; no se presenta como evidencia contra t2vec. La versión corregida usa normalización sobre todo el vocabulario y se guarda y evalúa por separado.

El notebook presenta las métricas completas de validación y test y una inspección de la estrategia elegida: cuatro aciertos de ambas clases y seis errores de test (las tres falsas alarmas y tres de las 25 diferencias no detectadas). Cada error incluye rutas completas, una ampliación, perfiles de distancia y contribuciones TreeSHAP, con los votos humanos y las reglas de selección. Los identificadores y explicaciones numéricas se conservan en [inspection.json](resultados/rutas/inspection.json). Esta inspección no cambia modelos ni umbrales.

El EDA incluye el reparto de votos decididos por viaje: 1.816 de los 2.386 viajes tienen un solo voto, 410 tienen dos y 19 tienen tres. Entre los que reúnen dos o más, 31 terminan en empate. El desacuerdo entre personas solo es observable cuando hay más de un voto, y el enunciado pide una valoración «según su propia percepción», sin criterio fijo de cuántos metros separan dos rutas distintas.

El rendimiento se reparte además por número de votos. Con el modelo y el umbral congelados, el acierto en validación pasa del 94,3 % con un voto al 100 % con dos o tres; en test, del 92,6 % al 98,6 % y al 100 %. Los 21 errores de validación y 27 de los 28 de test están en viajes de un solo voto, por encima de lo esperado por el tamaño del grupo (p = 0,016 y p = 0,027, prueba hipergeométrica descriptiva y posterior). La comparación no es limpia: un viaje con dos votos y mayoría definida es, por construcción, uno donde dos personas coincidieron, de modo que ruido de etiqueta y dificultad real no se distinguen con este archivo. Las cifras están en [errors_by_votes.json](resultados/rutas/errors_by_votes.json).

El EDA describe también el panel: ocho anotadores muy desiguales. El anotador 2 firma 918 anotaciones sobre 856 viajes y es el único votante en 612 de ellos, el 26 % del archivo; el anotador 5 hizo 21. Los criterios difieren: el anotador 1 marca «diferentes» el 65 % de las veces y el 7 el 39,7 %, y son justo el par que menos coincide entre los que comparten viajes suficientes (82,6 % de acuerdo, kappa de Cohen 0,66, sobre 46 viajes).

Eso se refleja en el modelo. Sobre viajes de validación y test con un único anotador decidido, el acierto va del 88,2 % con el anotador 1 (119 viajes) al 98,0 % con el anotador 7 (147 viajes): cuanto más exigente la persona, peor coincide el modelo. Es coherente con que sus errores sean sobre todo diferencias no detectadas. El reparto es descriptivo y los grupos contienen viajes distintos, así que parte de la brecha puede deberse a dificultad y no a criterio.

Los 7 empates de test se dibujan en una figura propia. En los siete el cuerpo central del trayecto coincide y la mayor separación cae en el primer o el último cuarto del recorrido: la discrepancia está en cómo empieza o acaba el viaje, no en el camino intermedio. El modelo no reproduce esa duda de forma fiable, y con siete casos la comparación no es concluyente.

El notebook añade dos secciones orientadas a la decisión. La **sección 9** traduce la desviación entre rutas a euros: la tarifa se fija antes del viaje, así que la base y la mínima se cancelan y basta con los precios por kilómetro y por minuto. Se usan las tarifas públicas de Cabify por ciudad, categoría estándar, consultadas el 19 de octubre de 2026, y duraciones de OSRM donde el encaje es fiable (1.398 de 2.386 viajes). Es una estimación de orden de magnitud, no una auditoría: las tarifas son de 2026 y el archivo de 2021, se ignoran peajes, tarifa dinámica y promociones, y Colombia y Ecuador usan valores estimados porque la fuente no publica una cifra única. Resultados en **resultados/rutas_euros/**.

La desviación mediana es de 0,11 €, el 3,1 % de una tarifa media de 4,63 €, con P95 de 1,24 €. El neto es −9 € sobre 8.759 € facturados, así que no hay fuga sistemática; pero la compensación ocurre en el agregado, no en la calle. El precio se cierra antes de arrancar: si el trayecto real sale más largo lo absorben Cabify y el conductor, y si sale más corto el cliente pagó de más sin reembolso. Un 8,9 % de los clientes pagó más de un 10 % por encima de lo que valía su trayecto y un 2,6 % de los viajes costó a Cabify más de un 25 % del precio. Frente a un margen EBITDA del grupo del 3,3 % en 2025 (27,2 M€ sobre 833,3 M€), el 48 % de los viajes se desvía más de lo que ese mismo viaje deja de beneficio y el bruto movido equivale a 2,1 veces el margen. El 3,3 % es margen de grupo, no de un trayecto, y la muestra está dominada por ciudades latinoamericanas con tarifas bajas: la comparación da escala, no contabilidad. El impacto se concentra —el 10 % de los viajes acumula el 55 %— y la correlación de Spearman entre la etiqueta humana y la desviación en euros es 0,45, aunque del 10 % de mayor impacto el 93 % está etiquetado «diferentes». Los errores del modelo caen donde hay poco dinero: las 25 diferencias no detectadas en test suman el 2,5 % del total, y el modelo recupera el 97,4 % del dinero en juego entre los viajes realmente distintos pese a detectar el 90,2 % de los casos: falla en los viajes baratos y acierta en los caros. Es una propiedad de este archivo, no un objetivo de entrenamiento, y habría que volver a comprobarla con datos nuevos.

La **sección 10** añade predicción conforme con CV+ sobre entrenamiento y calibración Mondrian en dos grupos, un voto y dos o más, porque los grupos de tres y cuatro votos tienen 8 y 2 casos. El intervalo sobre la proporción de votos se traduce en tres decisiones: iguales, diferentes o revisión humana. Con α = 0,10 el 12,9 % de los viajes va a revisión y el error de las decisiones automáticas baja del 6,3 % al 0,5 %; con α = 0,15, un 6,1 % de revisión deja un 1,9 % de error. En α = 0,05 la abstención llega al 84 % y el método deja de ser utilizable. El residuo en el percentil 90 es un 30 % menor cuando la etiqueta tiene dos votos, lo que pone cifra al valor de una segunda anotación. Implementación en [conformal.py](src/rutas/conformal.py) y resultados en **resultados/rutas_conformal/**.

La **sección 10.1** mide el coste de decidir una pareja nueva: 6,2 ms, de los que XGBoost consume 0,8 µs; el gasto está en la geometría. Un millón de viajes al día son 1,7 horas de CPU, así que el cómputo no es un obstáculo. La revisión humana sí: una abstención del 12,9 % sobre ese volumen son 129.000 revisiones diarias, inviable. La política que se deriva combina ambas secciones —revisar donde el modelo duda **y** hay dinero en juego, ordenando los casos abstenidos por desviación estimada y agotando el presupuesto diario—; no se implementa porque exige el presupuesto real de revisión y la fórmula de tarificación interna. La medición se reproduce con **PYTHONPATH=src python scripts/benchmark_inference.py** y se guarda en [inference_cost.json](resultados/rutas/inference_cost.json).

La **sección 11** cierra con una recomendación explícita: desplegar geometría básica con predicción conforme a α = 0,15, revisar por impacto económico en lugar de por tasa fija, y destinar la siguiente inversión a una segunda anotación antes que a un modelo mayor. No se recomienda desplegar t2vec ni la ampliación viaria: ninguna demuestra mejora y ambas multiplican el coste operativo.

El script **scripts/plot_val_errors.py** dibuja las parejas de validación donde la predicción no coincide con la mayoría humana, con sus votos y separaciones, y guarda el detalle en **resultados/rutas/val_errors_<modelo>.json**. Es una inspección descriptiva: validación intervino en elegir modelo, iteración y umbral, así que no es una medida de error independiente.

El EDA inicial revisa nulos, estructura de coordenadas, longitudes, puntos repetidos, densidad de muestreo, huecos del trazado, composición geográfica y disponibilidad de votos. Explica qué transformaciones ya utiliza cada método y por qué se conservan medidas absolutas y relativas. Las cifras se guardan en [eda.json](resultados/rutas/eda.json); esta ampliación descriptiva, realizada tras la evaluación, no modifica el entrenamiento.

La ampliación viaria utiliza extractos de OpenStreetMap distribuidos por Geofabrik, un motor OSRM local y un grafo dirigido de distancias. Compara variables viarias solas y combinadas con geometría. Sus informes están en **resultados/rutas_grafo/**. Es una ampliación exploratoria posterior a inspeccionar el test original; mantiene las particiones y los modelos anteriores. Las duraciones son estimaciones de un perfil fijo, sin tráfico histórico, y las distancias del grafo simplificado no incorporan restricciones de giro.

## Reproducir la segunda parte

Después de activar el entorno e instalar las dependencias:

~~~sh
python scripts/run_routes.py
python scripts/audit_t2vec_reconstruction.py
PYTHONPATH=src python -m rutas.t2vec_corrected
python scripts/audit_t2vec_reconstruction.py --corrected
python scripts/download_geofabrik.py
python scripts/prepare_road_extract.py
python scripts/start_local_osrm.py
PYTHONPATH=src python -m rutas.graph_extension
python scripts/evaluate_graph_extension.py
PYTHONPATH=src python -m rutas.pricing
PYTHONPATH=src python -m rutas.conformal
PYTHONPATH=src python scripts/benchmark_inference.py
python scripts/build_routes_notebook.py
python scripts/render_notebook.py 02_modelado_rutas.ipynb
PYTHONPATH=src python -m unittest discover -s tests -v
docker stop cabify-challenge-osrm
~~~

El pipeline descarga el JSON público si falta. Los datos brutos, cachés y modelos permanecen locales y se excluyen de Git. Los resultados, predicciones, partición, protocolo y semilla se guardan en **resultados/rutas/**. El entrenamiento del encoder conserva el decoder de reconstrucción; la clasificación usa una cabeza supervisada separada.

Para predecir una pareja nueva con el modelo elegido, ejecutar **python scripts/predict_route.py pareja.json**. El archivo debe contener los campos **estimated_route** y **real_route**, cada uno con una lista de coordenadas **[latitud, longitud]**.

La extensión requiere Docker en funcionamiento y varios GB de disco. El motor está fijado por digest en **scripts/start_local_osrm.py**; se guardan fechas y hashes de las descargas. Las URLs latest de Geofabrik cambian: conservar las cachés es necesario para reproducir exactamente esa cartografía. El servicio se publica solo en 127.0.0.1:5057; los trazados se procesan localmente. La extracción usa una envolvente de cada pareja más 0,08 grados de margen, conservando vías completas y sus referencias. El notebook necesita las cachés para regenerar figuras; el HTML ya exportado se puede leer sin ellas.

El pipeline original y la ampliación son independientes; no hace falta reentrenar los cuatro modelos originales si ya están guardados. **scripts/evaluate_graph_extension.py** comprueba los hashes originales registrados en **resultados/rutas_grafo/original_artifact_hashes.json**. La ampliación del EDA sí modifica su informe descriptivo; no los modelos. Los mapas son © OpenStreetMap contributors, bajo [ODbL](https://www.openstreetmap.org/copyright), distribuidos por [Geofabrik](https://download.geofabrik.de/).

La auditoría de reconstrucción usa 143 viajes de validación interna del encoder; ese conjunto intervino previamente en seleccionar el checkpoint y no se presenta como un test nuevo. Se comparan entradas limpias y alteradas, reconstrucción con y sin ayuda, terminación prematura, edición de secuencias y Fréchet espacial, mostrando explícitamente las salidas vacías. Los informes están en **resultados/rutas_reconstruccion/** y **resultados/rutas_reconstruccion_corregida/**. Los pesos, protocolo y métricas supervisadas corregidos están en **modelos/rutas_t2vec_corregido/** y **resultados/rutas_t2vec_corregido/**.
