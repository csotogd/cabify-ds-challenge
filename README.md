# Cabify · Data Science Challenge

Las dos partes del challenge: el experimento para decidir si Just Eat sigue pagando el servicio de fotografía, y un modelo que dice si dos trazados GPS son la misma ruta.

## Por dónde empezar

Hay bastante material. Esto es lo que yo leería según el tiempo disponible.

| Tienes | Lee |
|---|---|
| 5 minutos | Este README hasta «Lo que encontré». |
| 20 minutos | [02_modelado_rutas.ipynb](02_modelado_rutas.ipynb), el resumen de seis líneas del principio y las secciones 9, 10 y 11. Ahí está lo que de verdad cambia una decisión. |
| Una hora | Los dos notebooks enteros. El 01 va de causalidad y potencia, el 02 de modelado. |
| Quiero el código | `src/rutas/` para los modelos, `scripts/` para el pipeline, `tests/` para las 13 pruebas. |

Los dos notebooks están ejecutados y con salidas. También hay HTML por si prefieres el navegador, aunque las fórmulas cargan MathJax de una CDN y sin red se ven en crudo. El notebook no tiene ese problema.

## Lo que encontré

Dos encargos sin relación. En los dos, lo que más valor añadió no fue mejorar el método.

**Parte 1.** El diseño experimental sale bien: ciudades emparejadas, sorteo de acceso inmediato contra diferido, análisis por asignación inicial. El problema es que nadie había calculado cuánto tendría que crecer el margen para que el programa se pague. Sale 1,25 puntos. El techo que permite la aritmética de dilución es 1 punto. Y detectar 1,25 puntos necesitaría unas 1.010 ciudades. Con esos supuestos el experimento no puede distinguir «se paga» de «no se paga», y eso conviene saberlo antes de gastar meses.

**Parte 2.** Ocho medidas geométricas dan un 94,3 % de acierto equilibrado. Tres alternativas más complejas no mejoran de forma demostrable. Pero el 83 % de las etiquetas las firma una sola persona, y el panel no comparte criterio: uno marca «diferentes» el 65 % de las veces y otro el 39,7 %. El modelo acierta un 98 % con el segundo y un 88 % con el primero. Y al medir el problema en euros en vez de en aciertos, el modelo detecta el 90,2 % de los casos pero recupera el 97,4 % del dinero: falla en los viajes baratos.

## Qué hice yo y qué hizo la IA

Esto lo he hecho trabajando con IA, y me parece más útil contar cómo que fingir otra cosa.

**El código lo ha escrito la IA.** Todo: los modelos, el pipeline, las figuras, los tests. Yo no he escrito el código a mano. Lo que sí he hecho es leerlo y revisarlo entero, que es distinto de aceptarlo.

**Las ideas van en los dos sentidos.** Algunas las traigo yo y le pido que las desarrolle. Otras se las pido directamente: «¿qué se te ocurre aquí?». Luego evalúo lo que sale, monto experimentos para comprobarlo, me quedo con lo que aguanta y descarto lo demás. Varias cosas que se propusieron no están en el repositorio porque las probé y no se sostenían.

**Lo que aparece aquí ha pasado un filtro.** Cuando algo me convence a medias, lo refino prompteando hasta que lo entiendo o hasta que queda claro que no va a funcionar. Cuando una línea se agota, pivoto. Varias de las secciones que más me gustan —el umbral de rentabilidad de la Parte 1, medir el error en euros en la Parte 2— salieron de ese ida y vuelta, no de la primera respuesta.

**Las decisiones y las conclusiones son mías.** Qué modelo se elige, qué se recomienda, qué queda fuera y qué límites se declaran. Si algo de aquí está mal, es mío.

Para que esa revisión sea posible y no un acto de fe: el protocolo se congela antes de mirar el test, los resultados y las semillas están guardados en `resultados/`, y hay 13 pruebas. Todo lo de este repositorio se puede volver a ejecutar y comprobar.

## Contenido

| Archivo | Qué es |
|---|---|
| [01_fotografia_evaluacion_retrospectiva.ipynb](01_fotografia_evaluacion_retrospectiva.ipynb) | Parte 1. Retrospectivo como estudio complementario (§1–12), puente (§13), experimento prospectivo (§14–27). |
| [02_modelado_rutas.ipynb](02_modelado_rutas.ipynb) | Parte 2. EDA, cuatro modelos, inspección de errores, euros, predicción conforme y recomendación. |
| [enunciado_challenge.md](enunciado_challenge.md) | El enunciado original, por comodidad. |
| [src/rutas/](src/rutas/) · [scripts/](scripts/) · [tests/](tests/) | Implementación, pipeline y pruebas. |
| [resultados/](resultados/) | Métricas, predicciones, partición, protocolo y semillas. Todo reproducible. |
| [figuras/](figuras/) | Las figuras que generan los notebooks. |
| [requirements.txt](requirements.txt) | Python 3.12 y las versiones exactas que usé. |

---

## Parte 1 · El experimento

Los números de esta parte son **sintéticos**. No tengo registros internos de Just Eat y no pretendo tenerlos. Las cifras públicas dan escala y están fechadas; no son una muestra.

Primero miro el histórico, porque ya existe. La lista de espera da una comparación candidata: todos pidieron el servicio, unos publicaron y otros esperan. Pero todo depende de por qué unos fueron antes. Si el comercial priorizó a quien iba a lanzar una promoción, mido la promoción. Así que el retrospectivo sirve para estimar variabilidad, elegir métrica y dimensionar plazos. No para concluir.

El experimento prospectivo aleatoriza **ciudades**, no restaurantes. Si sorteo restaurantes dentro de una ciudad, el control deja de ser control: el tratamiento le quita pedidos. La unidad tiene que contener a la competencia. Se emparejan ciudades antes del sorteo y dentro de cada pareja una recibe acceso inmediato y la otra diferido con fecha anunciada. No se retira el servicio a nadie. Solicitarlo sigue siendo voluntario y los precios libres, porque lo que se financia es la política, no una versión de laboratorio.

El resultado es el margen de contribución semanal neto del coste del programa, ponderado por el margen previo de cada ciudad.

### El umbral de rentabilidad (§21.4)

Esta sección es la que cierra el encargo. El umbral es el coste semanal del programa en una ciudad dividido por su margen semanal previo, lo que lo deja en las mismas unidades que el efecto estimado:

| | Puntos de crecimiento | Ciudades para detectarlo |
|---|---:|---:|
| Efecto plausible según la dilución | 1,00 | 1.574 |
| Umbral de rentabilidad | **1,25** | **1.010** |
| Para lo que estaba dimensionado el diseño | 5,00 | 68 |

Supuestos: 32 publicaciones por ciudad, 40.000 € de margen semanal, 16 semanas, sesiones a 250 €. El punto de cruce está sobre los 200 € por sesión. Todo hipotético.

De ahí salen dos cosas. Una, que con el coste real por sesión y el margen de ciudad —que Finanzas tiene— la tabla se recalcula en una tarde y decide entre lanzar, rediseñar concentrando el servicio donde el margen expuesto es mayor, o recomendar el cierre sin gastar en el ensayo. Dos, que el contraste debería ser contra el umbral y no contra cero. El servicio ya está corriendo: la carga de la prueba es demostrar que se paga, no que hace algo. Con el nulo clásico puedes salir con un resultado significativo, positivo y ruinoso.

---

## Parte 2 · El modelo

Esta parte usa el archivo real enlazado en el challenge. 2.386 viajes, 3.027 anotaciones, 8 personas.

Antes de modelar: una fila es una anotación, no un viaje. Hay 123 registros repetidos del mismo anotador sobre el mismo viaje. Un voto por persona y viaje; quien se contradice consigo mismo no vota. Partición 60/20/20 por viaje (1.432 / 477 / 477), estratificada, fijada antes de cualquier ajuste. En test, 443 casos tienen mayoría definida.

| Modelo | Acierto equilibrado en test |
|---|---:|
| Geometría básica + XGBoost | 94,3 % |
| Vector interpretable (42) + XGBoost | 94,8 % |
| t2vec reducido + clasificador | 71,8 % |
| Vector + t2vec + XGBoost | 95,5 % |

La regla de selección se congeló en validación por log loss blanda y elige geometría básica. Las mejoras del vector ampliado y del híbrido tienen intervalos bootstrap pareados que incluyen cero, así que no se pueden defender.

**Sobre t2vec.** Mi adaptación no reproduce el artículo. Auditándola después encontré que la reconstrucción libre fallaba y que la pérdida muestreada tenía un defecto: el token de fin nunca participaba como negativo. Reentrené con normalización sobre todo el vocabulario y lo guardé aparte; mejora algunos indicadores y sigue sin reconstruir bien. Conservo el ensayo fallido documentado. Describe mi código, no el método publicado.

### Quién sostiene cada etiqueta

De los 2.386 viajes, 1.816 tienen un solo voto decidido, 410 tienen dos y 19 tienen tres. Entre los de dos o más, 31 acaban en empate.

El panel es muy desigual. El anotador 2 firma 918 anotaciones sobre 856 viajes y es el único votante en 612, el 26 % del archivo. El 5 hizo 21. Y los criterios difieren: el anotador 1 marca «diferentes» el 65 % de las veces y el 7 el 39,7 %, que son justo el par que menos coincide entre los que comparten viajes suficientes (82,6 % de acuerdo, kappa de Cohen 0,66, sobre 46 viajes).

Eso se nota en el modelo. Sobre viajes con un único anotador decidido, el acierto va del 88,2 % con el anotador 1 (n=119) al 98,0 % con el 7 (n=147). Cuanto más exigente la persona, peor coincide. Cautela: cada grupo tiene viajes distintos, así que parte de la brecha puede ser dificultad y no criterio.

El rendimiento también se reparte por número de votos. En validación pasa del 94,3 % con un voto al 100 % con dos o tres; en test, del 92,6 % al 98,6 % y 100 %. Los 21 errores de validación y 27 de los 28 de test caen en viajes de un solo voto, algo por encima de lo que explicaría el tamaño del grupo (p = 0,016 y p = 0,027, hipergeométrica, descriptiva y posterior). La comparación no es limpia: un viaje con dos votos y mayoría definida es, por construcción, uno donde dos personas coincidieron. Ruido de etiqueta y dificultad real no se separan con este archivo. Cifras en [errors_by_votes.json](resultados/rutas/errors_by_votes.json).

Los 7 empates de test están dibujados. En los siete el cuerpo central del trayecto coincide y la mayor separación cae en el primer o el último cuarto: la discrepancia está en cómo empieza o acaba el viaje. El modelo no reproduce esa duda de forma fiable, y con siete casos la comparación no concluye nada.

### En euros (§9)

La tarifa se fija antes del viaje, así que la base y la mínima se cancelan al comparar las dos rutas y basta con el precio por kilómetro y por minuto:

```
Δ€ = €/km · (km_real − km_est) + €/min · (min_real − min_est)
```

Tarifas públicas de Cabify por ciudad, categoría estándar, consultadas el 19 de octubre de 2026. Duraciones de OSRM donde el encaje es fiable (1.398 de 2.386). Esto es una estimación de orden de magnitud, no una auditoría: las tarifas son de 2026 y el archivo de 2021, ignoro peajes, tarifa dinámica y promociones, y Colombia y Ecuador usan valores estimados porque la fuente no publica cifra única. Resultados en `resultados/rutas_euros/`.

La desviación mediana es de 0,11 €, el 3,1 % de una tarifa media de 4,63 €. El neto es −9 € sobre 11.054 € facturados, así que no hay fuga sistemática. Pero la compensación ocurre en el agregado, no en la calle: si el trayecto real sale más largo lo absorben Cabify y el conductor, y si sale más corto el cliente pagó de más y no hay reembolso. Un 8,9 % de clientes pagó más de un 10 % por encima de lo que valía su trayecto.

Frente a un margen EBITDA del grupo del 3,3 % en 2025 (27,2 M€ sobre 833,3 M€), el 48 % de los viajes se desvía más de lo que ese mismo viaje deja de beneficio, y el bruto movido equivale a 2,1 veces el margen. Ese 3,3 % es margen de grupo, no de un trayecto, y la muestra está dominada por ciudades latinoamericanas con tarifas bajas: la comparación da escala, no contabilidad.

El impacto se concentra: el 10 % de los viajes acumula el 55 %. La correlación de Spearman entre la etiqueta humana y la desviación en euros es 0,45, pero del 10 % de mayor impacto el 93 % está etiquetado «diferentes». Y los errores del modelo caen donde hay poco dinero: las 25 diferencias no detectadas suman el 2,5 % del total. **Detecta el 90,2 % de los casos y recupera el 97,4 % del dinero.** Nadie lo entrenó para eso, es una propiedad de este archivo, y habría que reconfirmarla con datos nuevos.

### Predicción conforme (§10)

CV+ sobre entrenamiento, porque validación ya se gastó eligiendo modelo, iteración y umbral. Calibración Mondrian en dos grupos —un voto (n=1.085) y dos o más (n=263)— porque son regímenes de ruido distintos; los grupos de tres y cuatro votos tienen 8 y 2 casos y no llegan ni al mínimo teórico. El intervalo sobre la proporción de votos se traduce en tres decisiones: iguales, diferentes, o revisión humana.

| α | A revisión | Error en lo automático |
|---|---:|---:|
| hoy, umbral fijo | 0 % | 6,3 % |
| 0,20 | 4,1 % | 2,8 % |
| 0,15 | 6,1 % | 1,9 % |
| 0,10 | 12,9 % | 0,5 % |

Con α = 0,05 la abstención se va al 84 % y el método deja de servir. El residuo en el percentil 90 es un 30 % menor cuando la etiqueta tiene dos votos, lo que pone cifra a lo que vale una segunda anotación. Implementación en [conformal.py](src/rutas/conformal.py), resultados en `resultados/rutas_conformal/`.

### Coste y escala (§10.1)

Decidir una pareja nueva cuesta 6,2 ms, de los que XGBoost consume 0,8 µs. El gasto está en la geometría. Un millón de viajes al día son 1,7 horas de CPU, así que el cómputo no es el problema. La revisión humana sí: ese 12,9 % sobre un millón son 129.000 revisiones diarias.

La política que se deriva combina las dos secciones: revisar donde el modelo duda **y** hay dinero en juego, ordenando los casos abstenidos por desviación estimada hasta agotar el presupuesto. No la implemento porque hace falta el presupuesto real de revisión y la fórmula de tarificación interna. Medición en [inference_cost.json](resultados/rutas/inference_cost.json).

### Recomendación (§11)

Desplegar geometría básica con predicción conforme a α = 0,15, revisar por impacto económico en lugar de por tasa fija, y gastar la siguiente inversión en una segunda anotación antes que en un modelo mayor. No desplegaría t2vec ni la ampliación viaria: ninguna demuestra mejora y ambas multiplican el coste operativo.

### Otras cosas que hay dentro

- **EDA.** Nulos, estructura de coordenadas, longitudes, puntos repetidos, densidad de muestreo, huecos del trazado, composición geográfica y votos. El archivo está estructuralmente limpio —cero nulos, cero coordenadas inválidas— pero el 19,2 % de los pasos de las rutas reales tiene longitud cero y 133 rutas tienen saltos de más de 1 km. No eliminé ningún viaje ni recorté outliers. Cifras en [eda.json](resultados/rutas/eda.json).
- **Inspección de errores.** Cuatro aciertos y seis errores de test con rutas completas, ampliación, perfiles de distancia y contribuciones TreeSHAP. Identificadores en [inspection.json](resultados/rutas/inspection.json). No cambia modelos ni umbrales.
- **`scripts/plot_val_errors.py`** dibuja las discrepancias de validación. Descriptivo: validación intervino en la selección, así que no es una medida de error independiente.
- **Ampliación viaria.** Extractos de OpenStreetMap vía Geofabrik, OSRM local y un grafo dirigido de distancias. Exploratoria, posterior a inspeccionar el test, con las particiones y modelos anteriores intactos. Duraciones de un perfil fijo sin tráfico; el grafo simplificado no incorpora restricciones de giro. Informes en `resultados/rutas_grafo/`.

---

## Reproducir

Python 3.12. Desde la raíz del repositorio.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Parte 1:

```sh
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 01_fotografia_evaluacion_retrospectiva.ipynb
```

Parte 2, en orden:

```sh
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
```

`scripts/render_notebook.py` acepta cualquiera de los dos notebooks y genera el HTML maquetado.

Para predecir una pareja nueva: `python scripts/predict_route.py pareja.json`, con los campos `estimated_route` y `real_route`, cada uno una lista de coordenadas `[latitud, longitud]`.

**Notas prácticas.** El pipeline descarga el JSON público si falta. Los datos brutos, cachés y modelos se quedan fuera de Git. La ampliación viaria necesita Docker y varios GB de disco; el motor está fijado por digest en `scripts/start_local_osrm.py` y el servicio solo escucha en 127.0.0.1:5057, así que los trazados no salen de la máquina. Las URLs «latest» de Geofabrik cambian con el tiempo, de modo que conservar las cachés es lo único que garantiza reproducir exactamente esa cartografía. El pipeline original y la ampliación son independientes: `scripts/evaluate_graph_extension.py` comprueba los hashes registrados en `resultados/rutas_grafo/original_artifact_hashes.json` para asegurarse de que no se tocaron los modelos originales.

Mapas © OpenStreetMap contributors, bajo [ODbL](https://www.openstreetmap.org/copyright), distribuidos por [Geofabrik](https://download.geofabrik.de/).

## Lo que no está demostrado

- Que el modelo transfiera a una ciudad nueva. La partición es aleatoria y los trayectos comparten calles.
- Que la asimetría favorable en euros se mantenga fuera de este archivo.
- Que la garantía de la predicción conforme sea exacta. Exige intercambiabilidad y aquí es aproximada.
- Que las cifras en euros sean las reales. Son tarifas de 2026 sobre viajes de 2021.
- Nada de la Parte 1. Es un diseño con números hipotéticos, pendiente de validar disponibilidad de mercados, variabilidad histórica, participación e interferencia entre ciudades.
