# Cabify · Data Science Challenge

Propuesta de evaluación del servicio de fotografía profesional de Just Eat, preparada para la primera parte del challenge de Data Science.

El notebook combina explicaciones, gráficos, cálculos de tamaño de muestra y simulaciones. Presenta un estudio retrospectivo como evidencia complementaria y un experimento prospectivo aleatorizado por ciudades o áreas metropolitanas emparejadas.

## Contenido

- [Notebook ejecutado](01_fotografia_evaluacion_retrospectiva.ipynb): desarrollo completo, código y resultados. La Parte I trata la evaluación retrospectiva; la Parte II desarrolla el experimento prospectivo.
- [Versión HTML](01_fotografia_evaluacion_retrospectiva.html): lectura en un navegador después de descargar o clonar el repositorio.
- [Gráficos](figuras/): figuras generadas por el notebook.
- [Notas de preparación](notas_preparacion_experimento.json): decisiones metodológicas y referencias para la entrevista.
- [Dependencias](requirements.txt): versiones del entorno utilizado.

Los ejemplos numéricos y las simulaciones son sintéticos. Las referencias públicas se citan en el notebook y no sustituyen datos internos de Just Eat. La viabilidad real del experimento queda pendiente de validar la disponibilidad de mercados, la variabilidad histórica, la participación y la interferencia entre ciudades.

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

Primera parte desarrollada: evaluación retrospectiva complementaria y diseño experimental prospectivo. La segunda parte del challenge está pendiente.
