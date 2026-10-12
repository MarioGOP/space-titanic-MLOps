# space titanic

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# ITESM MNA 2026 

## Description
Predice si un pasajero logro ser teletransportado o no desde el Titanic espacial, utilizando un modelo de regresión logística.

## Preparar el ambiente
Configura un entorno virtual y activa el entorno virtual:

```bash
python3.13 -m venv venv
source venv/bin/activate
```

Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Descarga el dataset de Kaggle (https://www.kaggle.com/competitions/spaceship-titanic/data) y colócalo en la carpeta `data/raw/`. Se utiliza solo el dataset de entrenamiento para este proyecto. Renombra el archivo `train.csv` a `dataset.csv` y asegúrate de que esté en la ruta `data/raw/dataset.csv`.

Corre la limpieza de datos y la generación de características:

```bash
python -m space_titanic.dataset
python -m space_titanic.features
```

Ahora levanta el servidor de MLflow para monitorear los experimentos:

```bash
mlflow ui 
```

Si el servidor no esta corriendo el experimento no se podrá monitorear. El servidor de MLflow estará disponible en `http://localhost:5000`.

Por último, entrena el modelo de regresión logística:

```bash
python -m space_titanic.modeling.train
```

Los argumentos opcionales para el entrenamiento del modelo son:
- --max-iter: Número máximo de iteraciones para el optimizador (por defecto: 1000)
- --c: Inversa de la fuerza de regularización (por defecto: 1.0)
- --solver: Algoritmo a usar en el optimizador (por defecto: 'lbfgs'), opciones: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cholesky'
- --seed: Semilla para la reproducibilidad de los experimentos (por defecto: 42)

ejemplo de uso:

```bash
python -m space_titanic.modeling.train --max-iter 500 --c 0.5 --solver liblinear --seed 123
```

El seed para la reproducibilidad de los experimentos es 42 por defecto. Se puede cambiar al mandar el argumento --seed o cambiando la constante `RANDOM_STATE` en el archivo `config.py`. `RANDOM_STATE` también se utiliza para la división de los datos en entrenamiento y prueba.

El archivo `requirements.txt` contiene todas las versiones exactas de las dependencias necesarias para ejecutar el proyecto. Se requiere Python 3.13
