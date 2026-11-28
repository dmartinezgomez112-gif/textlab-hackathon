# textlab-hackathon

Plataforma web desarrollada en Python con Streamlit y Scikit-learn para diseñar, configurar, entrenar, evaluar, comparar y utilizar modelos de clasificación supervisada de textos sin fuga de información (data leakage) ni dependencia de nombres fijos de columnas o clases.



## URL de la aplicación

https://textlab-hackathon-2qpd9lyspwo4kcrf5zrhft.streamlit.app



## Dependencias

Las dependencias del proyecto se encuentran en `requirements.txt` (incluyendo `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `nltk`, entre otras librerías necesarias).



## Pasos para ejecutarlo



1. Clonar el repositorio desde GitHub:

git clone https://github.com/dmartinezgomez112-gif/textlab-hackathon.git



2. Entrar a la carpeta donde se encuentra el código de la aplicación:

cd textlab-hackathon/textlab-hackathon



3. Instalar las dependencias listadas en `requirements.txt`:

pip install -r requirements.txt



4. Ejecutar la aplicación web con Streamlit:

streamlit run app.py



5. Utilizar la plataforma en el navegador:

- Abrir la dirección local que muestre la terminal (por lo general `http://localhost:8501`).

- En la pestaña **1. Dataset y Exploración**, cargar un archivo CSV etiquetado (como `financial_sentiment_es.csv` u otro conjunto de datos de clasificación de textos), elegir las columnas de texto y etiqueta, y recorrer las secciones de entrenamiento, comparación de experimentos y predicción de nuevos textos.
