#  TextLab: Plataforma Web para la Clasificación Supervisada de Textos

Aplicación web interactiva desarrollada en Python y Streamlit que permite diseñar, ejecutar, evaluar y comparar experimentos de clasificación supervisada de textos mediante Aprendizaje Automático, garantizando un flujo libre de fugas de información (*data leakage*).

##  Características Principales
* **Gestión de Datasets:** Carga dinámica de archivos CSV, selección de columnas de texto y etiqueta, detección de valores faltantes/duplicados y visualización de distribución de clases.
* **Extracción de Características:** Soporte para n-gramas (1, 2, 3), filtrado de *stopwords* / *content words* y etiquetado gramatical (POS) con representaciones TF-IDF, frecuencia y binaria.
* **Modelado y Validación:** Entrenamiento con Regresión Logística, SVM, Random Forest y Naive Bayes; partición estratificada Train/Test y Validación Cruzada Estratificada (3, 5 o 10 folds) encapsulados en `sklearn.pipeline.Pipeline`.
* **Evaluación y Comparación:** Cálculo de Accuracy, Precision, Recall y F1-Score (macro y por clase), matriz de confusión interactiva con Plotly y tabla comparativa de experimentos en sesión.
* **Predicción:** Inferencia individual por texto o masiva mediante carga y descarga de archivos CSV.

##  Dependencias
* `streamlit==1.39.0`
* `pandas==2.2.3`
* `numpy==1.26.4`
* `scikit-learn==1.5.2`
* `plotly==5.24.1`
* `nltk==3.9.1`

##  Instrucciones de Ejecución Local

1. Clonar el repositorio:
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd textlab-hackathon
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. (Opcional) Descargar y filtrar el dataset de desarrollo en español:
```bash
pip install datasets
python preparar_dataset.py
```

4. Ejecutar la aplicación web:
```bash
streamlit run app.py
```

##  URL de la Aplicación Publicada

