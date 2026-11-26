import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import nltk
import re

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="TextLab | Clasificación Supervisada de Textos",
    page_icon="🧪",
    layout="wide",
)

# --- DESCARGA SILENCIOSA DE RECURSOS NLTK ---
@st.cache_resource
def setup_nltk():
    resources = [
        "stopwords",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

setup_nltk()

# --- INICIALIZACIÓN DE ESTADO DE SESIÓN ---
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

# --- TRANSFORMADOR PERSONALIZADO (EVITA DATA LEAKAGE DENTRO DEL PIPELINE) ---
class LinguisticPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_mode="words", language="spanish"):
        self.feature_mode = feature_mode
        self.language = language

    def fit(self, X, y=None):
        return self

    def _process_text(self, text):
        text = str(text).lower()
        tokens = re.findall(r"\b\w+\b", text)

        try:
            sw = set(nltk.corpus.stopwords.words(self.language))
        except Exception:
            sw = set(nltk.corpus.stopwords.words("english"))

        if self.feature_mode == "words":
            return " ".join(tokens)
        elif self.feature_mode == "stopwords":
            filtered = [t for t in tokens if t in sw]
            return " ".join(filtered) if filtered else "empty_sw"
        elif self.feature_mode == "content_words":
            filtered = [t for t in tokens if t not in sw]
            return " ".join(filtered) if filtered else "empty_cw"
        elif self.feature_mode == "pos":
            tagged = nltk.pos_tag(tokens)
            return " ".join([tag for _, tag in tagged]) if tagged else "NONE"
        return " ".join(tokens)

    def transform(self, X):
        return [self._process_text(doc) for doc in X]


# --- INTERFAZ PRINCIPAL ---
st.title("🧪 TextLab: Plataforma de Clasificación Supervisada de Textos")
st.markdown("Diseña, entrena, evalúa y compara modelos de NLP sin fugas de información (*data leakage*).")

tabs = st.tabs([
    "📂 1. Dataset y Exploración",
    "⚙️ 2. Entrenamiento y Evaluación",
    "📊 3. Comparación de Experimentos",
    "🔮 4. Predicción de Nuevos Textos",
])

# ==============================================================================
# PESTAÑA 1: GESTIÓN DEL CONJUNTO DE DATOS (Req. A)
# ==============================================================================
with tabs[0]:
    st.header("Gestión y Diagnóstico del Dataset")
    uploaded_file = st.file_uploader("Cargar archivo CSV etiquetado", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state["df_raw"] = df_raw

        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox("Selecciona la columna de TEXTO:", df_raw.columns, index=0)
        with col2:
            default_label_idx = 1 if len(df_raw.columns) > 1 else 0
            label_col = st.selectbox("Selecciona la columna de ETIQUETA (Clase):", df_raw.columns, index=default_label_idx)

        st.session_state["text_col"] = text_col
        st.session_state["label_col"] = label_col

        # Diagnóstico de calidad de datos
        st.subheader("🔍 Calidad de los Datos")
        missing_text = df_raw[text_col].isna().sum()
        missing_label = df_raw[label_col].isna().sum()
        duplicates = df_raw.duplicated(subset=[text_col, label_col]).sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Registros", len(df_raw))
        m2.metric("Nulos en Texto", int(missing_text))
        m3.metric("Nulos en Etiqueta", int(missing_label))
        m4.metric("Registros Duplicados", int(duplicates))

        clean_option = st.checkbox("Eliminar registros con valores nulos y duplicados automáticamente", value=True)
        df_clean = df_raw[[text_col, label_col]].copy()
        if clean_option:
            df_clean = df_clean.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col, label_col])
        else:
            df_clean = df_clean.dropna(subset=[text_col, label_col])

        st.session_state["df_clean"] = df_clean

        # Identificación automática de clases y distribución
        st.subheader("📊 Distribución de Clases Detectadas")
        class_counts = df_clean[label_col].value_counts().reset_index()
        class_counts.columns = ["Clase", "Cantidad"]
        class_counts["Porcentaje (%)"] = (class_counts["Cantidad"] / len(df_clean) * 100).round(2)

        c_chart, c_table = st.columns([2, 1])
        with c_chart:
            fig_dist = px.bar(
                class_counts,
                x="Clase",
                y="Cantidad",
                color="Clase",
                text="Cantidad",
                title="Frecuencia de Textos por Clase",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with c_table:
            st.dataframe(class_counts, use_container_width=True, hide_index=True)

        with st.expander("Vista previa de los datos procesados"):
            st.dataframe(df_clean.head(15), use_container_width=True)
    else:
        st.info("⬆️ Sube un archivo CSV para comenzar.")

# ==============================================================================
# PESTAÑA 2: EXTRACCIÓN, CONFIGURACIÓN Y ENTRENAMIENTO (Req. B, C, D)
# ==============================================================================
with tabs[1]:
    st.header("Configuración del Pipeline y Entrenamiento")

    if "df_clean" not in st.session_state:
        st.warning("⚠️ Primero carga y configura un dataset en la pestaña 1.")
    else:
        df_clean = st.session_state["df_clean"]
        text_col = st.session_state["text_col"]
        label_col = st.session_state["label_col"]

        col_feat, col_model, col_val = st.columns(3)

        # --- B. Extracción de características ---
        with col_feat:
            st.subheader("1. Extracción de Características")
            lang_choice = st.selectbox("Idioma Stopwords:", ["spanish", "english"])
            feat_type = st.selectbox(
                "Tipo de tokens / características:",
                options=["words", "content_words", "stopwords", "pos"],
                format_func=lambda x: {
                    "words": "Palabras completas (N-gramas)",
                    "content_words": "Content Words (Sin stopwords)",
                    "stopwords": "Solo Stopwords",
                    "pos": "Etiquetas gramaticales (POS)",
                }[x],
            )
            ngram_choice = st.selectbox(
                "Rango de N-gramas:",
                options=[(1, 1), (2, 2), (3, 3), (1, 2)],
                format_func=lambda x: {
                    (1, 1): "Unigramas (1-gramas)",
                    (2, 2): "Bigramas (2-gramas)",
                    (3, 3): "Trigramas (3-gramas)",
                    (1, 2): "Unigramas + Bigramas",
                }[x],
            )
            rep_type = st.selectbox(
                "Representación vectorial:",
                options=["tfidf", "frequency", "binary"],
                format_func=lambda x: {
                    "tfidf": "TF-IDF",
                    "frequency": "Frecuencia (Conteo)",
                    "binary": "Binaria (Presencia/Ausencia)",
                }[x],
            )

        # --- C. Clasificador e Hiperparámetros ---
        with col_model:
            st.subheader("2. Clasificador e Hiperparámetros")
            clf_name = st.selectbox(
                "Algoritmo Clasificador:",
                ["Regresión Logística", "SVM", "Random Forest", "Naive Bayes"],
            )

            params = {}
            if clf_name == "Regresión Logística":
                params["C"] = st.number_input("Regularización (C):", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
                params["max_iter"] = st.slider("Máx. Iteraciones:", 100, 1000, 300, step=100)
            elif clf_name == "SVM":
                params["C"] = st.number_input("Regularización (C):", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
                params["kernel"] = st.selectbox("Kernel:", ["linear", "rbf"])
            elif clf_name == "Random Forest":
                params["n_estimators"] = st.slider("Número de árboles (n_estimators):", 10, 300, 100, step=10)
                params["max_depth"] = st.selectbox("Profundidad máxima:", [None, 10, 20, 50])
            elif clf_name == "Naive Bayes":
                params["alpha"] = st.number_input("Suavizado Laplace (alpha):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

        # --- Validación y Partición ---
        with col_val:
            st.subheader("3. Partición y Validación")
            test_size = st.slider("Proporción Test (%):", min_value=10, max_value=50, value=20, step=5) / 100.0
            cv_option = st.selectbox(
                "Validación Cruzada Estratificada:",
                options=[0, 3, 5, 10],
                format_func=lambda x: "Desactivada" if x == 0 else f"{x}-Folds Estratificados",
            )
            random_seed = st.number_input("Semilla aleatoria (Random State):", value=42, step=1)

        run_btn = st.button("🚀 Entrenar y Evaluar Modelo", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Ejecutando pipeline sin fugas de información..."):
                X = df_clean[text_col].astype(str)
                y = df_clean[label_col]

                if rep_type == "tfidf":
                    vectorizer = TfidfVectorizer(ngram_range=ngram_choice)
                elif rep_type == "frequency":
                    vectorizer = CountVectorizer(ngram_range=ngram_choice, binary=False)
                else:
                    vectorizer = CountVectorizer(ngram_range=ngram_choice, binary=True)

                if clf_name == "Regresión Logística":
                    clf = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=random_seed)
                elif clf_name == "SVM":
                    clf = SVC(C=params["C"], kernel=params["kernel"], probability=True, random_state=random_seed)
                elif clf_name == "Random Forest":
                    clf = RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        random_state=random_seed,
                    )
                else:
                    clf = MultinomialNB(alpha=params["alpha"])

                pipeline = Pipeline([
                    ("prep", LinguisticPreprocessor(feature_mode=feat_type, language=lang_choice)),
                    ("vec", vectorizer),
                    ("clf", clf),
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=random_seed
                )

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

                cv_mean, cv_std = None, None
                if cv_option > 0:
                    skf = StratifiedKFold(n_splits=cv_option, shuffle=True, random_state=random_seed)
                    cv_results = cross_validate(pipeline, X_train, y_train, cv=skf, scoring="f1_macro")
                    cv_mean = np.mean(cv_results["test_score"])
                    cv_std = np.std(cv_results["test_score"])

                exp_id = len(st.session_state.experiments) + 1
                ngram_label = f"{ngram_choice[0]}-gramas" if ngram_choice[0] == ngram_choice[1] else "1-2 gramas"
                feat_desc = f"{feat_type} | {ngram_label} | {rep_type}"
                model_key = f"Exp #{exp_id}: {clf_name} ({feat_desc})"

                st.session_state.experiments.append({
                    "ID": f"Exp #{exp_id}",
                    "Clasificador": clf_name,
                    "Características": feat_desc,
                    "Accuracy (%)": round(acc * 100, 2),
                    "Precision Macro (%)": round(prec_macro * 100, 2),
                    "Recall Macro (%)": round(rec_macro * 100, 2),
                    "F1 Macro (%)": round(f1_macro * 100, 2),
                    "CV F1 Media (%)": round(cv_mean * 100, 2) if cv_mean is not None else "N/A",
                    "CV F1 Std (±%)": round(cv_std * 100, 2) if cv_std is not None else "N/A",
                })
                st.session_state.trained_models[model_key] = pipeline

                st.success(f"✅ Experimento #{exp_id} completado y guardado en el historial.")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Accuracy", f"{acc*100:.2f}%")
                k2.metric("Precision (Macro)", f"{prec_macro*100:.2f}%")
                k3.metric("Recall (Macro)", f"{rec_macro*100:.2f}%")
                k4.metric("F1-Score (Macro)", f"{f1_macro*100:.2f}%")

                if cv_mean is not None:
                    st.info(f"🔄 **Validación Cruzada ({cv_option}-Folds sobre Train):** F1-Macro Promedio = **{cv_mean*100:.2f}%** | Desviación Estándar = **±{cv_std*100:.2f}%**")

                g1, g2 = st.columns(2)
                classes = sorted(list(y.unique()))

                with g1:
                    cm = confusion_matrix(y_test, y_pred, labels=classes)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        x=[str(c) for c in classes],
                        y=[str(c) for c in classes],
                        labels=dict(x="Predicción", y="Real", color="Cantidad"),
                        title="Matriz de Confusión (Conjunto Test)",
                        color_continuous_scale="Blues",
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                with g2:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    class_metrics = []
                    for c in classes:
                        c_str = str(c)
                        if c_str in report:
                            class_metrics.append({"Clase": c_str, "Métrica": "Precision", "Valor": report[c_str]["precision"]})
                            class_metrics.append({"Clase": c_str, "Métrica": "Recall", "Valor": report[c_str]["recall"]})
                            class_metrics.append({"Clase": c_str, "Métrica": "F1-Score", "Valor": report[c_str]["f1-score"]})
                    df_metrics = pd.DataFrame(class_metrics)
                    fig_class = px.bar(
                        df_metrics,
                        x="Clase",
                        y="Valor",
                        color="Métrica",
                        barmode="group",
                        title="Desempeño Detallado por Clase",
                        range_y=[0, 1.05],
                    )
                    st.plotly_chart(fig_class, use_container_width=True)

# ==============================================================================
# PESTAÑA 3: COMPARACIÓN DE EXPERIMENTOS (Req. E)
# ==============================================================================
with tabs[2]:
    st.header("Comparación de Experimentos de la Sesión")

    if not st.session_state.experiments:
        st.info("Aún no has ejecutado experimentos. Entrena al menos un modelo en la pestaña 2.")
    else:
        df_exp = pd.DataFrame(st.session_state.experiments)
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

        fig_comp = px.bar(
            df_exp,
            x="ID",
            y=["Accuracy (%)", "F1 Macro (%)"],
            barmode="group",
            hover_data=["Clasificador", "Características"],
            title="Comparativa de Accuracy vs F1-Macro por Experimento",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        if st.button("🗑️ Limpiar historial de experimentos"):
            st.session_state.experiments = []
            st.session_state.trained_models = {}
            st.rerun()

# ==============================================================================
# PESTAÑA 4: PREDICCIÓN DE NUEVOS TEXTOS (Req. F)
# ==============================================================================
with tabs[3]:
    st.header("Predicción con Modelos Entrenados")

    if not st.session_state.trained_models:
        st.warning("⚠️ Entrena al menos un modelo antes de realizar predicciones.")
    else:
        selected_model_name = st.selectbox(
            "Selecciona un modelo entrenado en esta sesión:",
            list(st.session_state.trained_models.keys()),
        )
        model_pipeline = st.session_state.trained_models[selected_model_name]

        pred_mode = st.radio("Modo de entrada:", ["Texto individual", "Archivo CSV sin etiquetar"], horizontal=True)

        if pred_mode == "Texto individual":
            user_text = st.text_area("Escribe el texto a clasificar:", height=120)
            if st.button("Predecir Clase", type="primary"):
                if user_text.strip():
                    prediction = model_pipeline.predict([user_text])[0]
                    st.success(f"🏷️ **Clase Predicha:** `{prediction}`")
                else:
                    st.warning("Por favor ingresa un texto válido.")
        else:
            unlabeled_file = st.file_uploader("Sube un CSV con textos nuevos", type=["csv"], key="pred_csv")
            if unlabeled_file is not None:
                df_new = pd.read_csv(unlabeled_file)
                pred_text_col = st.selectbox("Selecciona la columna que contiene los textos:", df_new.columns)
                if st.button("Ejecutar Predicción Masiva", type="primary"):
                    texts_to_pred = df_new[pred_text_col].fillna("").astype(str)
                    df_new["Clase_Predicha"] = model_pipeline.predict(texts_to_pred)
                    st.dataframe(df_new, use_container_width=True)

                    csv_out = df_new.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Descargar resultados en CSV",
                        data=csv_out,
                        file_name="predicciones_textlab.csv",
                        mime="text/csv",
                    )
