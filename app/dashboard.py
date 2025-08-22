"""
Dashboard interactivo para el proyecto de churn.

Este módulo define una aplicación web utilizando Streamlit que permite explorar
el conjunto de datos de churn y evaluar la probabilidad de abandono de un
cliente mediante un modelo de aprendizaje automático. La aplicación está
pensada como complemento visual al análisis realizado en los notebooks de
este proyecto y está escrita en primera persona para reflejar mi
participación en todas las etapas.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV y aplica la misma limpieza
    empleada en los notebooks. Se utiliza el decorador `cache_data` para
    evitar recargar los datos en cada interacción.

    Args:
        csv_path: Ruta al archivo CSV con los datos.

    Returns:
        Un DataFrame limpio listo para el análisis y el modelado.
    """
    df = pd.read_csv(csv_path)
    # Conversión de tipos y creación del flag de churn
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['ChurnFlag'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower().startswith('churn') else 0)
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    df.dropna(inplace=True)
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """Entrena dos modelos (regresión logística y bosque aleatorio) con
    los datos proporcionados y devuelve los modelos entrenados junto con
    información de las columnas categóricas y numéricas para el preprocesado.

    Args:
        df: DataFrame limpio con la variable objetivo en la columna `ChurnFlag`.

    Returns:
        Una tupla con el preprocesador, el modelo de regresión logística,
        el modelo de bosque aleatorio, el AUC de cada modelo y una lista de
        columnas numéricas y categóricas.
    """
    X = df.drop(columns=['customerID', 'Churn', 'ChurnFlag'])
    y = df['ChurnFlag']
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Regresión logística
    log_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    log_pipe.fit(X_train, y_train)
    prob_log = log_pipe.predict_proba(X_test)[:, 1]
    auc_log = roc_auc_score(y_test, prob_log)
    # Bosque aleatorio
    rf_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    prob_rf = rf_pipe.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, prob_rf)
    return preprocessor, log_pipe, rf_pipe, auc_log, auc_rf, categorical_cols, numerical_cols


def main():
    st.set_page_config(page_title="Churn Analysis Dashboard", layout="wide")
    st.title("Dashboard de análisis de churn")
    st.markdown(
        """
        Este panel interactivo te permite explorar el conjunto de datos de churn
        de telecomunicaciones que desarrollé para mi portafolio y evaluar,
        mediante modelos de **regresión logística** y **bosques aleatorios**, la probabilidad
        de que un cliente abandone el servicio. Usa la barra lateral para
        navegar por las diferentes secciones.
        """
    )
    # Carga de datos
    data = load_data("data/sample_telco_churn.csv")
    # Entrenamiento de modelos
    preprocessor, log_model, rf_model, auc_log, auc_rf, cat_cols, num_cols = train_models(data)
    # Barra lateral
    menu = st.sidebar.radio("Secciones", (
        "Descripción de datos", "Exploración gráfica", "Predicción individual"
    ))

    if menu == "Descripción de datos":
        st.header("Descripción general de los datos")
        st.write(
            "El conjunto de datos contiene **{} clientes** con **{} características** y una variable objetivo (`ChurnFlag`)."
            .format(data.shape[0], data.shape[1] - 3)
        )
        with st.expander("Mostrar muestra aleatoria"):
            st.dataframe(data.sample(5), use_container_width=True)
        st.subheader("Distribución de la variable objetivo")
        churn_counts = data['Churn'].value_counts().rename(index={'Stayed': 'No churn', 'Churned': 'Churn'})
        st.bar_chart(churn_counts)
        st.markdown(
            f"Hay **{churn_counts.get('Churn', 0)}** clientes que han abandonado y **{churn_counts.get('No churn', 0)}** que permanecen."
        )
        st.subheader("Comparación de cargos mensuales")
        import altair as alt
        box_data = data[['Churn', 'MonthlyCharges']].copy()
        box_data['Churn'] = box_data['Churn'].replace({'Stayed': 'No churn', 'Churned': 'Churn'})
        box = alt.Chart(box_data).mark_boxplot(extent='min-max').encode(
            x=alt.X('Churn:N', title='Estado de churn'),
            y=alt.Y('MonthlyCharges:Q', title='Cargos mensuales')
        ).properties(width=600, height=300)
        st.altair_chart(box, use_container_width=True)
    elif menu == "Exploración gráfica":
        st.header("Exploración gráfica")
        st.write("Selecciona una característica categórica para comparar la distribución de churn:")
        feature = st.selectbox("Característica", cat_cols)
        # Tabla agrupada
        grouped = data.groupby([feature, 'Churn']).size().unstack(fill_value=0)
        grouped = grouped.rename(columns={'Stayed': 'No churn', 'Churned': 'Churn'})
        st.dataframe(grouped)
        # Gráfico de barras apiladas
        stacked = grouped.copy()
        stacked.index.name = feature
        st.bar_chart(stacked)
        # Mención de AUC
        st.markdown(
            f"Resultados del modelo en conjunto de prueba:\n\n"
            f"- **AUC regresión logística:** {auc_log:.2f}\n"
            f"- **AUC bosque aleatorio:** {auc_rf:.2f}"
        )
    elif menu == "Predicción individual":
        st.header("Predicción de churn para un cliente")
        st.write(
            "Introduce los datos de un cliente ficticio y selecciona el modelo con el que deseas predecir la probabilidad de churn."
        )
        # Entrada de usuario para variables
        user_input = {}
        with st.form("formulario"):
            # Variables categóricas
            for col in cat_cols:
                opciones = sorted(data[col].unique().tolist())
                user_input[col] = st.selectbox(col, opciones)
            # Variables numéricas
            for col in num_cols:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                mean_val = float(data[col].mean())
                user_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=mean_val)
            modelo = st.radio("Selecciona modelo", ("Regresión logística", "Bosque aleatorio"))
            submit = st.form_submit_button("Calcular probabilidad")
        if submit:
            # Crear DataFrame de una fila
            new_df = pd.DataFrame([user_input])
            # Aplicar predicción
            if modelo == "Regresión logística":
                prob = log_model.predict_proba(new_df)[0, 1]
            else:
                prob = rf_model.predict_proba(new_df)[0, 1]
            st.success(f"La probabilidad de churn estimada es de {prob * 100:.1f}%")


if __name__ == "__main__":
    main()