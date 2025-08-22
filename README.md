# Proyecto de análisis de churn

Este repositorio contiene un proyecto de ciencia de datos elaborado para demostrar mis habilidades como analista y científico de datos. El objetivo es predecir el abandono de clientes (*churn*) en un contexto de telecomunicaciones utilizando un conjunto de datos sintético inspirado en el *Telco Customer Churn* de IBM. La estructura del proyecto sigue un flujo de trabajo completo: desde la carga y limpieza de datos, pasando por el análisis exploratorio y el modelado, hasta la creación de un tablero interactivo.

## Estructura de carpetas

```
churn-portfolio-personal/
├── app/
│   └── dashboard.py        # Aplicación Streamlit para explorar datos y hacer predicciones
├── data/
│   └── sample_telco_churn.csv # Dataset sintético de 5 000 clientes
├── notebooks/
│   ├── 01_load_and_clean.ipynb     # Carga y limpieza de datos
│   ├── 02_eda_sql.ipynb            # Análisis exploratorio con SQL y visualizaciones
│   ├── 03_modeling.ipynb           # Construcción de modelos (regresión logística y bosque aleatorio)
│   └── 04_dashboard_export.ipynb    # Exportación de gráficas y tablas para el dashboard
├── reports/
│   ├── model_card.md         # Tarjeta informativa del modelo
│   └── resumen_ejecutivo.md  # Resumen ejecutivo del proyecto
├── requirements.txt          # Paquetes necesarios para reproducir el proyecto
└── README.md                 # Este documento
```

## Cómo ejecutar el proyecto

1. **Clona el repositorio** en tu máquina local y navega al directorio del proyecto.

2. **Crea un entorno virtual** (opcional pero recomendado) e instala las dependencias:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Explora los notebooks**: puedes abrir los archivos `.ipynb` con Jupyter Lab o Jupyter Notebook para seguir paso a paso la carga de datos, la limpieza, el análisis exploratorio, el modelado y la exportación de gráficos.

4. **Ejecuta el dashboard**: para lanzar la aplicación Streamlit y explorar el proyecto de manera interactiva, ejecuta:

   ```bash
   streamlit run app/dashboard.py
   ```

   La aplicación te permitirá visualizar la distribución del churn, explorar comparaciones por categorías y calcular la probabilidad de abandono para un cliente ficticio.

5. **Revisa los reportes**: en la carpeta `reports/` encontrarás una tarjeta informativa del modelo con detalles sobre su rendimiento y un resumen ejecutivo con los hallazgos principales y recomendaciones de negocio.

## Dependencias

Las dependencias principales utilizadas en este proyecto se encuentran en `requirements.txt` e incluyen:

* `pandas` y `numpy` para la manipulación de datos.
* `matplotlib` y `seaborn` para visualizaciones básicas.
* `scikit-learn` para el preprocesamiento y los modelos de clasificación.
* `sqlite3` integrado en Python para consultas SQL en memoria.
* `streamlit` y `altair` para construir el dashboard interactivo.
* `jupyter` para ejecutar los notebooks.

Puedes instalar todas las dependencias con `pip install -r requirements.txt`.

## Contribuciones

Este proyecto fue desarrollado como parte de mi portafolio personal y no está sujeto a contribuciones externas. Sin embargo, si encuentras algún problema o deseas sugerir mejoras, no dudes en abrir un *issue* o enviar una *pull request*.

## Licencia

El contenido de este proyecto se encuentra bajo la licencia MIT, lo que significa que eres libre de utilizar el código y los informes para fines educativos o personales. El dataset sintético se genera únicamente con fines demostrativos y no refleja datos reales de clientes.