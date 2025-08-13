# Análisis de Churn de Clientes — Proyecto Personal

Construí este proyecto para **demostrar mis habilidades** del track de *IBM Data Analyst* y *IBM Data Science* con un caso clásico de negocio: **churn de telecomunicaciones**. Mi objetivo fue cubrir **todo el ciclo**: ingestión de datos, limpieza, EDA, SQL, modelado con scikit‑learn y un **dashboard** para comunicar resultados.

> **Datos**: uso el dataset público *Telco Customer Churn*. El proyecto puede cargarlo de forma automática (vía API/URL) o, si estoy sin internet, se apoya en un *sample* incluido para que todo corra.

---

## Qué quise mostrar
- **Adquisición y limpieza** de datos con `pandas` (casting, imputación, normalización de columnas).
- **EDA** con gráficos y **consultas SQL** (SQLite en memoria) para responder preguntas de negocio.
- **Modelado** con `scikit-learn` usando `ColumnTransformer` + `Pipeline` + `GridSearchCV` y métricas (`ROC AUC`, `PR AUC`, matriz de confusión).
- **Dashboard** con Plotly Dash para explorar el **riesgo de churn** por umbral y segmento de contrato.
- **Robustez**: el dashboard tiene *fallbacks* para que siempre muestre algo (predicciones guardadas, datos limpios o sample).

---

## Cómo lo ejecuto (VS Code + Jupyter)
1) Crear y activar entorno:
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2) Notebooks en orden:
- `notebooks/01_load_and_clean.ipynb`  → deja `data/telco_churn_cleaned.csv`
- `notebooks/02_eda_sql.ipynb`         → EDA + SQL (SQLite en memoria)
- `notebooks/03_modeling.ipynb`        → pipeline + métricas
- `notebooks/04_dashboard_export.ipynb`→ crea `data/predictions.csv` (opcional; el dashboard funciona con fallbacks)

3) Dashboard:
```bash
python app/dashboard.py
```
En consola veré la fuente de datos que tomó:
```
[dashboard] Fuente: predictions.csv
[dashboard] Fuente: telco_churn_cleaned.csv + quick model
[dashboard] Fuente: sample_telco_churn.csv + quick model
```

---

## Estructura del repo
```
churn-portfolio-personal/
├─ app/
│  └─ dashboard.py
├─ data/
│  └─ sample_telco_churn.csv
├─ notebooks/
│  ├─ 01_load_and_clean.ipynb
│  ├─ 02_eda_sql.ipynb
│  ├─ 03_modeling.ipynb
│  └─ 04_dashboard_export.ipynb
├─ reports/
│  ├─ model_card.md
│  └─ resumen_ejecutivo.md
├─ requirements.txt
└─ README.md
```

---

## Decisiones y aprendizajes (resumen)
- Usé **Logistic Regression** como baseline por su interpretabilidad y **GridSearchCV** para ajustar `C`.
- Organicé el preprocesamiento con **ColumnTransformer**: OHE para categóricas y `StandardScaler` para numéricas.
- Calculé métricas de clasificación enfocándome en **áreas bajo curva** (ROC/PR) por ser más informativas con clases desbalanceadas.
- El **dashboard** expone un **umbral de decisión** y segmentación por `Contract` para ayudar a negocio a priorizar campañas.
- Añadí *fallbacks* para evitar errores cuando el archivo de predicciones no existe (muy útil al presentar).

---

## Trabajo futuro
- Añadir **importancia de variables** (permutation importance) y explicación local (SHAP).
- Optimizar el umbral con una **función de costo** (retención vs. gasto).
- Crear un pequeño test set con **temporal split** y registrar experimentos.

---

## Lo que destaco en mi CV
- Construí un **pipeline reproducible** de punta a punta: `pandas` → `SQL` → `ML` → `Dashboard`.
- Apliqué **validación cruzada** y medí `ROC AUC`/`PR AUC`; documenté resultados y limitaciones.
- Entregué un **dashboard interactivo** listo para stakeholders.
- Escribí una **model card** y un **resumen ejecutivo** para comunicación no técnica.
