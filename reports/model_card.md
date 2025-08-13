# Model Card — Clasificador de Churn (escrito en primera persona)

## Qué modelo construí
Entrené un modelo de clasificación binaria para predecir si un cliente hará **churn**. Como baseline utilicé **Logistic Regression** en un `Pipeline` con `ColumnTransformer` (One-Hot Encoding para categóricas y `StandardScaler` para numéricas).

## Objetivo de negocio
Priorizar a clientes con mayor **probabilidad de churn** para acciones de retención (descuentos, llamadas, ofertas personalizadas).

## Datos
- Fuente: dataset público **IBM Telco Customer Churn**.
- Variable objetivo: `Churn` (Yes/No o 1/0 tras normalización).
- Mezcla de columnas categóricas y numéricas (tenure, cargos, tipo de contrato, etc.).

## Métricas que reporto
- **ROC AUC** y **PR AUC** (me ayudan con clases desbalanceadas).
- **F1**, precisión, exhaustividad y **matriz de confusión** a un umbral por defecto (0.5).
- En el dashboard dejo el umbral **ajustable**.

## Riesgos y consideraciones
- Riesgo de **drift** si cambian políticas comerciales o contexto económico.
- Posible **sesgo** entre segmentos; debo revisar desempeño por subgrupos.
- La probabilidad estimada requiere **recalibración** si aplico nuevos datos.

## Mantenimiento
- Reentrenar con datos recientes y monitorear drift y métricas en producción.
- Revisar el umbral con feedback del equipo de negocio (costo–beneficio).
