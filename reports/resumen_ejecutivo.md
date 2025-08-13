# Resumen ejecutivo (en primera persona)

Mi análisis identifica patrones asociados a mayor probabilidad de **churn**. A nivel exploratorio, los clientes con contrato **Month-to-month** y cargos mensuales más altos tienden a presentar mayor riesgo. Entrené un modelo baseline (Logistic Regression con OHE/Scaler) y lo integré en un **dashboard** donde negocio puede:

- Ajustar el **umbral** de riesgo y dimensionar cuántos clientes quedarían marcados.
- Filtrar por **Contract** para priorizar segmentos (“Month-to-month”, “One year”, “Two year”).
- Visualizar la **distribución** de probabilidades de churn.

El entregable está preparado para presentaciones: si no encuentro el archivo de predicciones, el dashboard calcula proba al vuelo con el dataset limpio o, en última instancia, usa un **sample** incluido. Mi siguiente paso sería agregar explicación de variables (permutation importance/SHAP) y optimizar el umbral con una **función de costo** de retención.
