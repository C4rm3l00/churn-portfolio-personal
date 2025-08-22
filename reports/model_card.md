---
title: "Modelo de churn - tarjeta informativa"
---

# Tarjeta del modelo de predicción de churn

## Descripción general

Este documento describe el modelo que desarrollé para predecir la probabilidad de abandono (*churn*) de clientes de un proveedor de telecomunicaciones ficticio. Utilicé un conjunto de datos sintético inspirado en el conocido *Telco Customer Churn* de IBM; sin embargo, creé los datos desde cero para evitar restricciones de licencia y asegurar que todos los atributos fueran comprensibles para mí.

El objetivo principal del modelo es identificar qué clientes tienen mayor riesgo de abandonar el servicio para, posteriormente, diseñar estrategias de retención. Trabajé con dos algoritmos distintos:

* **Regresión logística:** Un modelo lineal que resulta interpretable y funciona bien cuando las relaciones entre las variables explicativas y la respuesta son lineales.
* **Bosque aleatorio:** Un ensamble de árboles de decisión que captura relaciones no lineales y posibles interacciones entre variables. Suele mejorar la precisión a costa de interpretabilidad.

## Datos de entrenamiento

* **Filas:** 5 000 clientes
* **Columnas:** 21 atributos, de los cuales 19 son características predictoras (como duración del servicio, servicios contratados, cargos mensuales, método de pago) y 2 columnas corresponden a la clave del cliente y la etiqueta `Churn` (abandonó o se quedó).
* **Procedencia:** los datos fueron generados mediante simulaciones con distribuciones aproximadas al caso real. Todas las variables categóricas incluyen valores como "Yes/No", "Month-to-month/One year/Two year" o distintos métodos de pago. La variable objetivo (`ChurnFlag`) se calculó combinando de forma aleatoria el *tenure*, los cargos mensuales y el tipo de contrato usando una función logística.

## Preprocesamiento

1. **Conversión de tipos:** Convertí `TotalCharges` a numérico y eliminé filas con valores nulos.
2. **Normalización y codificación:** Para modelos que lo requieren, escalé las variables numéricas a media 0 y varianza 1 y apliqué *one-hot encoding* a todas las variables categóricas con `handle_unknown='ignore'`.
3. **Partición:** Separé el conjunto en entrenamiento y prueba con una proporción 70/30 estratificada, para preservar el balance de clases.

## Métricas de rendimiento

Evalué los modelos sobre un conjunto de prueba estratificado y obtuve los siguientes resultados aproximados (los valores exactos pueden variar ligeramente debido a la aleatoriedad en la partición de datos):

| Modelo | Exactitud (accuracy) | Precisión (para la clase *Churn*) | Recall (para la clase *Churn*) | AUC |
|-------|----------------------|----------------------------------|--------------------------------|-----|
| Regresión logística | ≈ 0.61 | ≈ 0.54 | ≈ 0.35 | ≈ 0.61 |
| Bosque aleatorio    | ≈ 0.58 | ≈ 0.49 | ≈ 0.34 | ≈ 0.58 |

En este conjunto de datos la regresión logística supera ligeramente al bosque aleatorio tanto en exactitud como en área bajo la curva ROC. Las métricas relativamente bajas se deben a que los datos son sintéticos y el comportamiento de churn se generó a partir de una función logística simple; aun así, ambos modelos proporcionan una base útil para segmentar clientes según su probabilidad de abandonar.

## Consideraciones éticas y de equidad

El conjunto de datos sintético no incluye atributos sensibles como raza o género en la etiqueta de predicción. Aun así, **existe la posibilidad de que variables como el género influyan de forma indirecta** en la predicción si correlacionan con el churn. Para mitigar sesgos:

* Revisé la distribución de `gender` y verifiqué que no hubiera sesgo desproporcionado en los resultados; no identifiqué diferencias significativas entre "Male" y "Female".
* Puedo excluir `gender` del modelo para asegurar que el género no afecte la predicción. Dado que los modelos continúan funcionando adecuadamente, decidí eliminar esa variable en la versión final que expongo en el dashboard.

## Limitaciones

* **Datos sintéticos:** Aunque los datos se generaron con cuidado, no representan un entorno de negocio real. Para proyectos en producción sería necesario utilizar datos reales y validar su calidad.
* **Simplicidad de la función de churn:** El cálculo del churn en los datos de entrenamiento se basa en una función logística de pocas variables. Un entorno real incluiría muchos más factores (competencia, calidad del servicio, eventos macroeconómicos, etc.).
* **Evolución temporal:** El modelo no considera la evolución temporal ni el comportamiento histórico de los clientes. Para un análisis avanzado, sería recomendable incorporar series de tiempo y secuencias de interacciones.

## Uso previsto y contacto

Este modelo fue desarrollado con fines educativos y de portafolio para demostrar habilidades en análisis de datos, preprocesamiento, construcción de modelos de clasificación y visualización con Streamlit. **No debe ser utilizado para tomar decisiones comerciales reales** sin una validación adicional con datos auténticos y sin pasar por una evaluación ética más profunda.

Si tienes dudas o quieres colaborar en mejoras, puedes contactarme a través del repositorio de GitHub asociado a este proyecto.