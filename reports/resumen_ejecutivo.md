# Resumen ejecutivo del análisis de churn

## Contexto del proyecto

En el sector de las telecomunicaciones, la retención de clientes es uno de los desafíos más importantes. Un cliente que abandona (*churn*) representa ingresos futuros perdidos, además del costo de adquisición de un nuevo usuario para reemplazarlo. Con este proyecto quise simular un escenario real utilizando datos sintéticos similares al conjunto *Telco Customer Churn* de IBM. Mi objetivo era mostrar, en primera persona, cómo construir un flujo completo de ciencia de datos que incluya la preparación de datos, el análisis exploratorio, el modelado predictivo y la presentación de resultados a través de un tablero interactivo.

## Principales hallazgos

1. **Distribución de churn:** De los 5 000 clientes simulados, aproximadamente el 27 % abandonó el servicio. Este balance moderadamente desequilibrado refleja la realidad de muchas empresas, donde el churn es significativo pero no mayoritario.
2. **Factores determinantes:**
   * **Tipo de contrato:** Los clientes con contratos de mes a mes presentan una tasa de abandono mucho mayor en comparación con los que tienen contratos de uno o dos años. Esto sugiere que los compromisos a largo plazo ofrecen estabilidad y reducen el churn.
   * **Cargos mensuales:** Hay una relación positiva entre los cargos mensuales y la probabilidad de churn; los clientes que pagan tarifas más altas tienden a abandonar en mayor proporción, especialmente cuando no contratan servicios complementarios.
   * **Duración del servicio (tenure):** Los clientes con menos de 12 meses de antigüedad son más propensos a cancelar. A medida que aumenta la relación con la empresa, la tasa de churn disminuye.
3. **Rendimiento del modelo:** En el conjunto de prueba, la **regresión logística** logró una exactitud de alrededor del 61 % y un AUC ≈ 0.61, mientras que el **bosque aleatorio** obtuvo una exactitud cercana al 58 % y un AUC ≈ 0.58. Aunque estos valores son relativamente modestos, reflejan el carácter sintético y simplificado de los datos. La regresión logística resultó ligeramente superior y más interpretable.

## Recomendaciones

1. **Fomentar contratos de largo plazo:** Diseñar campañas para que los clientes migren de contratos mensuales a anuales o bianuales puede reducir la tasa de churn.
2. **Segmentar clientes de alto riesgo:** Utilizar el modelo predictivo para identificar clientes con una probabilidad de abandono superior al 60 % y ofrecerles descuentos, mejoras en su plan o servicios adicionales.
3. **Analizar planes premium:** Revisar la estructura de precios de los servicios con cargos mensuales altos. Ajustar precios o incluir beneficios adicionales podría disminuir el abandono entre estos clientes.

## Próximos pasos

Este proyecto constituye un ejercicio integral de análisis de datos. Para una implementación en producción recomiendo:

* **Validar con datos reales:** Utilizar registros históricos de una empresa real para ajustar el modelo y mejorar su precisión.
* **Incorporar variables externas:** Aspectos como la competencia, promociones de mercado y satisfacción del cliente pueden enriquecer el modelo.
* **Seguir la evolución temporal:** Construir modelos que incorporen series de tiempo, permitiendo detectar patrones de comportamiento a lo largo del ciclo de vida del cliente.

## Conclusión

Este estudio demuestra mi capacidad para diseñar y ejecutar un proyecto de ciencia de datos de principio a fin. El paquete incluye un conjunto de datos sintético, notebooks que documentan el proceso de análisis, modelos predictivos con validación cruzada, un tablero interactivo para explorar los resultados y documentación detallada. Todos estos elementos están listos para ser incluidos en un portafolio profesional, mostrando la combinación de análisis riguroso y comunicación clara.