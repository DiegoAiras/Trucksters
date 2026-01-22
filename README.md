# Código de Apoyo - Trabajo de Fin de Máster (TFM)

Este repositorio contiene los scripts y el análisis desarrollado para mi Trabajo de Fin de Máster. El objetivo principal del proyecto es el modelado y la predicción de series temporales aplicadas al sector logístico (Caso: Trucksters).

## Estructura del Proyecto

El código está organizado de forma secuencial para facilitar su consulta y replicación:

1. **`01_TFM_EDA.ipynb`**: Análisis Exploratorio de Datos (EDA). Contiene el estudio de las variables de la base de datos, análisis clúster y detección de atípicos.
2. **`02_TFM_Modelos.ipynb`**: Implementación de modelos predictivos sobre la base de datos original. Se aplican y comparan ajustes de **ARIMA** y **Prophet**.
3. **`03_TFM_Modelos_Outliers.ipynb`**: Análisis de robustez aplicando los mismos modelos (ARIMA y Prophet) sobre una versión del dataset tras un proceso de limpieza de valores atípicos (outliers).
4. **`Funciones.py`**: Módulo de Python que contiene todas las funciones personalizadas de preprocesamiento, cálculo de métricas y visualización utilizadas en los notebooks de modelado.

## Tecnologías Utilizadas

* **Lenguaje:** Python 3.13
* **Librerías principales:** * `pandas` y `numpy` para manipulación de datos.
    * `matplotlib` y `seaborn` para visualizaciones.
    * `statsmodels` (ARIMA) y `prophet` para el modelado de series temporales.

## Notas importantes

* **Privacidad de los datos:** Por motivos de confidencialidad con la empresa colaboradora, los datasets originales no han sido incluidos en este repositorio público/privado. El código está diseñado para ser consultado como apoyo metodológico al informe del TFM.
* **Resultados:** Los notebooks contienen los "outputs" guardados de las últimas ejecuciones, permitiendo visualizar las gráficas y resultados sin necesidad de ejecutar el código localmente.
