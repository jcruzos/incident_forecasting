# Fase 2: Modelado y Entrenamiento (Sección de Modelos)

## Objetivo
Diseñar una sección que permita instanciar, entrenar y evaluar múltiples modelos de Machine Learning utilizando los datos limpios de la Fase 1. El objetivo final es definir cuál es el modelo idóneo para su uso en la proyección oficial.

## Modelos a Evaluar

Dado que el problema de proyectar incidentes en el tiempo se clasifica como un análisis de **Series Temporales (Time Series Forecasting)**, se contemplan los siguientes modelos:

1. **Regresión Lineal / Polinómica:** Modelo base sencillo para captar una tendencia general ascendente o descendente.
2. **ARIMA / SARIMA (Statsmodels):** Modelos estadísticos clásicos excelentes para series con estacionalidad clara y dependencias lineales pasadas.
3. **Prophet (Meta):** Modelo robusto y muy utilizado para series temporales de negocio. Maneja muy bien días festivos, estacionalidades múltiples y valores atípicos.
4. **Random Forest Regressor / XGBoost:** Modelos basados en árboles, estructurando el problema mediante características de tiempo (día de la semana, mes, trimestre).

## Tareas a Desarrollar

1. **Sección de Configuración (Backend / Frontend):**
   - Habilitar una interfaz en el dashboard (Streamlit) donde el usuario pueda seleccionar los modelos que desea entrenar.
   - Definir parámetros como el "horizonte de predicción" (¿Proyectar a 7 días? ¿30 días? ¿3 meses?).

2. **División de los Datos (Train / Test Split):**
   - Separar el histórico de incidentes secuencialmente (ej: 80% para entrenamiento, 20% para prueba). *Nota: En series temporales no se puede hacer split aleatorio.*

3. **Entrenamiento y Evaluación Competitiva:**
   - El backend instanciará los modelos seleccionados y los entrenará con los datos de entrenamiento.
   - Generar predicciones sobre los datos de prueba.
   - Calcular métricas de error para cada modelo, tales como:
     - **MAE** (Error Absoluto Medio).
     - **RMSE** (Raíz del Error Cuadrático Medio).
     - **MAPE** (Error Porcentual Absoluto Medio).

4. **Selección del Modelo Idóneo:**
   - Comparar las métricas de todos los modelos entrenados.
   - El modelo con el menor error (ej: menor MAPE) será marcado como el "Idóneo" y se guardará (serialización con `joblib`) para futuras inferencias rápidas, de manera que no haya que reentrenar a menos que entren nuevos datos.

## Entregables de la Fase 2
- Código del Pipeline de Machine Learning en Python.
- Sección interactiva en el Dashboard para lanzar el entrenamiento de N modelos simultáneos.
- Tabla comparativa de métricas resultante tras el entrenamiento, resaltando el modelo campeón.
