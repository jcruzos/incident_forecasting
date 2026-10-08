# Fase 3: Proyección de Datos y Dashboard Gráfico

## Objetivo
Generar la visualización final de la proyección de incidentes utilizando el "Modelo Idóneo" (definido en la Fase 2) e integrar todo en un dashboard gráfico fácil de utilizar y estéticamente agradable.

## Tareas a Desarrollar

1. **Generación de la Proyección (Forecast):**
   - Utilizando el modelo campeón entrenado, predecir la cantidad de incidentes para los siguientes períodos definidos (por ejemplo, los próximos 30 días).
   - Generar un DataFrame que fusione:
     1. Los datos históricos reales (el pasado).
     2. Los datos proyectados/predichos (el futuro), incluyendo intervalos de confianza si el modelo lo permite (como Prophet).

2. **Desarrollo del Dashboard Gráfico (Streamlit):**
   - **Módulo de Carga (UI de Fase 1):** Un componente "Drag & Drop" para subir el archivo `TICKETS.xls`.
   - **Panel de KPIs:** Mostrar indicadores clave de rendimiento como: Total de incidentes históricos, Promedio diario, Modelo actual en uso y Tendencia proyectada (alza/baja).
   - **Gráfico Principal:** 
     - Utilizar bibliotecas gráficas interactivas como `Plotly` o `Altair` (integradas muy bien en Streamlit).
     - El gráfico debe ser una línea de tiempo (Time Series plot).
     - Línea sólida de un color (ej: azul) para representar los incidentes históricos ocurridos.
     - Línea punteada de otro color (ej: naranja) o área sombreada para representar la proyección a futuro y el rango de incertidumbre.

3. **Interactividad y Filtros:**
   - Controles para hacer zoom in/out en períodos específicos del tiempo.
   - Opción para exportar/descargar los datos proyectados en un nuevo archivo CSV/Excel.

4. **Integración Final de Sistema:**
   - Conectar fluidamente el Frontend (Dashboard) con el Backend (API). Cuando el usuario suba el Excel en el dashboard, la solicitud viaja al backend, se procesa, se selecciona/ejecuta el modelo, y retorna los datos tabulares que el dashboard finalmente grafica.

## Entregables de la Fase 3
- Aplicación de Dashboard (Frontend) terminada y vinculada a la API.
- Gráficos interactivos de líneas de tiempo demostrando claramente el corte entre el "hoy" y el "futuro".
- KPIs descriptivos que resuman la situación del servicio o plataforma según las proyecciones.
