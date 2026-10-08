# Fase 1: Carga de Excel y Limpieza de Datos

## Objetivo
Implementar el módulo capaz de recibir un archivo Excel (como `TICKETS.xls`), procesarlo automáticamente, aplicar reglas de limpieza y filtrar exclusivamente los registros catalogados como "Incidentes".

## Tareas a Desarrollar

1. **Endpoint de Carga (Upload API):**
   - Crear un endpoint en **FastAPI** (`POST /upload`) que acepte archivos `.xls` o `.xlsx`.
   - Guardar el archivo temporalmente en una carpeta segura dentro del contenedor o cargarlo directamente en memoria para su procesamiento.

2. **Lectura y Transformación Inicial (Pandas):**
   - Utilizar `pandas.read_excel()` para extraer los datos crudos.
   - Estandarizar los nombres de las columnas (ej: eliminar espacios, convertir a minúsculas) para evitar errores de tipeo.
   - Manejar valores nulos (NaT, NaN) imputándolos o eliminando las filas si la información es crítica (ej: fecha de creación nula).

3. **Filtrado Automático de Incidentes:**
   - Analizar la columna que define el "Tipo de Ticket" (ej: `Tipo`, `Clasificación`, `Categoría`).
   - Aplicar una regla condicional para conservar únicamente las filas donde el tipo corresponda a "Incidente".
   - Descartar "Requerimientos", "Cambios", "Problemas" o cualquier otro registro irrelevante para este análisis.

4. **Preparación de Datos para Series Temporales (Agrupación):**
   - Extraer la fecha (y hora, si aplica) de la creación del incidente.
   - Transformar la granularidad a un formato de serie temporal: Agrupar la cantidad de incidentes por día, semana o mes, dependiendo del volumen de los datos. 
   - El resultado será un dataset estructurado con dos columnas principales (ej: `fecha` y `cantidad_incidentes`), el cual es el formato idóneo para entrenar modelos predictivos.

## Entregables de la Fase 1
- Código fuente del backend (FastAPI + Script de Pandas).
- Endpoint REST funcional para subir el Excel y retornar una previsualización de los datos limpios en formato JSON.
- Tests unitarios básicos comprobando que los tipos de tickets no correspondientes a "Incidente" han sido descartados.
