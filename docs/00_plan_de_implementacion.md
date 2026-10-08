# Plan de Implementación: Sistema de Proyección de Incidentes

Este documento describe la arquitectura, el stack tecnológico y el flujo de trabajo general para el desarrollo del sistema de carga, análisis, predicción y visualización de incidentes a partir del archivo `TICKETS.xls`.

## Arquitectura y Stack Tecnológico Propuesto

Para mantener un ecosistema unificado y eficiente, especialmente enfocado en datos y Machine Learning, se propone un stack basado en **Python**:

1. **Backend y API:**
   - **FastAPI:** Framework web rápido y moderno para construir la API que recibirá el archivo Excel, orquestará la limpieza de datos y ejecutará los modelos de Machine Learning.
   
2. **Procesamiento y Análisis de Datos:**
   - **Pandas:** Para la lectura, manipulación y limpieza de los datos del archivo Excel.
   - **xlrd / openpyxl:** Dependencias necesarias para leer formatos `.xls` y `.xlsx`.

3. **Machine Learning (Modelado y Predicción):**
   - **Scikit-Learn:** Para preprocesamiento, métricas de evaluación y modelos base de regresión.
   - **Prophet (de Meta) o Statsmodels (ARIMA/SARIMA):** Modelos especializados en series temporales para proyectar la cantidad de incidentes a futuro.
   - **Joblib / Pickle:** Para guardar y cargar los modelos entrenados.

4. **Frontend / Dashboard Gráfico:**
   - **Streamlit:** Framework de Python ideal para crear aplicaciones web interactivas de datos y dashboards en tiempo récord sin necesidad de escribir HTML/CSS/JS (alternativamente, se podría usar **Dash** o un frontend separado en React/Vue, pero Streamlit acelerará el desarrollo drásticamente).

5. **Infraestructura y Despliegue:**
   - **Docker y Docker Compose:** Para contenerizar tanto el backend como el frontend, asegurando que el proyecto funcione idénticamente en cualquier entorno.

## Resumen de Fases del Proyecto

El proyecto se dividirá en 3 fases principales, detalladas en sus respectivos documentos:

- **Fase 1:** [Carga y Limpieza de Datos automatizada](01_fase_carga_limpieza.md).
- **Fase 2:** [Entrenamiento y Selección de Modelos de Machine Learning](02_fase_entrenamiento_modelos.md).
- **Fase 3:** [Proyección de Datos y Dashboard Gráfico](03_fase_proyeccion_dashboard.md).

Además, se incluye una **[Guía de Docker](04_guia_docker.md)** para la validación, instalación y uso de la infraestructura requerida.

## Estructura de Directorios (Propuesta)

```text
incident_dashboards/
│
├── backend/                  # Código de la API FastAPI y modelos ML
│   ├── main.py
│   ├── data_processor.py
│   ├── model_manager.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                 # Código del Dashboard en Streamlit
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── docs/                     # Documentación del proyecto (Fases y Plan)
│   ├── 00_plan_de_implementacion.md
│   ├── 01_fase_carga_limpieza.md
│   └── ...
│
├── data/                     # Carpeta local para almacenar TICKETS.xls temporalmente
├── docker-compose.yml        # Orquestador de contenedores
└── README.md
```
