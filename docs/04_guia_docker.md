# Guía de Infraestructura: Docker y Docker Compose

Todo el desarrollo de este sistema (carga de excel, modelos de ML y dashboard) está planteado para funcionar dentro de contenedores usando **Docker**. Esto previene el clásico problema de "funciona en mi máquina" y facilita un despliegue transparente.

## ¿Qué necesitas tener instalado?

1. **Docker Engine / Docker Desktop** (Dependiendo si usas Linux, Windows o Mac).
2. **Docker Compose** (Viene integrado con las versiones recientes de Docker Desktop).

## 1. Validar la Instalación de Docker

Abre una terminal (Símbolo del sistema, PowerShell o Bash) y ejecuta los siguientes comandos:

```bash
# Validar si Docker está instalado y ejecutándose
docker --version
# Debería mostrar algo como: Docker version 24.0.0, build ...

# Validar si Docker Compose está instalado
docker-compose --version 
# O en versiones más nuevas: 
docker compose version
# Debería mostrar algo como: Docker Compose version v2.18.1

# Comprobar que el daemon/servicio está corriendo correctamente (bajando una imagen ligera de prueba)
docker run hello-world
```
Si el último comando imprime un mensaje que dice "Hello from Docker!", tu instalación está correcta y validada.

## 2. Estructura de Contenedores del Proyecto

El proyecto utilizará dos contenedores simultáneos orquestados por un archivo `docker-compose.yml`:
1. **Contenedor "backend":** Correrá FastAPI y ejecutará los scripts pesados de Pandas y Machine Learning. Expondrá un puerto (ej. `8000`).
2. **Contenedor "frontend":** Correrá el dashboard de Streamlit. Expondrá el puerto `8501`.

## 3. ¿Cómo utilizar Docker en este Proyecto?

Una vez desarrollados los archivos `Dockerfile` y el `docker-compose.yml` (como parte de la implementación técnica), el uso del proyecto será sumamente sencillo.

### Levantar el Entorno (Start)
Ve a la terminal, posiciónate en la carpeta raíz (`c:\Proyectos\incident_dashboards`) y ejecuta:

```bash
docker-compose up --build
```
- El flag `--build` fuerza a Docker a descargar e instalar todas las dependencias de Python (pandas, scikit-learn, fastapi, etc.) la primera vez que se corre o cuando hay cambios.
- Podrás ver los logs en la terminal indicando que la API y el Dashboard están listos.

### Acceder al Sistema
- **Dashboard UI:** Abre tu navegador y dirígete a `http://localhost:8501`
- **Documentación de la API (Swagger):** Dirígete a `http://localhost:8000/docs`

### Apagar el Entorno (Stop)
Para detener el proyecto correctamente sin borrar las imágenes creadas, puedes pulsar `Ctrl + C` en la terminal donde se está ejecutando, o bien, si está en background, ejecuta:

```bash
docker-compose down
```

## Beneficios de este esquema:
- Nadie del equipo necesitará instalar Python, ni gestionar entornos virtuales (`venv`), ni lidiar con versiones conflictivas de dependencias en su máquina local. 
- Mover este proyecto a producción en un servidor cloud AWS/Azure/GCP será inmediato al entregar únicamente las configuraciones de Docker.
