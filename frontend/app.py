import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

# Configuración y conexión
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Dashboard Proyección de Incidentes", layout="wide")
st.title("📈 Sistema de Análisis y Proyección de Incidentes")
st.markdown("Carga tu archivo Excel de tickets, limpia los datos automáticamente y entrena modelos predictivos.")

# --- BARRA LATERAL ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube el archivo TICKETS.xls", type=["xls", "xlsx"])

if uploaded_file is not None:
    if st.sidebar.button("Procesar Archivo"):
        with st.spinner("Enviando y limpiando datos en el backend..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/vnd.ms-excel")}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.sidebar.success(data["mensaje"])
                    st.session_state['data_loaded'] = True
                    st.session_state['total_incidentes'] = data['total_incidentes']
                else:
                    st.sidebar.error(f"Error: {response.json().get('detail', 'Desconocido')}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("No se pudo conectar al Backend. Revisa que el contenedor FastAPI esté corriendo.")

if st.session_state.get('data_loaded', False):
    st.sidebar.markdown("---")
    
    st.sidebar.header("Descargar Datos Limpios")
    try:
        csv_clean_response = requests.get(f"{BACKEND_URL}/download-clean")
        if csv_clean_response.status_code == 200:
            st.sidebar.download_button(
                label="📥 Descargar CSV Limpio",
                data=csv_clean_response.content,
                file_name='incidentes_limpios.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.sidebar.error("No se pudo obtener el archivo limpio.")
        
    st.sidebar.markdown("---")
    st.sidebar.header("2. Entrenamiento y Proyección")
    horizonte = st.sidebar.slider("Días a proyectar (Horizonte)", min_value=7, max_value=90, value=30)
    
    if st.sidebar.button("Entrenar Modelos y Proyectar"):
        with st.spinner("Entrenando modelos de Machine Learning (Regresión Lineal, ES, Prophet, ARIMA, SARIMA, RF, XGBoost)..."):
            try:
                res = requests.post(f"{BACKEND_URL}/train", params={"horizonte": horizonte})
                if res.status_code == 200:
                    st.session_state['forecast_data'] = res.json()
                    st.sidebar.success("Modelos entrenados correctamente.")
                else:
                    error_msg = res.text
                    try:
                        error_msg = res.json().get('detail', res.text)
                    except Exception:
                        pass
                    st.sidebar.error(f"Error entrenando: {error_msg}")
            except Exception as e:
                st.sidebar.error(f"Error de conexión: {e}")

# --- ÁREA PRINCIPAL ---
if st.session_state.get('forecast_data'):
    st.header("Resultados de la Proyección")
    
    fdata = st.session_state['forecast_data']
    
    # 1. KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidentes Históricos", st.session_state.get('total_incidentes', 'N/A'))
    col2.metric("Mejor Modelo Encontrado", fdata['mejor_modelo'])
    col3.metric("Días Proyectados", fdata['horizonte_dias'])
    
    st.markdown("---")
    
    # 2. Gráfico Interactivo de Histórico + Predicción
    st.subheader(f"Gráfico de Tendencia Histórica y Proyección ({fdata['mejor_modelo']})")
    
    fechas_hist = pd.to_datetime(fdata['fechas_historicas'])
    valores_hist = fdata['valores_historicos']
    fechas_proy = pd.to_datetime(fdata['fechas_proyectadas'])
    valores_proy = fdata['valores_proyectados']
    
    fig = go.Figure()
    
    # Línea Histórica
    fig.add_trace(go.Scatter(x=fechas_hist, y=valores_hist,
                    mode='lines',
                    name='Histórico (Real)',
                    line=dict(color='#1f77b4', width=2)))
                    
    # Línea Predicción
    fig.add_trace(go.Scatter(x=fechas_proy, y=valores_proy,
                    mode='lines+markers',
                    name='Proyección (Futuro)',
                    line=dict(color='#ff7f0e', dash='dash', width=2)))
                    
    fig.update_layout(xaxis_title='Fecha',
                      yaxis_title='Cantidad de Incidentes por día',
                      template='plotly_white',
                      hovermode='x unified')
                      
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Datos Tabulares
    col_izq, col_der = st.columns(2)
    
    with col_izq:
        st.subheader("Evaluación de Modelos Entrenados")
        st.markdown("Comparativa de métricas: MAE, RMSE y R². El modelo con menor MAE es el seleccionado.")
        eval_df = pd.DataFrame(fdata['modelos_evaluados'])
        # Reordenar columnas para mejor lectura si existen
        if 'rmse' in eval_df.columns:
            eval_df = eval_df[['nombre', 'mae', 'rmse', 'r2']]
            st.dataframe(eval_df.style.highlight_min(subset=['mae', 'rmse'], color='lightgreen').highlight_max(subset=['r2'], color='lightgreen'), use_container_width=True)
        else:
            st.dataframe(eval_df.style.highlight_min(subset=['mae'], color='lightgreen'), use_container_width=True)
        
    with col_der:
        st.subheader("Planificacion de Personal (Capacidad)")
        
        #Parametro operativo_ Horas productivas por tecnico al dia
        horas_productivas = st.slider("Horas utiles por tecnico al dia (Sin refrigerio)", min_value=4.0, max_value=8.0,value=6.5, step=0.5)
        
        #Recuperar el tiempo promedio que mando el backend
        tiempo_prom_minutos = fdata.get('tiempo_promedio_historico', 45.0)
        st.markdown(f"**Tiempo medio historico por ticket:** {round(tiempo_prom_minutos, 1)} minutos")
        
        #Calcular capacidad necesaria
        df_capacidad = pd.DataFrame({
            'Fecha': fechas_proy.dt.strftime("%Y-%m-%d") if hasattr(fechas_proy, 'dt') else pd.to_datetime(fechas_proy).strftime("%Y-%m-%d"),
            'Tickets Estimados': valores_proy
        })
        
        # Formula: (Tickets * Minutos) / 60 = Horas necesarias. Luego Horas necesarias / Horas Productivas por persona
        df_capacidad['Carga (Horas)'] = (df_capacidad['Tickets Estimados'] * tiempo_prom_minutos) / 60
        # Redondeo hacia arriba para asegurar que siempre haya suficiente personal
        df_capacidad['Tecnicos Requeridos'] = (df_capacidad['Carga (Horas)'] / horas_productivas).apply(lambda x: max(1, int(x + 0.99)))
        
        # Formato visual
        df_capacidad['Carga (Horas)'] = df_capacidad['Carga (Horas)'].round(1)
        
        # Mostrar tabla coloreada (con fallback si falta matplotlib)
        try:
            st.dataframe(
                df_capacidad.style.background_gradient(subset=['Tecnicos Requeridos'], cmap='Oranges'),
                use_container_width=True
            )
        except Exception:
            st.dataframe(df_capacidad, use_container_width=True)
        
        # Boton de Descarga
        csv_capacidad = df_capacidad.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Plan de Personal CSV",
            data=csv_capacidad,
            file_name='plan_capacidad_soporte.csv',
            mime='text/csv',
        ) 
      #  st.subheader("Datos Proyectados (Futuro)")
       # df_proy = pd.DataFrame({
        #    'Fecha': fechas_proy.strftime("%Y-%m-%d"),
         #   'Incidentes Estimados': valores_proy
        #})
        #st.dataframe(df_proy, use_container_width=True)
        
        # Botón de Descarga
        #csv = df_proy.to_csv(index=False).encode('utf-8')
        #st.download_button(
         #   label="Descargar Proyección como CSV",
          #  data=csv,
           # file_name='proyeccion_incidentes.csv',
            #mime='text/csv',
        #)
else:
    # Pantalla de inicio si no hay datos
    st.info("👈 Comienza subiendo un archivo de Excel desde el panel lateral y procesa los datos.")
    
    # Explicación de los modelos
    st.markdown("""
    ### ¿Cómo funciona este sistema?
    1. **Limpieza de Datos:** El sistema lee tu archivo Excel, filtra unicamente los registros categorizados como "Incidentes" y los agrupa cronológicamente.
    2. **Entrenamiento de ML:** Se ejecutan simultáneamente distintos modelos (Regresión Lineal, Suavizado Exponencial, Prophet, ARIMA, SARIMA, Random Forest, XGBoost) en background para aprender de los patrones históricos.
    3. **Evaluación:** Se comparan usando el *Error Absoluto Medio (MAE)*, junto con el *RMSE* y *R²*, para identificar cuál representa mejor la realidad del negocio.
    4. **Proyección:** El modelo campeón predice los próximos *N* días para mostrar un pronóstico de volumen de incidencias.
    """)
