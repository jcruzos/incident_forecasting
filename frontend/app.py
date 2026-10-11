import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# Configuración y conexión
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Dashboard Proyección de Incidentes", layout="wide")
st.title("📈 Sistema Avanzado de Análisis y Proyección de Incidentes")
st.markdown("Carga tu archivo Excel de tickets, enriquece los datos automáticamente con variables de negocio y entrena modelos predictivos competitivos.")

# --- BARRA LATERAL ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube el archivo TICKETS.xls", type=["xls", "xlsx"])

if uploaded_file is not None:
    if st.sidebar.button("Procesar Archivo"):
        with st.spinner("Enviando, cruzando con matriz SLA y generando métricas operativas..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/vnd.ms-excel")}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.sidebar.success(data["mensaje"])
                    st.session_state['data_loaded'] = True
                    st.session_state['upload_metrics'] = data
                else:
                    st.sidebar.error(f"Error: {response.json().get('detail', 'Desconocido')}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("No se pudo conectar al Backend. Revisa que el contenedor FastAPI esté corriendo.")

if st.session_state.get('data_loaded', False):
    metrics = st.session_state.get('upload_metrics', {})
    
    st.sidebar.markdown("---")
    st.sidebar.header("Resumen del Dataset Cargado")
    st.sidebar.write(f"• **Total Tickets:** {metrics.get('total_tickets', 'N/A')}")
    st.sidebar.write(f"• **Incidentes:** {metrics.get('total_incidentes', 'N/A')}")
    st.sidebar.write(f"• **Solicitudes:** {metrics.get('total_solicitudes', 'N/A')}")
    st.sidebar.write(f"• **Soporte POS / Cajas:** {metrics.get('total_pos', 'N/A')}")
    st.sidebar.write(f"• **Cumplimiento SLA Promedio:** {metrics.get('tasa_cumplimiento_promedio', 'N/A')}%")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Descargar Datos Limpios")
    try:
        csv_clean_response = requests.get(f"{BACKEND_URL}/download-clean")
        if csv_clean_response.status_code == 200:
            st.sidebar.download_button(
                label="📥 Descargar CSV Enriquecido",
                data=csv_clean_response.content,
                file_name='incidentes_enriquecidos.csv',
                mime='text/csv'
            )
    except Exception:
        st.sidebar.error("No se pudo obtener el archivo limpio.")
        
    st.sidebar.markdown("---")
    st.sidebar.header("2. Entrenamiento y Proyección")
    
    opciones_series = {
        "Total de Tickets (Global)": "todos",
        "Solo Incidentes Críticos": "incidentes",
        "Solo Solicitudes": "solicitudes",
        "Solo Soporte ERP - POS": "pos"
    }
    serie_seleccionada_label = st.sidebar.selectbox("Selecciona qué deseas proyectar:", list(opciones_series.keys()))
    tipo_serie_val = opciones_series[serie_seleccionada_label]
    
    horizonte = st.sidebar.slider("Días a proyectar (Horizonte)", min_value=7, max_value=90, value=30)
    
    if st.sidebar.button("Entrenar Modelos y Proyectar"):
        with st.spinner(f"Entrenando y proyectando todos los modelos para '{serie_seleccionada_label}'..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/train",
                    params={"horizonte": horizonte, "tipo_serie": tipo_serie_val}
                )
                if res.status_code == 200:
                    st.session_state['forecast_data'] = res.json()
                    st.session_state['serie_label'] = serie_seleccionada_label
                    st.sidebar.success("¡Modelos entrenados y proyectados con éxito!")
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
    fdata = st.session_state['forecast_data']
    serie_label = st.session_state.get('serie_label', 'Proyección de Incidentes')
    
    st.header(f"Resultados de la Proyección: {serie_label}")
    
    # -------------------------------------------------------------
    # Selector de Modelo por el Usuario
    # -------------------------------------------------------------
    predicciones_dict = fdata.get('predicciones_por_modelo', {})
    mejor_modelo_rec = fdata.get('mejor_modelo', 'Desconocido')
    
    lista_nombres_modelos = list(predicciones_dict.keys()) if predicciones_dict else [mejor_modelo_rec]
    
    # Construir opciones poniendo el recomendado al inicio
    opciones_selector = [f"🏆 {mejor_modelo_rec} (Recomendado por menor error)"] + [m for m in lista_nombres_modelos if m != mejor_modelo_rec]
    
    col_sel_1, col_sel_2 = st.columns([2, 2])
    with col_sel_1:
        modelo_seleccionado_str = st.selectbox(
            "🎯 Selecciona el Modelo para la Proyección:",
            opciones_selector,
            index=0
        )
        
    modelo_activo = mejor_modelo_rec if "(Recomendado" in modelo_seleccionado_str else modelo_seleccionado_str
    valores_proy_activo = predicciones_dict.get(modelo_activo, fdata.get('valores_proyectados', []))
    
    with col_sel_2:
        modelos_para_comparar = [m for m in lista_nombres_modelos if m != modelo_activo]
        modelos_overlay = st.multiselect(
            "👁️ Superponer otros modelos para comparar curvas:",
            modelos_para_comparar
        )

    # 1. Tarjetas de KPIs Reactivas
    col1, col2, col3, col4 = st.columns(4)
    total_hist = sum(fdata['valores_historicos'])
    col1.metric("Volumen Histórico Total", f"{total_hist:,}")
    col2.metric("Modelo Activo en Gráfico", modelo_activo)
    col3.metric("Días Proyectados", f"{fdata['horizonte_dias']} días")
    total_proy = sum(valores_proy_activo)
    col4.metric("Volumen Proyectado Acumulado", f"{total_proy:,}")
    
    st.markdown("---")
    
    # 2. Gráfico Interactivo de Histórico + Predicción
    st.subheader(f"Tendencia Histórica y Proyección ({modelo_activo})")
    
    fechas_hist = pd.to_datetime(fdata['fechas_historicas'])
    valores_hist = fdata['valores_historicos']
    fechas_proy = pd.to_datetime(fdata['fechas_proyectadas'])
    
    fig = go.Figure()
    
    # Línea Histórica
    fig.add_trace(go.Scatter(
        x=fechas_hist,
        y=valores_hist,
        mode='lines',
        name='Histórico Real',
        line=dict(color='#1f77b4', width=2)
    ))
                     
    # Línea Predicción Futura del Modelo Seleccionado
    fig.add_trace(go.Scatter(
        x=fechas_proy,
        y=valores_proy_activo,
        mode='lines+markers',
        name=f"Proyección: {modelo_activo}",
        line=dict(color='#ff7f0e', dash='dash', width=2.5),
        marker=dict(size=6, symbol='circle')
    ))
    
    # Líneas Superpuestas para Comparar
    paleta_colores = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for idx_color, mod_comp in enumerate(modelos_overlay):
        if mod_comp in predicciones_dict:
            c = paleta_colores[idx_color % len(paleta_colores)]
            fig.add_trace(go.Scatter(
                x=fechas_proy,
                y=predicciones_dict[mod_comp],
                mode='lines+markers',
                name=f"Comparativa: {mod_comp}",
                line=dict(color=c, dash='dot', width=1.5),
                marker=dict(size=4, symbol='diamond')
            ))
                     
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Tickets Diarios',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Datos Tabulares: Evaluación Competitiva y Capacidad Operativa
    col_izq, col_der = st.columns([1, 1])
    
    with col_izq:
        st.subheader("📊 Evaluación Comparativa de Modelos")
        st.markdown("**Métricas calculadas:** MAE, RMSE, R² y MAPE (%). Los modelos con estacionalidad capturan los picos y valles semanales.")
        
        eval_df = pd.DataFrame(fdata['modelos_evaluados'])
        cols_mostrar = [c for c in ['nombre', 'mae', 'rmse', 'r2', 'mape'] if c in eval_df.columns]
        eval_df = eval_df[cols_mostrar]
        
        # Resaltado condicional
        styled_df = eval_df.style.highlight_min(subset=[c for c in ['mae', 'rmse', 'mape'] if c in eval_df.columns], color='lightgreen')
        if 'r2' in eval_df.columns:
            styled_df = styled_df.highlight_max(subset=['r2'], color='lightgreen')
            
        st.dataframe(styled_df, use_container_width=True)
        
        st.info(f"💡 **Modelo Activo Seleccionado:** `{modelo_activo}`. Si deseas probar cómo cambia el pronóstico, puedes cambiar el selector arriba.")
        
    with col_der:
        st.subheader(f"👥 Planificación de Capacidad ({modelo_activo})")
        
        horas_productivas = st.slider(
            "Horas útiles por técnico al día (Jornada neta sin refrigerio)",
            min_value=4.0, max_value=8.0, value=6.5, step=0.5
        )
        
        tiempo_prom_minutos = fdata.get('tiempo_promedio_historico', 45.0)
        st.markdown(f"**Tiempo medio por ticket:** `{round(tiempo_prom_minutos, 1)}` min | **Tickets proyectados:** `{sum(valores_proy_activo):,}`")
        
        df_capacidad = pd.DataFrame({
            'Fecha': fechas_proy.dt.strftime("%Y-%m-%d") if hasattr(fechas_proy, 'dt') else pd.to_datetime(fechas_proy).strftime("%Y-%m-%d"),
            'Tickets Estimados': valores_proy_activo
        })
        
        df_capacidad['Carga (Horas)'] = ((df_capacidad['Tickets Estimados'] * tiempo_prom_minutos) / 60).round(1)
        df_capacidad['Técnicos Requeridos'] = (df_capacidad['Carga (Horas)'] / horas_productivas).apply(lambda x: max(1, int(x + 0.99)))
        
        try:
            st.dataframe(
                df_capacidad.style.background_gradient(subset=['Técnicos Requeridos'], cmap='Oranges'),
                use_container_width=True
            )
        except Exception:
            st.dataframe(df_capacidad, use_container_width=True)
            
        csv_capacidad = df_capacidad.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Descargar Plan de Personal ({modelo_activo}) CSV",
            data=csv_capacidad,
            file_name=f'plan_personal_{modelo_activo.replace(" ", "_")}.csv',
            mime='text/csv'
        )

    # 4. Explicabilidad: Importancia de Variables (Feature Importance)
    imp_dict = fdata.get('importancia_variables', {})
    if imp_dict:
        st.markdown("---")
        st.subheader("🧠 Explicabilidad del Modelo: Importancia de Variables (Feature Importance)")
        st.markdown("Muestra qué variables de calendario, rezago o negocio tienen mayor peso en las decisiones del modelo.")
        
        df_imp = pd.DataFrame(list(imp_dict.items()), columns=['Variable', 'Importancia (%)'])
        df_imp = df_imp.sort_values('Importancia (%)', ascending=True)
        
        nombres_amigables = {
            'lag_1': 'Incidentes de ayer (Lag 1)',
            'lag_2': 'Incidentes hace 2 días (Lag 2)',
            'lag_7': 'Mismo día semana anterior (Lag 7)',
            'rolling_mean_7': 'Tendencia promedio 7 días',
            'rolling_std_7': 'Volatilidad últimos 7 días',
            'es_fin_de_semana': 'Fin de semana (Sáb/Dom)',
            'es_quincena': 'Días de Quincena (15/30)',
            'es_feriado': 'Día Feriado Oficial',
            'dia_semana': 'Día de la semana',
            'dia_mes': 'Día del mes',
            'sin_dia_semana': 'Ciclo semanal armónico (Seno)',
            'cos_dia_semana': 'Ciclo semanal armónico (Coseno)',
            'lag_sla': 'Cumplimiento SLA previo'
        }
        df_imp['Variable_Descriptiva'] = df_imp['Variable'].map(lambda x: nombres_amigables.get(x, x))
        
        fig_imp = px.bar(
            df_imp,
            x='Importancia (%)',
            y='Variable_Descriptiva',
            orientation='h',
            title=f"Peso Relativo de Características en el Modelo",
            color='Importancia (%)',
            color_continuous_scale='Blues'
        )
        fig_imp.update_layout(yaxis_title='', template='plotly_white')
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("👈 Comienza subiendo un archivo Excel (`TICKETS.xls`) desde el panel lateral y haz clic en **Procesar Archivo**.")
    
    st.markdown("""
    ### 🚀 Novedades del Motor Predictivo:
    1. **Selección Manual y Comparación Multi-Modelo:** Puedes elegir ver cualquier modelo entrenado y superponer varios en el gráfico simultáneamente.
    2. **Modelos con Ciclo Semanal Real:** Todos los modelos (SARIMA, Prophet, Random Forest, XGBoost, Holt-Winters) generan oscilaciones naturales acordes a la operación real.
    3. **Feature Engineering Avanzado:** Lags (`lag_1`, `lag_7`), tendencias móviles (`rolling_mean_7`), quincenas y feriados.
    4. **Planificación de Capacidad Dinámica:** Calcula el personal técnico necesario basándose en el modelo que tú decidas activar.
    """)
