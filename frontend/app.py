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

# Auto-detectar si ya existen datos procesados en el backend
if 'data_loaded' not in st.session_state:
    try:
        chk = requests.get(f"{BACKEND_URL}/download-clean", timeout=2)
        if chk.status_code == 200:
            st.session_state['data_loaded'] = True
    except Exception:
        pass

# --- ÁREA PRINCIPAL CON PESTAÑAS ---
tab_proyeccion, tab_evaluacion_mayo = st.tabs([
    "📈 Proyección y Capacidad",
    "🎯 Validación con Data Real de Mayo (TICKETS_MAYO.xlsx)"
])

with tab_proyeccion:
    if st.session_state.get('forecast_data'):
        fdata = st.session_state['forecast_data']
        serie_label = st.session_state.get('serie_label', 'Proyección de Incidentes')
        
        st.header(f"Resultados de la Proyección: {serie_label}")
        
        predicciones_dict = fdata.get('predicciones_por_modelo', {})
        mejor_modelo_rec = fdata.get('mejor_modelo', 'Desconocido')
        lista_nombres_modelos = list(predicciones_dict.keys()) if predicciones_dict else [mejor_modelo_rec]
        
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

        col1, col2, col3, col4 = st.columns(4)
        total_hist = sum(fdata['valores_historicos'])
        col1.metric("Volumen Histórico Total", f"{total_hist:,}")
        col2.metric("Modelo Activo en Gráfico", modelo_activo)
        col3.metric("Días Proyectados", f"{fdata['horizonte_dias']} días")
        total_proy = sum(valores_proy_activo)
        col4.metric("Volumen Proyectado Acumulado", f"{total_proy:,}")
        
        st.markdown("---")
        
        st.subheader(f"Tendencia Histórica y Proyección ({modelo_activo})")
        fechas_hist = pd.to_datetime(fdata['fechas_historicas'])
        valores_hist = fdata['valores_historicos']
        fechas_proy = pd.to_datetime(fdata['fechas_proyectadas'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fechas_hist, y=valores_hist, mode='lines', name='Histórico Real',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fechas_proy, y=valores_proy_activo, mode='lines+markers',
            name=f"Proyección: {modelo_activo}",
            line=dict(color='#ff7f0e', dash='dash', width=2.5),
            marker=dict(size=6, symbol='circle')
        ))
        
        paleta_colores = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for idx_color, mod_comp in enumerate(modelos_overlay):
            if mod_comp in predicciones_dict:
                c = paleta_colores[idx_color % len(paleta_colores)]
                fig.add_trace(go.Scatter(
                    x=fechas_proy, y=predicciones_dict[mod_comp], mode='lines+markers',
                    name=f"Comparativa: {mod_comp}",
                    line=dict(color=c, dash='dot', width=1.5),
                    marker=dict(size=4, symbol='diamond')
                ))
                         
        fig.update_layout(
            xaxis_title='Fecha', yaxis_title='Tickets Diarios', template='plotly_white',
            hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col_izq, col_der = st.columns([1, 1])
        with col_izq:
            st.subheader("📊 Evaluación Comparativa de Modelos (Cross-Validation)")
            eval_df = pd.DataFrame(fdata['modelos_evaluados'])
            cols_mostrar = [c for c in ['nombre', 'mae', 'rmse', 'r2', 'mape'] if c in eval_df.columns]
            eval_df = eval_df[cols_mostrar]
            styled_df = eval_df.style.highlight_min(subset=[c for c in ['mae', 'rmse', 'mape'] if c in eval_df.columns], color='lightgreen')
            if 'r2' in eval_df.columns:
                styled_df = styled_df.highlight_max(subset=['r2'], color='lightgreen')
            st.dataframe(styled_df, use_container_width=True)
            st.info(f"💡 **Modelo Activo Seleccionado:** `{modelo_activo}`.")
            
        with col_der:
            st.subheader(f"👥 Planificación de Capacidad ({modelo_activo})")
            horas_productivas = st.slider("Horas útiles por técnico al día (Jornada neta)", 4.0, 8.0, 6.5, 0.5)
            tiempo_prom_minutos = fdata.get('tiempo_promedio_historico', 45.0)
            st.markdown(f"**Tiempo medio por ticket:** `{round(tiempo_prom_minutos, 1)}` min | **Tickets proyectados:** `{sum(valores_proy_activo):,}`")
            
            df_capacidad = pd.DataFrame({
                'Fecha': fechas_proy.dt.strftime("%Y-%m-%d") if hasattr(fechas_proy, 'dt') else pd.to_datetime(fechas_proy).strftime("%Y-%m-%d"),
                'Tickets Estimados': valores_proy_activo
            })
            df_capacidad['Carga (Horas)'] = ((df_capacidad['Tickets Estimados'] * tiempo_prom_minutos) / 60).round(1)
            df_capacidad['Técnicos Requeridos'] = (df_capacidad['Carga (Horas)'] / horas_productivas).apply(lambda x: max(1, int(x + 0.99)))
            
            try:
                st.dataframe(df_capacidad.style.background_gradient(subset=['Técnicos Requeridos'], cmap='Oranges'), use_container_width=True)
            except Exception:
                st.dataframe(df_capacidad, use_container_width=True)
                
            csv_capacidad = df_capacidad.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"📥 Descargar Plan de Personal ({modelo_activo}) CSV", data=csv_capacidad, file_name=f'plan_personal_{modelo_activo.replace(" ", "_")}.csv', mime='text/csv')

        imp_dict = fdata.get('importancia_variables', {})
        if imp_dict:
            st.markdown("---")
            st.subheader("🧠 Explicabilidad del Modelo: Importancia de Variables (Feature Importance)")
            df_imp = pd.DataFrame(list(imp_dict.items()), columns=['Variable', 'Importancia (%)']).sort_values('Importancia (%)', ascending=True)
            nombres_amigables = {
                'lag_1': 'Incidentes de ayer (Lag 1)', 'lag_2': 'Incidentes hace 2 días (Lag 2)',
                'lag_7': 'Mismo día semana anterior (Lag 7)', 'rolling_mean_7': 'Tendencia promedio 7 días',
                'rolling_std_7': 'Volatilidad últimos 7 días', 'es_fin_de_semana': 'Fin de semana (Sáb/Dom)',
                'es_quincena': 'Días de Quincena (15/30)', 'es_feriado': 'Día Feriado Oficial',
                'dia_semana': 'Día de la semana', 'dia_mes': 'Día del mes',
                'sin_dia_semana': 'Ciclo semanal armónico (Seno)', 'cos_dia_semana': 'Ciclo semanal armónico (Coseno)',
                'lag_sla': 'Cumplimiento SLA previo'
            }
            df_imp['Variable_Descriptiva'] = df_imp['Variable'].map(lambda x: nombres_amigables.get(x, x))
            fig_imp = px.bar(df_imp, x='Importancia (%)', y='Variable_Descriptiva', orientation='h', title="Peso Relativo de Características en el Modelo", color='Importancia (%)', color_continuous_scale='Blues')
            fig_imp.update_layout(yaxis_title='', template='plotly_white')
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("👈 Selecciona en la barra lateral el tipo de ticket y haz clic en **Entrenar Modelos y Proyectar**.")
        st.markdown("""
        ### 🚀 Capacidades del Sistema:
        - **Selección de Modelos:** Elige y superpone cualquier modelo (Random Forest, XGBoost, Prophet, SARIMA, Suavizado Exponencial, Regresión Lineal).
        - **Planificación de Cuadrillas:** Dimensionamiento operativo según horas productivas y SLA.
        - **Validación con Data Real:** Ve a la pestaña **🎯 Validación con Data Real de Mayo** para contrastar con los datos ejecutados.
        """)

# =========================================================================
# PESTAÑA 2: EVALUACIÓN Y VALIDACIÓN CON DATOS REALES DE MAYO (TICKETS_MAYO)
# =========================================================================
with tab_evaluacion_mayo:
    st.header("🎯 Validación de Exactitud y Precisión con Datos Reales (Mayo 2026)")
    st.markdown("""
    Esta sección realiza una **auditoría y contraste empírico** contra la data real del archivo **`TICKETS_MAYO.xlsx`** (1 al 31 de mayo de 2026).
    Permite comprobar la exactitud diaria (MAE, RMSE, MAPE) y la precisión volumétrica global (WAPE y % de error acumulado).
    """)
    
    col_e1, col_e2 = st.columns([3, 1])
    with col_e1:
        opciones_eval = {
            "Total de Tickets (Global)": "todos",
            "Solo Incidentes Críticos": "incidentes",
            "Solo Solicitudes de Servicio": "solicitudes",
            "Solo Soporte ERP - POS / Cajas": "pos"
        }
        serie_eval_label = st.selectbox("Selecciona la serie a contrastar con Mayo Real:", list(opciones_eval.keys()), key="eval_serie_sel")
        tipo_serie_eval = opciones_eval[serie_eval_label]
        
    with col_e2:
        st.write("")
        st.write("")
        btn_evaluar = st.button("🧪 Ejecutar Evaluación con Mayo Real", use_container_width=True)
        
    if btn_evaluar or 'datos_mayo_eval' in st.session_state:
        if btn_evaluar or st.session_state.get('ultima_serie_eval') != tipo_serie_eval:
            with st.spinner("Cargando TICKETS_MAYO.xlsx y auditando proyecciones de todos los modelos..."):
                try:
                    res_eval = requests.get(f"{BACKEND_URL}/evaluar-mayo", params={"tipo_serie": tipo_serie_eval})
                    if res_eval.status_code == 200:
                        st.session_state['datos_mayo_eval'] = res_eval.json()
                        st.session_state['ultima_serie_eval'] = tipo_serie_eval
                    else:
                        st.error(f"Error en backend: {res_eval.text}")
                except Exception as ex:
                    st.error(f"Error de conexión con el backend: {ex}")
                    
        if st.session_state.get('datos_mayo_eval'):
            deval = st.session_state['datos_mayo_eval']
            resumen = deval['resumen_real']
            ranking = deval['ranking_modelos']
            mejor_diario = deval['mejor_modelo_exactitud_diaria']
            mejor_vol = deval['mejor_modelo_volumen_mensual']
            
            # Buscar métricas del mejor modelo
            info_mejor_diario = next((m for m in ranking if m['modelo'] == mejor_diario), ranking[0])
            info_mejor_vol = next((m for m in ranking if m['modelo'] == mejor_vol), ranking[0])
            
            # 1. Tarjetas de KPIs
            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tickets Reales en Mayo", f"{resumen['total_tickets_mayo']:,}", f"{resumen['promedio_diario']} tickets/día")
            k2.metric("Mejor Modelo Diario (MAE)", mejor_diario, f"MAE: {info_mejor_diario['mae']} tickets")
            k3.metric("Mejor Modelo Volumétrico", mejor_vol, f"Desvío: {info_mejor_vol['error_volumen_pct']:+}%")
            k4.metric("Exactitud Global WAPE", f"{info_mejor_diario['exactitud_wape']}%", "100 - WAPE")
            
            # 2. Gráfico Interactivo de Comparación Real vs Modelos
            st.markdown("---")
            st.subheader(f"Curva de Realidad vs Proyección: {serie_eval_label}")
            
            fechas_eval = pd.to_datetime(deval['fechas'])
            valores_reales = deval['valores_reales']
            
            todos_modelos_nom = [m['modelo'] for m in ranking]
            modelos_graficar = st.multiselect(
                "Selecciona qué modelos proyectados deseas visualizar junto a la data Real:",
                todos_modelos_nom,
                default=[mejor_diario] if mejor_diario in todos_modelos_nom else [todos_modelos_nom[0]]
            )
            
            fig_comp = go.Figure()
            # Línea Real Mayo en azul marino oscuro y sólido con marcadores
            fig_comp.add_trace(go.Scatter(
                x=fechas_eval,
                y=valores_reales,
                mode='lines+markers',
                name='⭐ REAL MAYO (TICKETS_MAYO.xlsx)',
                line=dict(color='#0d47a1', width=3.5),
                marker=dict(size=8, symbol='circle', color='#0d47a1')
            ))
            
            paleta_comp = ['#e65100', '#2e7d32', '#6a1b9a', '#c2185b', '#00838f', '#4e342e']
            for idx_m, mod_nom in enumerate(modelos_graficar):
                mod_data = next((m for m in ranking if m['modelo'] == mod_nom), None)
                if mod_data:
                    c = paleta_comp[idx_m % len(paleta_comp)]
                    fig_comp.add_trace(go.Scatter(
                        x=fechas_eval,
                        y=mod_data['predicciones'],
                        mode='lines+markers',
                        name=f"Pronóstico: {mod_nom}",
                        line=dict(color=c, dash='dash', width=2),
                        marker=dict(size=5, symbol='diamond')
                    ))
                    
            fig_comp.update_layout(
                xaxis_title='Fecha (Mayo 2026)',
                yaxis_title='Tickets por Día',
                template='plotly_white',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # 3. Tabla Ranking de Precisión
            st.markdown("---")
            st.subheader("📋 Tabla Comparativa de Exactitud y Precisión de Modelos")
            st.markdown("""
            - **MAE (Mean Absolute Error):** Error promedio de tickets equivocados por día (menor es mejor).
            - **RMSE:** Penaliza fuertemente errores aislados grandes (menor es mejor).
            - **Exactitud WAPE (%):** Precisión global ponderada sobre el volumen total ($100 - WAPE$).
            - **Error Volumen (%):** Desviación total entre la suma mensual proyectada y el real de mayo.
            """)
            
            df_ranking_view = pd.DataFrame([{
                'Modelo': r['modelo'],
                'Proyección Total': r['total_proyectado'],
                'Diferencia (Tickets)': f"{r['dif_tickets']:+d}",
                'Error Volumen (%)': f"{r['error_volumen_pct']:+.1f}%",
                'MAE (Tickets/Día)': r['mae'],
                'RMSE': r['rmse'],
                'MAPE (%)': f"{r['mape']:.1f}%",
                'Exactitud WAPE (%)': f"{r['exactitud_wape']:.1f}%",
                'R²': r['r2']
            } for r in ranking])
            
            st.dataframe(df_ranking_view, use_container_width=True)
            
            # 4. Desglose Diario Detallado
            with st.expander("🔍 Ver Desglose Diario Día por Día (Mayo 2026)"):
                df_detalle = pd.DataFrame(deval['detalle_diario'])
                st.dataframe(df_detalle, use_container_width=True)
                
            # 5. Descarga de Reporte
            csv_eval = df_ranking_view.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Descargar Reporte de Precisión Mayo ({serie_eval_label}) CSV",
                data=csv_eval,
                file_name=f'evaluacion_mayo_{tipo_serie_eval}.csv',
                mime='text/csv'
            )

