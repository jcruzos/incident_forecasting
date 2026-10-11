from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import io
import os
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from prophet import Prophet

app = FastAPI(title="Incident Forecaster API")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

processed_data_cache = None

# -------------------------------------------------------------
# Funciones Auxiliares de Feature Engineering y Calendario
# -------------------------------------------------------------

def obtener_feriados_peru(anios=[2025, 2026, 2027]) -> set:
    """Retorna un conjunto de fechas correspondientes a feriados nacionales de Perú."""
    feriados = {
        # Feriados fijos
        datetime.date(2025, 1, 1), datetime.date(2025, 5, 1), datetime.date(2025, 6, 29),
        datetime.date(2025, 7, 28), datetime.date(2025, 7, 29), datetime.date(2025, 8, 6),
        datetime.date(2025, 8, 30), datetime.date(2025, 10, 8), datetime.date(2025, 11, 1),
        datetime.date(2025, 12, 8), datetime.date(2025, 12, 9), datetime.date(2025, 12, 25),
        
        datetime.date(2026, 1, 1),
        # Semana Santa 2026 (Jueves y Viernes Santo)
        datetime.date(2026, 4, 2), datetime.date(2026, 4, 3),
        datetime.date(2026, 5, 1), datetime.date(2026, 6, 29),
        datetime.date(2026, 7, 28), datetime.date(2026, 7, 29), datetime.date(2026, 8, 6),
        datetime.date(2026, 8, 30), datetime.date(2026, 10, 8), datetime.date(2026, 11, 1),
        datetime.date(2026, 12, 8), datetime.date(2026, 12, 9), datetime.date(2026, 12, 25),
        
        datetime.date(2027, 1, 1), datetime.date(2027, 3, 25), datetime.date(2027, 3, 26),
        datetime.date(2027, 5, 1), datetime.date(2027, 7, 28), datetime.date(2027, 7, 29)
    }
    return feriados

def calcular_mape(y_true, y_pred) -> float:
    """Calcula el Error Porcentual Absoluto Medio (MAPE) protegiendo contra división por cero."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mascara = y_true > 0
    if not np.any(mascara):
        return 0.0
    return float(np.mean(np.abs((y_true[mascara] - y_pred[mascara]) / y_true[mascara])) * 100.0)

def construir_features_temporales(df_in: pd.DataFrame, target_col: str = "cantidad_incidentes") -> pd.DataFrame:
    """Genera variables de calendario, quincenas, estacionalidad armónica, rezagos (lags) y ventanas móviles."""
    df_feat = df_in.copy()
    feriados = obtener_feriados_peru()
    
    # Variables de calendario
    fechas = pd.to_datetime(df_feat['fecha'])
    df_feat['dia_semana'] = fechas.dt.dayofweek
    df_feat['es_fin_de_semana'] = (df_feat['dia_semana'] >= 5).astype(int)
    df_feat['dia_mes'] = fechas.dt.day
    df_feat['es_quincena'] = df_feat['dia_mes'].isin([14, 15, 16, 28, 29, 30, 31]).astype(int)
    df_feat['es_feriado'] = fechas.dt.date.apply(lambda d: 1 if d in feriados else 0)
    
    # Armónicos cíclicos semanales
    df_feat['sin_dia_semana'] = np.sin(2 * np.pi * df_feat['dia_semana'] / 7)
    df_feat['cos_dia_semana'] = np.cos(2 * np.pi * df_feat['dia_semana'] / 7)
    
    # Rezagos (Lags)
    df_feat['lag_1'] = df_feat[target_col].shift(1)
    df_feat['lag_2'] = df_feat[target_col].shift(2)
    df_feat['lag_7'] = df_feat[target_col].shift(7)
    df_feat['lag_14'] = df_feat[target_col].shift(14)
    
    # Ventanas móviles con shift(1) para evitar Data Leakage
    df_feat['rolling_mean_7'] = df_feat[target_col].shift(1).rolling(7, min_periods=1).mean()
    df_feat['rolling_std_7'] = df_feat[target_col].shift(1).rolling(7, min_periods=1).std().fillna(0)
    
    # Variables de arrastre operativo
    if 'tasa_cumplimiento_sla' in df_feat.columns:
        df_feat['lag_sla'] = df_feat['tasa_cumplimiento_sla'].shift(1).fillna(1.0)
    else:
        df_feat['lag_sla'] = 1.0
        
    if 'cantidad_pos' in df_feat.columns:
        df_feat['lag_pos'] = df_feat['cantidad_pos'].shift(1).fillna(0)
    else:
        df_feat['lag_pos'] = 0.0

    return df_feat

def predecir_recursivo_arbol(modelo, df_completo: pd.DataFrame, fechas_futuro: pd.DatetimeIndex, target_col: str, feature_cols: list) -> list:
    """Realiza una proyección autorregresiva recursiva multi-paso para modelos de Machine Learning (RF / XGBoost)."""
    df_sim = df_completo[['fecha', target_col, 'tasa_cumplimiento_sla', 'cantidad_pos']].copy() if 'tasa_cumplimiento_sla' in df_completo.columns and 'cantidad_pos' in df_completo.columns else df_completo[['fecha', target_col]].copy()
    
    if 'tasa_cumplimiento_sla' not in df_sim.columns:
        df_sim['tasa_cumplimiento_sla'] = 1.0
    if 'cantidad_pos' not in df_sim.columns:
        df_sim['cantidad_pos'] = 0.0
        
    predicciones = []
    
    for f in fechas_futuro:
        # Añadir temporalmente la fila futura con valor temporal nulo
        nueva_fila = pd.DataFrame([{
            'fecha': f,
            target_col: np.nan,
            'tasa_cumplimiento_sla': float(df_sim['tasa_cumplimiento_sla'].iloc[-7:].mean()),
            'cantidad_pos': float(df_sim['cantidad_pos'].iloc[-7:].mean())
        }])
        df_temp = pd.concat([df_sim, nueva_fila], ignore_index=True)
        
        # Calcular features en la fila más reciente
        df_features = construir_features_temporales(df_temp, target_col=target_col)
        fila_features = df_features.iloc[[-1]][feature_cols].fillna(0)
        
        # Inferencia
        pred_val = max(0, float(round(modelo.predict(fila_features)[0])))
        predicciones.append(int(pred_val))
        
        # Retroalimentar la serie para el siguiente lag
        df_sim.loc[df_sim.index[-1] + 1 if not df_sim.empty else 0] = {
            'fecha': f,
            target_col: pred_val,
            'tasa_cumplimiento_sla': float(df_sim['tasa_cumplimiento_sla'].iloc[-7:].mean()),
            'cantidad_pos': float(df_sim['cantidad_pos'].iloc[-7:].mean())
        }
        
    return predicciones


# -------------------------------------------------------------
# Endpoints de la API
# -------------------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global processed_data_cache
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # 1. Cargar matriz SLA buscando en posibles rutas del servidor
        posibles_rutas_sla = [
            os.path.join(os.path.dirname(__file__), "data", "sla_atencion_soporte.xlsx"),
            os.path.join("data", "sla_atencion_soporte.xlsx"),
            os.path.join(os.path.dirname(__file__), "data", "SLA DE ATENCION SOPORTE.xlsx"),
            os.path.join("data", "SLA DE ATENCION SOPORTE.xlsx"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "SLA DE ATENCION SOPORTE.xlsx"),
            os.path.join(".", "SLA DE ATENCION SOPORTE.xlsx"),
            os.path.join(".", "sla_atencion_soporte.xlsx")
        ]
        ruta_sla = next((p for p in posibles_rutas_sla if os.path.exists(p)), None)
        
        if not ruta_sla:
            raise HTTPException(status_code=500, detail="Falta el archivo maestro SLA en el servidor.")
        
        df_sla = pd.read_excel(ruta_sla)
        
        # Estandarizar nombres de columnas
        df.columns = df.columns.str.strip().str.upper()
        df_sla.columns = df_sla.columns.str.strip().str.upper()
        
        # 2. Cruce con Matriz SLA
        if 'CATEGORIA' in df.columns and 'CATEGORIA' in df_sla.columns:
            df = df.merge(df_sla[['CATEGORIA', 'SLA_MINUTOS']], on='CATEGORIA', how='left')
        else:
            df['SLA_MINUTOS'] = 120
            
        df['SLA_MINUTOS'] = df['SLA_MINUTOS'].fillna(120)
        
        # 3. Cálculo de tiempos reales y cumplimiento SLA
        if 'FECHAHORACREACION' in df.columns and 'FECHAHORACIERRE' in df.columns:
            df['FECHAHORACREACION'] = pd.to_datetime(df['FECHAHORACREACION'], errors='coerce')
            df['FECHAHORACIERRE'] = pd.to_datetime(df['FECHAHORACIERRE'], errors='coerce')
            
            df['MINUTOS_REALES'] = (df['FECHAHORACIERRE'] - df['FECHAHORACREACION']).dt.total_seconds() / 60
            df['CUMPLE_SLA'] = (df['MINUTOS_REALES'] <= df['SLA_MINUTOS']).astype(int)
        else:
            df['MINUTOS_REALES'] = 45.0
            df['CUMPLE_SLA'] = 1
            
        validos = df.loc[df['MINUTOS_REALES'] > 0, 'MINUTOS_REALES']
        tiempo_promedio = float(validos.mean()) if not validos.empty and not pd.isna(validos.mean()) else 45.0
        
        # 4. Agrupación Diaria Enriquecida
        if 'FECHAHORACREACION' in df.columns:
            df['FECHA_CORTA'] = df['FECHAHORACREACION'].dt.date
            
            # Normalizar columnas de tipo y servicio
            df['TIPO_STR'] = df['TIPO'].astype(str).str.upper() if 'TIPO' in df.columns else ""
            df['SERVICE_STR'] = df['TIPOSERVICE'].astype(str).str.upper() if 'TIPOSERVICE' in df.columns else ""
            df['DERIVADO_NUM'] = df['DERIVADO'].fillna(0).astype(int) if 'DERIVADO' in df.columns else 0
            
            df_agrupado = df.groupby('FECHA_CORTA').agg(
                cantidad_total=('TICKETID', 'count') if 'TICKETID' in df.columns else ('FECHA_CORTA', 'count'),
                cantidad_incidentes=('TIPO_STR', lambda x: (x == 'INCIDENTE').sum()),
                cantidad_solicitudes=('TIPO_STR', lambda x: (x == 'SOLICITUD').sum()),
                cantidad_pos=('SERVICE_STR', lambda x: (x.str.contains('POS|ERP', na=False)).sum()),
                cantidad_derivados=('DERIVADO_NUM', lambda x: (x == 1).sum()),
                tasa_cumplimiento_sla=('CUMPLE_SLA', 'mean'),
                minutos_promedio=('MINUTOS_REALES', lambda x: float(x[x > 0].mean()) if (x > 0).any() else 45.0)
            ).reset_index()
            
            df_agrupado = df_agrupado.sort_values('FECHA_CORTA')
            df_agrupado['FECHA_CORTA'] = pd.to_datetime(df_agrupado['FECHA_CORTA'])
            df_agrupado.set_index('FECHA_CORTA', inplace=True)
            
            # Reindex continuo para evitar huecos en la serie de tiempo
            idx = pd.date_range(df_agrupado.index.min(), df_agrupado.index.max())
            df_agrupado = df_agrupado.reindex(idx).fillna({
                'cantidad_total': 0,
                'cantidad_incidentes': 0,
                'cantidad_solicitudes': 0,
                'cantidad_pos': 0,
                'cantidad_derivados': 0,
                'tasa_cumplimiento_sla': 1.0,
                'minutos_promedio': tiempo_promedio
            })
            df_agrupado.reset_index(inplace=True)
            df_agrupado.rename(columns={'index': 'fecha'}, inplace=True)
            
            # Guardar tiempo promedio y datasets limpios
            with open("data/tiempo_promedio.txt", "w") as f:
                f.write(str(tiempo_promedio))
                
            df_agrupado.to_csv("data/incidentes_enriquecidos.csv", index=False)
            df_agrupado.to_csv("data/incidentes_limpios.csv", index=False)
            
            processed_data_cache = df_agrupado
            
            return {
                "mensaje": "Datos procesados y enriquecidos con éxito.",
                "total_tickets": int(df_agrupado['cantidad_total'].sum()),
                "total_incidentes": int(df_agrupado['cantidad_incidentes'].sum()),
                "total_solicitudes": int(df_agrupado['cantidad_solicitudes'].sum()),
                "total_pos": int(df_agrupado['cantidad_pos'].sum()),
                "tasa_cumplimiento_promedio": round(float(df_agrupado['tasa_cumplimiento_sla'].mean()) * 100, 1),
                "tiempo_promedio_historico": round(tiempo_promedio, 2),
                "dias_registrados": len(df_agrupado)
            }
        else:
            raise HTTPException(status_code=400, detail="No se encontró la columna de fecha de creación en el archivo.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")


@app.get("/download-clean")
async def download_clean():
    file_path = "data/incidentes_enriquecidos.csv"
    if not os.path.exists(file_path):
        file_path = "data/incidentes_limpios.csv"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/csv", filename="incidentes_enriquecidos.csv")
    raise HTTPException(status_code=404, detail="Archivo procesado no encontrado. Sube un archivo primero.")


@app.post("/train")
async def train_models(horizonte: int = 30, tipo_serie: str = "todos"):
    global processed_data_cache
    if processed_data_cache is None:
        try:
            ruta_datos = "data/incidentes_enriquecidos.csv" if os.path.exists("data/incidentes_enriquecidos.csv") else "data/incidentes_limpios.csv"
            processed_data_cache = pd.read_csv(ruta_datos)
            processed_data_cache['fecha'] = pd.to_datetime(processed_data_cache['fecha'])
        except Exception:
            raise HTTPException(status_code=400, detail="No hay datos procesados. Sube un archivo Excel primero.")
            
    df = processed_data_cache.copy()
    
    # Seleccionar la columna objetivo según el filtro solicitado
    mapeo_columnas = {
        "todos": "cantidad_total" if "cantidad_total" in df.columns else "cantidad_incidentes",
        "incidentes": "cantidad_incidentes",
        "solicitudes": "cantidad_solicitudes" if "cantidad_solicitudes" in df.columns else "cantidad_incidentes",
        "pos": "cantidad_pos" if "cantidad_pos" in df.columns else "cantidad_incidentes"
    }
    target_col = mapeo_columnas.get(tipo_serie.lower(), "cantidad_incidentes")
    if target_col not in df.columns:
        target_col = "cantidad_incidentes"
        
    # Feature Engineering con Lags, Rolling y Calendario
    df_feat = construir_features_temporales(df, target_col=target_col)
    
    # Columnas de características para los modelos tabulares
    feature_cols = [
        'dia_semana', 'es_fin_de_semana', 'dia_mes', 'es_quincena', 'es_feriado',
        'sin_dia_semana', 'cos_dia_semana', 'lag_1', 'lag_2', 'lag_7',
        'rolling_mean_7', 'rolling_std_7', 'lag_sla'
    ]
    
    # Remover los primeros registros afectados por los lags para un entrenamiento limpio
    df_modelos = df_feat.dropna(subset=['lag_7', 'rolling_mean_7']).copy().reset_index(drop=True)
    
    if len(df_modelos) < 15:
        # Fallback a menos variables si la muestra es pequeña
        feature_cols = ['dia_semana', 'es_fin_de_semana', 'dia_mes', 'es_quincena', 'sin_dia_semana', 'cos_dia_semana']
        df_modelos = df_feat.copy()

    # Split cronológico 80/20
    train_size = int(len(df_modelos) * 0.8)
    train, test = df_modelos.iloc[:train_size], df_modelos.iloc[train_size:]
    
    if len(train) < 5 or len(test) < 2:
        raise HTTPException(status_code=400, detail="No hay suficientes datos históricos para validar el entrenamiento.")
        
    resultados_modelos = []
    feature_importances_dict = {}

    # --- Modelo 1: Regresión Lineal (con Lags y Calendario) ---
    try:
        lr = LinearRegression()
        lr.fit(train[feature_cols], train[target_col])
        lr_preds = np.maximum(0, lr.predict(test[feature_cols]))
        
        resultados_modelos.append({
            "nombre": "Regresión Lineal",
            "mae": float(mean_absolute_error(test[target_col], lr_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], lr_preds))),
            "r2": float(r2_score(test[target_col], lr_preds)),
            "mape": float(calcular_mape(test[target_col], lr_preds)),
            "tipo": "linear"
        })
    except Exception as e:
        print("Error en Regresión Lineal:", e)

    # --- Modelo 2: Suavizado Exponencial (Holt-Winters con Tendencia Amortiguada) ---
    try:
        ts_train = train[target_col].values
        es = ExponentialSmoothing(ts_train, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True, initialization_method="estimated")
        es_fit = es.fit()
        es_preds = np.maximum(0, es_fit.forecast(len(test)))
        
        resultados_modelos.append({
            "nombre": "Suavizado Exponencial",
            "mae": float(mean_absolute_error(test[target_col], es_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], es_preds))),
            "r2": float(r2_score(test[target_col], es_preds)),
            "mape": float(calcular_mape(test[target_col], es_preds)),
            "tipo": "statsmodels"
        })
    except Exception as e:
        print("Error en Suavizado Exponencial:", e)

    # --- Modelo 3: Prophet con Regresores Exógenos ---
    try:
        df_prophet_train = train[['fecha', target_col, 'es_quincena', 'es_fin_de_semana', 'es_feriado']].copy()
        df_prophet_train.rename(columns={'fecha': 'ds', target_col: 'y'}, inplace=True)
        
        m_prophet = Prophet(
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.08
        )
        m_prophet.add_regressor('es_quincena')
        m_prophet.add_regressor('es_fin_de_semana')
        m_prophet.add_regressor('es_feriado')
        m_prophet.fit(df_prophet_train)
        
        df_prophet_test = test[['fecha', 'es_quincena', 'es_fin_de_semana', 'es_feriado']].copy()
        df_prophet_test.rename(columns={'fecha': 'ds'}, inplace=True)
        forecast_test = m_prophet.predict(df_prophet_test)
        prophet_preds = np.maximum(0, forecast_test['yhat'].values)
        
        resultados_modelos.append({
            "nombre": "Prophet",
            "mae": float(mean_absolute_error(test[target_col], prophet_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], prophet_preds))),
            "r2": float(r2_score(test[target_col], prophet_preds)),
            "mape": float(calcular_mape(test[target_col], prophet_preds)),
            "tipo": "prophet"
        })
    except Exception as e:
        print("Error en Prophet:", e)

    # --- Modelo 4: ARIMA (Búsqueda de Orden Óptimo por AIC) ---
    try:
        ts_train = train[target_col].values
        mejor_aic = float('inf')
        mejor_orden = (1, 1, 1)
        mejor_arima_fit = None
        
        for p in [1, 2]:
            for d in [0, 1]:
                for q in [0, 1]:
                    try:
                        m_cand = ARIMA(ts_train, order=(p, d, q))
                        res_cand = m_cand.fit()
                        if res_cand.aic < mejor_aic:
                            mejor_aic = res_cand.aic
                            mejor_orden = (p, d, q)
                            mejor_arima_fit = res_cand
                    except Exception:
                        continue
                        
        if mejor_arima_fit is None:
            mejor_arima_fit = ARIMA(ts_train, order=(1, 1, 1)).fit()
            
        arima_preds = np.maximum(0, mejor_arima_fit.forecast(steps=len(test)))
        resultados_modelos.append({
            "nombre": f"ARIMA {mejor_orden}",
            "mae": float(mean_absolute_error(test[target_col], arima_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], arima_preds))),
            "r2": float(r2_score(test[target_col], arima_preds)),
            "mape": float(calcular_mape(test[target_col], arima_preds)),
            "tipo": "arima",
            "orden": mejor_orden
        })
    except Exception as e:
        print("Error en ARIMA:", e)

    # --- Modelo 5: SARIMAX con Variables Exógenas ---
    try:
        ts_train = train[target_col].values
        exog_cols = ['es_fin_de_semana', 'es_quincena']
        exog_train = train[exog_cols].values
        exog_test = test[exog_cols].values
        
        sarima = SARIMAX(
            ts_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_fit = sarima.fit(disp=False)
        sarima_preds = np.maximum(0, sarima_fit.forecast(steps=len(test), exog=exog_test))
        
        resultados_modelos.append({
            "nombre": "SARIMA (Exógeno)",
            "mae": float(mean_absolute_error(test[target_col], sarima_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], sarima_preds))),
            "r2": float(r2_score(test[target_col], sarima_preds)),
            "mape": float(calcular_mape(test[target_col], sarima_preds)),
            "tipo": "sarima"
        })
    except Exception as e:
        print("Error en SARIMA:", e)

    # --- Modelo 6: Random Forest (con Lags y Regularización) ---
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42)
        rf.fit(train[feature_cols], train[target_col])
        rf_preds = np.maximum(0, rf.predict(test[feature_cols]))
        
        # Guardar importancia de variables
        rf_imp = dict(zip(feature_cols, [round(float(v) * 100, 2) for v in rf.feature_importances_]))
        feature_importances_dict["Random Forest"] = dict(sorted(rf_imp.items(), key=lambda x: x[1], reverse=True))
        
        resultados_modelos.append({
            "nombre": "Random Forest",
            "mae": float(mean_absolute_error(test[target_col], rf_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], rf_preds))),
            "r2": float(r2_score(test[target_col], rf_preds)),
            "mape": float(calcular_mape(test[target_col], rf_preds)),
            "tipo": "random_forest"
        })
    except Exception as e:
        print("Error en Random Forest:", e)

    # --- Modelo 7: XGBoost (con Regularización y Lags) ---
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(train[feature_cols], train[target_col])
        xgb_preds = np.maximum(0, xgb_model.predict(test[feature_cols]))
        
        xgb_imp = dict(zip(feature_cols, [round(float(v) * 100, 2) for v in xgb_model.feature_importances_]))
        feature_importances_dict["XGBoost"] = dict(sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True))
        
        resultados_modelos.append({
            "nombre": "XGBoost",
            "mae": float(mean_absolute_error(test[target_col], xgb_preds)),
            "rmse": float(np.sqrt(mean_squared_error(test[target_col], xgb_preds))),
            "r2": float(r2_score(test[target_col], xgb_preds)),
            "mape": float(calcular_mape(test[target_col], xgb_preds)),
            "tipo": "xgboost"
        })
    except Exception as e:
        print("Error en XGBoost:", e)

    if not resultados_modelos:
        raise HTTPException(status_code=500, detail="Fallo el entrenamiento de todos los modelos.")

    # Selección del mejor modelo (menor MAE)
    mejor_modelo_info = min(resultados_modelos, key=lambda x: x['mae'])
    
    # --- Proyección Futura a N días ---
    fechas_futuro = pd.date_range(start=df['fecha'].max() + pd.Timedelta(days=1), periods=horizonte)
    predicciones_futuro = []
    
    if mejor_modelo_info['tipo'] in ["random_forest", "xgboost", "linear"]:
        if mejor_modelo_info['tipo'] == "random_forest":
            modelo_final = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42)
        elif mejor_modelo_info['tipo'] == "xgboost":
            modelo_final = xgb.XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        else:
            modelo_final = LinearRegression()
            
        modelo_final.fit(df_modelos[feature_cols], df_modelos[target_col])
        predicciones_futuro = predecir_recursivo_arbol(modelo_final, df, fechas_futuro, target_col, feature_cols)
        
    elif mejor_modelo_info['tipo'] == "statsmodels":
        ts_data_all = df[target_col].values
        try:
            es_final = ExponentialSmoothing(ts_data_all, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True, initialization_method="estimated")
            es_final_fit = es_final.fit()
        except Exception:
            es_final = ExponentialSmoothing(ts_data_all, trend='add', seasonal=None, damped_trend=True, initialization_method="estimated")
            es_final_fit = es_final.fit()
        preds = es_final_fit.forecast(horizonte)
        predicciones_futuro = [max(0, int(round(p))) for p in preds]
        
    elif mejor_modelo_info['tipo'] == "prophet":
        df_prophet_all = construir_features_temporales(df, target_col=target_col)
        df_prophet_all = df_prophet_all[['fecha', target_col, 'es_quincena', 'es_fin_de_semana', 'es_feriado']].copy()
        df_prophet_all.rename(columns={'fecha': 'ds', target_col: 'y'}, inplace=True)
        
        m_final = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.08)
        m_final.add_regressor('es_quincena')
        m_final.add_regressor('es_fin_de_semana')
        m_final.add_regressor('es_feriado')
        m_final.fit(df_prophet_all)
        
        feriados = obtener_feriados_peru()
        df_futuro_prophet = pd.DataFrame({'ds': fechas_futuro})
        df_futuro_prophet['dia_semana'] = df_futuro_prophet['ds'].dt.dayofweek
        df_futuro_prophet['es_fin_de_semana'] = (df_futuro_prophet['dia_semana'] >= 5).astype(int)
        df_futuro_prophet['dia_mes'] = df_futuro_prophet['ds'].dt.day
        df_futuro_prophet['es_quincena'] = df_futuro_prophet['dia_mes'].isin([14, 15, 16, 28, 29, 30, 31]).astype(int)
        df_futuro_prophet['es_feriado'] = df_futuro_prophet['ds'].dt.date.apply(lambda d: 1 if d in feriados else 0)
        
        preds_df = m_final.predict(df_futuro_prophet)
        predicciones_futuro = [max(0, int(round(p))) for p in preds_df['yhat']]
        
    elif mejor_modelo_info['tipo'] == "arima":
        ts_data_all = df[target_col].values
        orden = mejor_modelo_info.get('orden', (1, 1, 1))
        arima_final = ARIMA(ts_data_all, order=orden)
        arima_final_fit = arima_final.fit()
        preds = arima_final_fit.forecast(steps=horizonte)
        predicciones_futuro = [max(0, int(round(p))) for p in preds]
        
    elif mejor_modelo_info['tipo'] == "sarima":
        ts_data_all = df[target_col].values
        df_feat_all = construir_features_temporales(df, target_col=target_col)
        exog_cols = ['es_fin_de_semana', 'es_quincena']
        exog_all = df_feat_all[exog_cols].values
        
        df_futuro_feat = pd.DataFrame({'fecha': fechas_futuro})
        df_futuro_feat = construir_features_temporales(df_futuro_feat, target_col=target_col)
        exog_futuro = df_futuro_feat[exog_cols].values
        
        sarima_final = SARIMAX(ts_data_all, exog=exog_all, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), enforce_stationarity=False, enforce_invertibility=False)
        sarima_final_fit = sarima_final.fit(disp=False)
        preds = sarima_final_fit.forecast(steps=horizonte, exog=exog_futuro)
        predicciones_futuro = [max(0, int(round(p))) for p in preds]

    # Tiempo promedio histórico de resolución
    tiempo_prom_historico = 45.0
    for p in ["data/tiempo_promedio.txt", os.path.join(os.path.dirname(__file__), "data", "tiempo_promedio.txt")]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    tiempo_prom_historico = float(f.read().strip())
                break
            except Exception:
                pass

    return {
        "modelos_evaluados": [
            {
                "nombre": m['nombre'],
                "mae": round(m['mae'], 2),
                "rmse": round(m['rmse'], 2),
                "r2": round(m['r2'], 4),
                "mape": round(m['mape'], 2)
            }
            for m in resultados_modelos
        ],
        "mejor_modelo": mejor_modelo_info['nombre'],
        "horizonte_dias": horizonte,
        "tipo_serie": tipo_serie,
        "importancia_variables": feature_importances_dict.get(mejor_modelo_info['nombre'], feature_importances_dict.get("XGBoost", feature_importances_dict.get("Random Forest", {}))),
        "fechas_historicas": df['fecha'].dt.strftime("%Y-%m-%d").tolist(),
        "valores_historicos": df[target_col].tolist(),
        "fechas_proyectadas": [f.strftime("%Y-%m-%d") for f in fechas_futuro],
        "valores_proyectados": predicciones_futuro,
        "tiempo_promedio_historico": round(tiempo_prom_historico, 2)
    }
