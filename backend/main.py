from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from prophet import Prophet

app = FastAPI(title="Incident Forecaster API")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Almacenamiento en memoria para demostración (ideal: Base de datos o AWS S3)
processed_data_cache = None

@app.post("/upload")
async def upload_file(file: UploadFile = File (...)):
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
        print(f"Ruta SLA seleccionada: {ruta_sla}")
        
        if not ruta_sla:
            carpetas_debug = [os.path.join(os.path.dirname(__file__), "data"), "data"]
            for c in carpetas_debug:
                if os.path.exists(c):
                    print(f"Archivos en carpeta {c}: {os.listdir(c)}")
            raise HTTPException(status_code=500, detail="Falta el archivo maestro SLA en el servidor.")
        
        df_sla = pd.read_excel(ruta_sla)
        
        # Estandarizar nombres de columnas
        df.columns = df.columns.str.strip().str.upper()
        df_sla.columns = df_sla.columns.str.strip().str.upper()
        
        # 2. Filtrar solo incidencias y solicitudes (excluir tareas programadas)
        if 'TIPO' in df.columns:
            df = df[df['TIPO'].astype(str).str.upper().isin(['INCIDENTE','SOLICITUD'])]  
        
        # 3. Cruce de dataframes por Categoria
        if 'CATEGORIA' in df.columns and 'CATEGORIA' in df_sla.columns:
            df = df.merge(df_sla[['CATEGORIA','SLA_MINUTOS']], on='CATEGORIA', how='left')
        else:
            df['SLA_MINUTOS'] = 120 # Valor por defecto si fallan las columnas
            
        # 4. Calculo de tiempos reales y cumplimiento SLA
        if 'FECHAHORACREACION' in df.columns and 'FECHAHORACIERRE' in df.columns:
            df['FECHAHORACREACION'] = pd.to_datetime(df['FECHAHORACREACION'], errors='coerce')
            df['FECHAHORACIERRE'] = pd.to_datetime(df['FECHAHORACIERRE'], errors='coerce')
            
            # Calcular minutos reales
            df['MINUTOS_REALES'] = (df['FECHAHORACIERRE'] - df['FECHAHORACREACION']).dt.total_seconds() / 60
            
            # Etiquetar cumplimiento (1 = cumple, 0 = no cumple)
            df['CUMPLE_SLA'] = (df['MINUTOS_REALES'] <= df['SLA_MINUTOS']).astype(int)
            
        # Extraer el tiempo promedio historico de resolucion (ignorando valores nulos o negativos)
        if 'MINUTOS_REALES' in df.columns:
            validos = df.loc[df['MINUTOS_REALES'] > 0, 'MINUTOS_REALES']
            tiempo_promedio = float(validos.mean()) if not validos.empty and not pd.isna(validos.mean()) else 45.0
        else:
            tiempo_promedio = 45.0
        
        # 5. Agrupar por dia para el entrenamiento de ML
        if 'FECHAHORACREACION' in df.columns:
            df['FECHA_CORTA'] = df['FECHAHORACREACION'].dt.date
            df_agrupado = df.groupby('FECHA_CORTA').agg(
                cantidad_incidentes=('TICKETID', 'count'),
                tasa_cumplimiento_sla=('CUMPLE_SLA','mean')
            ).reset_index()
            
            df_agrupado = df_agrupado.sort_values('FECHA_CORTA')
            df_agrupado['FECHA_CORTA'] = pd.to_datetime(df_agrupado['FECHA_CORTA'])
            df_agrupado.set_index('FECHA_CORTA', inplace=True)
            
            # Rellenar fechas vacias con 0
            idx = pd.date_range(df_agrupado.index.min(), df_agrupado.index.max())
            df_agrupado = df_agrupado.reindex(idx).fillna({'cantidad_incidentes': 0, 'tasa_cumplimiento_sla': 1.0})
            df_agrupado.reset_index(inplace=True)
            df_agrupado.rename(columns={'index': 'fecha'}, inplace=True)
            
            # Guardar tiempo promedio en un archivo auxiliar para usarlo en la proyeccion
            with open("data/tiempo_promedio.txt", "w") as f:
                f.write(str(tiempo_promedio))
                
            df_agrupado.to_csv("data/incidentes_limpios.csv", index=False)
            
            return {
                "mensaje": "Cruce con SLA exitoso y datos agrupados.",
                "total_incidentes": int(df_agrupado['cantidad_incidentes'].sum()),
                "tiempo_promedio_historico": round(tiempo_promedio, 2),
                "dias_registrados": len(df_agrupado)
            }
        else:
            raise HTTPException(status_code=400, detail="No se encontro la columna de fecha de creacion.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")
#async def upload_file(file: UploadFile = File(...)):
 #   global processed_data_cache
  #  if not file.filename.endswith(('.xls', '.xlsx')):
   #     raise HTTPException(status_code=400, detail="Solo se permiten archivos Excel (.xls, .xlsx)")
    
    #contents = await file.read()
    #try:
     #   df = pd.read_excel(io.BytesIO(contents))
        
        # 1. Estandarizar columnas a minúsculas para encontrar los datos más fácil
      #  df.columns = df.columns.str.lower().str.strip()
        
        # 2. Buscar columna de tipo de ticket y fecha dinámicamente
       # tipo_col = next((col for col in df.columns if 'tipo' in col or 'type' in col or 'clasificacion' in col), None)
        #fecha_col = next((col for col in df.columns if 'fecha' in col or 'date' in col or 'creacion' in col or 'creación' in col), None)
        
        #if not fecha_col:
            # Fallback: tomar la primera columna que sea tipo datetime
         #   for col in df.columns:
          #      if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(df[col]):
           #         fecha_col = col
            #        break
           # if not fecha_col:
            #    raise HTTPException(status_code=400, detail="No se encontró una columna de fecha en el Excel para agrupar en el tiempo.")
            
        # 3. Limpiar y Filtrar
       # if tipo_col:
        #    df = df[df[tipo_col].astype(str).str.lower().str.contains("incidente|incident", na=False)]
            
        # 4. Asegurar formato de fecha
       # df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
       # df = df.dropna(subset=[fecha_col])
        
        # 5. Agrupar por día
       # df['fecha_corta'] = df[fecha_col].dt.date
       # df_agrupado = df.groupby('fecha_corta').size().reset_index(name='cantidad_incidentes')
        #df_agrupado = df_agrupado.sort_values('fecha_corta')
        #df_agrupado['fecha_corta'] = pd.to_datetime(df_agrupado['fecha_corta'])
        #df_agrupado.set_index('fecha_corta', inplace=True)
        
        # Rellenar fechas vacías con 0 incidentes para tener una serie temporal continua
        #idx = pd.date_range(df_agrupado.index.min(), df_agrupado.index.max())
        #df_agrupado = df_agrupado.reindex(idx, fill_value=0)
        #df_agrupado.reset_index(inplace=True)
        #df_agrupado.rename(columns={'index': 'fecha'}, inplace=True)
        
        #processed_data_cache = df_agrupado
        
        # Guardar copia limpia temporal
       # df_agrupado.to_csv("data/incidentes_limpios.csv", index=False)
        
       # return {
       #     "mensaje": "Archivo procesado y limpiado correctamente.",
      #      "total_incidentes": int(df_agrupado['cantidad_incidentes'].sum()),
     #       "dias_registrados": len(df_agrupado),
    #        "head": df_agrupado.head(5).to_dict(orient="records")
   #     }
        
  #  except Exception as e:
 #       raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

@app.get("/download-clean")
async def download_clean():
    file_path = "data/incidentes_limpios.csv"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/csv", filename="incidentes_limpios.csv")
    raise HTTPException(status_code=404, detail="Archivo procesado no encontrado. Sube un archivo primero.")
@app.post("/train")
async def train_models(horizonte: int = 30):
    global processed_data_cache
    if processed_data_cache is None:
        try:
            processed_data_cache = pd.read_csv("data/incidentes_limpios.csv")
            processed_data_cache['fecha'] = pd.to_datetime(processed_data_cache['fecha'])
        except Exception:
            raise HTTPException(status_code=400, detail="No hay datos procesados. Sube un Excel primero.")
            
    df = processed_data_cache.copy()
    
    # Split 80/20 cronológico
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    if len(train) < 5:
         raise HTTPException(status_code=400, detail="No hay suficientes datos históricos para entrenar.")
         
    resultados_modelos = []
    
    # --- Modelo 1: Regresión Lineal ---
    try:
        df_lr = df.copy()
        df_lr['dia_ordinal'] = df_lr['fecha'].map(pd.Timestamp.toordinal)
        train_lr, test_lr = df_lr.iloc[:train_size], df_lr.iloc[train_size:]
        
        lr = LinearRegression()
        lr.fit(train_lr[['dia_ordinal']], train_lr['cantidad_incidentes'])
        lr_preds = lr.predict(test_lr[['dia_ordinal']])
        
        lr_mae = mean_absolute_error(test_lr['cantidad_incidentes'], lr_preds)
        lr_rmse = np.sqrt(mean_squared_error(test_lr['cantidad_incidentes'], lr_preds))
        lr_r2 = r2_score(test_lr['cantidad_incidentes'], lr_preds)
        resultados_modelos.append({
            "nombre": "Regresión Lineal",
            "mae": float(lr_mae),
            "rmse": float(lr_rmse),
            "r2": float(lr_r2),
            "tipo": "sklearn"
        })
    except Exception as e:
        print("Error en Regresion Lineal:", e)

    # --- Modelo 2: Exponential Smoothing (Statsmodels) ---
    try:
        ts_data = train['cantidad_incidentes'].values
        es = ExponentialSmoothing(ts_data, trend='add', seasonal=None, initialization_method="estimated")
        es_fit = es.fit()
        es_preds = es_fit.forecast(len(test))
        
        es_mae = mean_absolute_error(test['cantidad_incidentes'], es_preds)
        es_rmse = np.sqrt(mean_squared_error(test['cantidad_incidentes'], es_preds))
        es_r2 = r2_score(test['cantidad_incidentes'], es_preds)
        resultados_modelos.append({
            "nombre": "Suavizado Exponencial",
            "mae": float(es_mae),
            "rmse": float(es_rmse),
            "r2": float(es_r2),
            "tipo": "statsmodels"
        })
    except Exception as e:
         print("Error en ES:", e)
         
    # --- Modelo 3: Prophet ---
    try:
        df_prophet = train.copy().reset_index(drop=True)
        df_prophet = df_prophet[['fecha', 'cantidad_incidentes']]
        df_prophet.rename(columns={'fecha': 'ds', 'cantidad_incidentes': 'y'}, inplace=True)
        
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        
        future = test[['fecha']].copy()
        future.rename(columns={'fecha': 'ds'}, inplace=True)
        forecast = m.predict(future)
        
        prophet_mae = mean_absolute_error(test['cantidad_incidentes'], forecast['yhat'])
        prophet_rmse = np.sqrt(mean_squared_error(test['cantidad_incidentes'], forecast['yhat']))
        prophet_r2 = r2_score(test['cantidad_incidentes'], forecast['yhat'])
        resultados_modelos.append({
            "nombre": "Prophet",
            "mae": float(prophet_mae),
            "rmse": float(prophet_rmse),
            "r2": float(prophet_r2),
            "tipo": "prophet"
        })
    except Exception as e:
        print("Error en Prophet:", e)

    # --- Modelo 4: ARIMA ---
    try:
        ts_data = train['cantidad_incidentes'].values
        arima = ARIMA(ts_data, order=(5, 1, 0))
        arima_fit = arima.fit()
        arima_preds = arima_fit.forecast(steps=len(test))
        
        arima_mae = mean_absolute_error(test['cantidad_incidentes'], arima_preds)
        arima_rmse = np.sqrt(mean_squared_error(test['cantidad_incidentes'], arima_preds))
        arima_r2 = r2_score(test['cantidad_incidentes'], arima_preds)
        resultados_modelos.append({
            "nombre": "ARIMA",
            "mae": float(arima_mae),
            "rmse": float(arima_rmse),
            "r2": float(arima_r2),
            "tipo": "arima"
        })
    except Exception as e:
        print("Error en ARIMA:", e)
        
    # --- Modelo 5: SARIMA ---
    try:
        ts_data = train['cantidad_incidentes'].values
        sarima = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        sarima_fit = sarima.fit(disp=False)
        sarima_preds = sarima_fit.forecast(steps=len(test))
        
        sarima_mae = mean_absolute_error(test['cantidad_incidentes'], sarima_preds)
        sarima_rmse = np.sqrt(mean_squared_error(test['cantidad_incidentes'], sarima_preds))
        sarima_r2 = r2_score(test['cantidad_incidentes'], sarima_preds)
        resultados_modelos.append({
            "nombre": "SARIMA",
            "mae": float(sarima_mae),
            "rmse": float(sarima_rmse),
            "r2": float(sarima_r2),
            "tipo": "sarima"
        })
    except Exception as e:
        print("Error en SARIMA:", e)

    # --- Modelo 6: Random Forest ---
    try:
        df_rf = df.copy()
        df_rf['dia_ordinal'] = df_rf['fecha'].map(pd.Timestamp.toordinal)
        train_rf, test_rf = df_rf.iloc[:train_size], df_rf.iloc[train_size:]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_rf[['dia_ordinal']], train_rf['cantidad_incidentes'])
        rf_preds = rf.predict(test_rf[['dia_ordinal']])
        
        rf_mae = mean_absolute_error(test_rf['cantidad_incidentes'], rf_preds)
        rf_rmse = np.sqrt(mean_squared_error(test_rf['cantidad_incidentes'], rf_preds))
        rf_r2 = r2_score(test_rf['cantidad_incidentes'], rf_preds)
        resultados_modelos.append({
            "nombre": "Random Forest",
            "mae": float(rf_mae),
            "rmse": float(rf_rmse),
            "r2": float(rf_r2),
            "tipo": "random_forest"
        })
    except Exception as e:
        print("Error en Random Forest:", e)

    # --- Modelo 7: XGBoost ---
    try:
        df_xgb = df.copy()
        df_xgb['dia_ordinal'] = df_xgb['fecha'].map(pd.Timestamp.toordinal)
        train_xgb, test_xgb = df_xgb.iloc[:train_size], df_xgb.iloc[train_size:]
        
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(train_xgb[['dia_ordinal']], train_xgb['cantidad_incidentes'])
        xgb_preds = xgb_model.predict(test_xgb[['dia_ordinal']])
        
        xgb_mae = mean_absolute_error(test_xgb['cantidad_incidentes'], xgb_preds)
        xgb_rmse = np.sqrt(mean_squared_error(test_xgb['cantidad_incidentes'], xgb_preds))
        xgb_r2 = r2_score(test_xgb['cantidad_incidentes'], xgb_preds)
        resultados_modelos.append({
            "nombre": "XGBoost",
            "mae": float(xgb_mae),
            "rmse": float(xgb_rmse),
            "r2": float(xgb_r2),
            "tipo": "xgboost"
        })
    except Exception as e:
        print("Error en XGBoost:", e)
         
    if not resultados_modelos:
        raise HTTPException(status_code=500, detail="Fallo el entrenamiento de los modelos.")
        
    # Seleccionar el modelo idóneo (menor error absoluto)
    mejor_modelo_info = min(resultados_modelos, key=lambda x: x['mae'])
    
    # Reentrenar el mejor modelo con TODOS los datos y predecir el horizonte
    predicciones_futuro = []
    fechas_futuro = pd.date_range(start=df['fecha'].max() + pd.Timedelta(days=1), periods=horizonte)
    
    if mejor_modelo_info['tipo'] == "sklearn":
        modelo_final = LinearRegression()
        df['dia_ordinal'] = df['fecha'].map(pd.Timestamp.toordinal)
        modelo_final.fit(df[['dia_ordinal']], df['cantidad_incidentes'])
        
        df_futuro = pd.DataFrame({'fecha': fechas_futuro})
        df_futuro['dia_ordinal'] = df_futuro['fecha'].map(pd.Timestamp.toordinal)
        preds = modelo_final.predict(df_futuro[['dia_ordinal']])
        predicciones_futuro = [max(0, int(p)) for p in preds] 
        
    elif mejor_modelo_info['tipo'] == "statsmodels":
        ts_data_all = df['cantidad_incidentes'].values
        es_final = ExponentialSmoothing(ts_data_all, trend='add', seasonal=None, initialization_method="estimated")
        es_final_fit = es_final.fit()
        preds = es_final_fit.forecast(horizonte)
        predicciones_futuro = [max(0, int(p)) for p in preds]
        
    elif mejor_modelo_info['tipo'] == "prophet":
        df_prophet_all = df[['fecha', 'cantidad_incidentes']].copy()
        df_prophet_all.rename(columns={'fecha': 'ds', 'cantidad_incidentes': 'y'}, inplace=True)
        m_final = Prophet(daily_seasonality=True)
        m_final.fit(df_prophet_all)
        df_futuro_prophet = pd.DataFrame({'ds': fechas_futuro})
        preds_df = m_final.predict(df_futuro_prophet)
        predicciones_futuro = [max(0, int(p)) for p in preds_df['yhat']]
        
    elif mejor_modelo_info['tipo'] == "arima":
        ts_data_all = df['cantidad_incidentes'].values
        arima_final = ARIMA(ts_data_all, order=(5, 1, 0))
        arima_final_fit = arima_final.fit()
        preds = arima_final_fit.forecast(steps=horizonte)
        predicciones_futuro = [max(0, int(p)) for p in preds]
        
    elif mejor_modelo_info['tipo'] == "sarima":
        ts_data_all = df['cantidad_incidentes'].values
        sarima_final = SARIMAX(ts_data_all, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        sarima_final_fit = sarima_final.fit(disp=False)
        preds = sarima_final_fit.forecast(steps=horizonte)
        predicciones_futuro = [max(0, int(p)) for p in preds]
        
    elif mejor_modelo_info['tipo'] == "random_forest":
        rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
        df['dia_ordinal'] = df['fecha'].map(pd.Timestamp.toordinal)
        rf_final.fit(df[['dia_ordinal']], df['cantidad_incidentes'])
        df_futuro = pd.DataFrame({'fecha': fechas_futuro})
        df_futuro['dia_ordinal'] = df_futuro['fecha'].map(pd.Timestamp.toordinal)
        preds = rf_final.predict(df_futuro[['dia_ordinal']])
        predicciones_futuro = [max(0, int(p)) for p in preds]
        
    elif mejor_modelo_info['tipo'] == "xgboost":
        xgb_final = xgb.XGBRegressor(n_estimators=100, random_state=42)
        df['dia_ordinal'] = df['fecha'].map(pd.Timestamp.toordinal)
        xgb_final.fit(df[['dia_ordinal']], df['cantidad_incidentes'])
        df_futuro = pd.DataFrame({'fecha': fechas_futuro})
        df_futuro['dia_ordinal'] = df_futuro['fecha'].map(pd.Timestamp.toordinal)
        preds = xgb_final.predict(df_futuro[['dia_ordinal']])
        predicciones_futuro = [max(0, int(p)) for p in preds]
    
    # Leer el tiempo promedio historico guardado
    tiempo_prom_historico = 45.0 # Valor por defecto de seguridad
    posibles_rutas_tiempo = [
        "data/tiempo_promedio.txt",
        os.path.join(os.path.dirname(__file__), "data", "tiempo_promedio.txt")
    ]
    ruta_tiempo = next((p for p in posibles_rutas_tiempo if os.path.exists(p)), None)
    if ruta_tiempo:
        try:
            with open(ruta_tiempo, "r") as f:
                tiempo_prom_historico = float(f.read().strip())
        except Exception:
            pass     
    return {
        "modelos_evaluados": [{"nombre": m['nombre'], "mae": round(m['mae'], 2), "rmse": round(m['rmse'], 2), "r2": round(m['r2'], 4)} for m in resultados_modelos],
        "mejor_modelo": mejor_modelo_info['nombre'],
        "horizonte_dias": horizonte,
        "fechas_historicas": df['fecha'].dt.strftime("%Y-%m-%d").tolist(),
        "valores_historicos": df['cantidad_incidentes'].tolist(),
        "fechas_proyectadas": [f.strftime("%Y-%m-%d") for f in fechas_futuro],
        "valores_proyectados": predicciones_futuro,
        "tiempo_promedio_historico": tiempo_prom_historico #NUEVA LINEA AÑADIDA
    }
