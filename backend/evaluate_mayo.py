import urllib.request
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar datos reales de Mayo
df_mayo = pd.read_excel('data/TICKETS_MAYO.xlsx')
df_mayo.columns = df_mayo.columns.str.strip().str.upper()
df_mayo['FECHAHORACREACION'] = pd.to_datetime(df_mayo['FECHAHORACREACION'], errors='coerce')
df_mayo['FECHA_CORTA'] = pd.to_datetime(df_mayo['FECHAHORACREACION'].dt.date)
df_mayo['TIPO_STR'] = df_mayo['TIPO'].astype(str).str.upper()
df_mayo['SERVICE_STR'] = df_mayo['TIPOSERVICE'].astype(str).str.upper()

agrup_mayo = df_mayo.groupby('FECHA_CORTA').agg(
    cantidad_total=('TICKETID', 'count'),
    cantidad_incidentes=('TIPO_STR', lambda x: (x == 'INCIDENTE').sum()),
    cantidad_solicitudes=('TIPO_STR', lambda x: (x == 'SOLICITUD').sum()),
    cantidad_pos=('SERVICE_STR', lambda x: (x.str.contains('POS|ERP', na=False)).sum())
).reindex(pd.date_range('2026-05-01', '2026-05-31')).fillna(0)

series_a_evaluar = [
    ('todos', 'cantidad_total', 'TOTAL DE TICKETS (GLOBAL)'),
    ('incidentes', 'cantidad_incidentes', 'INCIDENTES CRÍTICOS'),
    ('solicitudes', 'cantidad_solicitudes', 'SOLICITUDES DE SERVICIO'),
    ('pos', 'cantidad_pos', 'SOPORTE ERP - POS / CAJAS')
]

for cod_serie, col_name, titulo in series_a_evaluar:
    y_real = agrup_mayo[col_name].values
    url = f'http://localhost:8000/train?horizonte=31&tipo_serie={cod_serie}'
    req = urllib.request.Request(url, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
    
    total_real = int(y_real.sum())
    prom_real = float(y_real.mean())
    print("\n" + "=" * 96)
    print(f"SERIE: {titulo}")
    print(f"Total Real en Mayo: {total_real} tickets | Promedio diario real: {prom_real:.1f} tickets/día")
    print("=" * 96)
    print(f"{'Modelo':<23} | {'Proy Mayo':<10} | {'Desv Total':<13} | {'MAE':<6} | {'RMSE':<6} | {'MAPE':<8} | {'Exactitud WAPE':<15} | {'R2':<6}")
    print("-" * 96)
    
    preds_dict = res['predicciones_por_modelo']
    ranking = []
    for mod_nom, p_list in preds_dict.items():
        p_arr = np.array(p_list)
        mae = float(mean_absolute_error(y_real, p_arr))
        rmse = float(np.sqrt(mean_squared_error(y_real, p_arr)))
        
        mask = y_real > 0
        mape = float(np.mean(np.abs((y_real[mask] - p_arr[mask]) / y_real[mask])) * 100) if np.any(mask) else 0.0
        wape = float(np.sum(np.abs(y_real - p_arr)) / np.sum(y_real) * 100) if np.sum(y_real) > 0 else 0.0
        exactitud = max(0.0, 100.0 - wape)
        r2 = float(r2_score(y_real, p_arr))
        tot_proy = int(p_arr.sum())
        dif_vol = tot_proy - total_real
        err_vol_pct = (dif_vol / total_real) * 100 if total_real > 0 else 0.0
        
        ranking.append({
            'modelo': mod_nom,
            'tot_proy': tot_proy,
            'dif_vol': dif_vol,
            'err_vol_pct': err_vol_pct,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'exactitud': exactitud,
            'r2': r2
        })
        
    ranking.sort(key=lambda x: x['mae'])
    for r in ranking:
        desv_str = f"{r['dif_vol']:+d} ({r['err_vol_pct']:+.1f}%)"
        print(f"{r['modelo']:<23} | {r['tot_proy']:<10} | {desv_str:<13} | {r['mae']:<6.2f} | {r['rmse']:<6.2f} | {r['mape']:<7.2f}% | {r['exactitud']:<14.2f}% | {r['r2']:<6.2f}")
