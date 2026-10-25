# -*- coding: utf-8 -*-
"""
preparar_focos.py -- convierte el CSV crudo de focos de calor (formato IDEAM/"amazonica") en una tabla
ordenada y sin ambigüedades, lista para que el script de extracción GOES tome la posición y la hora.

Entrada : CSV crudo (separador ';', decimal ',', fecha local UTC-5 'YYYY-MM-DD HH:MM').
Salida  : focos_amazonica_2024_limpio.csv  (coma, punto decimal, UTF-8, nombres de columna sin tildes)
          focos_amazonica_2024_qc.txt      (conteos y avisos de calidad)

NO se elimina ninguna fila ni se filtra por confianza: cada decisión de filtrado queda para el
script de etiquetado, donde se documenta. Solo se reordenan, tipan y derivan columnas.

Uso: python preparar_focos.py "<CSV crudo>" ["<carpeta de salida>"]
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ================= CONFIGURACIÓN =================

IN = r"C:\Users\User\Desktop\Nuevo Manuscript\focos_amazonica_2024.csv"

OUT_DIR = os.path.dirname(os.path.abspath(IN))
OUT_CSV = os.path.join(OUT_DIR, "focos_amazonica_2024_limpio.csv")
OUT_QC = os.path.join(OUT_DIR, "focos_amazonica_2024_qc.txt")

# =================================================
OUT_CSV = os.path.join(OUT_DIR, "focos_amazonica_2024_limpio.csv")
OUT_QC = os.path.join(OUT_DIR, "focos_amazonica_2024_qc.txt")
UTC_OFFSET_H = 5          # la columna 'Fecha (UTC-5)' está en hora local UTC-5 -> UTC = local + 5 h
CLUSTER_KM = 2.0          # radio para agrupar focos contiguos del mismo paso de satélite (píxel GOES ~2 km)

raw = pd.read_csv(IN, sep=";", dtype=str, encoding="utf-8-sig")
raw.columns = [c.strip() for c in raw.columns]
col = {  # nombres del archivo crudo (se comprueban para no inventar columnas)
    "fecha": "Fecha (UTC-5)", "lat": "Lat", "lon": "Lon", "fuente": "Fuente",
    "temp": "Temperatura (C)", "temp_alt": "Temperatura Alt* (C)",
    "frp": [c for c in raw.columns if c.startswith("Radiaci")][0],
    "conf": "Confianza", "dn": "Captura (Dia-Noche)",
    "scan": "Scan - real pixel size (km)", "track": "Track - real pixel size (km)",
}
missing = [v for v in col.values() if isinstance(v, str) and v not in raw.columns]
assert not missing, f"Faltan columnas en el CSV crudo: {missing}. Hay: {list(raw.columns)}"

num = lambda s: pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
dt_local = pd.to_datetime(raw[col["fecha"]], format="%Y-%m-%d %H:%M", errors="coerce")
assert dt_local.notna().all(), "Hay fechas que no siguen el formato YYYY-MM-DD HH:MM"
dt_utc = dt_local + pd.Timedelta(hours=UTC_OFFSET_H)

conf = raw[col["conf"]].str.strip()
is_pct = conf.str.match(r"^\d+\s*%$")
conf_pct = pd.to_numeric(conf.where(is_pct).str.replace(r"[^\d]", "", regex=True), errors="coerce")
cat_map = {"Baja": "low", "Nominal": "nominal", "Alta": "high"}
conf_cat = conf.where(~is_pct).map(cat_map)
assert (conf_cat.notna() | conf_pct.notna()).all(), f"Confianza no reconocida: {conf[~(conf_cat.notna() | conf_pct.notna())].unique()}"

fuente = raw[col["fuente"]].str.strip()
out = pd.DataFrame({
    "hotspot_id": np.arange(1, len(raw) + 1),                # posición en el archivo crudo (trazabilidad)
    "datetime_local": dt_local.dt.strftime("%Y-%m-%d %H:%M:%S"),
    "datetime_utc": dt_utc.dt.strftime("%Y-%m-%d %H:%M:%S"),
    "year_utc": dt_utc.dt.year, "doy_utc": dt_utc.dt.dayofyear,
    "hour_utc": dt_utc.dt.hour, "minute_utc": dt_utc.dt.minute,
    "lat": num(raw[col["lat"]]), "lon": num(raw[col["lon"]]),
    "source": fuente,
    "sensor": fuente.str.split("-").str[0],
    "temp_c": num(raw[col["temp"]]), "temp_alt_c": num(raw[col["temp_alt"]]),
    "frp_mw": num(raw[col["frp"]]),
    "confidence_raw": conf, "confidence_class": conf_cat, "confidence_pct": conf_pct,
    "day_night": raw[col["dn"]].str.strip(),
    "scan_km": num(raw[col["scan"]]), "track_km": num(raw[col["track"]]),
})
assert out[["lat", "lon"]].notna().all().all(), "Lat/Lon no numéricos"

# Paso de satélite: mismos (fuente, instante) = un paso. Útil para etiquetar por imagen, no por foco.
out["overpass_id"] = out.groupby(["source", "datetime_utc"], sort=False).ngroup() + 1
# Slot nominal de 10 min (solo para contar imágenes; el script de extracción usará los archivos reales de S3)
out["goes_slot_utc_nominal"] = dt_utc.dt.round("10min").dt.strftime("%Y-%m-%d %H:%M:%S")

# Grupos de focos contiguos del mismo paso (VIIRS 375 m: varios focos caen en un píxel GOES de 2 km)
lat0 = out["lat"].mean()
kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
cid = np.zeros(len(out), dtype=np.int64)
nxt = 0
for _, idx in out.groupby("overpass_id").indices.items():
    pts = np.column_stack([out["lon"].to_numpy()[idx] * kx, out["lat"].to_numpy()[idx] * ky])
    pairs = cKDTree(pts).query_pairs(CLUSTER_KM, output_type="ndarray")
    n = len(idx)
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        k, lab = connected_components(g, directed=False)
    else:
        k, lab = n, np.arange(n)
    cid[idx] = lab + nxt
    nxt += k
out["cluster_id_2km"] = cid + 1

out = out.sort_values(["datetime_utc", "hotspot_id"]).reset_index(drop=True)
os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding="utf-8")

# ---------------- QC ----------------
L = []
def p(s=""): L.append(s); print(s)
p(f"Filas: {len(out)}  | columnas: {out.shape[1]}")
p(f"Rango local: {dt_local.min()} -> {dt_local.max()}   | rango UTC: {dt_utc.min()} -> {dt_utc.max()}")
p(f"Lat {out.lat.min():.3f}..{out.lat.max():.3f} | Lon {out.lon.min():.3f}..{out.lon.max():.3f}")
p(f"Filas con FRP vacío: {int(out.frp_mw.isna().sum())}")
p(f"Duplicados exactos (fecha, lat, lon, fuente): {int(out.duplicated(['datetime_utc','lat','lon','source']).sum())}")
p(f"Pasos de satélite (fuente, instante): {out.overpass_id.nunique()} | instantes UTC distintos: {out.datetime_utc.nunique()} | slots nominales de 10 min: {out.goes_slot_utc_nominal.nunique()}")
p(f"Grupos de focos contiguos (<= {CLUSTER_KM} km, mismo paso): {out.cluster_id_2km.nunique()}  (focos por grupo: mediana {out.groupby('cluster_id_2km').size().median():.0f}, máx {out.groupby('cluster_id_2km').size().max()})")
p("\nFocos por mes (UTC):"); p(out.groupby(pd.to_datetime(out.datetime_utc).dt.month).size().to_string())
p("\nFocos por sensor / fuente:"); p(out.source.value_counts().to_string())
p("\nConfianza VIIRS (categoría):"); p(out.confidence_class.value_counts(dropna=False).to_string())
p(f"Confianza MODIS (%): n={int(out.confidence_pct.notna().sum())}, mediana={out.confidence_pct.median():.0f}")
p("\nHora UTC del foco:"); p(out.hour_utc.value_counts().sort_index().to_string())
p(f"\nDía/noche: {out.day_night.value_counts().to_dict()}")
dec31 = (pd.to_datetime(out.datetime_utc).dt.date == pd.Timestamp("2024-12-31").date()).sum()
p(f"Focos con fecha UTC 2024-12-31: {dec31}  (el crudo termina el {dt_local.max().date()} local)")
p(f"Focos con fecha UTC en 2025: {int((out.year_utc != 2024).sum())}")
open(OUT_QC, "w", encoding="utf-8").write("\n".join(L))
print(f"\nGuardado:\n  {OUT_CSV}\n  {OUT_QC}")