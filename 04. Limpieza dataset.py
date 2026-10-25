# -*- coding: utf-8 -*-
"""
limpiar_dataset.py  --  limpieza del CSV de focos ANTES de entrenar (no re-extrae nada).

Qué corrige (cada paso está contado en la salida y guardado en limpieza_log.csv):

  [L1] Etiquetas contradictorias. Cada foco se etiquetó por separado, así que un mismo
       píxel-imagen (x, y, hora UTC) puede ser 1 para un foco y 0 para otro. Si un píxel-imagen
       es positivo para ALGÚN foco, se conserva como positivo y se descartan sus copias negativas.
  [L2] Filas repetidas. El mismo píxel-imagen aparece una vez por cada foco cercano. Se deja UNA fila
       por (x, y, hora UTC). Se verifica que las copias tengan valores iguales de T7 antes de colapsar.
  [L3] Negativos contaminados. Un negativo debía estar a >10 km de SU foco, pero pudo quedar cerca de
       OTRO foco de la misma hora. Se descartan los negativos a menos de EXCL_KM de cualquier foco de
       la misma hora UTC (la lista de focos sale del propio CSV).
  [L4] Se agrega la columna fire_minute_in_hour (minuto del foco dentro de la hora). NO es predictor:
       sirve para evaluar por franjas de desfase foco-imagen. El script de modelado usa una lista fija
       de variables, así que esta columna no entra al modelo.

Límites que conviene declarar:
  * La lista de focos de [L3] solo contiene los focos que dejaron filas en el CSV. Los focos sin
    píxeles válidos o filtrados (p. ej. por agua) no cuentan, así que la limpieza de negativos es
    un mínimo, no una garantía.
  * x e y se tratan como lon/lat en grados (así los escribe el script de extracción tras reproyectar a
    EPSG:4326). La distancia usa una proyección equirectangular local; error << 1 % en esta región.

Uso:   python limpiar_dataset.py "<ruta CSV de entrada>" ["<ruta CSV de salida>"]
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_IN = r"C:\Users\User\Desktop\Modelo incendios\focos_sin_agua.csv"
IN = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(IN), "focos_limpio.csv")
LOG = os.path.join(os.path.dirname(OUT), "limpieza_log.csv")
EXCL_KM = 10.0            # mismo radio de exclusión que usó la extracción (BUFFER_EXCL_M)
XY_DECIMALS = 6           # solo para comparar flotantes de x,y; no cambia los datos guardados

NEED = ["label", "x", "y", "fire_latitude", "fire_longitude", "fire_datetime_utc"]

log = []
def rec(step, **kw):
    row = {"step": step}; row.update(kw); log.append(row)
    print(f"[{step}] " + ", ".join(f"{k}={v}" for k, v in kw.items()))

print(f"Leyendo {IN}")
df = pd.read_csv(IN)
missing = [c for c in NEED if c not in df.columns]
if missing:
    sys.exit(f"Faltan columnas en el CSV: {missing}. Columnas disponibles: {list(df.columns)}")

t = pd.to_datetime(df["fire_datetime_utc"], errors="coerce")
n_nat = int(t.isna().sum())
if n_nat:
    print(f"[AVISO] {n_nat} filas con fecha no interpretable; se descartan (no se pueden asignar a una imagen).")
    keep = ~t.isna()
    df, t = df[keep].reset_index(drop=True), t[keep].reset_index(drop=True)

df["_img_t"] = t.dt.floor("h").values
df["_xr"] = df["x"].round(XY_DECIMALS)
df["_yr"] = df["y"].round(XY_DECIMALS)
KEY = ["_xr", "_yr", "_img_t"]

def stats(d):
    g = d.groupby(KEY)["label"]
    return dict(filas=len(d), pos=int((d.label == 1).sum()), neg=int((d.label == 0).sum()),
                pct_pixel_imagen_repetido=round(float((g.transform("size") > 1).mean()), 4),
                pct_ambas_clases=round(float((g.transform("nunique") > 1).mean()), 4))

rec("antes", **stats(df))

# Tabla de focos (una fila por foco y hora) ANTES de descartar nada: se usa en [L3]
events = df[["fire_latitude", "fire_longitude", "_img_t"]].drop_duplicates()
rec("focos_unicos", n=len(events))

# ---------- [L1] + [L2] ----------
dupmask = df.duplicated(KEY, keep=False)
if "T7" in df.columns and dupmask.any():
    nun = df[dupmask].groupby(KEY)["T7"].nunique()
    rec("verificacion_copias", grupos_repetidos=len(nun), pct_con_T7_distinto=round(float((nun > 1).mean()), 6))
else:
    print("[AVISO] No se verificó que las copias tengan igual T7 (falta la columna T7 o no hay copias).")

has_pos = df.groupby(KEY)["label"].transform("max") == 1
n_neg_conflicto = int(((df.label == 0) & has_pos).sum())
df = df.sort_values("label", ascending=False, kind="stable")      # positivos primero
before = len(df)
df = df.drop_duplicates(KEY, keep="first").sort_index()
rec("L1_L2", negativos_descartados_por_conflicto=n_neg_conflicto,
    filas_colapsadas_en_total=before - len(df), **stats(df))

# ---------- [L3] ----------
lat0 = float(events["fire_latitude"].mean())
kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
ev_xy = np.column_stack([events["fire_longitude"].to_numpy() * kx, events["fire_latitude"].to_numpy() * ky])
ev_by_t = {k: ev_xy[idx] for k, idx in events.reset_index(drop=True).groupby("_img_t").indices.items()}

neg_pos = np.flatnonzero(df["label"].to_numpy() == 0)
neg_xy = np.column_stack([df["x"].to_numpy()[neg_pos] * kx, df["y"].to_numpy()[neg_pos] * ky])
neg_t = df["_img_t"].to_numpy()[neg_pos]
too_close = np.zeros(len(neg_pos), dtype=bool)
order = pd.Series(np.arange(len(neg_pos))).groupby(neg_t).indices
for tt, idx in order.items():
    pts = ev_by_t.get(pd.Timestamp(tt))
    if pts is None:
        continue
    d, _ = cKDTree(pts).query(neg_xy[idx])
    too_close[idx] = d < EXCL_KM
drop_rows = df.index[neg_pos[too_close]]
df = df.drop(index=drop_rows)
rec("L3", negativos_descartados_cercanos_a_otro_foco=int(too_close.sum()), **stats(df))

# ---------- verificaciones finales ----------
g = df.groupby(KEY)["label"]
assert (g.transform("size") == 1).all(), "quedan píxel-imagen repetidos"
assert (g.transform("nunique") == 1).all(), "quedan píxel-imagen con ambas clases"

df["fire_minute_in_hour"] = pd.to_datetime(df["fire_datetime_utc"]).dt.minute
df = df.drop(columns=["_img_t", "_xr", "_yr"]).reset_index(drop=True)
rec("despues", filas=len(df), pos=int((df.label == 1).sum()), neg=int((df.label == 0).sum()),
    razon_neg_pos=round(float((df.label == 0).sum() / max((df.label == 1).sum(), 1)), 3))

df.to_csv(OUT, index=False)
pd.DataFrame(log).to_csv(LOG, index=False)
print(f"\nGuardado: {OUT}\nRegistro: {LOG}")
print("\nSiguiente paso: ajustar N_SAMPLE_CLASS en el script de modelado a un valor <= min(pos, neg) del paso 'despues'.")