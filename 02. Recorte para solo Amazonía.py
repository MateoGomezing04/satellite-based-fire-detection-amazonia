# -*- coding: utf-8 -*-
"""
recortar_archivo_amazonia.py -- Recorta UN archivo (CSV o Excel) al polígono de la Amazonía.

- CSV: lee el archivo completo. Detecta solo el separador (, ; o tabulador) y la coma decimal.
- Excel: lee TODAS las hojas sin importar su nombre y une las que tengan columnas de coordenadas.
- Conserva las filas cuyo punto (x = longitud, y = latitud) cae dentro del polígono o sobre su borde.
- Guarda el resultado como CSV junto al original, con el sufijo "_amazonia" (no modifica el original).

USO (PowerShell):  python recortar_archivo_amazonia.py
                   python recortar_archivo_amazonia.py --archivo "C:\\...\\otro.csv" --shape "C:\\...\\Amazonia.shp"
Requiere: pip install geopandas shapely pandas openpyxl
"""
import os, sys, csv, argparse
import numpy as np
import pandas as pd

# ============================== RUTAS (EDITE AQUÍ) ==============================
ARCHIVO = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\2024\salida_goes_areas\csv\archivo final\Dataset 2024 tipo csv.csv"
SHAPE_AMAZONIA = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\Amazonia.shp"
SALIDA = None              # None = mismo nombre + "_amazonia.csv" en la misma carpeta
BUFFER_KM = 0.0            # 0 = estrictamente dentro; > 0 incluye un margen hacia afuera
COL_LON, COL_LAT = None, None   # None = detectar (x/y, lon/lat, longitud/latitud...)
CRS_SI_FALTA_PRJ = None    # p. ej. "EPSG:4326", SOLO si el .shp no trae .prj

CANDIDATAS_LON = ["x", "lon", "longitud", "longitude", "long", "fire_longitude"]
CANDIDATAS_LAT = ["y", "lat", "latitud", "latitude", "fire_latitude"]

def leer_csv(ruta):
    """Lee el CSV detectando separador, decimal y codificación."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(ruta, "r", encoding=enc, newline="") as f: muestra = f.read(200_000)
            break
        except UnicodeDecodeError: continue
    try: sep = csv.Sniffer().sniff(muestra.splitlines()[0], delimiters=",;\t").delimiter
    except Exception: sep = ";" if muestra.splitlines()[0].count(";") > muestra.splitlines()[0].count(",") else ","
    decimal = "," if sep == ";" else "."
    print(f"CSV: separador '{sep}' | decimal '{decimal}' | codificación {enc}")
    df = pd.read_csv(ruta, sep=sep, decimal=decimal, encoding=enc, low_memory=False)
    return df, sep, decimal

def leer_excel(ruta):
    hojas = pd.read_excel(ruta, sheet_name=None)
    print(f"Excel: {len(hojas)} hoja(s): {list(hojas)}")
    partes = []
    for n, h in hojas.items():
        if buscar_col(h, CANDIDATAS_LON, COL_LON, silencioso=True) and buscar_col(h, CANDIDATAS_LAT, COL_LAT, silencioso=True):
            print(f"  hoja '{n}': {len(h):,} filas -> se usa"); partes.append(h)
        else: print(f"  hoja '{n}': sin columnas de coordenadas -> se omite")
    if not partes: sys.exit("[DETENIDO] Ninguna hoja tiene columnas de longitud/latitud.")
    return pd.concat(partes, ignore_index=True)

def buscar_col(df, candidatas, forzada=None, silencioso=False):
    if forzada:
        if forzada in df.columns: return forzada
        if silencioso: return None
        sys.exit(f"[DETENIDO] No existe la columna '{forzada}'. Columnas: {list(df.columns)}")
    low = {str(c).strip().lower(): c for c in df.columns}
    for c in candidatas:
        if c in low: return low[c]
    if silencioso: return None
    sys.exit(f"[DETENIDO] No se encontró columna de coordenadas ({candidatas}). Columnas: {list(df.columns)}. "
             "Defina COL_LON y COL_LAT al inicio del script.")

def a_numero(s):
    """Convierte a número aunque venga como texto con coma decimal ("-72,5")."""
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")

def cargar_poligono(ruta, buffer_km=0.0, crs_si_falta=None):
    import geopandas as gpd
    if not os.path.exists(ruta): sys.exit(f"[DETENIDO] No existe el shape: {ruta}")
    g = gpd.read_file(ruta)
    if g.crs is None:
        if not crs_si_falta: sys.exit("[DETENIDO] El shape no tiene .prj. Defina CRS_SI_FALTA_PRJ solo si conoce el sistema.")
        g = g.set_crs(crs_si_falta)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy(); g["geometry"] = g.geometry.make_valid()
    if buffer_km > 0:
        c = g.to_crs("EPSG:4326").union_all().centroid
        g = g.to_crs(f"+proj=aeqd +lat_0={c.y:.4f} +lon_0={c.x:.4f} +datum=WGS84 +units=m +no_defs")
        g["geometry"] = g.buffer(buffer_km * 1000)
    geom = g.to_crs("EPSG:4326").union_all()
    lo0, la0, lo1, la1 = geom.bounds
    print(f"Polígono: lon {lo0:.2f} a {lo1:.2f} | lat {la0:.2f} a {la1:.2f}")
    return geom

def dentro(lon, lat, geom):
    import shapely
    lon = np.asarray(lon, float); lat = np.asarray(lat, float); ok = np.isfinite(lon) & np.isfinite(lat)
    out = np.zeros(len(lon), bool); shapely.prepare(geom)
    out[ok] = shapely.intersects_xy(geom, lon[ok], lat[ok]); return out

def resumen(df, col):
    return df[col].value_counts().sort_index().to_dict() if col in df.columns else {}

def main():
    ap = argparse.ArgumentParser(description="Recorta un CSV/Excel al polígono de la Amazonía")
    ap.add_argument("--archivo", default=ARCHIVO); ap.add_argument("--shape", default=SHAPE_AMAZONIA)
    ap.add_argument("--salida", default=SALIDA); ap.add_argument("--buffer-km", type=float, default=BUFFER_KM)
    args = ap.parse_args()
    if not os.path.exists(args.archivo): sys.exit(f"[DETENIDO] No existe el archivo: {args.archivo}")
    ext = os.path.splitext(args.archivo)[1].lower()
    sep, dec = ",", "."
    if ext in (".csv", ".txt"): df, sep, dec = leer_csv(args.archivo)
    elif ext in (".xlsx", ".xlsm", ".xls"): df = leer_excel(args.archivo)
    else: sys.exit(f"[DETENIDO] Formato no soportado: {ext}")
    clon, clat = buscar_col(df, CANDIDATAS_LON, COL_LON), buscar_col(df, CANDIDATAS_LAT, COL_LAT)
    lon, lat = a_numero(df[clon]), a_numero(df[clat])
    print(f"Filas leídas: {len(df):,} | coordenadas: '{clon}' (lon), '{clat}' (lat) | sin coordenada válida: {int((lon.isna() | lat.isna()).sum()):,}")
    if lon.notna().any() and (lon.abs() > 180).mean() > 0.5:
        sys.exit("[DETENIDO] La columna de longitud tiene valores > 180: no parecen grados. Revise COL_LON/COL_LAT.")
    geom = cargar_poligono(args.shape, args.buffer_km, CRS_SI_FALTA_PRJ)
    m = dentro(lon.to_numpy(), lat.to_numpy(), geom)
    out_df = df[m].reset_index(drop=True)
    salida = args.salida or os.path.splitext(args.archivo)[0] + "_amazonia.csv"
    out_df.to_csv(salida, index=False, sep=sep, decimal=dec, encoding="utf-8-sig")   # mismo formato que el original
    print(f"\nFilas dentro de la Amazonía: {len(out_df):,} de {len(df):,} ({100 * len(out_df) / max(len(df), 1):.1f} %)")
    for col in ("neg_type", "label"):
        a, d = resumen(df, col), resumen(out_df, col)
        if a: print(f"{col}: " + " | ".join(f"{k}: {a[k]:,} -> {d.get(k, 0):,}" for k in a)); break
    print(f"Guardado: {salida}")

if __name__ == "__main__":
    main()