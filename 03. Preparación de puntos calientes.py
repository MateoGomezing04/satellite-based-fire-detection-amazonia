# -*- coding: utf-8 -*-
"""
preparar_puntos_ideam.py -- Confirma los puntos calientes del IDEAM con las áreas quemadas mensuales y deja el archivo
listo para extraer_goes16_limpio.py (--focos).

ENTRADA: puntos_ideam_2024_limpio.csv (salida de limpiar_puntos_ideam.py) + carpeta de shapes de área quemada mensual.

REGLA DE CONFIRMACIÓN (documentar en el artículo)
  1. Por punto: el punto cae dentro del área quemada de SU MES (con margen BUFFER_M por el tamaño del píxel;
     opcionalmente también del mes anterior/siguiente con --meses-vecinos).
  2. Por evento: si al menos un punto de un evento (puntos a <= 2 km y <= 2 días, de limpiar_puntos_ideam.py) quedó
     confirmado, se confirman también los demás puntos del evento que estén a <= DIST_EVENTO_KM del área quemada.
     Así no se pierden los puntos del borde del fuego, y un evento muy extenso no confirma puntos lejanos.
  3. estado:  confirmado    -> candidato a POSITIVO (si su confianza no es baja)
              no_confirmado -> NO es positivo; el extractor lo usa solo para no tomar negativos cerca
              sin_area      -> mes sin shape de área quemada (se trata como no_confirmado)

SALIDAS
  puntos_ideam_2024_confirmados.csv  -> para extraer_goes16_limpio.py: los no confirmados van con confianza 'low'
  puntos_ideam_2024_estado.csv       -> todos los puntos con estado, distancia al área quemada y confianza original
  resumen_confirmacion.txt

USO (PowerShell):  python preparar_puntos_ideam.py
Luego:             python extraer_goes16_limpio.py --focos "...\\puntos_ideam_2024_confirmados.csv" --rios "...\\rios.shp" --plan
Requiere: pip install pandas numpy geopandas shapely
"""
import os, sys, re, glob, argparse
import numpy as np
import pandas as pd

# ============================== RUTAS (EDITE AQUÍ) ==============================
PUNTOS_LIMPIOS = r"C:\Users\User\Desktop\Nuevo Manuscript\Puntos calientes\Puntos calientes IDEAM 2024.csv"
AREAS_DIR = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\2024\Areas\Areaa"
RECORTE = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\Amazonia.shp"   # None = sin recorte
SALIDA = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Puntos calientes\preparados"

BUFFER_M = 500             # margen alrededor del polígono para el punto (VIIRS 375 m, MODIS 1 km, área quemada 500 m)
DIST_EVENTO_KM = 5.0       # distancia máxima al área quemada para confirmar por evento
MESES_VECINOS = False

_MCD64 = re.compile(r"\.A(\d{4})(\d{3})\.", re.I)
MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
         "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}

def mes_de_shape(ruta):
    b = os.path.basename(ruta); m = _MCD64.search(b)
    if m: return (pd.Timestamp(year=int(m.group(1)), month=1, day=1) + pd.Timedelta(days=int(m.group(2)) - 1)).month
    hits = {v for k, v in MESES.items() if k in b.lower()}
    return hits.pop() if len(hits) == 1 else None

def confirmar(estado_punto, dist_km, evento, dist_evento_km):
    """Lógica pura. estado_punto: 1 dentro, 0 fuera, NaN sin shape. Devuelve (confirmado_por, estado)."""
    dentro = np.nan_to_num(np.asarray(estado_punto, float)) == 1
    ev = pd.Series(np.asarray(evento))
    ev_conf = ev.map(pd.Series(dentro).groupby(ev).any()).to_numpy()
    por_evento = ev_conf & ~dentro & (np.asarray(dist_km, float) <= dist_evento_km)
    conf = dentro | por_evento
    estado = np.where(conf, "confirmado", np.where(np.isnan(np.asarray(estado_punto, float)), "sin_area", "no_confirmado"))
    via = np.where(dentro, "punto", np.where(por_evento, "evento", ""))
    return via, estado

def main():
    ap = argparse.ArgumentParser(description="Confirma puntos calientes con áreas quemadas")
    ap.add_argument("--puntos", default=PUNTOS_LIMPIOS); ap.add_argument("--areas-dir", default=AREAS_DIR)
    ap.add_argument("--recorte", default=RECORTE); ap.add_argument("--out", default=SALIDA)
    ap.add_argument("--buffer-m", type=float, default=BUFFER_M); ap.add_argument("--dist-evento-km", type=float, default=DIST_EVENTO_KM)
    ap.add_argument("--meses-vecinos", action="store_true", default=MESES_VECINOS)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    import geopandas as gpd, shapely

    if not os.path.exists(args.puntos): sys.exit(f"[DETENIDO] No existe {args.puntos}. Corra primero limpiar_puntos_ideam.py")
    df = pd.read_csv(args.puntos)
    if "event_id" not in df.columns: sys.exit("[DETENIDO] El archivo no viene de limpiar_puntos_ideam.py (falta event_id).")
    df["_t"] = pd.to_datetime(df.datetime_utc); n0 = len(df)

    if args.recorte and str(args.recorte).lower() not in ("none", "ninguno"):
        r = gpd.read_file(args.recorte)
        if r.crs is None: sys.exit(f"[DETENIDO] {args.recorte} no tiene .prj")
        rg = r.to_crs("EPSG:4326").geometry.make_valid().union_all(); shapely.prepare(rg)
        df = df[shapely.intersects_xy(rg, df.lon.values, df.lat.values)].reset_index(drop=True)
    print(f"Puntos: {n0:,} | dentro del recorte: {len(df):,}")

    c = shapely.Point(df.lon.mean(), df.lat.mean())
    aeqd = f"+proj=aeqd +lat_0={c.y:.4f} +lon_0={c.x:.4f} +datum=WGS84 +units=m +no_defs"
    pts = gpd.GeoSeries(gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(aeqd).values
    lo0, la0, lo1, la1 = df.lon.min() - 0.5, df.lat.min() - 0.5, df.lon.max() + 0.5, df.lat.max() + 0.5
    shapes = {}
    for f in sorted(glob.glob(os.path.join(args.areas_dir, "*.shp"))):
        m = mes_de_shape(f)
        if m is None: print(f"  [AVISO] mes no reconocido: {os.path.basename(f)}"); continue
        g = gpd.read_file(f)
        if g.crs is None: sys.exit(f"[DETENIDO] {f} no tiene .prj")
        g = g.to_crs("EPSG:4326").cx[lo0:lo1, la0:la1]; g = g[g.geometry.notna() & ~g.geometry.is_empty]
        shapes[m] = g.to_crs(aeqd).geometry.make_valid().values
        print(f"  mes {m:02d}: {len(shapes[m]):,} polígonos ({os.path.basename(f)})")
    if not shapes: sys.exit(f"[DETENIDO] No hay shapes en {args.areas_dir}")

    mes = df._t.dt.month.to_numpy(); dist = np.full(len(df), np.inf); evaluado = np.zeros(len(df), bool)
    for m in range(1, 13):
        sel = np.flatnonzero(mes == m)
        if not len(sel): continue
        geoms = [x for k in ([m] + ([m - 1, m + 1] if args.meses_vecinos else [])) if k in shapes for x in shapes[k]]
        if m in shapes: evaluado[sel] = True
        if not geoms: continue
        tree = shapely.STRtree(np.asarray(geoms, dtype=object))
        (qi, _), d = tree.query_nearest(pts[sel], return_distance=True, all_matches=False, max_distance=(args.dist_evento_km + 1) * 1000)
        dist[sel[qi]] = d
    dist_km = dist / 1000
    en_area = np.where(evaluado, (dist_km <= args.buffer_m / 1000).astype(float), np.nan)
    via, estado = confirmar(en_area, dist_km, df.event_id.to_numpy(), args.dist_evento_km)
    df["dist_area_quemada_km"] = np.where(np.isfinite(dist_km), dist_km.round(3), np.nan)
    df["en_area_quemada"] = en_area; df["confirmado_por"] = via; df["estado"] = estado
    df["positivo_candidato"] = ((df.estado == "confirmado") & (df.retained == 1)).astype(int)
    df = df.drop(columns="_t")
    df.to_csv(os.path.join(args.out, "puntos_ideam_2024_estado.csv"), index=False)

    ext = df.copy(); no = ext.estado != "confirmado"
    ext.loc[no, "confidence_class"] = "low"; ext.loc[no, "confidence_pct"] = 0
    # extraer_goes16_limpio.py usa confidence_class para VIIRS y confidence_pct para MODIS: para MODIS deja la clase vacía
    modis = ext.sensor.eq("MODIS") & ~no
    ext.loc[modis, "confidence_class"] = np.nan
    ext.to_csv(os.path.join(args.out, "puntos_ideam_2024_confirmados.csv"), index=False)

    ev = df.groupby("event_id").estado.first()
    L = [f"Puntos analizados: {len(df):,}", "", "Estado de los puntos:", df.estado.value_counts().to_string(),
         "Confirmados por:", df[df.estado == "confirmado"].confirmado_por.value_counts().to_string(),
         f"Positivos candidatos (confirmados y confianza no baja): {int(df.positivo_candidato.sum()):,}", "",
         f"Eventos: {df.event_id.nunique():,} | eventos con algún punto confirmado: {int(df.groupby('event_id').estado.apply(lambda s: (s == 'confirmado').any()).sum()):,}", "",
         "Por mes:", df.groupby("mes").agg(puntos=("hotspot_id", "size"), confirmados=("estado", lambda s: int((s == "confirmado").sum())),
                                          positivos=("positivo_candidato", "sum")).assign(pct_confirmado=lambda t: (100 * t.confirmados / t.puntos).round(1)).to_string(), "",
         "Por sensor:", df.groupby("sensor").agg(puntos=("hotspot_id", "size"), confirmados=("estado", lambda s: int((s == "confirmado").sum()))).to_string(), "",
         "Confirmación según FRP (MW):", df.assign(frp=pd.cut(df.frp_mw, [0, 5, 10, 25, 50, 100, 1e4])).groupby("frp", observed=True)
            .estado.apply(lambda s: round(100 * (s == "confirmado").mean(), 1)).to_string(),
         "", f"Parámetros: BUFFER_M={args.buffer_m}, DIST_EVENTO_KM={args.dist_evento_km}, MESES_VECINOS={args.meses_vecinos}"]
    open(os.path.join(args.out, "resumen_confirmacion.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\nGuardado en {args.out}:\n  puntos_ideam_2024_confirmados.csv (para la extracción GOES)\n  puntos_ideam_2024_estado.csv\n  resumen_confirmacion.txt")

if __name__ == "__main__":
    main()