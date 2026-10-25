# -*- coding: utf-8 -*-
"""
descargar_goes_puntos_ideam.py -- Descarga GOES-16 (ABI) en las coordenadas y horas de los puntos calientes confirmados
del IDEAM y genera las variables para el entrenamiento (C07, C14, T7, T14, BTD, Ratio, ND, Cloud_Probabilities,
FireMask, FRP, LST, TPW, LST_BT_DIFF ...).

PENSADO PARA SPYDER: edite la sección CONFIGURACIÓN y ejecute con F5 (no necesita argumentos de consola).

MODOS (variable MODO):
  "check"    -> comprueba el acceso a NOAA (S3) y los nombres de las variables. No descarga imágenes completas.
  "plan"     -> cuenta cuántas imágenes GOES hay que procesar por mes. No descarga imágenes.
  "piloto"   -> procesa solo las fechas de FECHAS_PILOTO y luego une los resultados.
  "completo" -> procesa los meses de MESES y luego une los resultados.
  "unir"     -> solo une los CSV por imagen ya descargados (dataset_goes16_2024.csv + un CSV por mes).
Si la corrida se corta, ejecute de nuevo: las imágenes ya procesadas se saltan.

LÓGICA (decisiones metodológicas; documentar en el artículo)
  - Cada punto confirmado se asigna al escaneo GOES cuyo punto medio está más cerca de su hora (<= TOL_SCAN_MIN).
  - Positivo = píxel GOES que contiene un punto confirmado con confianza no baja.
  - Los puntos NO confirmados (marcados 'low' en el archivo) no son positivos, pero sí impiden negativos cercanos.
  - Negativos 'near' (NEAR_MIN_KM..NEAR_MAX_KM del punto más cercano de la imagen) y 'far' (> FAR_MIN_KM de todo
    punto de la imagen, >= RING_KM de todo punto a +-FAR_TIME_H h y <= FAR_MAX_KM de algún punto del año).
  - Se eliminan píxeles a <= WATER_KM de un río y píxeles fuera del recorte (Amazonía).
  - Filtros de calidad: DQF C07/C14 == 0, DQF nubes == 0, DQF fuego <= 1, DQF LST/TPW NaN o 0.
    OJO: exigir cielo despejado sesga la muestra a escenas sin nube.

INSTALACIÓN (Anaconda Prompt, una vez):
  conda install -c conda-forge geopandas pyproj xarray h5netcdf s3fs scipy
"""
import os, sys, re, json, time, hashlib, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
FOCOS = r"C:\Users\User\Desktop\Nuevo Manuscript\Puntos calientes\Puntos calientes\preparados\puntos_ideam_2024_confirmados.csv"
RIOS = r"C:\Users\User\Desktop\Manuscript JI\Insumos\SHP Amazonia\Division\Rios  Colombia\rios.shp"   # verifique la ruta
RECORTE = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\Amazonia.shp"                     # None = sin recorte
SALIDA = r"C:\Users\User\Desktop\Nuevo Manuscript\Extraccion GOES puntos calientes"
CRS_SI_FALTA_PRJ = None    # p. ej. "EPSG:4326", SOLO si un .shp no trae .prj y usted conoce su sistema

MODO = "completo"             # "check" -> "plan" -> "piloto" -> "completo"  (o "unir")
FECHAS_PILOTO = ["2024-02-23", "2024-01-15", "2024-03-10"]   # fechas UTC con muchos puntos confirmados
MESES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# ============================== PARÁMETROS ==============================
SATELLITE = "GOES-16"; BUCKET = "noaa-goes16"; LON_0 = -75.0
GOES_CRS = f"+proj=geos +h=35786023 +lon_0={LON_0} +sweep=x +datum=WGS84"
H_SAT = 35786023.0
PRODUCTS = {"rad": "ABI-L1b-RadF", "acm": "ABI-L2-ACMF", "fdc": "ABI-L2-FDCF",
            "lst": "ABI-L2-LSTF", "tpw": "ABI-L2-TPWF"}
VARS = {"c07": ("rad", ["Rad", "DQF", "planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"]),
        "c14": ("rad", ["Rad", "DQF", "planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"]),
        "acm": ("acm", ["Cloud_Probabilities", "DQF"]),
        "fdc": ("fdc", ["Mask", "Power", "DQF"]),
        "lst": ("lst", ["LST", "DQF"]),
        "tpw": ("tpw", ["TPW", "DQF_Overall"])}

TOL_SCAN_MIN = 10.0        # |hora del foco - punto medio del escaneo| máximo
TOL_COMPANION_MIN = 5.0    # C14/ACM/FDC deben empezar a <= 5 min del inicio de C07
TOL_LST_TPW_MIN = 35.0     # LST/TPW: escaneo más cercano dentro de esta tolerancia (cadencia por verificar en el piloto)
MODIS_MIN_PCT = 30         # MODIS con confianza < 30 % = baja (verificar contra la documentación de la fuente)
POS_MODE = "pixel"         # 'pixel' (píxel que contiene el foco) | 'radius'
POS_RADIUS_KM = 2.0        # solo si POS_MODE='radius'
NEAR_MIN_KM, NEAR_MAX_KM = 10.0, 30.0
FAR_MIN_KM, RING_KM, FAR_TIME_H, FAR_MAX_KM = 30.0, 5.0, 24.0, 60.0   # FAR_MAX_KM=60 es un parámetro SIN fuente
NEAR_PER_POS, FAR_PER_POS = 2.0, 2.0
WATER_KM = 2.0
MARGIN_KM = 70.0           # margen de la ventana de lectura alrededor de los focos de la imagen (>= FAR_MAX_KM)
FILTER_LST_DQF = True; FILTER_TPW_DQF = True
SEED = 42

# ============================== UTILIDADES ==============================
class Detenido(Exception):
    pass

def detener(msg):
    raise Detenido(msg)

_tok = re.compile(r"_s(\d{13})\d?_e(\d{13})\d?_")
def parse_scan_times(name):
    """OR_..._sYYYYDDDHHMMSSt_eYYYYDDDHHMMSSt_c... -> (inicio, fin) en UTC (pd.Timestamp)."""
    m = _tok.search(os.path.basename(name))
    if not m: raise ValueError(f"No se pudo leer s/e del nombre: {name}")
    f = lambda s: pd.Timestamp(year=int(s[:4]), month=1, day=1) + pd.Timedelta(
        days=int(s[4:7]) - 1, hours=int(s[7:9]), minutes=int(s[9:11]), seconds=int(s[11:13]))
    return f(m.group(1)), f(m.group(2))

def to_km_factory(lat0):
    kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
    return lambda lon, lat: np.column_stack([np.asarray(lon, float).ravel() * kx, np.asarray(lat, float).ravel() * ky])

def planck_bt(rad, fk1, fk2, bc1, bc2):
    with np.errstate(all="ignore"):
        return (fk2 / np.log(fk1 / rad + 1.0) - bc1) / bc2

# ============================== GRILLA GOES ==============================
class Grid:
    """Malla fija GOES de referencia (ángulos de escaneo en rad). índice <-> lon/lat vía pyproj."""
    def __init__(self, x, y, transformer_fwd=None, transformer_inv=None):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.nx, self.ny = len(self.x), len(self.y)
        self.dx = (self.x[-1] - self.x[0]) / (self.nx - 1); self.dy = (self.y[-1] - self.y[0]) / (self.ny - 1)
        if transformer_fwd is None:
            from pyproj import Transformer
            transformer_fwd = Transformer.from_crs("EPSG:4326", GOES_CRS, always_xy=True)
            transformer_inv = Transformer.from_crs(GOES_CRS, "EPSG:4326", always_xy=True)
        self.fwd, self.inv = transformer_fwd, transformer_inv
    def lonlat_to_index(self, lon, lat):
        X, Y = self.fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        ix = np.rint((np.asarray(X) / H_SAT - self.x[0]) / self.dx); iy = np.rint((np.asarray(Y) / H_SAT - self.y[0]) / self.dy)
        ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ix = np.where(ok, ix, -1).astype(np.int64); iy = np.where(ok, iy, -1).astype(np.int64)
        return ix, iy, ok
    def index_to_lonlat(self, ix, iy):
        lon, lat = self.inv.transform(self.x[ix] * H_SAT, self.y[iy] * H_SAT)
        return np.asarray(lon), np.asarray(lat)

def nearest_index(ref_coord, prod_coord):
    """Para cada valor de ref_coord, índice del valor más cercano en prod_coord (monótono, asc. o desc.)."""
    prod_coord = np.asarray(prod_coord, float); sgn = 1.0 if prod_coord[-1] >= prod_coord[0] else -1.0
    p = prod_coord * sgn; r = np.asarray(ref_coord, float) * sgn
    j = np.clip(np.searchsorted(p, r), 1, len(p) - 1)
    j = j - (np.abs(r - p[j - 1]) <= np.abs(r - p[j]))
    return j.astype(np.int64)

# ============================== AGUA Y RECORTE ==============================
def _leer_shape(ruta, crs_override=None, que="shape"):
    import geopandas as gpd
    if not ruta or not os.path.exists(ruta): detener(f"No existe el {que}: {ruta}")
    g = gpd.read_file(ruta)
    if g.crs is None:
        if not crs_override:
            detener(f"{ruta} no tiene CRS (.prj ausente). Copie el .prj junto al .shp o defina CRS_SI_FALTA_PRJ "
                    "SOLO si conoce el sistema de coordenadas.")
        g = g.set_crs(crs_override)
    g = g.to_crs("EPSG:4326"); g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.make_valid()
    return g

def mascara_agua(lon, lat, ruta_rios, crs_override=None, water_km=WATER_KM):
    """True donde el centro del píxel está a <= water_km de un río (línea) o dentro de un río (polígono)."""
    import shapely
    from pyproj import Transformer
    g = _leer_shape(ruta_rios, crs_override, "shape de ríos").explode(index_parts=False)
    lo, la = np.asarray(lon, float).ravel(), np.asarray(lat, float).ravel(); ok = np.isfinite(lo) & np.isfinite(la)
    aeqd = f"+proj=aeqd +lat_0={np.nanmean(la):.4f} +lon_0={np.nanmean(lo):.4f} +datum=WGS84 +units=km +no_defs"
    geoms = g.to_crs(aeqd).geometry.values
    X, Y = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True).transform(lo[ok], la[ok])
    tree = shapely.STRtree(geoms)
    (qi, _), d = tree.query_nearest(shapely.points(X, Y), return_distance=True, all_matches=False, max_distance=water_km + 0.5)
    out = np.zeros(lo.shape, bool); idx = np.flatnonzero(ok)[qi]; out[idx] = d <= water_km
    return out.reshape(np.shape(lon))

def mascara_recorte(lon, lat, ruta, crs_override=None):
    import shapely
    if not ruta or str(ruta).lower() in ("none", "ninguno"): return np.isfinite(lon)
    geom = _leer_shape(ruta, crs_override, "shape de recorte").geometry.union_all(); shapely.prepare(geom)
    lo, la = np.asarray(lon, float).ravel(), np.asarray(lat, float).ravel(); ok = np.isfinite(lo) & np.isfinite(la)
    out = np.zeros(lo.shape, bool); out[ok] = shapely.intersects_xy(geom, lo[ok], la[ok])
    return out.reshape(np.shape(lon))

# ============================== ETIQUETADO (lógica pura, probada con datos sintéticos) ==============================
def build_labels(valid, XK, YK, pos_flat, tree_img, tree_t24, tree_all, rng, P=None):
    """valid: (ny,nx) píxeles que pasan calidad y no son agua. XK,YK: centros en km. pos_flat: índices planos
    de positivos ya validados. Devuelve índices planos pos/near/far y distancias."""
    P = P or {}
    g = lambda k, d: P.get(k, d)
    vflat = np.flatnonzero(valid.ravel())
    pv = np.column_stack([XK.ravel()[vflat], YK.ravel()[vflat]])
    d_img, nn = tree_img.query(pv)
    is_pos = np.isin(vflat, pos_flat)
    near_c = (~is_pos) & (d_img >= g("NEAR_MIN_KM", NEAR_MIN_KM)) & (d_img <= g("NEAR_MAX_KM", NEAR_MAX_KM))
    far_c = np.zeros(len(vflat), bool)
    cand = np.flatnonzero((~is_pos) & (d_img > g("FAR_MIN_KM", FAR_MIN_KM)))
    if len(cand):
        d24, _ = tree_t24.query(pv[cand]); ok = d24 >= g("RING_KM", RING_KM)
        cand = cand[ok]
        if len(cand):
            dall, _ = tree_all.query(pv[cand], distance_upper_bound=g("FAR_MAX_KM", FAR_MAX_KM))
            far_c[cand[np.isfinite(dall)]] = True
    npos = len(pos_flat)
    n_near = int(round(g("NEAR_PER_POS", NEAR_PER_POS) * npos)); n_far = int(round(g("FAR_PER_POS", FAR_PER_POS) * npos))
    pick = lambda mask, n: (rng.choice(np.flatnonzero(mask), size=min(n, int(mask.sum())), replace=False) if mask.any() and n > 0 else np.array([], int))
    s_near, s_far = pick(near_c, n_near), pick(far_c, n_far)
    return dict(pos=pos_flat, near=vflat[s_near], far=vflat[s_far], d_img_near=d_img[s_near], d_img_far=d_img[s_far],
                nn_near=nn[s_near], nn_far=nn[s_far], cand_near=int(near_c.sum()), cand_far=int(far_c.sum()))

# ============================== S3 ==============================
class S3Goes:
    def __init__(self, cache_path):
        import s3fs
        self.fs = s3fs.S3FileSystem(anon=True, default_fill_cache=False, default_block_size=2 * 1024 * 1024)
        self.cache_path = cache_path; self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self._dirty = 0
    def _list(self, product, ts):
        key = f"{product}/{ts.year}/{ts.dayofyear:03d}/{ts.hour:02d}"
        if key not in self.cache:
            for attempt in range(3):
                try: names = [p for p in self.fs.ls(f"{BUCKET}/{key}/") if p.endswith(".nc")]; break
                except FileNotFoundError: names = []; break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(2 * (attempt + 1))
            self.cache[key] = names; self._dirty += 1
            if self._dirty >= 200: self.save()
        return self.cache[key]
    def save(self):
        json.dump(self.cache, open(self.cache_path, "w")); self._dirty = 0
    def scans(self, product, ts, band=None):
        """Escaneos de la hora de ts y de las horas vecinas -> lista (inicio, fin, ruta) ordenada y sin duplicar."""
        out = {}
        for dh in (-1, 0, 1):
            for p in self._list(product, ts.floor("h") + pd.Timedelta(hours=dh)):
                if band and f"C{band:02d}_G16" not in os.path.basename(p): continue
                s, e = parse_scan_times(p); out[s] = (s, e, p)
        return [out[k] for k in sorted(out)]

def assign_scans(hot, s3):
    """Asigna cada foco al escaneo C07 cuyo punto medio está más cerca. Devuelve (scan_start, offset_min) por foco."""
    t = pd.to_datetime(hot["datetime_utc"]); start = np.full(len(hot), np.datetime64("NaT"), "datetime64[ns]"); off = np.full(len(hot), np.nan)
    fl = t.dt.floor("h")
    for h, grp in hot.groupby(fl):
        cands = s3.scans(PRODUCTS["rad"], h, band=7)
        if not cands: continue
        mids = np.array([(s + (e - s) / 2).value for s, e, _ in cands], dtype="int64")
        ht = pd.to_datetime(grp["datetime_utc"]).astype("datetime64[ns]").astype("int64").to_numpy()
        diff = (ht[:, None] - mids[None, :]) / 6e10        # minutos
        j = np.abs(diff).argmin(axis=1); o = diff[np.arange(len(j)), j]
        pos = hot.index.get_indexer(grp.index)
        start[pos] = np.array([cands[k][0].to_datetime64() for k in j]); off[pos] = o
    ok = np.isfinite(off) & (np.abs(off) <= TOL_SCAN_MIN)
    start[~ok] = np.datetime64("NaT"); off[~ok] = np.nan
    return pd.Series(start, index=hot.index), pd.Series(off, index=hot.index)

def pick_companion(s3, product, ts_start, tol_min, band=None):
    c = s3.scans(product, ts_start, band=band)
    if not c: return None, np.nan
    d = np.array([(s - ts_start).total_seconds() / 60 for s, _, _ in c]); j = np.abs(d).argmin()
    return (c[j][2], float(d[j])) if abs(d[j]) <= tol_min else (None, np.nan)

def read_product(s3, path, names, ref_grid, win, mapcache):
    """Lee `names` en la ventana win=(iy0,iy1,ix0,ix1) de la malla de referencia. Si la malla del producto
    difiere, usa el vecino más cercano (se marca en 'mapping')."""
    import xarray as xr
    iy0, iy1, ix0, ix1 = win
    with s3.fs.open(path, "rb") as f, xr.open_dataset(f, engine="h5netcdf") as ds:
        missing = [n for n in names if n not in ds.variables]
        if missing: raise KeyError(f"{os.path.basename(path)}: faltan variables {missing}. Disponibles: {list(ds.variables)}")
        px, py = np.asarray(ds["x"].values, float), np.asarray(ds["y"].values, float)
        sig = (len(px), len(py), round(px[0], 9), round(py[0], 9), round(px[-1], 9))
        if sig not in mapcache:
            same = (len(px) == ref_grid.nx and len(py) == ref_grid.ny and np.abs(px - ref_grid.x).max() < 0.01 * abs(ref_grid.dx)
                    and np.abs(py - ref_grid.y).max() < 0.01 * abs(ref_grid.dy))
            mapcache[sig] = ("identica", None, None) if same else ("vecino", nearest_index(ref_grid.x, px), nearest_index(ref_grid.y, py))
        kind, cmap, rmap = mapcache[sig]
        cols = np.arange(ix0, ix1) if cmap is None else cmap[ix0:ix1]; rows = np.arange(iy0, iy1) if rmap is None else rmap[iy0:iy1]
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        out = {"mapping": kind}
        for n in names:
            da = ds[n]
            if da.ndim == 0: out[n] = float(da.values); continue
            arr = np.asarray(da.isel(y=slice(int(r0), int(r1)), x=slice(int(c0), int(c1))).values)
            out[n] = arr[np.ix_(rows - r0, cols - c0)].astype(np.float64)
    return out

# ============================== PROCESO POR IMAGEN ==============================
def process_scan(start, hot_img, ctx):
    """hot_img: todos los focos asignados a esta imagen (cualquier confianza). Devuelve (DataFrame, dict_log)."""
    t0 = time.time(); grid, reg, P = ctx["grid"], ctx["region"], ctx["P"]
    mid = None; log = dict(image_id=f"G16_{start:%Y%j%H%M%S}", scan_start_utc=str(start), n_hot_img=len(hot_img))
    paths = {}
    path07, _ = pick_companion(ctx["s3"], PRODUCTS["rad"], start, 0.5, band=7)
    if path07 is None: log["status"] = "sin_C07"; return None, log
    s7, e7 = parse_scan_times(path07); mid = s7 + (e7 - s7) / 2
    paths["c07"] = path07
    for k, band in (("c14", 14),):
        p, d = pick_companion(ctx["s3"], PRODUCTS["rad"], start, TOL_COMPANION_MIN, band=band); paths[k] = p
    for k in ("acm", "fdc"):
        p, d = pick_companion(ctx["s3"], PRODUCTS[k], start, TOL_COMPANION_MIN); paths[k] = p; log[f"{k}_offset_min"] = d
    for k in ("lst", "tpw"):
        p, d = pick_companion(ctx["s3"], PRODUCTS[k], start, TOL_LST_TPW_MIN); paths[k] = p; log[f"{k}_offset_min"] = d
    core_missing = [k for k in ("c14", "acm", "fdc") if paths[k] is None]
    if core_missing: log["status"] = "faltan_" + "_".join(core_missing); return None, log

    # focos: índices, agua, retención
    ix, iy, ok = grid.lonlat_to_index(hot_img["lon"].to_numpy(), hot_img["lat"].to_numpy())
    h = hot_img.assign(ix=ix, iy=iy, ingrid=ok)
    h = h[h.ingrid]
    ret = h[h.retained & ~h.water_hot]
    log["n_hot_retained"] = int(h.retained.sum()); log["n_hot_water"] = int((h.retained & h.water_hot).sum())
    if ret.empty: log["status"] = "sin_focos_retenidos"; return None, log

    m = int(np.ceil(MARGIN_KM / 1.9))
    iy0 = max(reg["iy0"], h.iy.min() - m); iy1 = min(reg["iy1"], h.iy.max() + m + 1)
    ix0 = max(reg["ix0"], h.ix.min() - m); ix1 = min(reg["ix1"], h.ix.max() + m + 1)
    win = (int(iy0), int(iy1), int(ix0), int(ix1)); sl = (slice(iy0 - reg["iy0"], iy1 - reg["iy0"]), slice(ix0 - reg["ix0"], ix1 - reg["ix0"]))
    lon, lat, water, en_rec = reg["lon"][sl], reg["lat"][sl], reg["water"][sl], reg["en_recorte"][sl]

    D = {}; consts = {}
    for key in ("c07", "c14"):
        r = read_product(ctx["s3"], paths[key], VARS[key][1], grid, win, ctx["mapcache"]); consts[key] = {c: r[c] for c in r if c.startswith("planck")}; D[key] = r
    D["acm"] = read_product(ctx["s3"], paths["acm"], VARS["acm"][1], grid, win, ctx["mapcache"])
    D["fdc"] = read_product(ctx["s3"], paths["fdc"], VARS["fdc"][1], grid, win, ctx["mapcache"])
    shp = lon.shape; nanarr = np.full(shp, np.nan)
    D["lst"] = read_product(ctx["s3"], paths["lst"], VARS["lst"][1], grid, win, ctx["mapcache"]) if paths["lst"] else {"LST": nanarr, "DQF": nanarr}
    D["tpw"] = read_product(ctx["s3"], paths["tpw"], VARS["tpw"][1], grid, win, ctx["mapcache"]) if paths["tpw"] else {"TPW": nanarr, "DQF_Overall": nanarr}
    log["tpw_mapping"] = D["tpw"].get("mapping", "NA"); log["lst_mapping"] = D["lst"].get("mapping", "NA")

    C07, C14 = D["c07"]["Rad"], D["c14"]["Rad"]; DQF7, DQF14 = D["c07"]["DQF"], D["c14"]["DQF"]
    CP, DQFC = D["acm"]["Cloud_Probabilities"], D["acm"]["DQF"]
    MASK, POW, DQFF = D["fdc"]["Mask"], D["fdc"]["Power"], D["fdc"]["DQF"]
    LST, DQFL, TPW, DQFT = D["lst"]["LST"], D["lst"]["DQF"], D["tpw"]["TPW"], D["tpw"]["DQF_Overall"]
    c7, c14c = consts["c07"], consts["c14"]
    T7 = planck_bt(C07, c7["planck_fk1"], c7["planck_fk2"], c7["planck_bc1"], c7["planck_bc2"])
    T14 = planck_bt(C14, c14c["planck_fk1"], c14c["planck_fk2"], c14c["planck_bc1"], c14c["planck_bc2"])
    BTD = T7 - T14
    with np.errstate(all="ignore"):
        RATIO = T7 / (T14 + 1e-6); den = T7 + T14; den = np.where(np.abs(den) > 1e-6, den, 1e-6); ND = (T7 - T14) / den
        valid = (np.isfinite(C07) & np.isfinite(C14) & np.isfinite(CP) & (DQF7 == 0) & (DQF14 == 0) & (DQFC == 0) & (DQFF <= 1)
                 & (C07 > 0) & (C14 > 0) & np.isfinite(T7) & np.isfinite(T14) & np.isfinite(BTD) & np.isfinite(RATIO) & np.isfinite(ND) & ~water & en_rec)
        if FILTER_LST_DQF: valid &= (np.isnan(DQFL) | (DQFL == 0))
        if FILTER_TPW_DQF: valid &= (np.isnan(DQFT) | (DQFT == 0))
    log["n_valid_window"] = int(valid.sum())

    to_km = ctx["to_km"]; XYK = to_km(lon, lat); XK, YK = XYK[:, 0].reshape(shp), XYK[:, 1].reshape(shp)
    hk_img = to_km(h.lon, h.lat); tree_img = cKDTree(hk_img)
    t_c = mid.value; lo, hi = t_c - int(FAR_TIME_H * 3.6e12), t_c + int(FAR_TIME_H * 3.6e12)
    a, b = np.searchsorted(ctx["hot_t_ns"], [lo, hi]); tree_t24 = cKDTree(ctx["hot_km"][a:b])

    # positivos
    r_iy, r_ix = ret.iy.to_numpy() - win[0], ret.ix.to_numpy() - win[2]
    inwin = (r_iy >= 0) & (r_iy < shp[0]) & (r_ix >= 0) & (r_ix < shp[1])
    if POS_MODE == "pixel":
        pos_flat_all = r_iy[inwin] * shp[1] + r_ix[inwin]
    else:
        tr = cKDTree(to_km(ret.lon, ret.lat)); vf = np.flatnonzero(valid.ravel())
        dd, _ = tr.query(np.column_stack([XK.ravel()[vf], YK.ravel()[vf]])); pos_flat_all = vf[dd <= POS_RADIUS_KM]
    vflat = valid.ravel()
    pos_flat = np.unique(pos_flat_all[vflat[pos_flat_all]])
    log["n_hot_pos_invalid_quality"] = int(len(np.unique(pos_flat_all)) - len(pos_flat))
    if len(pos_flat) == 0: log["status"] = "sin_positivos_validos"; return None, log

    rng = np.random.default_rng([SEED, int(start.value % (2**31))])
    L = build_labels(valid, XK, YK, pos_flat, tree_img, tree_t24, ctx["tree_all"], rng, P)
    flat = np.concatenate([L["pos"], L["near"], L["far"]])
    label = np.concatenate([np.ones(len(L["pos"])), np.zeros(len(L["near"]) + len(L["far"]))]).astype(int)
    neg_type = np.array(["pos"] * len(L["pos"]) + ["near"] * len(L["near"]) + ["far"] * len(L["far"]))
    d_nearest = np.concatenate([np.full(len(L["pos"]), np.nan), L["d_img_near"], L["d_img_far"]])
    nn_all = np.concatenate([np.full(len(L["pos"]), -1), L["nn_near"], L["nn_far"]]).astype(int)
    rr, cc = np.unravel_index(flat, shp)
    # para positivos: foco retenido dentro del píxel (o el más cercano)
    hot_nn = h.reset_index(drop=True)
    hot_lon, hot_lat = hot_nn.lon.to_numpy(), hot_nn.lat.to_numpy()
    d_pos, nn_pos = cKDTree(hk_img).query(np.column_stack([XK.ravel()[L["pos"]], YK.ravel()[L["pos"]]]))
    nn_all[:len(L["pos"])] = nn_pos; d_nearest[:len(L["pos"])] = d_pos
    # agregación de focos por píxel positivo
    pos_info = {}
    key_pos = (ret.iy.to_numpy() - win[0]) * shp[1] + (ret.ix.to_numpy() - win[2])
    for fl, grp in ret.assign(_k=key_pos).groupby("_k"):
        pos_info[fl] = dict(hotspot_ids=";".join(map(str, grp.hotspot_id)), n_hotspots=len(grp), cluster_ids=";".join(map(str, sorted(set(grp.cluster_id_2km)))),
                            overpass_id=int(grp.overpass_id.iloc[0]), source=" + ".join(sorted(set(grp.source))), confidence_class=grp.confidence_class.dropna().iloc[0] if grp.confidence_class.notna().any() else np.nan,
                            confidence_pct=grp.confidence_pct.max(), frp_mw=grp.frp_mw.max(), day_night=grp.day_night.iloc[0],
                            hotspot_offset_min=float(((pd.to_datetime(grp.datetime_utc) - mid).dt.total_seconds() / 60).mean()))
    sel = lambda A: A[rr, cc]
    ixg, iyg = cc + win[2], rr + win[0]
    frp = pd.Series(sel(POW)).replace([-999, -1e30], np.nan).clip(lower=0)
    df = pd.DataFrame({
        "image_id": log["image_id"], "scan_start_utc": str(start), "scan_mid_utc": str(mid), "fire_datetime_utc": str(mid),
        "fire_hour": mid.hour, "satellite": SATELLITE, "label": label, "neg_type": neg_type,
        "x": lon[rr, cc], "y": lat[rr, cc], "pix_ix": ixg, "pix_iy": iyg, "pixel_id": [f"{a}_{b}" for a, b in zip(ixg, iyg)],
        "nearest_hotspot_km": d_nearest, "fire_latitude": hot_lat[nn_all], "fire_longitude": hot_lon[nn_all],
        "C07": sel(C07), "C14": sel(C14), "DQF7": sel(DQF7), "DQF14": sel(DQF14),
        "Cloud_Probabilities": sel(CP), "DQF_cloud": sel(DQFC), "Cloud_Weight": 1 - sel(CP),
        "FireMask": sel(MASK), "FRP": frp.values, "DQF_fire": sel(DQFF),
        "LST": sel(LST), "DQF_LST": sel(DQFL), "TPW": sel(TPW), "DQF_TPW": sel(DQFT),
        "T7": sel(T7), "T14": sel(T14), "BTD": sel(BTD), "Ratio": sel(RATIO), "ND": sel(ND)})
    df["FRP_detected"] = df["FRP"].notna().astype(int); df["FRP_log"] = np.log1p(df["FRP"]); df["LST_BT_DIFF"] = df["LST"] - df["T14"]
    df["lst_disponible"] = int(paths["lst"] is not None); df["tpw_disponible"] = int(paths["tpw"] is not None)
    for k in ("fdc", "lst", "tpw"): df[f"{k}_offset_min"] = log[f"{k}_offset_min"]
    extra_cols = ["hotspot_ids", "n_hotspots", "cluster_ids", "overpass_id", "source", "confidence_class", "confidence_pct", "frp_mw", "day_night", "hotspot_offset_min"]
    for c in extra_cols: df[c] = np.nan if c not in ("hotspot_ids", "cluster_ids", "source", "confidence_class", "day_night") else ""
    for i, fl in enumerate(flat[:len(L["pos"])]):
        for c, v in pos_info.get(fl, {}).items(): df.loc[i, c] = v
    med_off = float(((pd.to_datetime(h.datetime_utc) - mid).dt.total_seconds() / 60).median())
    df.loc[df.label == 0, "hotspot_offset_min"] = med_off
    log.update(status="ok", n_pos=int(len(L["pos"])), n_near=int(len(L["near"])), n_far=int(len(L["far"])), cand_near=L["cand_near"], cand_far=L["cand_far"],
               scan_mid_utc=str(mid), seconds=round(time.time() - t0, 1))
    return df, log

# ============================== PROGRAMA PRINCIPAL ==============================
def load_hotspots(path):
    hot = pd.read_csv(path)
    need = ["hotspot_id", "datetime_utc", "lat", "lon", "source", "confidence_class", "confidence_pct", "frp_mw", "day_night", "overpass_id", "cluster_id_2km"]
    miss = [c for c in need if c not in hot.columns]
    if miss: detener(f"Faltan columnas en el CSV de focos: {miss}. Ejecute antes preparar_focos.py. Hay: {list(hot.columns)}")
    hot["datetime_utc"] = pd.to_datetime(hot["datetime_utc"]).astype("datetime64[ns]")   # ns explícito (pandas >= 3 usa us)
    hot["retained"] = ((hot.confidence_class.notna() & (hot.confidence_class != "low")) | (hot.confidence_class.isna() & (hot.confidence_pct >= MODIS_MIN_PCT)))
    hot = hot.sort_values("datetime_utc").reset_index(drop=True)
    if hot.datetime_utc.max() >= pd.Timestamp("2025-04-08"): detener("Hay focos posteriores al cambio a GOES-19; este script es solo GOES-16.")
    return hot

def build_context(args, hot, s3):
    """Malla de referencia, máscara de agua y cache de la región."""
    import xarray as xr
    c = s3.scans(PRODUCTS["rad"], hot.datetime_utc.iloc[len(hot) // 2], band=7)
    with s3.fs.open(c[0][2], "rb") as f, xr.open_dataset(f, engine="h5netcdf") as ds:
        grid = Grid(ds["x"].values, ds["y"].values)
    to_km = to_km_factory(float(hot.lat.mean()))
    ix, iy, ok = grid.lonlat_to_index(hot.lon.to_numpy(), hot.lat.to_numpy())
    assert ok.all(), f"{(~ok).sum()} focos caen fuera del disco GOES"
    m = int(np.ceil((MARGIN_KM + 5) / 1.9))
    reg = dict(iy0=int(max(0, iy.min() - m)), iy1=int(min(grid.ny, iy.max() + m + 1)), ix0=int(max(0, ix.min() - m)), ix1=int(min(grid.nx, ix.max() + m + 1)))
    rec_mt = os.path.getmtime(args.recorte) if args.recorte and os.path.exists(str(args.recorte)) else 0
    sig = hashlib.md5(json.dumps([reg, WATER_KM, os.path.getmtime(args.rios), str(args.recorte), rec_mt, round(grid.x[0], 9), round(grid.y[0], 9), grid.nx]).encode()).hexdigest()
    cache = os.path.join(args.out, "cache_malla_agua.npz")
    z = np.load(cache, allow_pickle=True) if os.path.exists(cache) else None
    if z is not None and str(z["sig"]) == sig and "en_recorte" in z.files:
        lon, lat, water, en_rec = z["lon"], z["lat"], z["water"], z["en_recorte"]; print("Cache de malla, agua y recorte: cargado.")
    else:
        print("Calculando malla lon/lat y máscara de agua (una vez)...")
        II, JJ = np.meshgrid(np.arange(reg["ix0"], reg["ix1"]), np.arange(reg["iy0"], reg["iy1"]))
        lon, lat = grid.index_to_lonlat(II.ravel(), JJ.ravel()); lon, lat = lon.reshape(II.shape), lat.reshape(II.shape)
        # verificación ida y vuelta (detecta errores de índice/proyección)
        s = np.random.default_rng(0).choice(lon.size, 2000, replace=False)
        i2, j2, ok2 = grid.lonlat_to_index(lon.ravel()[s], lat.ravel()[s])
        assert ok2.all() and (i2 == II.ravel()[s]).all() and (j2 == JJ.ravel()[s]).all(), "Ida y vuelta de índices GOES falló"
        water = mascara_agua(lon, lat, args.rios, args.rios_crs)
        en_rec = mascara_recorte(lon, lat, args.recorte, args.rios_crs)
        np.savez_compressed(cache, lon=lon, lat=lat, water=water, en_recorte=en_rec, sig=sig)
        print(f"  píxeles región: {water.size:,} | agua (<= {WATER_KM} km de un río): {water.mean():.1%} | dentro del recorte: {en_rec.mean():.1%}")
    return grid, dict(reg, lon=lon, lat=lat, water=water, en_recorte=en_rec), to_km

class _Args:
    pass

def ejecutar(modo=None):
    args = _Args(); args.focos, args.rios, args.rios_crs, args.out, args.recorte = FOCOS, RIOS, CRS_SI_FALTA_PRJ, SALIDA, RECORTE
    modo = (modo or MODO).lower()
    args.dates = FECHAS_PILOTO if modo == "piloto" else None
    args.months = MESES if modo == "completo" else None
    os.makedirs(args.out, exist_ok=True)
    print(f"MODO: {modo} | salida: {args.out}")
    if modo == "unir": return merge(args.out)
    for nombre, ruta in (("archivo de puntos", args.focos), ("shape de ríos", args.rios)):
        if not os.path.exists(ruta): detener(f"No existe el {nombre}: {ruta}")
    if args.recorte and str(args.recorte).lower() not in ("none", "ninguno") and not os.path.exists(args.recorte):
        detener(f"No existe el shape de recorte: {args.recorte} (ponga RECORTE = None para no recortar)")
    hot = load_hotspots(args.focos)
    print(f"Puntos en el archivo: {len(hot):,} | candidatos a positivo (confirmados, confianza no baja): {int(hot.retained.sum()):,}")
    s3 = S3Goes(os.path.join(args.out, "listados_s3.json"))
    if modo == "check": return check(hot, s3)
    if modo not in ("plan", "piloto", "completo"): detener(f"MODO no reconocido: {modo}")
    print("Asignando cada punto a su escaneo GOES (lista archivos en S3; la primera vez tarda)...")
    start, off = assign_scans(hot, s3); s3.save()
    hot["scan_start"] = start; hot["offset_min"] = off
    print(f"Puntos: {len(hot):,} | sin escaneo dentro de {TOL_SCAN_MIN} min: {int(start.isna().sum())} | escaneos distintos: {start.nunique()}")
    print(f"Desfase punto-escaneo (min): mediana {off.median():.1f}, p5 {off.quantile(.05):.1f}, p95 {off.quantile(.95):.1f}")
    hot = hot[hot.scan_start.notna()].copy()
    if args.dates: hot = hot[hot.scan_start.dt.strftime("%Y-%m-%d").isin(args.dates)]
    if args.months: hot = hot[hot.scan_start.dt.month.isin(args.months)]
    # solo hacen falta imágenes donde haya al menos un positivo candidato
    con_pos = hot.groupby("scan_start").retained.transform("any"); hot = hot[con_pos]
    nscan = hot.scan_start.nunique(); print(f"Escaneos a procesar (con al menos un punto confirmado): {nscan}")
    if modo == "plan":
        print(hot.groupby(hot.scan_start.dt.month).agg(escaneos=("scan_start", "nunique"), positivos=("retained", "sum")).to_string()); return
    if nscan == 0: detener("No hay escaneos para procesar con esa selección (revise FECHAS_PILOTO / MESES).")
    hot_all = load_hotspots(args.focos)
    grid, region, to_km = build_context(args, hot_all, s3)
    ixh, iyh, okh = grid.lonlat_to_index(hot.lon.to_numpy(), hot.lat.to_numpy())
    hot["water_hot"] = region["water"][iyh - region["iy0"], ixh - region["ix0"]]
    ctx = dict(s3=s3, grid=grid, region=region, to_km=to_km, P={}, mapcache={},
               hot_t_ns=hot_all.datetime_utc.astype("int64").to_numpy(), hot_km=to_km(hot_all.lon, hot_all.lat), tree_all=cKDTree(to_km(hot_all.lon, hot_all.lat)))
    imgdir = os.path.join(args.out, "imagenes"); os.makedirs(imgdir, exist_ok=True)
    logp = os.path.join(args.out, "extraccion_log.csv"); t_ini = time.time()
    for k, (st, g) in enumerate(hot.groupby("scan_start"), 1):
        fn = os.path.join(imgdir, f"G16_{st:%Y%j%H%M%S}.csv")
        if os.path.exists(fn): continue
        try:
            df, lg = process_scan(st, g.reset_index(drop=True), ctx)
        except Exception as e:
            df, lg = None, dict(image_id=f"G16_{st:%Y%j%H%M%S}", status=f"ERROR {type(e).__name__}: {e}"); print("  ", lg["status"])
        if not lg["status"].startswith("ERROR"):          # con ERROR no se escribe archivo -> se reintenta en la próxima corrida
            (df if df is not None else pd.DataFrame()).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
        pd.DataFrame([lg]).to_csv(logp, mode="a", header=not os.path.exists(logp), index=False)
        if k % 10 == 0 or k == nscan:
            print(f"[{k}/{nscan}] {lg['image_id']} {lg['status']} pos={lg.get('n_pos', 0)} near={lg.get('n_near', 0)} far={lg.get('n_far', 0)} "
                  f"| {(time.time() - t_ini) / 60:.1f} min"); s3.save()
    s3.save(); print("Descarga terminada. Uniendo resultados...")
    return merge(args.out)

def check(hot, s3):
    """Imprime variables reales de un archivo de cada producto y comprueba los nombres que usa el script."""
    import xarray as xr
    ts = hot.datetime_utc.iloc[len(hot) // 2]; print("Hora de prueba:", ts)
    for key, (prod, names) in VARS.items():
        band = {"c07": 7, "c14": 14}.get(key)
        c = s3.scans(PRODUCTS[prod], ts, band=band)
        if not c: print(f"[{key}] SIN ARCHIVOS en S3 para esa hora"); continue
        with s3.fs.open(c[0][2], "rb") as f, xr.open_dataset(f, engine="h5netcdf") as ds:
            miss = [n for n in names if n not in ds.variables]
            print(f"[{key}] {os.path.basename(c[0][2])}\n   dims={dict(ds.sizes)} | variables={list(ds.data_vars)}\n   {'OK' if not miss else 'FALTAN: ' + str(miss)}")
            for dq in ("DQF", "DQF_Overall"):
                if dq in ds.variables: print(f"   {dq} atributos:", {a: ds[dq].attrs[a] for a in ds[dq].attrs if a in ("flag_values", "flag_meanings", "long_name")})
    s3.save()

def merge(out):
    d = os.path.join(out, "imagenes")
    if not os.path.isdir(d): detener(f"No hay carpeta de imágenes en {out}: primero ejecute 'piloto' o 'completo'.")
    fs_ = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".csv")]
    parts = [pd.read_csv(f) for f in fs_ if os.path.getsize(f) > 5]
    if not parts: detener("Las imágenes procesadas no tienen datos (revise extraccion_log.csv, columna status).")
    df = pd.concat(parts, ignore_index=True)
    key = ["image_id", "pixel_id"]
    assert not df.duplicated(key).any(), "Hay píxel-imagen repetidos"
    L = []; p = lambda s="": (L.append(s), print(s))
    p(f"Imágenes con datos: {len(parts)} de {len(fs_)} archivos | filas: {len(df):,}")
    p("Etiquetas / tipo:"); p(df.groupby(["label", "neg_type"]).size().to_string())
    p(f"Píxeles únicos: {df.pixel_id.nunique():,} | positivos únicos por píxel: {df[df.label==1].pixel_id.nunique():,}")
    p("Filas por mes (hora de escaneo):"); p(df.groupby(pd.to_datetime(df.scan_start_utc).dt.month).label.agg(["size", "sum"]).rename(columns={"sum": "pos"}).to_string())
    p("Valores faltantes por columna (>0):"); na = df.isna().sum(); p(na[na > 0].to_string())
    p(f"DQF_fire==0 entre positivos: {(df[df.label==1].DQF_fire==0).mean():.3f} | entre negativos: {(df[df.label==0].DQF_fire==0).mean():.3f}")
    p(f"Desfase foco-escaneo en positivos (min): mediana {df[df.label==1].hotspot_offset_min.median():.1f}")
    fuego = list(range(10, 16)) + list(range(30, 36))
    p(f"Positivos con fuego según GOES (FireMask 10-15/30-35): {df[df.label==1].FireMask.isin(fuego).mean():.3f} | negativos: {df[df.label==0].FireMask.isin(fuego).mean():.3f}")
    df.to_csv(os.path.join(out, "dataset_goes16_2024.csv"), index=False)
    mdir = os.path.join(out, "por_mes"); os.makedirs(mdir, exist_ok=True)
    for m, g in df.groupby(pd.to_datetime(df.scan_start_utc).dt.month): g.to_csv(os.path.join(mdir, f"dataset_goes16_2024_mes_{m:02d}.csv"), index=False)
    open(os.path.join(out, "dataset_qc.txt"), "w", encoding="utf-8").write("\n".join(L))
    print(f"Guardado en {out}: dataset_goes16_2024.csv, por_mes\\*.csv y dataset_qc.txt")
    return df

if __name__ == "__main__":
    try:
        resultado = ejecutar()
    except Detenido as e:
        print("\n[DETENIDO]", e)