# -*- coding: utf-8 -*-
"""
14_escenas_completas_y_vecindad.py -- Revisión, ronda 1.
  MODO "escenas"  (R1-C3, R1-C4, R2-C2, R3-C6): aplica HGB y RF a TODOS los píxeles válidos del área de estudio en
                  ~20 escenas completas y cuenta las detecciones y las falsas alarmas por escena y por 10 000 km².
  MODO "vecindad" (R3-C4): recall de GOES-FDC, HGB y RF con tolerancia de vecindad 3×3 (el fuego cuenta como
                  detectado si aparece en el píxel o en cualquiera de sus 8 vecinos), y la tasa de falsos positivos
                  con la misma tolerancia, en las escenas del dataset.
NO modifica el modelo del artículo ni sus salidas.

CÓMO FUNCIONA
  - Importa "01. Descarga imágenes GOES.py" (lectura de S3, malla ABI, ventanas, máscaras de ríos y recorte, Planck) y
    "07. Entrenamiento del modelo.py" (modelos, pesos, umbral de máximo F1). Mismas variables y filtros de calidad.
  - Entrena los modelos de los 5 pliegues por fecha (como el script 07) y un modelo final con todo el dataset; los
    guarda en CARPETA_SALIDA\\modelos_14.joblib para no reentrenar en la siguiente corrida.
    * Escena de una fecha del dataset -> se usa el modelo del pliegue en que esa fecha fue de PRUEBA (no la vio).
    * Escena de una fecha sin muestras -> se usa el modelo final.
  - Umbrales: máximo F1 (el del pliegue, como en el artículo) y los de FPR = 1 % y 0.1 % (de las probabilidades
    out-of-fold de los no fuegos del dataset).
  - Clasificación de cada detección en una escena (con TODOS los focos IDEAM, confirmados o no, a +-12 h):
      a <= 2 km de un foco        -> "en_foco"
      2-10 km                     -> "cerca_de_fuego" (zona que el muestreo de entrenamiento excluyó)
      > 10 km                     -> "aislada" = cota superior de las falsas alarmas (puede incluir fuegos no
                                     detectados por VIIRS/MODIS)
  - Los píxeles a <= 2 km de ríos se procesan aparte (se excluyeron del entrenamiento por el reflejo del agua).

PENSADO PARA SPYDER: edite CONFIGURACIÓN, elija MODO y ejecute con F5. Si se corta, ejecute de nuevo: las escenas
ya procesadas se saltan. Primero corra MODO = "escenas" (unos minutos por escena), luego MODO = "vecindad".
SALIDA (CARPETA_SALIDA): escenas_completas.xlsx, vecindad_3x3.xlsx, carpeta escenas\\ (un CSV de detecciones por escena)
"""
import os, sys, time, json, importlib.util, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
SCRIPT_01 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\01. Descarga imagenes GOES.py"
SCRIPT_07 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\07. Entrenamiento del modelo.py"
DATASET = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
ESTADO = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Puntos calientes\Puntos calientes\preparados\puntos_ideam_2024_estado.csv"
RIOS = r"C:\Users\User\Desktop\Manuscript JI\Insumos\SHP Amazonia\Division\Rios  Colombia\rios.shp"   # la misma del script 01
RECORTE = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\Amazonia.shp"                     # la misma del script 01
CRS_SI_FALTA_PRJ = None
CARPETA_SALIDA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\vecindad_3x3"

MODO = "escenas"          # "escenas" | "vecindad"
MODELOS = ["hgb", "rf"]

# Escenas: 12 de días con fuego del dataset (6 de día y 6 de noche, elegidas automáticamente: las de más fuegos,
# una por fecha) + escenas de época de lluvias (hora UTC; 17:00 UTC = mediodía local, 07:00 UTC = 2 a. m.).
N_ESCENAS_FUEGO_DIA, N_ESCENAS_FUEGO_NOCHE = 6, 6
ESCENAS_LLUVIAS = ["2024-06-12 17:00", "2024-06-26 07:00", "2024-07-10 17:00", "2024-07-24 07:00",
                   "2024-08-07 17:00", "2024-05-15 07:00", "2024-05-29 17:00", "2024-09-11 07:00"]
VENTANA_FOCOS_H = 12.0    # focos a +-12 h de la escena para clasificar las detecciones
R_FOCO_KM, R_CERCA_KM = 2.0, 10.0
N_BOOTSTRAP = 1000
SEMILLA = 42

# ============================== UTILIDADES ==============================
def importar(ruta, nombre):
    if not os.path.exists(ruta): sys.exit(f"[DETENIDO] No existe {ruta}")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def es_noche(hora_utc):
    """Hora local (UTC-5) entre 18:00 y 06:00."""
    h = (np.asarray(hora_utc) - 5) % 24
    return (h >= 18) | (h < 6)

# ============================== MODELOS (pliegues + final) ==============================
def entrenar_modelos(m07, df):
    import joblib
    from sklearn.model_selection import StratifiedGroupKFold
    ruta = os.path.join(CARPETA_SALIDA, "modelos_14.joblib")
    if os.path.exists(ruta):
        M = joblib.load(ruta)
        if M.get("n_filas") == len(df) and M.get("modelos") == MODELOS:
            print("Modelos cargados de", ruta); return M
    X, y, g = df[m07.VARIABLES], df.label.to_numpy(), df.fecha.to_numpy()
    cv = list(StratifiedGroupKFold(n_splits=m07.N_PLIEGUES, shuffle=True, random_state=m07.SEMILLA).split(X, y, g))
    M = dict(n_filas=len(df), modelos=MODELOS, pliegue_de_fecha={}, oof={}, umbral_pliegue={}, modelo_pliegue={}, final={}, umbral_fpr={})
    for k, (tr, te) in enumerate(cv, 1):
        for f in np.unique(g[te]): M["pliegue_de_fecha"][f] = k
    for n in MODELOS:
        t0 = time.time(); prob = np.full(len(df), np.nan); ths, mods = [], []
        for k, (tr, te) in enumerate(cv, 1):
            cv_in = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=m07.SEMILLA + k)
            p_in = np.full(len(tr), np.nan)
            for a, b in cv_in.split(X.iloc[tr], y[tr], g[tr]):
                p_in[b] = m07.ajustar(m07.crear_modelo(n), n, X.iloc[tr[a]], y[tr[a]]).predict_proba(X.iloc[tr[b]])[:, 1]
            th = m07.umbral_max_f1(y[tr], p_in); ths.append(th)
            mdl = m07.ajustar(m07.crear_modelo(n), n, X.iloc[tr], y[tr]); mods.append(mdl)
            prob[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        M["oof"][n] = prob; M["umbral_pliegue"][n] = ths; M["modelo_pliegue"][n] = mods
        M["final"][n] = (m07.ajustar(m07.crear_modelo(n), n, X, y), float(np.mean(ths)))
        neg = np.sort(prob[y == 0])
        M["umbral_fpr"][n] = {f: float(neg[int(np.ceil((1 - f) * len(neg))) - 1]) for f in (0.01, 0.001)}
        pred = np.zeros(len(df), int)
        for k, (tr, te) in enumerate(cv, 1): pred[te] = (prob[te] >= ths[k - 1]).astype(int)
        mt = m07.metricas(y, pred, prob)
        print(f"[{n}] OOF F1 = {mt['f1']:.3f} | PR-AUC = {mt['pr_auc']:.3f} | umbrales pliegue {np.round(ths, 2)} | "
              f"umbral FPR 1 % = {M['umbral_fpr'][n][0.01]:.3f}, 0.1 % = {M['umbral_fpr'][n][0.001]:.3f} | {time.time() - t0:.0f} s")
        ref = {"hgb": (0.858, 0.932), "rf": (0.858, 0.931)}[n]
        if abs(mt["f1"] - ref[0]) > 0.005 or abs(mt["pr_auc"] - ref[1]) > 0.005:
            print(f"[AVISO] {n} no reproduce el artículo (F1 {ref[0]}, PR-AUC {ref[1]}). Revise antes de interpretar.")
    joblib.dump(M, ruta); return M

def modelo_para(M, n, fecha):
    """Modelo que no vio la fecha: el del pliegue de prueba de esa fecha; si la fecha no está en el dataset, el final."""
    k = M["pliegue_de_fecha"].get(fecha)
    if k is None:
        mdl, th = M["final"][n]; return mdl, th, "final"
    return M["modelo_pliegue"][n][k - 1], M["umbral_pliegue"][n][k - 1], f"pliegue_{k}"

# ============================== LECTURA DE UNA ESCENA (funciones del script 01) ==============================
def leer_escena(m01, ctx, start, win):
    """Lee C07, C14, ACM, FDC, LST y TPW de la escena que empieza en `start` dentro de win=(iy0,iy1,ix0,ix1)
    y calcula las variables del modelo. Devuelve dict de arreglos 2-D + 'mid' o None si falta algún producto."""
    s3, grid = ctx["s3"], ctx["grid"]
    p07, _ = m01.pick_companion(s3, m01.PRODUCTS["rad"], start, 0.5, band=7)
    if p07 is None: return None, "sin_C07"
    s7, e7 = m01.parse_scan_times(p07); mid = s7 + (e7 - s7) / 2
    paths = {"c07": p07}
    paths["c14"], _ = m01.pick_companion(s3, m01.PRODUCTS["rad"], start, m01.TOL_COMPANION_MIN, band=14)
    for k in ("acm", "fdc"): paths[k], _ = m01.pick_companion(s3, m01.PRODUCTS[k], start, m01.TOL_COMPANION_MIN)
    for k in ("lst", "tpw"): paths[k], _ = m01.pick_companion(s3, m01.PRODUCTS[k], start, m01.TOL_LST_TPW_MIN)
    falt = [k for k in ("c14", "acm", "fdc") if paths[k] is None]
    if falt: return None, "faltan_" + "_".join(falt)
    rd = lambda k: m01.read_product(s3, paths[k], m01.VARS[k][1], grid, win, ctx["mapcache"])
    c7, c14, acm, fdc = rd("c07"), rd("c14"), rd("acm"), rd("fdc")
    shp = c7["Rad"].shape; nan = np.full(shp, np.nan)
    lst = rd("lst") if paths["lst"] else {"LST": nan, "DQF": nan}
    tpw = rd("tpw") if paths["tpw"] else {"TPW": nan, "DQF_Overall": nan}
    T7 = m01.planck_bt(c7["Rad"], c7["planck_fk1"], c7["planck_fk2"], c7["planck_bc1"], c7["planck_bc2"])
    T14 = m01.planck_bt(c14["Rad"], c14["planck_fk1"], c14["planck_fk2"], c14["planck_bc1"], c14["planck_bc2"])
    with np.errstate(all="ignore"):
        BTD = T7 - T14; RATIO = T7 / (T14 + 1e-6); den = T7 + T14; den = np.where(np.abs(den) > 1e-6, den, 1e-6); ND = BTD / den
        calidad = (np.isfinite(c7["Rad"]) & np.isfinite(c14["Rad"]) & np.isfinite(acm["Cloud_Probabilities"]) & (c7["DQF"] == 0)
                   & (c14["DQF"] == 0) & (acm["DQF"] == 0) & (fdc["DQF"] <= 1) & (c7["Rad"] > 0) & (c14["Rad"] > 0)
                   & np.isfinite(T7) & np.isfinite(T14) & np.isfinite(BTD) & np.isfinite(RATIO) & np.isfinite(ND))
        if m01.FILTER_LST_DQF: calidad &= (np.isnan(lst["DQF"]) | (lst["DQF"] == 0))
        if m01.FILTER_TPW_DQF: calidad &= (np.isnan(tpw["DQF_Overall"]) | (tpw["DQF_Overall"] == 0))
    V = dict(C07=c7["Rad"], C14=c14["Rad"], T7=T7, T14=T14, BTD=BTD, Ratio=RATIO, ND=ND, Cloud_Probabilities=acm["Cloud_Probabilities"],
             LST=lst["LST"], TPW=tpw["TPW"], LST_BT_DIFF=lst["LST"] - T14, fire_hour=np.full(shp, mid.hour, float),
             FireMask=fdc["Mask"], calidad=calidad, mid=mid)
    return V, "ok"

def predecir(mdl, V, variables, mascara):
    X = pd.DataFrame({v: V[v][mascara] for v in variables})
    p = np.full(mascara.shape, np.nan)
    if len(X): p[mascara] = mdl.predict_proba(X)[:, 1]
    return p

def area_pixel_km2(XK, YK):
    dxx, dxy = np.gradient(XK, axis=1), np.gradient(YK, axis=1)
    dyx, dyy = np.gradient(XK, axis=0), np.gradient(YK, axis=0)
    return np.abs(dxx * dyy - dxy * dyx)

# ============================== CONTEXTO COMÚN ==============================
def preparar(m01, m07):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    m07.VARIABLES = ["C07", "C14", "T7", "T14", "BTD", "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF", "fire_hour"]
    df = m07.cargar(DATASET)
    e = pd.read_csv(ESTADO, low_memory=False); e["datetime_utc"] = pd.to_datetime(e.datetime_utc).astype("datetime64[ns]")
    for c in ("confidence_class", "confidence_pct", "frp_mw", "day_night", "overpass_id", "cluster_id_2km", "source"):
        if c not in e.columns: e[c] = np.nan
    e = e.sort_values("datetime_utc").reset_index(drop=True)
    s3 = m01.S3Goes(os.path.join(CARPETA_SALIDA, "listados_s3.json"))
    args = type("A", (), {})(); args.rios, args.rios_crs, args.out, args.recorte = RIOS, CRS_SI_FALTA_PRJ, CARPETA_SALIDA, RECORTE
    grid, reg, to_km = m01.build_context(args, e, s3)
    XYK = to_km(reg["lon"], reg["lat"]); reg["XK"] = XYK[:, 0].reshape(reg["lon"].shape); reg["YK"] = XYK[:, 1].reshape(reg["lon"].shape)
    reg["area_km2"] = area_pixel_km2(reg["XK"], reg["YK"])
    ctx = dict(s3=s3, grid=grid, mapcache={})
    focos = dict(t=e.datetime_utc.astype("int64").to_numpy(), km=to_km(e.lon, e.lat), estado=e.estado.to_numpy())
    M = entrenar_modelos(m07, df)
    return df, e, reg, ctx, focos, M, to_km

def focos_ventana(focos, mid, horas):
    lo, hi = mid.value - int(horas * 3.6e12), mid.value + int(horas * 3.6e12)
    a, b = np.searchsorted(focos["t"], [lo, hi]); return focos["km"][a:b]

# ============================== MODO ESCENAS ==============================
def elegir_escenas(df):
    pos = df[df.label == 1].copy(); pos["t"] = pd.to_datetime(pos.scan_start_utc)
    por = pos.groupby("scan_start_utc").agg(fuegos=("label", "size"), fecha=("fecha", "first"), t=("t", "first")).reset_index()
    por["noche"] = es_noche(por.t.dt.hour)
    sel = []
    for noche, n in ((False, N_ESCENAS_FUEGO_DIA), (True, N_ESCENAS_FUEGO_NOCHE)):
        s = por[por.noche == noche].sort_values("fuegos", ascending=False).drop_duplicates("fecha").head(n)
        sel += [dict(inicio=pd.Timestamp(r.scan_start_utc), tipo="fuego_" + ("noche" if noche else "dia"), fuegos_dataset=int(r.fuegos)) for r in s.itertuples()]
    for t in ESCENAS_LLUVIAS:
        t = pd.Timestamp(t); sel.append(dict(inicio=t, tipo="lluvias_" + ("noche" if es_noche(t.hour) else "dia"), fuegos_dataset=0))
    return pd.DataFrame(sel)

def inicio_real(m01, ctx, t):
    """Escaneo C07 cuyo inicio está más cerca de t (para las escenas de lluvias dadas por hora)."""
    c = ctx["s3"].scans(m01.PRODUCTS["rad"], t, band=7)
    if not c: return None
    d = [abs((s - t).total_seconds()) for s, _, _ in c]; j = int(np.argmin(d))
    return c[j][0] if d[j] <= 600 else None

def modo_escenas(m01, m07):
    df, e, reg, ctx, focos, M, to_km = preparar(m01, m07)
    FUEGO = set(m07.FUEGO_FDC)
    esc = elegir_escenas(df); print("\nEscenas:\n" + esc.to_string(index=False))
    cdir = os.path.join(CARPETA_SALIDA, "escenas"); os.makedirs(cdir, exist_ok=True)
    win = (reg["iy0"], reg["iy1"], reg["ix0"], reg["ix1"])
    filas = []
    for r in esc.itertuples():
        t0 = time.time(); ini = r.inicio if r.tipo.startswith("fuego") else inicio_real(m01, ctx, r.inicio)
        if ini is None: print(f"  {r.inicio}: sin escaneo C07 a <= 10 min"); continue
        fn = os.path.join(cdir, f"G16_{ini:%Y%j%H%M%S}.json")
        if os.path.exists(fn):
            filas += json.load(open(fn)); print(f"  {ini}: ya procesada"); continue
        V, st = leer_escena(m01, ctx, ini, win)
        if V is None: print(f"  {ini}: {st}"); continue
        mid = V["mid"]; fecha = f"{ini:%Y-%m-%d}"
        base = V["calidad"] & reg["en_recorte"]
        tierra, rio = base & ~reg["water"], base & reg["water"]
        fk = focos_ventana(focos, mid, VENTANA_FOCOS_H)
        if len(fk):
            d, _ = cKDTree(fk).query(np.column_stack([reg["XK"].ravel(), reg["YK"].ravel()])); d = d.reshape(tierra.shape)
        else:
            d = np.full(tierra.shape, np.inf)
        clase = np.where(d <= R_FOCO_KM, "en_foco", np.where(d <= R_CERCA_KM, "cerca_de_fuego", "aislada"))
        nube = V["Cloud_Probabilities"]
        detectores = {"GOES-FDC": (np.isin(V["FireMask"], list(FUEGO)), "-")}
        probs = {}
        for n in MODELOS:
            mdl, th, origen = modelo_para(M, n, fecha); p = predecir(mdl, V, m07.VARIABLES, base); probs[n] = p
            for nombre_u, u in (("max_F1", th), ("FPR_1%", M["umbral_fpr"][n][0.01]), ("FPR_0.1%", M["umbral_fpr"][n][0.001])):
                detectores[f"{n.upper()} {nombre_u}"] = (np.nan_to_num(p, nan=-1) >= u, origen)
        area_tierra = float(reg["area_km2"][tierra].sum())
        fil_esc = []
        for nombre, (det, origen) in detectores.items():
            for zona, m in (("tierra", tierra), ("rios_2km", rio)):
                dz = det & m
                fil = dict(escena=f"G16_{ini:%Y%j%H%M%S}", inicio_utc=str(ini), tipo=r.tipo, noche=bool(es_noche(mid.hour)), fecha=fecha,
                           fuegos_dataset=int(r.fuegos_dataset), focos_12h=len(fk), detector=nombre, modelo_usado=origen, zona=zona,
                           pixeles_validos=int(m.sum()), area_valida_km2=float(reg["area_km2"][m].sum()),
                           detecciones=int(dz.sum()), en_foco=int((dz & (clase == "en_foco")).sum()),
                           cerca_de_fuego=int((dz & (clase == "cerca_de_fuego")).sum()), aisladas=int((dz & (clase == "aislada")).sum()),
                           aisladas_nube_mayor_50=int((dz & (clase == "aislada") & (nube > 0.5)).sum()))
                fil["aisladas_por_10000_km2"] = 1e4 * fil["aisladas"] / max(fil["area_valida_km2"], 1e-9)
                fil["tasa_aisladas_por_pixel"] = fil["aisladas"] / max(int((m & (clase == "aislada")).sum()), 1)
                fil["pct_en_foco"] = 100 * fil["en_foco"] / max(fil["detecciones"], 1)
                fil_esc.append(fil)
        json.dump(fil_esc, open(fn, "w")); filas += fil_esc
        # detecciones de la escena (para mapas)
        cand = tierra | rio
        any_det = np.zeros_like(cand)
        for det, _ in detectores.values(): any_det |= det
        sel = cand & any_det
        out = pd.DataFrame(dict(lon=reg["lon"][sel], lat=reg["lat"][sel], zona=np.where(rio[sel], "rios_2km", "tierra"),
                                clase=clase[sel], dist_foco_km=np.round(d[sel], 2), nube=nube[sel], fdc=detectores["GOES-FDC"][0][sel].astype(int),
                                **{f"prob_{n}": np.round(probs[n][sel], 4) for n in MODELOS}))
        out.to_csv(fn.replace(".json", "_detecciones.csv"), index=False)
        tt = [f for f in fil_esc if f["zona"] == "tierra"]
        print(f"  {ini} {r.tipo:<13} válidos {tt[0]['pixeles_validos']:>7,} | " +
              " | ".join(f"{f['detector']}: {f['detecciones']} ({f['aisladas']} aisl.)" for f in tt if f["detector"] in ("GOES-FDC", "HGB max_F1", "HGB FPR_0.1%", "RF max_F1"))
              + f" | {time.time() - t0:.0f} s")
    ctx["s3"].save()
    res = pd.DataFrame(filas)
    if res.empty: sys.exit("[DETENIDO] No se procesó ninguna escena.")
    t = res[res.zona == "tierra"].copy(); t["grupo"] = t.tipo.str.split("_").str[0]
    agg = lambda g: pd.Series(dict(escenas=g.escena.nunique(), pixeles_validos_mediana=g.pixeles_validos.median(),
                                   detecciones_media=g.detecciones.mean(), detecciones_mediana=g.detecciones.median(),
                                   aisladas_media=g.aisladas.mean(), aisladas_mediana=g.aisladas.median(), aisladas_max=g.aisladas.max(),
                                   aisladas_por_10000_km2_media=g.aisladas_por_10000_km2.mean(),
                                   tasa_aisladas_por_pixel=g.aisladas.sum() / max(g.pixeles_validos.sum(), 1),
                                   pct_en_foco=100 * g.en_foco.sum() / max(g.detecciones.sum(), 1),
                                   pct_cerca_de_fuego=100 * g.cerca_de_fuego.sum() / max(g.detecciones.sum(), 1),
                                   pct_aisladas=100 * g.aisladas.sum() / max(g.detecciones.sum(), 1),
                                   pct_aisladas_con_nube_mayor_50=100 * g.aisladas_nube_mayor_50.sum() / max(g.aisladas.sum(), 1)))
    tot = t.groupby(["detector"]).apply(agg).reset_index()
    por_tipo = t.groupby(["detector", "tipo"]).apply(agg).reset_index()
    rios = res.groupby(["detector", "zona"]).apply(lambda g: pd.Series(dict(pixeles=g.pixeles_validos.sum(), detecciones=g.detecciones.sum(),
                                                                         aisladas=g.aisladas.sum(), tasa_deteccion_por_pixel=g.detecciones.sum() / max(g.pixeles_validos.sum(), 1),
                                                                         tasa_aisladas_por_pixel=g.aisladas.sum() / max(g.pixeles_validos.sum(), 1)))).reset_index()
    salida = os.path.join(CARPETA_SALIDA, "escenas_completas.xlsx")
    with pd.ExcelWriter(salida) as w:
        tot.to_excel(w, sheet_name="resumen_por_detector", index=False); por_tipo.to_excel(w, sheet_name="por_tipo_de_escena", index=False)
        rios.to_excel(w, sheet_name="tierra_vs_rios", index=False); res.to_excel(w, sheet_name="por_escena", index=False)
        pd.DataFrame({"parametro": ["dataset", "estado", "ventana_focos_h", "r_foco_km", "r_cerca_km", "umbral_fpr", "nota"],
                      "valor": [DATASET, ESTADO, VENTANA_FOCOS_H, R_FOCO_KM, R_CERCA_KM, json.dumps({n: M["umbral_fpr"][n] for n in MODELOS}),
                                "aisladas = detecciones a > 10 km de todo foco IDEAM (confirmado o no) a +-12 h: cota superior de falsas alarmas"]}
                     ).to_excel(w, sheet_name="configuracion", index=False)
    pd.set_option("display.width", 240)
    print("\n================ ESCENAS COMPLETAS (píxeles de tierra) ================")
    print(tot.round(2).to_string(index=False)); print(f"\nGuardado: {salida}")
    return tot

# ============================== MODO VECINDAD 3x3 ==============================
def modo_vecindad(m01, m07):
    df, e, reg, ctx, focos, M, to_km = preparar(m01, m07)
    FUEGO = set(m07.FUEGO_FDC)
    for c in ("pix_ix", "pix_iy", "image_id", "scan_start_utc"):
        if c not in df.columns: sys.exit(f"[DETENIDO] Falta la columna {c} en el dataset")
    vdir = os.path.join(CARPETA_SALIDA, "vecindad"); os.makedirs(vdir, exist_ok=True)
    y = df.label.to_numpy(); partes = []
    escenas = df.groupby("image_id"); n_esc = escenas.ngroups; t_ini = time.time()
    for k, (img, g) in enumerate(escenas, 1):
        fn = os.path.join(vdir, f"{img}.csv")
        if os.path.exists(fn): partes.append(pd.read_csv(fn)); continue
        ini = pd.Timestamp(g.scan_start_utc.iloc[0]); fecha = g.fecha.iloc[0]
        ix, iy = g.pix_ix.to_numpy(int), g.pix_iy.to_numpy(int)
        win = (int(iy.min() - 1), int(iy.max() + 2), int(ix.min() - 1), int(ix.max() + 2))
        try:
            V, st = leer_escena(m01, ctx, ini, win)
        except Exception as ex:
            print(f"  {img}: ERROR {type(ex).__name__}: {ex}"); continue
        if V is None: print(f"  {img}: {st}"); continue
        ry, rx = iy - win[0], ix - win[2]
        # región de los vecinos (recorte, agua) en la misma ventana
        sl = (slice(win[0] - reg["iy0"], win[1] - reg["iy0"]), slice(win[2] - reg["ix0"], win[3] - reg["ix0"]))
        valido = V["calidad"] & reg["en_recorte"][sl] & ~reg["water"][sl]
        fdc = np.isin(V["FireMask"], list(FUEGO))
        res = dict(image_id=img, pixel_id=g.pixel_id.to_numpy(), label=g.label.to_numpy(), fecha=fecha,
                   T7_dataset=g.T7.to_numpy(), T7_releido=V["T7"][ry, rx],
                   fdc_centro=fdc[ry, rx].astype(int), fdc_3x3=np.zeros(len(g), int))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                res["fdc_3x3"] |= fdc[ry + dy, rx + dx].astype(int)
        for n in MODELOS:
            mdl, th, _ = modelo_para(M, n, fecha); p = predecir(mdl, V, m07.VARIABLES, valido)
            # el centro se evalúa siempre (es la muestra del dataset); los vecinos solo si son válidos
            centro = pd.DataFrame({v: V[v][ry, rx] for v in m07.VARIABLES}); p_c = mdl.predict_proba(centro)[:, 1]
            vec = np.full((len(g), 9), np.nan); j = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    vec[:, j] = p_c if (dy == 0 and dx == 0) else p[ry + dy, rx + dx]; j += 1
            res[f"prob_centro_{n}"] = p_c; res[f"prob_max3x3_{n}"] = np.nanmax(vec, axis=1); res[f"umbral_{n}"] = th
        out = pd.DataFrame(res); out.to_csv(fn, index=False); partes.append(out)
        if k % 10 == 0 or k == n_esc:
            print(f"[{k}/{n_esc}] {img} | {(time.time() - t_ini) / 60:.1f} min"); ctx["s3"].save()
    ctx["s3"].save()
    R = pd.concat(partes, ignore_index=True)
    dif_t7 = (R.T7_releido - R.T7_dataset).abs()
    print(f"\nVerificación: |T7 releído - T7 del dataset| mediana {dif_t7.median():.4f} K, máx {dif_t7.max():.4f} K (debe ser ~0)")
    det = {"GOES-FDC": (R.fdc_centro.astype(bool), R.fdc_3x3.astype(bool))}
    for n in MODELOS:
        det[n.upper()] = (R[f"prob_centro_{n}"] >= R[f"umbral_{n}"], R[f"prob_max3x3_{n}"] >= R[f"umbral_{n}"])
    yb = R.label.to_numpy() == 1; idx = R.groupby("fecha").indices; fe = np.array(list(idx)); rng = np.random.default_rng(SEMILLA)
    boots = [np.concatenate([idx[x] for x in rng.choice(fe, len(fe), replace=True)]) for _ in range(N_BOOTSTRAP)]
    filas = []
    for nombre, (c, v3) in det.items():
        c, v3 = np.asarray(c), np.asarray(v3)
        rc, fc = c[yb].mean(), c[~yb].mean(); r3, f3 = v3[yb].mean(), v3[~yb].mean()
        B = np.array([(v3[b][yb[b]].mean() - c[b][yb[b]].mean(), v3[b][~yb[b]].mean() - c[b][~yb[b]].mean()) for b in boots])
        filas.append(dict(detector=nombre, recall_pixel=rc, recall_3x3=r3, dif_recall=r3 - rc,
                          dif_recall_ic=f"[{np.quantile(B[:, 0], .025):.3f}, {np.quantile(B[:, 0], .975):.3f}]",
                          fpr_pixel=fc, fpr_3x3=f3, dif_fpr=f3 - fc, dif_fpr_ic=f"[{np.quantile(B[:, 1], .025):.4f}, {np.quantile(B[:, 1], .975):.4f}]",
                          fuegos=int(yb.sum()), no_fuegos=int((~yb).sum())))
    tab = pd.DataFrame(filas)
    salida = os.path.join(CARPETA_SALIDA, "vecindad_3x3.xlsx")
    with pd.ExcelWriter(salida) as w:
        tab.to_excel(w, sheet_name="recall_y_fpr_3x3", index=False)
        pd.DataFrame([dict(muestras=len(R), muestras_dataset=len(df), escenas=R.image_id.nunique(), escenas_dataset=n_esc,
                           T7_dif_mediana=dif_t7.median(), T7_dif_max=dif_t7.max())]).to_excel(w, sheet_name="verificacion", index=False)
        pd.DataFrame({"parametro": ["nota"], "valor": ["3x3: detectado si el píxel o cualquiera de sus 8 vecinos es fuego (FDC) o supera el "
                      "umbral del pliegue (HGB, RF). Vecinos de los modelos: solo píxeles válidos (calidad, recorte, sin ríos). "
                      "IC: bootstrap pareado por fecha (3x3 menos píxel)."]}).to_excel(w, sheet_name="configuracion", index=False)
    pd.set_option("display.width", 240)
    print("\n================ VECINDAD 3x3 ================"); print(tab.round(4).to_string(index=False)); print(f"\nGuardado: {salida}")
    return tab

# ============================== PRINCIPAL ==============================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    m01 = importar(SCRIPT_01, "descarga01"); m07 = importar(SCRIPT_07, "entrenamiento07")
    m01.WATER_KM = 2.0
    try:
        resultado = modo_escenas(m01, m07) if MODO == "escenas" else modo_vecindad(m01, m07)
    except m01.Detenido as ex:
        print("\n[DETENIDO]", ex)