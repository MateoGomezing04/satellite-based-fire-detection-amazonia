# -*- coding: utf-8 -*-
"""
15_anio_independiente_2025.py -- Revisión, ronda 1 (R1-C6). Evaluación en un año independiente.
Aplica los modelos HGB y RF entrenados con TODO 2024 (sin reentrenar) al dataset de enero-marzo de 2025, construido con
la misma cadena (focos IDEAM 2025 -> confirmación con áreas quemadas 2025 -> extracción GOES-16 con el script 01), y
lo compara con GOES-FDC en las mismas muestras. NO modifica el modelo del artículo ni sus salidas.

CÓMO FUNCIONA
  - Usa los modelos y umbrales guardados por el script 14 (modelos_14.joblib): modelo final 2024 con el umbral medio de
    máximo F1 de los pliegues, y los umbrales de FPR = 1 % y 0.1 % de 2024. Si no existe, los entrena (mismo código).
  - Métricas 2025 con IC 95 % por bootstrap de fechas; diferencias pareadas HGB-FDC, RF-FDC y HGB-RF en 2025;
    recall por FRP, día/noche y mes; recall a FPR fijo medido en 2025; comparación con la validación cruzada 2024;
    cambio de distribución de las variables entre años.
GOES-19 (desde el 8 de abril de 2025): córralo aparte con ETIQUETA = "goes19_2025" y su dataset. Esa evaluación mezcla
  año y sensor nuevos (otro instrumento ABI, otra calibración), así que se reporta por separado de la de GOES-16.
PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5.
SALIDA: CARPETA_SALIDA\\anio_independiente_2025.xlsx
"""
import os, sys, importlib.util, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
SCRIPT_07 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\07. Entrenamiento del modelo.py"
SCRIPT_14 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Revisión 1\scripts\14_escenas_completas_y_vecindad.py"
CARPETA_14 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Escenas completas"      # donde está modelos_14.joblib
DATASET_2024 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
DATASETS_2025 = [   # uno o varios datasets de 2025 (salidas del script 01); se evalúan por satélite y juntos
    r"C:\Users\User\Desktop\Nuevo Manuscript\Extraccion GOES puntos calientes 2025\GOES-16\dataset_goes16_2025.csv",
    r"C:\Users\User\Desktop\Nuevo Manuscript\Extraccion GOES puntos calientes 2025\GOES-19\dataset_goes19_2025.csv",
]
ESTADO_2025 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Puntos calientes\Puntos calientes\2025\preparados\puntos_ideam_2025_estado.csv"     # el script 03 lo nombra así; None = omitir
CARPETA_SALIDA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Año independiente 2025"
CAMBIO_G16_G19 = pd.Timestamp("2025-04-07")   # para repartir los focos de ESTADO_2025 entre satélites
MODELOS = ["hgb", "rf"]
N_BOOTSTRAP, N_BOOTSTRAP_GRUPOS = 1000, 500
SEMILLA = 42

# ============================== UTILIDADES ==============================
def importar(ruta, nombre):
    if not os.path.exists(ruta): sys.exit(f"[DETENIDO] No existe {ruta}")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def metr(y, pred, prob=None):
    from sklearn.metrics import (precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score,
                                 roc_auc_score, accuracy_score)
    neg = y == 0
    m = dict(recall=recall_score(y, pred, zero_division=0), precision=precision_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred), accuracy=accuracy_score(y, pred),
             fpr=float(pred[neg].mean()) if neg.any() else np.nan)
    if prob is not None and 0 < y.mean() < 1:
        m.update(pr_auc=average_precision_score(y, prob), roc_auc=roc_auc_score(y, prob))
    return m

def bootstrap(df, fun, n, semilla=SEMILLA, exigir_dos_clases=True):
    idx = df.groupby("fecha").indices; fe = np.array(list(idx)); rng = np.random.default_rng(semilla); R = []
    y = df.label.to_numpy()
    for _ in range(n):
        f = np.concatenate([idx[x] for x in rng.choice(fe, len(fe), replace=True)])
        if exigir_dos_clases and y[f].min() == y[f].max(): continue
        R.append(fun(f))
    return np.array(R)

def ic(a): return f"[{np.nanquantile(a, .025):.3f}–{np.nanquantile(a, .975):.3f}]"

def recall_a_fpr(y, prob, f):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, prob); i = np.searchsorted(fpr, f, side="right") - 1
    return float(tpr[i])

# ============================== PRINCIPAL ==============================
def ejecutar():
    warnings.filterwarnings("ignore"); os.makedirs(CARPETA_SALIDA, exist_ok=True)
    m07 = importar(SCRIPT_07, "entrenamiento07"); s14 = importar(SCRIPT_14, "escenas14")
    m07.VARIABLES = ["C07", "C14", "T7", "T14", "BTD", "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF", "fire_hour"]
    d24 = m07.cargar(DATASET_2024)
    partes = []
    for ruta in DATASETS_2025:
        if not os.path.exists(ruta): print(f"[AVISO] No existe {ruta}: se omite"); continue
        partes.append(m07.cargar(ruta))
    if not partes: sys.exit("[DETENIDO] Ninguno de DATASETS_2025 existe.")
    todos = pd.concat(partes, ignore_index=True)
    if pd.to_datetime(todos.scan_start_utc).dt.year.min() < 2025: sys.exit("[DETENIDO] DATASETS_2025 contiene fechas anteriores a 2025.")
    # satélite de cada muestra: columna 'satellite' del script 01 o prefijo G16_/G19_ del image_id
    sat = todos["satellite"].astype(str) if "satellite" in todos.columns else todos.image_id.astype(str).str[:3].map({"G16": "GOES-16", "G19": "GOES-19"})
    todos["sat"] = sat.str.upper().str.replace("GOES", "GOES-", regex=False).str.replace("--", "-", regex=False)
    print(f"2024: {len(d24):,} muestras, {int(d24.label.sum()):,} fuegos, {d24.fecha.nunique()} fechas")
    print(todos.groupby("sat").agg(muestras=("label", "size"), fuegos=("label", "sum"), fechas=("fecha", "nunique")).to_string())
    s14.CARPETA_SALIDA = CARPETA_14; os.makedirs(CARPETA_14, exist_ok=True); s14.MODELOS = MODELOS
    M = s14.entrenar_modelos(m07, d24)          # modelos de 2024 (GOES-16); 2025 NUNCA se usa para entrenar
    subconjuntos = [(f"{x.lower().replace('-', '')}_2025", todos[todos.sat == x]) for x in sorted(todos.sat.dropna().unique())]
    if len(subconjuntos) > 1: subconjuntos.append(("ambos_2025", todos))
    res = {}
    for etiqueta, d25 in subconjuntos:
        res[etiqueta] = evaluar(m07, s14, M, d24, d25.reset_index(drop=True), etiqueta)
    return res

def evaluar(m07, s14, M, d24, d25, ETIQUETA):
    """Evalúa los modelos de 2024 (sin reentrenar) y GOES-FDC sobre un subconjunto de 2025."""
    y = d25.label.to_numpy(); X = d25[m07.VARIABLES]
    preds = {"GOES-FDC": (d25.goes_fdc.to_numpy(), None)}; umbrales = {}
    for n in MODELOS:
        mdl, th = M["final"][n]; p = mdl.predict_proba(X)[:, 1]
        for nom, u in (("max_F1", th), ("FPR_1%_2024", M["umbral_fpr"][n][0.01]), ("FPR_0.1%_2024", M["umbral_fpr"][n][0.001])):
            preds[f"{n.upper()} {nom}"] = ((p >= u).astype(int), p); umbrales[f"{n.upper()} {nom}"] = u
        d25[f"prob_{n}"] = p
    # ---- métricas globales 2025 con IC
    filas = []
    for nom, (pr, pb) in preds.items():
        m = metr(y, pr, pb)
        B = bootstrap(d25, lambda f: [metr(y[f], pr[f], None if pb is None else pb[f]).get(k, np.nan) for k in ("recall", "precision", "f1", "fpr", "pr_auc")], N_BOOTSTRAP)
        filas.append(dict(detector=nom, umbral=umbrales.get(nom, np.nan), **m, recall_ic=ic(B[:, 0]), precision_ic=ic(B[:, 1]), f1_ic=ic(B[:, 2]),
                          fpr_ic=f"[{np.nanquantile(B[:, 3], .025):.4f}–{np.nanquantile(B[:, 3], .975):.4f}]",
                          pr_auc_ic=(ic(B[:, 4]) if pb is not None else ""),
                          fpr_near=float(pr[(y == 0) & (d25.neg_type == "near").to_numpy()].mean()),
                          fpr_far=float(pr[(y == 0) & (d25.neg_type == "far").to_numpy()].mean())))
    glob = pd.DataFrame(filas)
    # ---- referencia 2024 (validación cruzada, mismas métricas)
    y24 = d24.label.to_numpy(); ref = [dict(detector="GOES-FDC", **metr(y24, d24.goes_fdc.to_numpy()))]
    for n in MODELOS:
        pred24 = np.zeros(len(d24), int)
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=m07.N_PLIEGUES, shuffle=True, random_state=m07.SEMILLA).split(d24[m07.VARIABLES], y24, d24.fecha.to_numpy())
        for k, (tr, te) in enumerate(cv, 1): pred24[te] = (M["oof"][n][te] >= M["umbral_pliegue"][n][k - 1]).astype(int)
        ref.append(dict(detector=f"{n.upper()} max_F1", **metr(y24, pred24, M["oof"][n])))
    ref = pd.DataFrame(ref)
    # ---- recall a FPR fijo medido en 2025 (curva ROC de 2025)
    rf_tab = pd.DataFrame([dict(modelo=n.upper(), **{f"recall_a_FPR_{f:.1%}": recall_a_fpr(y, d25[f"prob_{n}"].to_numpy(), f) for f in (0.01, 0.001, 0.0001)}) for n in MODELOS])
    # ---- diferencias pareadas en 2025
    par = []
    comp = [(f"{n.upper()} max_F1", "GOES-FDC") for n in MODELOS] + ([("HGB max_F1", "RF max_F1")] if len(MODELOS) == 2 else [])
    for a, b in comp:
        pa, pb_ = preds[a][0], preds[b][0]
        d = lambda f: [metr(y[f], pa[f])[k] - metr(y[f], pb_[f])[k] for k in ("recall", "fpr", "f1")]
        base = d(np.arange(len(y))); B = bootstrap(d25, d, N_BOOTSTRAP)
        par.append(dict(comparacion=f"{a} - {b}", dif_recall=base[0], dif_recall_ic=ic(B[:, 0]), dif_fpr=base[1],
                        dif_fpr_ic=f"[{np.quantile(B[:, 1], .025):.4f}–{np.quantile(B[:, 1], .975):.4f}]", dif_f1=base[2], dif_f1_ic=ic(B[:, 2]),
                        fuegos_solo_primero=int(((pa == 1) & (pb_ == 0) & (y == 1)).sum()), fuegos_solo_segundo=int(((pa == 0) & (pb_ == 1) & (y == 1)).sum())))
    par = pd.DataFrame(par)
    # ---- recall por grupo
    pos = d25[d25.label == 1].copy(); grupos = []
    if "frp_mw" in pos.columns:
        pos["FRP"] = pd.cut(pos.frp_mw, [0, 5, 10, 25, 50, np.inf], labels=["<5", "5–10", "10–25", "25–50", ">50"])
    if "day_night" in pos.columns and pos.day_night.notna().any():      # marca D/N del foco, como en la Tabla A4
        pos["periodo"] = np.where(pos.day_night.astype(str).str.upper().str[0] == "N", "Night", "Day")
    else:
        pos["periodo"] = np.where(s14.es_noche(pd.to_datetime(pos.scan_start_utc).dt.hour), "Night", "Day")
    pos["Mes"] = pos.mes.astype(str)
    for col in [c for c in ("FRP", "periodo", "Mes") if c in pos.columns]:
        for cat, g in pos.groupby(col, observed=True):
            fila = dict(grupo=col, categoria=str(cat), n=len(g))
            for nom in ["GOES-FDC"] + [f"{n.upper()} max_F1" for n in MODELOS]:
                pr = preds[nom][0][g.index.to_numpy()]
                B = bootstrap(g.reset_index(drop=True), lambda f: [pr[f].mean()], N_BOOTSTRAP_GRUPOS, exigir_dos_clases=False) if g.fecha.nunique() > 1 else np.array([[pr.mean()]])
                fila[f"recall_{nom}"] = pr.mean(); fila[f"ic_{nom}"] = ic(B[:, 0])
            grupos.append(fila)
    grupos = pd.DataFrame(grupos)
    # ---- cambio de distribución de las variables
    dist = []
    for v in m07.VARIABLES:
        for lab, nom in ((1, "fuego"), (0, "no fuego")):
            a, b = d24.loc[d24.label == lab, v], d25.loc[d25.label == lab, v]
            dist.append(dict(variable=v, clase=nom, mediana_2024=a.median(), mediana_2025=b.median(), p25_2024=a.quantile(.25), p75_2024=a.quantile(.75),
                             p25_2025=b.quantile(.25), p75_2025=b.quantile(.75), pct_faltante_2024=100 * a.isna().mean(), pct_faltante_2025=100 * b.isna().mean()))
    dist = pd.DataFrame(dist)
    # ---- confirmación 2025 (script 03)
    conf = None
    if ESTADO_2025 and os.path.exists(ESTADO_2025):
        e = pd.read_csv(ESTADO_2025, low_memory=False); te = pd.to_datetime(e.datetime_utc)
        if ETIQUETA.startswith("goes16"): e = e[te < CAMBIO_G16_G19]
        elif ETIQUETA.startswith("goes19"): e = e[te >= CAMBIO_G16_G19 + pd.Timedelta(days=1)]
        conf = pd.DataFrame([dict(focos=len(e), confirmados=int((e.estado == "confirmado").sum()), pct_confirmados=100 * (e.estado == "confirmado").mean(),
                                  por_regla_i=int((e.confirmado_por == "punto").sum()), por_regla_ii=int((e.confirmado_por == "evento").sum()),
                                  positivos_candidatos=int(e.get("positivo_candidato", pd.Series(dtype=int)).sum()))])
    salida = os.path.join(CARPETA_SALIDA, f"anio_independiente_{ETIQUETA}.xlsx")
    with pd.ExcelWriter(salida) as w:
        glob.to_excel(w, sheet_name="metricas_2025", index=False); ref.to_excel(w, sheet_name="referencia_cv_2024", index=False)
        par.to_excel(w, sheet_name="diferencias_pareadas_2025", index=False); rf_tab.to_excel(w, sheet_name="recall_a_FPR_fijo_2025", index=False)
        grupos.to_excel(w, sheet_name="recall_por_grupo_2025", index=False); dist.to_excel(w, sheet_name="distribucion_variables", index=False)
        if conf is not None: conf.to_excel(w, sheet_name="confirmacion_2025", index=False)
        pd.DataFrame([dict(muestras_2025=len(d25), fuegos_2025=int(y.sum()), fechas_2025=d25.fecha.nunique(), prevalencia_2025=float(y.mean()),
                           muestras_2024=len(d24), fuegos_2024=int(y24.sum()), modelos="entrenados con todo 2024, sin reentrenar",
                           umbrales=str({k: round(v, 4) for k, v in umbrales.items()}))]).T.reset_index().rename(columns={"index": "parametro", 0: "valor"}
                        ).to_excel(w, sheet_name="configuracion", index=False)
    pd.set_option("display.width", 240)
    print(f"\n================ {ETIQUETA} (modelos de 2024 sin reentrenar) ================")
    print(glob[["detector", "recall", "recall_ic", "precision", "f1", "f1_ic", "fpr", "pr_auc"]].round(3).to_string(index=False))
    print("\nReferencia: validación cruzada 2024"); print(ref[["detector", "recall", "precision", "f1", "fpr", "pr_auc"]].round(3).to_string(index=False))
    print("\nDiferencias pareadas en 2025:"); print(par.round(3).to_string(index=False))
    print(f"\nGuardado: {salida}")
    return glob

if __name__ == "__main__":
    resultado = ejecutar()