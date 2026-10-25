# -*- coding: utf-8 -*-
"""
entrenar_modelo_fuego.py -- Entrena y evalúa un modelo de detección de fuego activo con GOES-16, usando como verdad
los puntos calientes del IDEAM confirmados por las áreas quemadas (salida de descargar_goes_puntos_ideam.py).

PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5.

DISEÑO (documentar en el artículo)
  - Entrada: solo variables radiométricas/atmosféricas de GOES (VARIABLES). NO se usan FireMask, FRP ni DQF_fire
    (son el producto de fuego de GOES: filtrarían la respuesta) ni distancias, ids o tipo de negativo.
  - Validación cruzada agrupada por FECHA (StratifiedGroupKFold): todas las imágenes de un mismo día quedan en el
    mismo pliegue. Evita que el mismo incendio (o el mismo píxel visto por varios satélites el mismo día) esté a la
    vez en entrenamiento y prueba, lo que inflaría las métricas.
  - Modelos: Random Forest y HistGradientBoosting (este último admite valores faltantes de LST/TPW sin imputar).
    Clases desbalanceadas (1 positivo : 4 negativos) -> pesos balanceados.
  - Línea base: el producto operativo de fuego de GOES (FDC, FireMask 10-15 / 30-35) evaluado en las MISMAS filas.
  - Métricas: precisión, recall (sensibilidad), F1, PR-AUC, ROC-AUC; falsos positivos por tipo de negativo
    (near/far); recall según la potencia del fuego (FRP de VIIRS/MODIS) y según si GOES-FDC lo detectó o no.
  - Umbral: 0.5 y el umbral de máximo F1 elegido DENTRO de cada pliegue de entrenamiento (no con los datos de prueba).

SALIDAS (en CARPETA_SALIDA):  metricas.xlsx, predicciones_oof.csv, importancia_variables.csv, figuras PNG,
                               modelo_final_<nombre>.joblib (entrenado con todos los datos)
INSTALACIÓN: conda install scikit-learn joblib matplotlib openpyxl
"""
import os, json, time, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
DATASET = r"C:\Users\User\Desktop\Nuevo Manuscript\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
CARPETA_SALIDA = r"C:\Users\User\Desktop\Nuevo Manuscript\Modelo fuego GOES"

VARIABLES = ["C07", "C14", "T7", "T14", "BTD", "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF", "fire_hour"]
MODELOS = ["hgb", "rf"]      # "hgb" = HistGradientBoosting, "rf" = Random Forest
N_PLIEGUES = 5
SEMILLA = 42
GUARDAR_MODELO_FINAL = True

FUEGO_FDC = list(range(10, 16)) + list(range(30, 36))   # códigos de fuego de FireMask (GOES FDC)
PROHIBIDAS = {"label", "neg_type", "FireMask", "FRP", "FRP_log", "FRP_detected", "DQF_fire", "nearest_hotspot_km",
              "hotspot_ids", "n_hotspots", "cluster_ids", "frp_mw", "confidence_pct", "confidence_class", "pixel_id",
              "image_id", "pix_ix", "pix_iy", "x", "y", "fire_latitude", "fire_longitude"}

# ============================== FUNCIONES ==============================
def cargar(ruta):
    if not os.path.exists(ruta): raise FileNotFoundError(f"No existe el dataset: {ruta}")
    df = pd.read_csv(ruta, low_memory=False)
    falt = [c for c in VARIABLES + ["label", "neg_type", "scan_start_utc", "FireMask"] if c not in df.columns]
    if falt: raise KeyError(f"Faltan columnas en el dataset: {falt}")
    mal = set(VARIABLES) & PROHIBIDAS
    if mal: raise ValueError(f"Estas variables filtran la respuesta y no pueden ser entrada: {sorted(mal)}")
    df["fecha"] = pd.to_datetime(df.scan_start_utc).dt.strftime("%Y-%m-%d")
    df["mes"] = pd.to_datetime(df.scan_start_utc).dt.month
    df["goes_fdc"] = df.FireMask.isin(FUEGO_FDC).astype(int)
    return df

def crear_modelo(nombre):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    if nombre == "hgb":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                              random_state=SEMILLA)
    if nombre == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                                    class_weight="balanced", n_jobs=-1, random_state=SEMILLA))
    raise ValueError(f"Modelo no reconocido: {nombre}")

def pesos(y):
    y = np.asarray(y); w = np.ones(len(y)); p = y.mean()
    w[y == 1] = 0.5 / p; w[y == 0] = 0.5 / (1 - p); return w

def ajustar(modelo, nombre, X, y):
    if nombre == "hgb": modelo.fit(X, y, sample_weight=pesos(y))
    else: modelo.fit(X, y)
    return modelo

def umbral_max_f1(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[np.nanargmax(f1)]) if len(th) else 0.5

def metricas(y, pred, prob=None):
    from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0), tasa_falsos_positivos=fp / max(fp + tn, 1), TP=tp, FP=fp, FN=fn, TN=tn)
    if prob is not None and len(np.unique(y)) == 2:
        m.update(pr_auc=average_precision_score(y, prob), roc_auc=roc_auc_score(y, prob))
    return m

def validar(df, nombre):
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = df[VARIABLES], df.label.to_numpy(), df.fecha.to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    prob = np.full(len(df), np.nan); pred_opt = np.zeros(len(df), int); pliegue = np.full(len(df), -1); umbrales = []
    for k, (tr, te) in enumerate(cv.split(X, y, g), 1):
        t0 = time.time()
        # umbral de máximo F1 con validación interna sobre el entrenamiento (sin mirar el pliegue de prueba)
        cv_in = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA + k)
        p_in = np.full(len(tr), np.nan)
        for a, b in cv_in.split(X.iloc[tr], y[tr], g[tr]):
            mdl = ajustar(crear_modelo(nombre), nombre, X.iloc[tr[a]], y[tr[a]]); p_in[b] = mdl.predict_proba(X.iloc[tr[b]])[:, 1]
        th = umbral_max_f1(y[tr], p_in); umbrales.append(th)
        mdl = ajustar(crear_modelo(nombre), nombre, X.iloc[tr], y[tr])
        prob[te] = mdl.predict_proba(X.iloc[te])[:, 1]; pred_opt[te] = (prob[te] >= th).astype(int); pliegue[te] = k
        print(f"  [{nombre}] pliegue {k}/{N_PLIEGUES}: {len(np.unique(g[te]))} fechas, {int(y[te].sum())} positivos de prueba, "
              f"umbral={th:.3f}, {time.time() - t0:.0f}s")
    return prob, pred_opt, pliegue, umbrales

def tablas_detalle(df, col_pred, nombre):
    filas = []
    for t in ("near", "far"):
        s = df[df.neg_type == t]
        filas.append(dict(modelo=nombre, grupo=f"negativos {t}", n=len(s), tasa_falsos_positivos=s[col_pred].mean()))
    pos = df[df.label == 1]
    for v, s in pos.groupby(pos.goes_fdc):
        filas.append(dict(modelo=nombre, grupo=f"positivos con GOES-FDC={'fuego' if v else 'sin fuego'}", n=len(s), recall=s[col_pred].mean()))
    if "frp_mw" in df.columns and pos.frp_mw.notna().any():
        cortes = pd.cut(pos.frp_mw, [0, 5, 10, 25, 50, 100, np.inf], labels=["<5", "5-10", "10-25", "25-50", "50-100", ">100"])
        for v, s in pos.groupby(cortes, observed=True):
            filas.append(dict(modelo=nombre, grupo=f"positivos FRP {v} MW", n=len(s), recall=s[col_pred].mean()))
    for m, s in pos.groupby(pos.mes):
        filas.append(dict(modelo=nombre, grupo=f"positivos mes {m:02d}", n=len(s), recall=s[col_pred].mean()))
    return filas

def graficas(df, nombres, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in nombres:
        pr, rc, _ = precision_recall_curve(df.label, df[f"prob_{n}"]); ax.plot(rc, pr, label=n.upper())
    m = metricas(df.label, df.goes_fdc)
    ax.scatter([m["recall"]], [m["precision"]], color="black", zorder=5, label="GOES-FDC (operativo)")
    ax.axhline(df.label.mean(), color="grey", ls=":", lw=1, label="azar")
    ax.set_xlabel("Recall (sensibilidad)"); ax.set_ylabel("Precisión"); ax.set_title("Curva precisión-recall (validación cruzada por fecha)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="lower left"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "curva_precision_recall.png"), dpi=200); plt.close(fig)

def importancia(df, nombre, out):
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = df[VARIABLES], df.label.to_numpy(), df.fecha.to_numpy()
    tr, te = next(StratifiedGroupKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA).split(X, y, g))
    mdl = ajustar(crear_modelo(nombre), nombre, X.iloc[tr], y[tr])
    r = permutation_importance(mdl, X.iloc[te], y[te], scoring="average_precision", n_repeats=10, random_state=SEMILLA)
    imp = pd.DataFrame({"variable": VARIABLES, "importancia_media": r.importances_mean, "desv_est": r.importances_std}).sort_values("importancia_media", ascending=False)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5)); s = imp.iloc[::-1]
    ax.barh(s.variable, s.importancia_media, xerr=s.desv_est, color="#4C72B0")
    ax.set_xlabel("Caída de PR-AUC al permutar la variable"); ax.set_title(f"Importancia por permutación ({nombre.upper()})")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(out, f"importancia_{nombre}.png"), dpi=200); plt.close(fig)
    return imp.assign(modelo=nombre)

def ejecutar():
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    df = cargar(DATASET)
    print(f"Filas: {len(df):,} | positivos: {int(df.label.sum()):,} | negativos: {int((df.label == 0).sum()):,} | fechas: {df.fecha.nunique()}")
    print("Faltantes por variable:", {c: int(v) for c, v in df[VARIABLES].isna().sum().items() if v})
    filas_glob, filas_det, imps = [], [], []
    base = metricas(df.label, df.goes_fdc); filas_glob.append(dict(modelo="GOES-FDC (operativo)", umbral="-", **base))
    filas_det += tablas_detalle(df.assign(pred_fdc=df.goes_fdc), "pred_fdc", "GOES-FDC")
    resumen = {}
    for n in MODELOS:
        print(f"\nValidación cruzada: {n.upper()}")
        prob, pred_opt, pliegue, ths = validar(df, n)
        df[f"prob_{n}"] = prob; df[f"pred_{n}"] = pred_opt; df[f"pliegue_{n}"] = pliegue
        m05 = metricas(df.label, (prob >= 0.5).astype(int), prob); mopt = metricas(df.label, pred_opt, prob)
        filas_glob.append(dict(modelo=n.upper(), umbral="0.5", **m05))
        filas_glob.append(dict(modelo=n.upper(), umbral=f"máx F1 interno (media {np.mean(ths):.2f})", **mopt))
        # métricas por pliegue: variabilidad
        for k in sorted(set(pliegue)):
            s = pliegue == k; mk = metricas(df.label[s], pred_opt[s], prob[s])
            filas_glob.append(dict(modelo=f"{n.upper()} pliegue {k}", umbral=f"{ths[k - 1]:.2f}", **mk))
        filas_det += tablas_detalle(df, f"pred_{n}", n.upper())
        imps.append(importancia(df, n, CARPETA_SALIDA))
        resumen[n] = dict(umbral=float(np.mean(ths)), **{k: float(v) for k, v in mopt.items()})
    glob = pd.DataFrame(filas_glob); det = pd.DataFrame(filas_det); imp = pd.concat(imps, ignore_index=True)
    graficas(df, MODELOS, CARPETA_SALIDA)
    with pd.ExcelWriter(os.path.join(CARPETA_SALIDA, "metricas.xlsx")) as w:
        glob.to_excel(w, sheet_name="metricas_globales", index=False); det.to_excel(w, sheet_name="detalle", index=False)
        imp.to_excel(w, sheet_name="importancia", index=False)
        pd.DataFrame({"parametro": ["dataset", "variables", "pliegues", "agrupacion", "semilla"],
                      "valor": [DATASET, ", ".join(VARIABLES), N_PLIEGUES, "fecha (día UTC del escaneo)", SEMILLA]}).to_excel(w, sheet_name="configuracion", index=False)
    cols = ["image_id", "pixel_id", "fecha", "mes", "x", "y", "label", "neg_type", "goes_fdc"] + [c for c in df.columns if c.startswith(("prob_", "pred_", "pliegue_"))]
    df[[c for c in cols if c in df.columns]].to_csv(os.path.join(CARPETA_SALIDA, "predicciones_oof.csv"), index=False)
    imp.to_csv(os.path.join(CARPETA_SALIDA, "importancia_variables.csv"), index=False)
    if GUARDAR_MODELO_FINAL:
        import joblib
        for n in MODELOS:
            mdl = ajustar(crear_modelo(n), n, df[VARIABLES], df.label.to_numpy())
            joblib.dump(dict(modelo=mdl, variables=VARIABLES, umbral=resumen[n]["umbral"]), os.path.join(CARPETA_SALIDA, f"modelo_final_{n}.joblib"))
    pd.set_option("display.width", 200)
    print("\n================ RESULTADOS (validación cruzada por fecha) ================")
    print(glob[~glob.modelo.str.contains("pliegue")][["modelo", "umbral", "precision", "recall", "f1", "pr_auc", "roc_auc", "tasa_falsos_positivos"]].round(3).to_string(index=False))
    print("\nDetalle (recall por grupo de positivos / falsos positivos por tipo de negativo):")
    print(det.round(3).to_string(index=False))
    print("\nImportancia de variables:"); print(imp.round(4).to_string(index=False))
    print(f"\nGuardado en {CARPETA_SALIDA}")
    return df, glob, det, imp

if __name__ == "__main__":
    datos, metricas_globales, detalle, importancias = ejecutar()