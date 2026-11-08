# -*- coding: utf-8 -*-
"""
11_validacion_espacial_y_variantes.py -- Revisión, ronda 1. Una sola corrida para cuatro observaciones:
  R2-C1 / R3-C9  validación por BLOQUES ESPACIALES (el mismo píxel o incendio nunca queda en entrenamiento y prueba)
                 + cuántos píxeles de fuego de prueba ya estaban en entrenamiento con la validación por fecha (fuga)
  R3-C9          early stopping de HGB AGRUPADO (número de iteraciones elegido con grupos, no con un 15 % aleatorio)
  R3-C3          ángulo cenital solar (SZA) en lugar de fire_hour
  R3-C10         líneas base simples: umbral sobre BTD y regresión logística con T7 y BTD
NO modifica el modelo del artículo ni sus salidas.

CÓMO FUNCIONA
  - Importa "07. Entrenamiento del modelo.py" y usa sus funciones (cargar, crear_modelo, ajustar, umbral_max_f1, metricas):
    mismos hiperparámetros y mismo umbral de máximo F1 elegido con validación interna de 3 pliegues.
  - validar_grupos() repite exactamente la lógica de validar() del script 07, pero con el agrupamiento que se le indique.
  - El experimento "control" (grupos = fecha, 12 variables) DEBE reproducir el artículo (HGB F1 = 0.858, PR-AUC = 0.932;
    RF F1 = 0.858, PR-AUC = 0.931). Si no, no interprete nada.
  - Bloques espaciales: celdas de BLOQUE_GRADOS × BLOQUE_GRADOS grados según la posición (x = lon, y = lat) del píxel.
    Todas las fechas de un bloque van al mismo pliegue, así que un píxel nunca aparece en entrenamiento y prueba a la vez.

PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5. Tiempo: HGB 1-3 min por experimento; RF bastante más.
Desactive experimentos en EXPERIMENTOS si quiere una corrida más corta.
SALIDA (CARPETA_SALIDA): validacion_espacial_y_variantes.xlsx (metricas, diferencias_vs_control, pliegues, fuga_pixeles,
        configuracion) y predicciones_<experimento>.csv con las probabilidades out-of-fold.
"""
import os, sys, time, importlib.util, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
SCRIPT_07 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\07. Entrenamiento del modelo.py"
DATASET = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
CARPETA_SALIDA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Validacion espacial y variantes"
BLOQUE_GRADOS = 0.5           # tamaño principal del bloque espacial (0.5° ≈ 55 km)
BLOQUE_GRADOS_SENS = 1.0      # tamaño alternativo (sensibilidad al tamaño del bloque)
N_BOOTSTRAP = 1000
SEMILLA = 42

BASE = ["C07", "C14", "T7", "T14", "BTD", "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF", "fire_hour"]
CON_SZA = [v for v in BASE if v != "fire_hour"] + ["SZA"]

# (nombre, agrupamiento, modelo, variables, observación)   agrupamiento: "fecha" | "bloque" | "bloque_sens"
# modelo: "hgb" | "rf" (como en el script 07) | "hgb_es_grupos" | "umbral_btd" | "logistica"
EXPERIMENTOS = [
    ("control_hgb",              "fecha",       "hgb",           BASE,          "control (artículo)"),
    ("control_rf",               "fecha",       "rf",            BASE,          "control (artículo)"),
    ("espacial_hgb",             "bloque",      "hgb",           BASE,          "R2-C1, R3-C9"),
    ("espacial_rf",              "bloque",      "rf",            BASE,          "R2-C1, R3-C9"),
    ("espacial_1grado_hgb",      "bloque_sens", "hgb",           BASE,          "R2-C1 (tamaño de bloque)"),
    ("espacial_1grado_rf",       "bloque_sens", "rf",            BASE,          "R2-C1 (tamaño de bloque)"),
    ("es_grupos_hgb",            "fecha",       "hgb_es_grupos", BASE,          "R3-C9"),
    ("es_grupos_espacial_hgb",   "bloque",      "hgb_es_grupos", BASE,          "R3-C9"),
    ("sza_hgb",                  "fecha",       "hgb",           CON_SZA,       "R3-C3"),
    ("sza_rf",                   "fecha",       "rf",            CON_SZA,       "R3-C3"),
    ("linea_base_btd",           "fecha",       "umbral_btd",    ["BTD"],       "R3-C10"),
    ("linea_base_logistica",     "fecha",       "logistica",     ["T7", "BTD"], "R3-C10"),
    ("linea_base_btd_espacial",  "bloque",      "umbral_btd",    ["BTD"],       "R3-C10"),
    ("linea_base_logistica_esp", "bloque",      "logistica",     ["T7", "BTD"], "R3-C10"),
]
FPR_FIJA = 0.01

# ============================== UTILIDADES ==============================
def importar_07(ruta):
    if not os.path.exists(ruta): sys.exit(f"[DETENIDO] No existe el script 07: {ruta}")
    spec = importlib.util.spec_from_file_location("entrenamiento07", ruta)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def angulo_cenital_solar(lat, lon, t_utc):
    """Ángulo cenital solar (grados), algoritmo de NOAA (ecuación del tiempo y declinación). t_utc: Serie datetime UTC."""
    t = pd.to_datetime(pd.Series(t_utc).astype(str), format="ISO8601")   # admite "18:05:00" y "18:05:00.500000"
    doy = t.dt.dayofyear.to_numpy(); h = (t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (h - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = h * 60 + eqt + 4 * np.asarray(lon, float)
    ha = np.radians(tst / 4 - 180); la = np.radians(np.asarray(lat, float))
    c = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def recall_a_fpr(y, prob, f):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, prob); i = np.searchsorted(fpr, f, side="right") - 1
    return float(tpr[i])

# ============================== MODELOS ADICIONALES ==============================
class UmbralBTD:
    """Línea base: la 'probabilidad' es el propio BTD; el umbral se elige igual que en los modelos (máximo F1 interno)."""
    def fit(self, X, y): return self
    def predict_proba(self, X):
        s = X["BTD"].to_numpy(float); return np.column_stack([-s, s])

def logistica():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

class HGBEarlyStoppingGrupos:
    """HGB con los hiperparámetros del script 07, pero el número de iteraciones se elige con validación interna AGRUPADA
    (los mismos grupos del pliegue): se entrena sin early stopping y se toma la iteración de menor log-loss en los grupos
    retenidos. Así ninguna fecha o bloque se usa a la vez para ajustar y para detener el entrenamiento."""
    def __init__(self, m07, n_iter=None): self.m07, self.n_iter = m07, n_iter
    def _base(self, max_iter):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=1.0, early_stopping=False, random_state=self.m07.SEMILLA)
    def elegir_iteraciones(self, X, y, g):
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import log_loss
        mejores = []
        for a, b in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA + 100).split(X, y, g):
            m = self._base(400).fit(X.iloc[a], y[a], sample_weight=self.m07.pesos(y[a]))
            perd = [log_loss(y[b], p[:, 1], sample_weight=self.m07.pesos(y[b]), labels=[0, 1]) for p in m.staged_predict_proba(X.iloc[b])]
            mejores.append(int(np.argmin(perd)) + 1)
        self.n_iter = int(np.median(mejores)); return self.n_iter
    def fit(self, X, y):
        self.modelo = self._base(self.n_iter or 400).fit(X, y, sample_weight=self.m07.pesos(y)); return self
    def predict_proba(self, X): return self.modelo.predict_proba(X)

# ============================== VALIDACIÓN (misma lógica que validar() del script 07) ==============================
def validar_grupos(m07, df, modelo, variables, grupos):
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = df[variables], df.label.to_numpy(), np.asarray(grupos)
    cv = StratifiedGroupKFold(n_splits=m07.N_PLIEGUES, shuffle=True, random_state=m07.SEMILLA)
    prob = np.full(len(df), np.nan); pred = np.zeros(len(df), int); pliegue = np.full(len(df), -1); ths, iters = [], []

    n_it = None
    def nuevo(Xtr=None, ytr=None, gtr=None):
        if modelo in ("hgb", "rf"): return m07.crear_modelo(modelo)
        if modelo == "umbral_btd": return UmbralBTD()
        if modelo == "logistica": return logistica()
        if modelo == "hgb_es_grupos": return HGBEarlyStoppingGrupos(m07, n_it)
        raise ValueError(modelo)
    def ajustar(mdl, Xa, ya):
        if modelo in ("hgb", "rf"): return m07.ajustar(mdl, modelo, Xa, ya)
        return mdl.fit(Xa, ya)

    for k, (tr, te) in enumerate(cv.split(X, y, g), 1):
        # early stopping agrupado: el número de iteraciones se elige UNA vez por pliegue, con grupos retenidos del entrenamiento
        n_it = HGBEarlyStoppingGrupos(m07).elegir_iteraciones(X.iloc[tr], y[tr], g[tr]) if modelo == "hgb_es_grupos" else None
        cv_in = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=m07.SEMILLA + k)
        p_in = np.full(len(tr), np.nan)
        for a, b in cv_in.split(X.iloc[tr], y[tr], g[tr]):
            mdl = ajustar(nuevo(X.iloc[tr[a]], y[tr[a]], g[tr[a]]), X.iloc[tr[a]], y[tr[a]])
            p_in[b] = mdl.predict_proba(X.iloc[tr[b]])[:, 1]
        th = m07.umbral_max_f1(y[tr], p_in); ths.append(th)
        mdl = ajustar(nuevo(X.iloc[tr], y[tr], g[tr]), X.iloc[tr], y[tr])
        if modelo == "hgb_es_grupos": iters.append(mdl.n_iter)
        elif modelo == "hgb": iters.append(int(mdl.n_iter_))
        prob[te] = mdl.predict_proba(X.iloc[te])[:, 1]; pred[te] = (prob[te] >= th).astype(int); pliegue[te] = k
    return prob, pred, pliegue, ths, iters

# ============================== MÉTRICAS Y DIAGNÓSTICOS ==============================
def resumen(m07, df, prob, pred, ths, iters, nombre, agrup, modelo, variables, obs):
    from sklearn.metrics import matthews_corrcoef
    y = df.label.to_numpy(); m = m07.metricas(y, pred, prob); neg = df.label == 0
    return dict(experimento=nombre, observacion=obs, agrupamiento=agrup, modelo=modelo, n_variables=len(variables),
                umbral_medio=float(np.mean(ths)), iteraciones_hgb=(np.median(iters) if iters else np.nan),
                precision=m["precision"], recall=m["recall"], f1=m["f1"], mcc=matthews_corrcoef(y, pred), pr_auc=m["pr_auc"],
                roc_auc=m["roc_auc"], fpr=m["tasa_falsos_positivos"],
                fpr_near=float(pred[(neg & (df.neg_type == "near")).to_numpy()].mean()),
                fpr_far=float(pred[(neg & (df.neg_type == "far")).to_numpy()].mean()),
                **{f"recall_a_FPR_{FPR_FIJA:.0%}": recall_a_fpr(y, prob, FPR_FIJA)}, TP=m["TP"], FP=m["FP"], FN=m["FN"], TN=m["TN"])

def ic_bootstrap(df, prob, pred, unidad):
    """IC 95 % de PR-AUC, F1 y recall remuestreando la unidad de agrupamiento (fecha o bloque)."""
    from sklearn.metrics import average_precision_score, f1_score, recall_score
    rng = np.random.default_rng(SEMILLA); idx = pd.Series(np.arange(len(df))).groupby(np.asarray(unidad)).indices; u = np.array(list(idx))
    y = df.label.to_numpy(); R = []
    for _ in range(N_BOOTSTRAP):
        f = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if y[f].min() == y[f].max(): continue
        R.append((average_precision_score(y[f], prob[f]), f1_score(y[f], pred[f]), recall_score(y[f], pred[f])))
    R = np.array(R); q = lambda j, p: float(np.quantile(R[:, j], p))
    return dict(pr_auc_ic=f"[{q(0, .025):.3f}–{q(0, .975):.3f}]", f1_ic=f"[{q(1, .025):.3f}–{q(1, .975):.3f}]",
                recall_ic=f"[{q(2, .025):.3f}–{q(2, .975):.3f}]")

def diferencia_pareada(df, pa, ya, pb, yb):
    """(experimento − control del mismo algoritmo), bootstrap pareado por FECHA, como en el artículo."""
    from sklearn.metrics import average_precision_score, f1_score
    rng = np.random.default_rng(SEMILLA); idx = df.groupby("fecha").indices; fe = np.array(list(idx)); y = df.label.to_numpy()
    d = lambda f: (average_precision_score(y[f], pa[f]) - average_precision_score(y[f], pb[f]), f1_score(y[f], ya[f]) - f1_score(y[f], yb[f]))
    base = d(np.arange(len(y))); R = []
    for _ in range(N_BOOTSTRAP):
        f = np.concatenate([idx[x] for x in rng.choice(fe, len(fe), replace=True)])
        if y[f].min() == y[f].max(): continue
        R.append(d(f))
    R = np.array(R)
    return dict(dif_pr_auc=base[0], dif_pr_auc_ic_inf=np.quantile(R[:, 0], .025), dif_pr_auc_ic_sup=np.quantile(R[:, 0], .975),
                dif_f1=base[1], dif_f1_ic_inf=np.quantile(R[:, 1], .025), dif_f1_ic_sup=np.quantile(R[:, 1], .975))

def composicion_pliegues(df, pliegue, grupos, nombre):
    filas = []
    for k in sorted(set(pliegue)):
        te = pliegue == k; tr = ~te; pos = df.label.to_numpy() == 1
        px_te = set(df.pixel_id[te & pos]); px_tr = set(df.pixel_id[tr & pos])
        filas.append(dict(experimento=nombre, pliegue=k, fechas=df.fecha[te].nunique(), grupos=len(set(np.asarray(grupos)[te])),
                          muestras=int(te.sum()), fuegos=int((te & pos).sum()), pixeles_fuego_distintos=len(px_te),
                          pixeles_fuego_tambien_en_entrenamiento=len(px_te & px_tr),
                          pct_muestras_fuego_con_pixel_en_entrenamiento=100 * float(df.pixel_id[te & pos].isin(px_tr).mean()) if (te & pos).any() else np.nan))
    return filas

# ============================== PRINCIPAL ==============================
def ejecutar():
    warnings.filterwarnings("ignore")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    m07 = importar_07(SCRIPT_07); m07.VARIABLES = BASE
    df = m07.cargar(DATASET)
    falt = [c for c in ("x", "y", "pixel_id", "scan_mid_utc", "neg_type") if c not in df.columns]
    if falt: sys.exit(f"[DETENIDO] Faltan columnas en el dataset: {falt}")
    df["SZA"] = angulo_cenital_solar(df.y, df.x, df.scan_mid_utc)
    bloque = lambda s: (np.floor(df.x / s).astype(int).astype(str) + "_" + np.floor(df.y / s).astype(int).astype(str)).to_numpy()
    grupos = {"fecha": df.fecha.to_numpy(), "bloque": bloque(BLOQUE_GRADOS), "bloque_sens": bloque(BLOQUE_GRADOS_SENS)}
    print(f"Filas: {len(df):,} | fuegos: {int(df.label.sum()):,} | fechas: {df.fecha.nunique()} | "
          f"bloques {BLOQUE_GRADOS}°: {len(set(grupos['bloque']))} | bloques {BLOQUE_GRADOS_SENS}°: {len(set(grupos['bloque_sens']))}")
    print(f"SZA: día {df.SZA[df.fire_hour.between(12, 23)].median():.0f}° | noche {df.SZA[~df.fire_hour.between(12, 23)].median():.0f}° (medianas)")

    filas, preds, pl_filas = [], {}, []
    for nombre, agrup, modelo, variables, obs in EXPERIMENTOS:
        t0 = time.time(); print(f"\n[{nombre}] grupos={agrup} modelo={modelo} variables={len(variables)}")
        prob, pred, pliegue, ths, iters = validar_grupos(m07, df, modelo, variables, grupos[agrup])
        preds[nombre] = (prob, pred)
        r = resumen(m07, df, prob, pred, ths, iters, nombre, agrup, modelo, variables, obs)
        r.update(ic_bootstrap(df, prob, pred, grupos[agrup])); filas.append(r)
        pl_filas += composicion_pliegues(df, pliegue, grupos[agrup], nombre)
        pd.DataFrame({"image_id": df.get("image_id"), "pixel_id": df.pixel_id, "fecha": df.fecha, "label": df.label, "neg_type": df.neg_type,
                      "goes_fdc": df.goes_fdc, "pliegue": pliegue, "prob": prob, "pred": pred}).to_csv(
            os.path.join(CARPETA_SALIDA, f"predicciones_{nombre}.csv"), index=False)
        print(f"   F1 = {r['f1']:.3f} {r['f1_ic']} | PR-AUC = {r['pr_auc']:.3f} {r['pr_auc_ic']} | recall = {r['recall']:.3f} | {time.time() - t0:.0f} s")
    met = pd.DataFrame(filas); plg = pd.DataFrame(pl_filas)

    control = {"hgb": "control_hgb", "rf": "control_rf", "hgb_es_grupos": "control_hgb", "umbral_btd": "control_hgb", "logistica": "control_hgb"}
    dif = []
    for nombre, agrup, modelo, variables, obs in EXPERIMENTOS:
        c = control[modelo]
        if nombre == c or c not in preds: continue
        dif.append(dict(experimento=nombre, comparado_con=c, **diferencia_pareada(df, *preds[nombre], *preds[c])))
    dif = pd.DataFrame(dif)
    fuga = plg.groupby("experimento")[["fuegos", "pixeles_fuego_distintos", "pixeles_fuego_tambien_en_entrenamiento"]].sum()
    fuga["pct_pixeles_fuego_de_prueba_vistos_en_entrenamiento"] = 100 * fuga.pixeles_fuego_tambien_en_entrenamiento / fuga.pixeles_fuego_distintos
    fuga = fuga.reset_index()

    salida = os.path.join(CARPETA_SALIDA, "validacion_espacial_y_variantes.xlsx")
    with pd.ExcelWriter(salida) as w:
        met.to_excel(w, sheet_name="metricas", index=False); dif.to_excel(w, sheet_name="diferencias_vs_control", index=False)
        plg.to_excel(w, sheet_name="pliegues", index=False); fuga.to_excel(w, sheet_name="fuga_pixeles", index=False)
        pd.DataFrame({"parametro": ["script_07", "dataset", "bloque_grados", "bloque_grados_sens", "n_bootstrap", "semilla", "linea_base_pr_auc",
                                    "nota_ic", "nota_diferencias"],
                      "valor": [SCRIPT_07, DATASET, BLOQUE_GRADOS, BLOQUE_GRADOS_SENS, N_BOOTSTRAP, SEMILLA, round(float(df.label.mean()), 3),
                                "IC de cada experimento: bootstrap de su unidad de agrupamiento (fechas o bloques)",
                                "Diferencias: bootstrap pareado por fecha frente al control del mismo algoritmo"]}).to_excel(w, sheet_name="configuracion", index=False)

    pd.set_option("display.width", 240)
    print("\n================ RESULTADOS ================")
    print(met[["experimento", "agrupamiento", "modelo", "recall", "precision", "f1", "f1_ic", "mcc", "pr_auc", "pr_auc_ic", "fpr", "iteraciones_hgb"]].round(3).to_string(index=False))
    print("\nDiferencias frente al control del mismo algoritmo (bootstrap pareado por fecha):"); print(dif.round(3).to_string(index=False))
    print("\nPíxeles de fuego de prueba que también estaban en entrenamiento:"); print(fuga.round(1).to_string(index=False))
    for nm, ref in (("control_hgb", (0.858, 0.932)), ("control_rf", (0.858, 0.931))):
        r = met[met.experimento == nm]
        if len(r) and (abs(r.f1.iloc[0] - ref[0]) > 0.005 or abs(r.pr_auc.iloc[0] - ref[1]) > 0.005):
            print(f"[AVISO] {nm} NO reproduce el artículo (F1 {ref[0]}, PR-AUC {ref[1]}). Revise antes de interpretar.")
    print(f"\nGuardado: {salida}")
    return met, dif, plg, fuga

if __name__ == "__main__":
    metricas, diferencias, pliegues, fuga = ejecutar()