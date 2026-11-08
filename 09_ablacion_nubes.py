# -*- coding: utf-8 -*-
"""
09_ablacion_nubes.py -- Bloque 0 de la revisión (R3-C7 y R3-C8). NO modifica el modelo del artículo ni sus salidas.

R3-C7: ¿el modelo separa las clases en parte por la nubosidad? (los negativos lejanos tienen más nubes que los fuegos)
       -> se repite EXACTAMENTE la validación del script 07 cambiando solo la lista de variables:
          completo (12 variables, el modelo del artículo) | sin nubes | sin nubes ni LST_BT_DIFF | solo nubes
R3-C8: ¿los faltantes de LST/TPW podrían delatar la etiqueta? -> porcentaje de faltantes por clase (sin entrenar).

CÓMO FUNCIONA
  - Importa "07. Entrenamiento del modelo.py" y usa sus funciones (cargar, crear_modelo, validar, metricas):
    mismos hiperparámetros, mismos pliegues por fecha (semilla 42) y mismo umbral de máximo F1 interno.
  - Como los pliegues dependen solo de la etiqueta y de la fecha, son idénticos en todas las variantes: la comparación
    es pareada, muestra por muestra.
  - La variante "completo" debe reproducir las cifras del artículo (HGB: F1 = 0.858, PR-AUC = 0.932). Si no coinciden,
    el dataset o el script no son los de la versión enviada: deténgase y revise antes de interpretar nada.

PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5. Tiempo aprox.: 1-3 min por variante con HGB; RF tarda más.
SALIDA: ablacion_nubes.xlsx en CARPETA_SALIDA (hojas: metricas, diferencias_vs_completo, faltantes_R3C8).
"""
import os, sys, time, importlib.util, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
SCRIPT_07 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\07. Entrenamiento del modelo.py"
DATASET = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
CARPETA_SALIDA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\Ablación nubes ambos modelos"
MODELOS = ["hgb","rf"]                    # añada "rf" si también quiere RF (más lento)
N_BOOTSTRAP = 1000                   # remuestreos de FECHAS para el IC de las diferencias
SEMILLA = 42

BASE = ["C07", "C14", "T7", "T14", "BTD", "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF", "fire_hour"]
VARIANTES = {
    "completo": BASE,                                                               # el modelo del artículo
    "sin_nubes": [v for v in BASE if v != "Cloud_Probabilities"],
    "sin_nubes_ni_LST_BT_DIFF": [v for v in BASE if v not in ("Cloud_Probabilities", "LST_BT_DIFF")],  # r = 0.63 con nubes
    "solo_nubes": ["Cloud_Probabilities"],
}
FPR_FIJA = 0.01

# ============================== FUNCIONES ==============================
def importar_07(ruta):
    if not os.path.exists(ruta): sys.exit(f"[DETENIDO] No existe el script 07: {ruta}")
    spec = importlib.util.spec_from_file_location("entrenamiento07", ruta)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)   # no ejecuta ejecutar(): está bajo __main__
    return m

def recall_a_fpr(y, prob, f):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, prob); i = np.searchsorted(fpr, f, side="right") - 1
    return float(tpr[i])

def resumen(m07, df, prob, pred, umbrales, variante, modelo):
    from sklearn.metrics import matthews_corrcoef
    y = df.label.to_numpy(); m = m07.metricas(y, pred, prob)
    neg = df.label == 0
    return dict(variante=variante, modelo=modelo.upper(), n_variables=len(VARIANTES[variante]),
                umbral_medio=round(float(np.mean(umbrales)), 3), precision=m["precision"], recall=m["recall"], f1=m["f1"],
                mcc=matthews_corrcoef(y, pred), pr_auc=m["pr_auc"], roc_auc=m["roc_auc"], fpr=m["tasa_falsos_positivos"],
                fpr_near=float(pred[(neg & (df.neg_type == "near")).to_numpy()].mean()),
                fpr_far=float(pred[(neg & (df.neg_type == "far")).to_numpy()].mean()),
                **{f"recall_a_FPR_{FPR_FIJA:.0%}": recall_a_fpr(y, prob, FPR_FIJA)}, TP=m["TP"], FP=m["FP"], FN=m["FN"], TN=m["TN"])

def bootstrap_diferencia(df, prob_a, pred_a, prob_b, pred_b):
    """IC 95 % de (variante - completo) remuestreando FECHAS completas, como el script 08."""
    from sklearn.metrics import average_precision_score, f1_score
    rng = np.random.default_rng(SEMILLA); idx = df.groupby("fecha").indices; fechas = np.array(list(idx))
    y = df.label.to_numpy()
    def d(f):
        return (average_precision_score(y[f], prob_a[f]) - average_precision_score(y[f], prob_b[f]),
                f1_score(y[f], pred_a[f]) - f1_score(y[f], pred_b[f]))
    base = d(np.arange(len(y))); res = []
    for _ in range(N_BOOTSTRAP):
        f = np.concatenate([idx[x] for x in rng.choice(fechas, len(fechas), replace=True)])
        if y[f].min() == y[f].max(): continue
        res.append(d(f))
    R = np.array(res)
    return dict(dif_pr_auc=base[0], dif_pr_auc_ic_inf=np.quantile(R[:, 0], .025), dif_pr_auc_ic_sup=np.quantile(R[:, 0], .975),
                dif_f1=base[1], dif_f1_ic_inf=np.quantile(R[:, 1], .025), dif_f1_ic_sup=np.quantile(R[:, 1], .975), n_bootstrap=len(R))

def faltantes(df):
    filas = []
    for c in ("LST", "TPW", "LST_BT_DIFF"):
        for nombre, s in (("fuego", df[df.label == 1]), ("negativo near", df[df.neg_type == "near"]),
                          ("negativo far", df[df.neg_type == "far"]), ("todos", df)):
            filas.append(dict(variable=c, grupo=nombre, n=len(s), faltantes=int(s[c].isna().sum()), pct_faltantes=100 * s[c].isna().mean()))
    return pd.DataFrame(filas)

# ============================== PRINCIPAL ==============================
def ejecutar():
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    m07 = importar_07(SCRIPT_07)
    m07.VARIABLES = BASE
    df = m07.cargar(DATASET)                       # mismas comprobaciones de columnas que el script 07
    print(f"Filas: {len(df):,} | positivos: {int(df.label.sum()):,} | fechas: {df.fecha.nunique()}")

    falt = faltantes(df)                           # R3-C8 (no entrena nada)
    print("\nR3-C8, faltantes (%):"); print(falt.pivot(index="grupo", columns="variable", values="pct_faltantes").round(1).to_string())

    filas, preds = [], {}
    for modelo in MODELOS:
        for variante, cols in VARIANTES.items():
            m07.VARIABLES = cols                   # validar() lee esta lista global del script 07
            t0 = time.time(); print(f"\n[{modelo.upper()}] {variante}: {len(cols)} variables")
            prob, pred, pliegue, ths = m07.validar(df, modelo)
            preds[(modelo, variante)] = (prob, pred, pliegue)
            filas.append(resumen(m07, df, prob, pred, ths, variante, modelo))
            print(f"   F1 = {filas[-1]['f1']:.3f} | PR-AUC = {filas[-1]['pr_auc']:.3f} | {time.time() - t0:.0f} s")
    m07.VARIABLES = BASE
    met = pd.DataFrame(filas)

    # comprobación: mismos pliegues en todas las variantes (comparación pareada)
    for modelo in MODELOS:
        ref = preds[(modelo, "completo")][2]
        assert all((preds[(modelo, v)][2] == ref).all() for v in VARIANTES), "Los pliegues cambiaron entre variantes"

    dif = []
    for modelo in MODELOS:
        pa, ya = preds[(modelo, "completo")][:2]
        for variante in VARIANTES:
            if variante == "completo": continue
            pb, yb = preds[(modelo, variante)][:2]
            dif.append(dict(modelo=modelo.upper(), variante=variante, **bootstrap_diferencia(df, pb, yb, pa, ya)))
    dif = pd.DataFrame(dif)

    salida = os.path.join(CARPETA_SALIDA, "ablacion_nubes.xlsx")
    with pd.ExcelWriter(salida) as w:
        met.to_excel(w, sheet_name="metricas", index=False)
        dif.to_excel(w, sheet_name="diferencias_vs_completo", index=False)
        falt.to_excel(w, sheet_name="faltantes_R3C8", index=False)
        pd.DataFrame({"parametro": ["script_07", "dataset", "variantes", "n_bootstrap", "semilla", "linea_base_pr_auc"],
                      "valor": [SCRIPT_07, DATASET, "; ".join(f"{k}: {', '.join(v)}" for k, v in VARIANTES.items()), N_BOOTSTRAP, SEMILLA,
                                round(float(df.label.mean()), 3)]}).to_excel(w, sheet_name="configuracion", index=False)

    pd.set_option("display.width", 220)
    print("\n================ ABLACIÓN DE NUBES (validación agrupada por fecha) ================")
    print(met[["modelo", "variante", "n_variables", "umbral_medio", "recall", "precision", "f1", "mcc", "pr_auc", "fpr", "fpr_near", "fpr_far"]].round(3).to_string(index=False))
    print("\nDiferencia respecto al modelo completo (IC 95 %, bootstrap por fecha):")
    print(dif.round(3).to_string(index=False))
    print(f"\nLínea base de PR-AUC (prevalencia de fuego): {df.label.mean():.3f}")
    c = met[(met.variante == "completo") & (met.modelo == "HGB")]
    if len(c) and (abs(c.f1.iloc[0] - 0.858) > 0.005 or abs(c.pr_auc.iloc[0] - 0.932) > 0.005):
        print("[AVISO] La variante 'completo' NO reproduce el artículo (HGB F1 0.858, PR-AUC 0.932). Revise dataset y script antes de interpretar.")
    print(f"\nGuardado: {salida}")
    return met, dif, falt

if __name__ == "__main__":
    metricas_ablacion, diferencias, faltantes_tabla = ejecutar()
