# -*- coding: utf-8 -*-
"""
10_comparacion_FDC.py -- Revisión, ronda 1. Comparación con GOES-FDC y soporte estadístico, SIN reentrenar.
  R2-C4  (a) puntos de operación de FDC por categoría de máscara, para comparar curva contra curva con HGB y RF
         (b) pruebas pareadas sobre las mismas muestras: HGB vs FDC, RF vs FDC y HGB vs RF
             (bootstrap pareado por fecha, que respeta la dependencia entre muestras del mismo día, y McNemar exacto
              como referencia, aunque supone muestras independientes)
  R2-C2  precisión corregida por prevalencia y falsas alarmas esperadas por cada 10 000 píxeles evaluados
  Tabla A4  verificación de los grupos (51, 54, 62) e IC exactos de la fila VIIRS + MODIS, con la misma lógica del script 08

ENTRADAS (no se descarga ni entrena nada):
  - predicciones_oof.csv del script 07 (prob_hgb, pred_hgb, prob_rf, pred_rf, goes_fdc, label, fecha, neg_type ...)
  - dataset_goes16_2024.csv del script 01 (FireMask, frp_mw, source, day_night ...)
PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5. Tarda pocos minutos.
SALIDA: comparacion_FDC.xlsx en CARPETA_SALIDA.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
PREDICCIONES = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Modelo fuego GOES\predicciones_oof.csv"
DATASET = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
CARPETA_SALIDA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Comparacion FDC"
MODELOS = ["hgb", "rf"]
N_BOOTSTRAP = 1000
N_BOOTSTRAP_GRUPOS = 500          # como el script 08 para el recall por grupo (Tabla A4)
SEMILLA = 42
PREVALENCIAS = [0.20, 0.05, 0.01, 0.001, 0.0001]   # 0.20 = la de la muestra; las demás, escenarios
FPR_FIJAS = [0.01, 0.001]

# Categorías de la máscara de FDC (ATBD / producto ABI-L2-FDC). 30-35 = las mismas, marcadas como "filtradas".
CATEGORIAS = {10: "Processed", 11: "Saturated", 12: "Cloud contaminated", 13: "High probability",
              14: "Medium probability", 15: "Low probability"}
# Puntos de operación acumulados, de la definición más estricta a la usada en el artículo (10-15 y 30-35)
NIVELES_FDC = [
    ("FDC-1: processed + saturated", [10, 11]),
    ("FDC-2: + high probability", [10, 11, 13]),
    ("FDC-3: + medium probability", [10, 11, 13, 14]),
    ("FDC-4: + low probability", [10, 11, 13, 14, 15]),
    ("FDC-5: + cloud contaminated (10-15)", [10, 11, 12, 13, 14, 15]),
    ("FDC-6: + filtered codes 30-35 (artículo)", [10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35]),
]

# ============================== DATOS ==============================
def cargar():
    for r in (PREDICCIONES, DATASET):
        if not os.path.exists(r): sys.exit(f"[DETENIDO] No existe: {r}")
    p = pd.read_csv(PREDICCIONES); d = pd.read_csv(DATASET, low_memory=False)
    need = {"image_id", "pixel_id", "label", "fecha", "neg_type", "goes_fdc"} | {f"{c}_{m}" for m in MODELOS for c in ("prob", "pred")}
    falt = need - set(p.columns)
    if falt: sys.exit(f"[DETENIDO] Faltan columnas en predicciones: {sorted(falt)}")
    extra = [c for c in ("FireMask", "frp_mw", "source", "day_night", "scan_start_utc") if c in d.columns and c not in p.columns]
    df = p.merge(d[["image_id", "pixel_id"] + extra], on=["image_id", "pixel_id"], how="left", validate="one_to_one")
    if df.FireMask.isna().all(): sys.exit("[DETENIDO] No se pudo unir predicciones y dataset (¿son de la misma corrida?).")
    df["FireMask"] = df.FireMask.fillna(-1).astype(int)
    chk = df.FireMask.isin(NIVELES_FDC[-1][1]).astype(int)
    if (chk != df.goes_fdc).any(): print(f"[AVISO] goes_fdc difiere de FireMask 10-15/30-35 en {(chk != df.goes_fdc).sum()} filas")
    s = df.get("source", pd.Series("", index=df.index)).fillna("").astype(str)
    df["sensor"] = np.select([s.str.contains("VIIRS") & s.str.contains("MODIS"), s.str.contains("VIIRS"), s.str.contains("MODIS")],
                             ["VIIRS + MODIS", "VIIRS", "MODIS"], "")
    hl = (pd.to_datetime(df.scan_start_utc).dt.hour - 5) % 24
    dn = df.get("day_night").astype(str).str.upper()
    df["periodo"] = np.where(dn.str.startswith("N"), "Night", np.where(dn.str.startswith("D"), "Day", np.where((hl >= 6) & (hl < 18), "Day", "Night")))
    df["mes"] = pd.to_datetime(df.fecha).dt.month
    print(f"Filas: {len(df):,} | fuegos: {int(df.label.sum()):,} | no fuego: {int((df.label == 0).sum()):,} | fechas: {df.fecha.nunique()}")
    return df

# ============================== MÉTRICAS ==============================
def conf(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1); pre = tp / max(tp + fp, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, recall=rec, fpr=fpr, precision=pre, f1=2 * pre * rec / max(pre + rec, 1e-12))

def umbral_para_fpr(y, prob, fpr_obj):
    """Umbral más bajo cuya FPR no supera fpr_obj (recall máximo con esa tolerancia)."""
    neg = np.sort(np.asarray(prob)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(fpr_obj * len(neg)))                 # negativos permitidos
    return np.nextafter(neg[k], np.inf) if k < len(neg) else -np.inf

def recall_para_fpr(y, prob, fpr_obj):
    th = umbral_para_fpr(y, prob, fpr_obj); return conf(y, (np.asarray(prob) >= th).astype(int))

def fpr_para_recall(y, prob, rec_obj):
    pos = np.sort(np.asarray(prob)[np.asarray(y) == 1])[::-1]
    k = max(int(np.ceil(rec_obj * len(pos))) - 1, 0); th = pos[k]
    return conf(y, (np.asarray(prob) >= th).astype(int))

def bootstrap_fechas(df, f, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEMILLA); idx = df.groupby("fecha").indices; fe = np.array(list(idx)); R = []
    for _ in range(n):
        rows = np.concatenate([idx[x] for x in rng.choice(fe, len(fe), replace=True)])
        s = df.iloc[rows]
        if s.label.nunique() < 2: continue
        R.append(f(s))
    return pd.DataFrame(R)

def ic(R, c): return float(np.quantile(R[c], .025)), float(np.quantile(R[c], .975))

def mcnemar_exacto(a_ok, b_ok):
    """McNemar exacto (binomial) sobre aciertos pareados. Supone independencia: solo como referencia."""
    from scipy.stats import binomtest
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0)

# ============================== ANÁLISIS ==============================
def puntos_operacion_fdc(df):
    y = df.label.to_numpy(); filas = []
    for nombre, codigos in NIVELES_FDC:
        fd = df.FireMask.isin(codigos).astype(int).to_numpy(); m = conf(y, fd)
        fila = dict(punto_FDC=nombre, codigos=", ".join(map(str, codigos)), **{k: m[k] for k in ("recall", "fpr", "precision", "f1", "TP", "FP")})
        for mod in MODELOS:
            pr = df[f"prob_{mod}"].to_numpy()
            a = recall_para_fpr(y, pr, m["fpr"]); b = fpr_para_recall(y, pr, m["recall"])
            fila[f"recall_{mod}_misma_FPR"] = a["recall"]; fila[f"FP_{mod}_misma_FPR"] = a["FP"]
            fila[f"FPR_{mod}_mismo_recall"] = b["fpr"]; fila[f"FP_{mod}_mismo_recall"] = b["FP"]
        # IC pareado de (recall del modelo a la misma FPR - recall de FDC)
        for mod in MODELOS:
            def dif(s, mod=mod):
                ys = s.label.to_numpy(); f_ = s.FireMask.isin(codigos).astype(int).to_numpy(); mf = conf(ys, f_)
                return {"d": recall_para_fpr(ys, s[f"prob_{mod}"].to_numpy(), mf["fpr"])["recall"] - mf["recall"]}
            R = bootstrap_fechas(df, dif); lo, hi = ic(R, "d")
            fila[f"dif_recall_{mod}_menos_FDC"] = fila[f"recall_{mod}_misma_FPR"] - fila["recall"]
            fila[f"dif_recall_{mod}_menos_FDC_IC"] = f"[{lo:.3f}, {hi:.3f}]"
        filas.append(fila)
    cat = []
    for c, nm in CATEGORIAS.items():
        for cc, filt in ((c, "no"), (c + 20, "sí")):
            s = df[df.FireMask == cc]
            cat.append(dict(codigo=cc, categoria=nm, filtrado=filt, n=len(s), fuegos=int(s.label.sum()), no_fuego=int((s.label == 0).sum())))
    return pd.DataFrame(filas), pd.DataFrame(cat)

def pruebas_pareadas(df):
    y = df.label.to_numpy(); filas = []
    pares = [(m, "fdc") for m in MODELOS] + ([("hgb", "rf")] if set(MODELOS) >= {"hgb", "rf"} else [])
    col = lambda m: "goes_fdc" if m == "fdc" else f"pred_{m}"
    for a, b in pares:
        pa, pb = df[col(a)].to_numpy(), df[col(b)].to_numpy()
        ma, mb = conf(y, pa), conf(y, pb)
        def f(s, a=a, b=b):
            ys = s.label.to_numpy(); x, z = conf(ys, s[col(a)].to_numpy()), conf(ys, s[col(b)].to_numpy())
            out = {k: x[k] - z[k] for k in ("recall", "fpr", "f1", "precision")}
            if a != "fdc" and b != "fdc":
                from sklearn.metrics import average_precision_score
                out["pr_auc"] = average_precision_score(ys, s[f"prob_{a}"]) - average_precision_score(ys, s[f"prob_{b}"])
            return out
        R = bootstrap_fechas(df, f)
        pos, neg = y == 1, y == 0
        b1, c1, p1 = mcnemar_exacto(pa[pos] == 1, pb[pos] == 1)           # recall: fuegos detectados por uno y no por otro
        b2, c2, p2 = mcnemar_exacto(pa[neg] == 0, pb[neg] == 0)           # especificidad
        fila = dict(comparacion=f"{a.upper()} - {b.upper()}")
        for k in ("recall", "fpr", "f1", "precision"):
            lo, hi = ic(R, k); fila[f"dif_{k}"] = ma[k] - mb[k]; fila[f"dif_{k}_IC95"] = f"[{lo:.4f}, {hi:.4f}]"
        if "pr_auc" in R:
            from sklearn.metrics import average_precision_score
            lo, hi = ic(R, "pr_auc")
            fila["dif_pr_auc"] = average_precision_score(y, df[f"prob_{a}"]) - average_precision_score(y, df[f"prob_{b}"])
            fila["dif_pr_auc_IC95"] = f"[{lo:.4f}, {hi:.4f}]"
        fila.update(fuegos_solo_primero=b1, fuegos_solo_segundo=c1, mcnemar_p_fuegos=p1,
                    noFuego_correcto_solo_primero=b2, noFuego_correcto_solo_segundo=c2, mcnemar_p_noFuego=p2,
                    p_unilateral_bootstrap_dif_recall_le_0=float((R["recall"] <= 0).mean()))
        filas.append(fila)
    return pd.DataFrame(filas)

def precision_por_prevalencia(df):
    y = df.label.to_numpy(); puntos = []
    for m in MODELOS:
        puntos.append((f"{m.upper()} (umbral máx. F1)", conf(y, df[f"pred_{m}"].to_numpy())))
        for f in FPR_FIJAS: puntos.append((f"{m.upper()} a FPR = {f:.1%}", recall_para_fpr(y, df[f"prob_{m}"].to_numpy(), f)))
    puntos.append(("GOES-FDC", conf(y, df.goes_fdc.to_numpy())))
    filas = []
    for nombre, m in puntos:
        fila = dict(detector=nombre, recall=m["recall"], fpr_muestral=m["fpr"], FP=m["FP"],
                    falsas_alarmas_por_10000_pixeles=1e4 * m["fpr"])
        for pi in PREVALENCIAS:
            num = m["recall"] * pi; den = num + m["fpr"] * (1 - pi)
            fila[f"precision_prev_{pi:g}"] = num / den if den > 0 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)

def verificar_tabla_A4(df):
    """Misma lógica que el análisis 3 del script 08 (grupos, bootstrap por fecha con semilla 42)."""
    pos = df[df.label == 1].copy()
    pos["FRP (MW)"] = pd.cut(pos.frp_mw, [0, 5, 10, 25, 50, np.inf], labels=["<5", "5–10", "10–25", "25–50", ">50"])
    pos["Mes"] = pos.mes.map(lambda v: f"{int(v):02d}")
    def boot(s, col, n):
        rng = np.random.default_rng(SEMILLA); fe = s.fecha.unique(); idx = s.groupby("fecha").indices; R = []
        for _ in range(n):
            f = rng.choice(fe, len(fe), replace=True); R.append(s.iloc[np.concatenate([idx[x] for x in f])][col].mean())
        return s[col].mean(), np.quantile(R, .025), np.quantile(R, .975)
    filas = []
    for grupo, col in (("FRP", "FRP (MW)"), ("día/noche", "periodo"), ("sensor", "sensor"), ("mes", "Mes")):
        for cat, s in pos.groupby(col, observed=True):
            if str(cat) == "" or len(s) == 0: continue
            fila = dict(grupo=grupo, categoria=str(cat), n=len(s))
            for serie, c in ((MODELOS[0], f"pred_{MODELOS[0]}"), ("fdc", "goes_fdc")):
                for n in (N_BOOTSTRAP_GRUPOS, N_BOOTSTRAP):
                    v, lo, hi = boot(s, c, n); fila[f"{serie}_recall"] = v; fila[f"{serie}_IC_{n}"] = f"[{lo:.3f}–{hi:.3f}]"
            filas.append(fila)
    tab = pd.DataFrame(filas)
    resumen = pd.DataFrame([
        dict(chequeo="fuegos totales", valor=len(pos)),
        dict(chequeo="fuegos sin FRP (excluidos de los grupos FRP)", valor=int(pos.frp_mw.isna().sum())),
        dict(chequeo="fuegos con FRP = 0 o negativo (también excluidos)", valor=int((pos.frp_mw <= 0).sum())),
        dict(chequeo="fuegos VIIRS + MODIS", valor=int((pos.sensor == "VIIRS + MODIS").sum())),
        dict(chequeo="fuegos sin sensor reconocido", valor=int((pos.sensor == "").sum())),
        dict(chequeo="fuegos fuera de enero, febrero, marzo y octubre", valor=int((~pos.mes.isin([1, 2, 3, 10])).sum())),
    ])
    return tab, resumen

# ============================== PRINCIPAL ==============================
def ejecutar():
    warnings.filterwarnings("ignore")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    df = cargar()
    print("\n[1] Puntos de operación de FDC por categoría de máscara...")
    puntos, categorias = puntos_operacion_fdc(df)
    print("[2] Pruebas pareadas (bootstrap por fecha y McNemar)...")
    pares = pruebas_pareadas(df)
    print("[3] Precisión corregida por prevalencia...")
    prev = precision_por_prevalencia(df)
    print("[4] Verificación de la Tabla A4...")
    a4, a4_res = verificar_tabla_A4(df)
    salida = os.path.join(CARPETA_SALIDA, "comparacion_FDC.xlsx")
    with pd.ExcelWriter(salida) as w:
        puntos.to_excel(w, sheet_name="FDC_puntos_operacion", index=False)
        categorias.to_excel(w, sheet_name="FDC_categorias", index=False)
        pares.to_excel(w, sheet_name="pruebas_pareadas", index=False)
        prev.to_excel(w, sheet_name="precision_prevalencia", index=False)
        a4.to_excel(w, sheet_name="tabla_A4_verificacion", index=False)
        a4_res.to_excel(w, sheet_name="tabla_A4_chequeos", index=False)
        pd.DataFrame({"parametro": ["predicciones", "dataset", "modelos", "n_bootstrap", "n_bootstrap_grupos", "semilla", "prevalencias",
                                    "nota_mcnemar"],
                      "valor": [PREDICCIONES, DATASET, ", ".join(MODELOS), N_BOOTSTRAP, N_BOOTSTRAP_GRUPOS, SEMILLA, ", ".join(map(str, PREVALENCIAS)),
                                "McNemar supone muestras independientes; la inferencia principal es el bootstrap pareado por fecha"]}).to_excel(w, sheet_name="configuracion", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n================ PUNTOS DE OPERACIÓN DE FDC ================")
    print(puntos.round(4).to_string(index=False))
    print("\n================ PRUEBAS PAREADAS ================"); print(pares.round(4).to_string(index=False))
    print("\n================ PRECISIÓN SEGÚN PREVALENCIA ================"); print(prev.round(4).to_string(index=False))
    print("\n================ TABLA A4 ================"); print(a4_res.to_string(index=False))
    print(a4[a4.categoria == "VIIRS + MODIS"].to_string(index=False))
    print(f"\nGuardado: {salida}")
    return puntos, pares, prev, a4

if __name__ == "__main__":
    puntos, pares, prevalencia, tabla_a4 = ejecutar()