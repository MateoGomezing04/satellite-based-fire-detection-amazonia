# -*- coding: utf-8 -*-
"""
analisis_articulo.py -- Cinco análisis para el artículo, a partir de los resultados ya obtenidos (sin descargar nada).

  1. Curva recall vs. tasa de falsos positivos (eje log) con GOES-FDC como punto, y recall a FPR fija.
  2. Intervalos de confianza al 95 % por bootstrap AGRUPADO POR FECHA (se remuestrean días completos, no filas,
     porque las filas de un mismo día no son independientes).
  3. Recall por potencia del fuego (FRP), día/noche, sensor que confirmó (VIIRS/MODIS), mes y detección de GOES-FDC.
  4. Importancia por GRUPOS de variables (permutación conjunta del grupo, validación cruzada por fecha).
  5. Mapa de aciertos (TP), omisiones (FN) y falsas alarmas (FP) en la Amazonía.

PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5.
ENTRADAS: predicciones_oof.csv (de entrenar_modelo_fuego.py) y dataset_goes16_2024.csv (de la extracción).
SALIDAS (CARPETA_SALIDA): tablas_articulo.xlsx y figuras fig1..fig5 en PNG (300 dpi) y PDF.
"""
import os, time, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
PREDICCIONES = r"C:\Users\User\Desktop\Nuevo Manuscript\Modelo fuego GOES\predicciones_oof.csv"
DATASET = r"C:\Users\User\Desktop\Nuevo Manuscript\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
RECORTE = r"C:\Users\User\Desktop\Manuscript JI\Insumos\Area quemada\Amazonia.shp"   # contorno del mapa; None = sin contorno
CARPETA_SALIDA = r"C:\Users\User\Desktop\Nuevo Manuscript\Analisis articulo"

MODELOS = ["hgb", "rf"]                 # columnas prob_<m> / pred_<m> de predicciones_oof.csv
MODELO_PRINCIPAL = "hgb"                # el que se usa en mapa y análisis por grupos
N_BOOTSTRAP = 1000
FPR_OBJETIVO = [0.01, 0.001, 0.0001]
SEMILLA = 42
IDIOMA_FIGURAS = "en"                   # "es" o "en" (para la versión en inglés del artículo)

GRUPOS_VARIABLES = {                    # análisis 4
    "3.9 µm (C07, T7)": ["C07", "T7"],
    "11.2 µm (C14, T14)": ["C14", "T14"],
    "Contraste 3.9–11.2 (BTD, Ratio, ND)": ["BTD", "Ratio", "ND"],
    "Nubes (Cloud_Probabilities)": ["Cloud_Probabilities"],
    "Superficie/atmósfera (LST, TPW, LST_BT_DIFF)": ["LST", "TPW", "LST_BT_DIFF"],
    "Hora (fire_hour)": ["fire_hour"],
}
N_REPETICIONES_PERM = 5
FUEGO_FDC = list(range(10, 16)) + list(range(30, 36))

# colores (paleta categórica validada: primeros tres puestos; texto siempre en tinta neutra)
AZUL, NARANJA, AGUA, TINTA, TINTA2, REJILLA = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
T = {"es": dict(recall="Recall (fracción de incendios detectados)", fpr="Tasa de falsos positivos (escala log)",
                fdc="GOES-FDC (operativo)", frp="FRP del punto caliente (MW)", imp="Caída de PR-AUC al permutar el grupo",
                tp="Detectado (TP)", fn="Omitido (FN)", fp="Falsa alarma (FP)", lon="Longitud", lat="Latitud",
                dn="Día / noche", sensor="Sensor que confirmó", mes="Mes"),
     "en": dict(recall="Recall (fraction of fires detected)", fpr="False positive rate (log scale)",
                fdc="GOES-FDC (operational)", frp="Hotspot FRP (MW)", imp="PR-AUC drop when the group is permuted",
                tp="Detected (TP)", fn="Missed (FN)", fp="False alarm (FP)", lon="Longitude", lat="Latitude",
                dn="Day / night", sensor="Confirming sensor", mes="Month")}

# ============================== DATOS ==============================
def cargar():
    for r in (PREDICCIONES, DATASET):
        if not os.path.exists(r): raise FileNotFoundError(f"No existe: {r}")
    p = pd.read_csv(PREDICCIONES); d = pd.read_csv(DATASET, low_memory=False)
    extra = [c for c in ("frp_mw", "day_night", "source", "fire_hour", "scan_start_utc", "FireMask", "C07", "C14", "T7", "T14", "BTD",
                         "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF") if c in d.columns and c not in p.columns]
    df = p.merge(d[["image_id", "pixel_id"] + extra], on=["image_id", "pixel_id"], how="left", validate="one_to_one")
    if df[extra[0]].isna().all(): raise ValueError("No se pudo unir predicciones y dataset (revise que sean de la misma corrida).")
    # día/noche: el del punto caliente si existe; si no, por la hora local del escaneo (UTC-5)
    hl = (pd.to_datetime(df.scan_start_utc).dt.hour - 5) % 24
    df["periodo"] = np.where(df.get("day_night").astype(str).str.upper().str.startswith("N"), "Noche",
                             np.where(df.get("day_night").astype(str).str.upper().str.startswith("D"), "Día",
                                      np.where((hl >= 6) & (hl < 18), "Día", "Noche")))
    s = df.get("source").fillna("").astype(str)
    df["sensor"] = np.select([s.str.contains("VIIRS") & s.str.contains("MODIS"), s.str.contains("VIIRS"), s.str.contains("MODIS")],
                             ["VIIRS + MODIS", "VIIRS", "MODIS"], "")
    df["goes_fdc"] = df.goes_fdc.astype(int)
    print(f"Filas: {len(df):,} | positivos: {int(df.label.sum()):,} | fechas: {df.fecha.nunique()}")
    return df

# ============================== MÉTRICAS ==============================
def recall_a_fpr(y, prob, fpr_obj):
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, prob)
    i = np.searchsorted(fpr, fpr_obj, side="right") - 1
    return float(tpr[i]), float(th[i]) if np.isfinite(th[i]) else 1.0

def metricas(y, pred, prob=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    m = dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12), fpr=fp / max(fp + tn, 1))
    if prob is not None and len(np.unique(y)) == 2:
        m["pr_auc"] = average_precision_score(y, prob); m["roc_auc"] = roc_auc_score(y, prob)
        for f in FPR_OBJETIVO: m[f"recall@FPR{f:.2%}"] = recall_a_fpr(y, prob, f)[0]
    return m

def bootstrap_por_fecha(df, funcion, n=N_BOOTSTRAP, semilla=SEMILLA):
    """Remuestrea FECHAS con reemplazo y recalcula funcion(df_muestra) -> dict. Devuelve (estimación, IC 2.5 %, IC 97.5 %)."""
    rng = np.random.default_rng(semilla)
    fechas = df.fecha.unique(); idx = df.groupby("fecha").indices
    base = funcion(df); res = []
    for _ in range(n):
        f = rng.choice(fechas, len(fechas), replace=True)
        filas = np.concatenate([idx[x] for x in f])
        try: res.append(funcion(df.iloc[filas]))
        except Exception: continue
    R = pd.DataFrame(res)
    return {k: (v, R[k].quantile(0.025) if k in R else np.nan, R[k].quantile(0.975) if k in R else np.nan) for k, v in base.items()}

def tabla_ic(dic, nombre):
    return pd.DataFrame([dict(modelo=nombre, metrica=k, valor=v, ic95_inf=a, ic95_sup=b) for k, (v, a, b) in dic.items()])

def fmt(v, a, b, pct=True):
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)} [{f(a)}–{f(b)}]"

# ============================== FIGURAS ==============================
def estilo(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=9); ax.grid(color=REJILLA, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TINTA); ax.yaxis.label.set_color(TINTA)

def guardar(fig, nombre):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(CARPETA_SALIDA, f"{nombre}.{ext}"), dpi=300, bbox_inches="tight")

def fig_curva(df, tx):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    colores = {"hgb": AZUL, "rf": NARANJA}; nombres = {"hgb": "HistGradientBoosting", "rf": "Random Forest"}
    for m in MODELOS:
        fpr, tpr, _ = roc_curve(df.label, df[f"prob_{m}"]); ok = fpr > 0
        ax.plot(fpr[ok], tpr[ok], color=colores.get(m, AGUA), lw=2, label=nombres.get(m, m.upper()))
    mf = metricas(df.label, df.goes_fdc)
    ax.scatter([max(mf["fpr"], 1e-5)], [mf["recall"]], s=70, color=TINTA, zorder=5, label=tx["fdc"])
    for f in FPR_OBJETIVO: ax.axvline(f, color=TINTA2, lw=0.8, ls=":")
    nneg = int((df.label == 0).sum())
    ax.axvspan(1e-5, 1 / nneg, color=REJILLA, alpha=0.6, lw=0)
    ax.text(1.1e-5, 0.03, f"< 1 negativo\n(n = {nneg:,})" if IDIOMA_FIGURAS == "es" else f"< 1 negative\n(n = {nneg:,})",
            fontsize=7.5, color=TINTA2, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0, 1.0)
    ax.set_xlabel(tx["fpr"]); ax.set_ylabel(tx["recall"]); estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TINTA); fig.tight_layout(); guardar(fig, "fig1_recall_vs_fpr"); plt.close(fig)

def fig_grupos(tab, grupo, nombre, tx, etiqueta_x):
    import matplotlib.pyplot as plt
    t = tab[tab.grupo_tipo == grupo]
    if t.empty: return
    cats = list(dict.fromkeys(t.categoria)); x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(cats) + 2), 4.2))
    for j, (serie, color, lab) in enumerate([(MODELO_PRINCIPAL, AZUL, MODELO_PRINCIPAL.upper()), ("fdc", TINTA2, tx["fdc"])]):
        s = t[t.serie == serie].set_index("categoria").reindex(cats)
        err = np.vstack([s.recall - s.ic95_inf, s.ic95_sup - s.recall])
        ax.bar(x + (j - 0.5) * w, s.recall, w * 0.92, color=color, label=lab, yerr=err, capsize=2.5,
               error_kw=dict(ecolor=TINTA, lw=0.8))
    n = t[t.serie == MODELO_PRINCIPAL].set_index("categoria").reindex(cats).n
    ax.set_xticks(x, [f"{c}\n(n={int(k):,})" for c, k in zip(cats, n)], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel(tx["recall"]); ax.set_xlabel(etiqueta_x); estilo(ax); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); guardar(fig, nombre); plt.close(fig)

def fig_importancia(imp, tx):
    import matplotlib.pyplot as plt
    s = imp.sort_values("caida_pr_auc")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(s.grupo, s.caida_pr_auc, xerr=[s.caida_pr_auc - s.ic_inf, s.ic_sup - s.caida_pr_auc], color=AZUL, height=0.6,
            error_kw=dict(ecolor=TINTA, lw=0.8, capsize=2.5))
    ax.set_xlabel(tx["imp"]); estilo(ax); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5, colors=TINTA)
    fig.tight_layout(); guardar(fig, "fig4_importancia_grupos"); plt.close(fig)

def fig_mapa(df, tx):
    """Tres paneles (aciertos, omisiones, falsas alarmas) con todos los píxeles en gris de fondo: evita que una clase tape a otra."""
    import matplotlib.pyplot as plt
    m = MODELO_PRINCIPAL; pr = df[f"pred_{m}"]
    paneles = [(df[(df.label == 1) & (pr == 1)], AZUL, tx["tp"]), (df[(df.label == 1) & (pr == 0)], NARANJA, tx["fn"]),
               (df[(df.label == 0) & (pr == 1)], AGUA, tx["fp"])]
    contorno = None
    if RECORTE and os.path.exists(str(RECORTE)):
        try:
            import geopandas as gpd
            contorno = gpd.read_file(RECORTE).to_crs("EPSG:4326")
        except Exception as e: print("  (sin contorno del recorte:", e, ")")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True)
    for ax, (d, c, lab) in zip(axs, paneles):
        if contorno is not None: contorno.boundary.plot(ax=ax, color=TINTA2, lw=0.7)
        ax.scatter(df.x, df.y, s=3, color=REJILLA, lw=0, zorder=1)
        ax.scatter(d.x, d.y, s=9, color=c, edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_title(f"{lab}  (n = {len(d):,})", fontsize=10, color=TINTA, loc="left")
        ax.set_aspect("equal"); estilo(ax); ax.set_xlabel(tx["lon"])
    axs[0].set_ylabel(tx["lat"]); fig.tight_layout(); guardar(fig, "fig5_mapa_errores"); plt.close(fig)

# ============================== ANÁLISIS 4: importancia por grupos ==============================
def importancia_grupos(df):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import average_precision_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    variables = [v for g in GRUPOS_VARIABLES.values() for v in g]
    falt = [v for v in variables if v not in df.columns]
    if falt: raise KeyError(f"Faltan variables para la importancia por grupos: {falt}")
    X, y, g = df[variables].reset_index(drop=True), df.label.to_numpy(), df.fecha.to_numpy()
    rng = np.random.default_rng(SEMILLA); caidas = {k: [] for k in GRUPOS_VARIABLES}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    for k, (tr, te) in enumerate(cv.split(X, y, g), 1):
        w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                             l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=SEMILLA)
        mdl.fit(X.iloc[tr], y[tr], sample_weight=w)
        Xte = X.iloc[te].copy(); base = average_precision_score(y[te], mdl.predict_proba(Xte)[:, 1])
        for nombre, cols in GRUPOS_VARIABLES.items():
            for _ in range(N_REPETICIONES_PERM):
                Xp = Xte.copy(); perm = rng.permutation(len(Xp))
                Xp[cols] = Xte[cols].to_numpy()[perm]          # se permuta el grupo COMPLETO (mantiene su correlación interna)
                caidas[nombre].append(base - average_precision_score(y[te], mdl.predict_proba(Xp)[:, 1]))
        print(f"  importancia por grupos: pliegue {k}/5 listo (PR-AUC base {base:.3f})")
    return pd.DataFrame([dict(grupo=k, caida_pr_auc=np.mean(v), ic_inf=np.quantile(v, 0.025), ic_sup=np.quantile(v, 0.975),
                              variables=", ".join(GRUPOS_VARIABLES[k])) for k, v in caidas.items()]).sort_values("caida_pr_auc", ascending=False)

# ============================== PRINCIPAL ==============================
# ============================== ANÁLISIS 6: figuras complementarias ==============================
NOMBRE_MODELO = {"hgb": "HistGradientBoosting", "rf": "Random Forest", "fdc": "GOES-FDC"}
COLOR_MODELO = {"hgb": AZUL, "rf": NARANJA, "fdc": TINTA2}
VARIABLES_CORR = ["C07", "C14", "T7", "T14", "BTD", "Ratio", "ND", "Cloud_Probabilities", "LST", "TPW", "LST_BT_DIFF", "fire_hour"]

def metricas_completas(y, pred, prob=None):
    """Exactitud, precisión, recall, especificidad, F1, MCC (+ PR-AUC y ROC-AUC si hay probabilidad)."""
    from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
    y = np.asarray(y); pred = np.asarray(pred)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    m = {"Exactitud": (tp + tn) / len(y), "Precisión": pr, "Recall": rc, "Especificidad": tn / max(tn + fp, 1),
         "F1": 2 * pr * rc / max(pr + rc, 1e-12), "MCC": matthews_corrcoef(y, pred)}
    if prob is not None:
        m["PR-AUC"] = average_precision_score(y, prob); m["ROC-AUC"] = roc_auc_score(y, prob)
    return m

def fig_barras_metricas(tab, tx):
    """Barras agrupadas por métrica (una barra por modelo), con valor encima e IC 95 % por bootstrap de fechas."""
    import matplotlib.pyplot as plt
    orden = ["Exactitud", "Precisión", "Recall", "Especificidad", "F1", "MCC", "PR-AUC", "ROC-AUC"]
    en = {"Exactitud": "Accuracy", "Precisión": "Precision", "Especificidad": "Specificity"}
    modelos = list(dict.fromkeys(tab.modelo)); x = np.arange(len(orden)); w = 0.8 / len(modelos)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for j, mo in enumerate(modelos):
        t = tab[tab.modelo == mo].set_index("metrica").reindex(orden)
        xs = x + (j - (len(modelos) - 1) / 2) * w; ok = t.valor.notna().to_numpy()
        err = np.vstack([t.valor - t.ic95_inf, t.ic95_sup - t.valor])[:, ok]
        ax.bar(xs[ok], t.valor[ok], w * 0.9, color=COLOR_MODELO[mo], label=NOMBRE_MODELO[mo], yerr=err, capsize=2,
               error_kw=dict(ecolor=TINTA, lw=0.7))
        for xi, v, hi in zip(xs[ok], t.valor[ok], t.ic95_sup[ok]):
            ax.text(xi, hi + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=TINTA, rotation=90)
        for xi in xs[~ok]: ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color=TINTA2, rotation=90)
    ax.set_xticks(x, [en.get(o, o) if IDIOMA_FIGURAS == "en" else o for o in orden], fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Valor" if IDIOMA_FIGURAS == "es" else "Score")
    ax.set_xlabel("Métrica de desempeño" if IDIOMA_FIGURAS == "es" else "Performance metric")
    estilo(ax); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(modelos))
    fig.tight_layout(); guardar(fig, "fig6_metricas_comparacion"); plt.close(fig)

def fig_matrices_confusion(df, series):
    """Una matriz por modelo: conteos y porcentaje por fila (verdadero). Escala secuencial de un solo tono."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#f4f7fc", "#a9c7ee", AZUL, "#123f75"])
    cls = ["No fuego", "Fuego"] if IDIOMA_FIGURAS == "es" else ["No fire", "Fire"]
    fig, axs = plt.subplots(1, len(series), figsize=(3.6 * len(series), 3.6))
    tabla = []
    for ax, (clave, pred) in zip(np.atleast_1d(axs), series):
        y = df.label.to_numpy(); p = np.asarray(pred)
        cm = np.array([[((y == 0) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum()], [((y == 1) & (p == 0)).sum(), ((y == 1) & (p == 1)).sum()]])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * pct[i, j]:.1f} %)", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.55 else TINTA)
        ax.set_xticks([0, 1], cls, fontsize=8.5); ax.set_yticks([0, 1], cls, fontsize=8.5)
        ax.set_xlabel("Predicho" if IDIOMA_FIGURAS == "es" else "Predicted", color=TINTA)
        ax.set_ylabel("Verdadero" if IDIOMA_FIGURAS == "es" else "Actual", color=TINTA)
        ax.set_title(NOMBRE_MODELO[clave], fontsize=10, color=TINTA, loc="left"); ax.tick_params(colors=TINTA2, length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        tabla.append(dict(modelo=NOMBRE_MODELO[clave], TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=cm[1, 1]))
    fig.tight_layout(); guardar(fig, "fig7_matrices_confusion"); plt.close(fig)
    return pd.DataFrame(tabla)

def fig_correlacion(df):
    """Matriz de correlación de Pearson de las variables de entrada + la etiqueta (punto-biserial). Paleta divergente."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cols = [c for c in VARIABLES_CORR if c in df.columns] + ["label"]
    C = df[cols].corr(method="pearson")
    nombres = [("Etiqueta (fuego)" if IDIOMA_FIGURAS == "es" else "Label (fire)") if c == "label" else c for c in cols]
    cmap = LinearSegmentedColormap.from_list("div", [AZUL, "#f2f1ee", "#c4314b"])
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    im = ax.imshow(np.where(mask, np.nan, C.values), cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i + 1):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else TINTA)
    ax.set_xticks(range(len(cols)), nombres, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols)), nombres, fontsize=8.5); ax.tick_params(colors=TINTA, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("r de Pearson" if IDIOMA_FIGURAS == "es" else "Pearson r", color=TINTA)
    cb.outline.set_visible(False); fig.tight_layout(); guardar(fig, "fig8_correlacion_pearson"); plt.close(fig)
    return C.round(3).reset_index().rename(columns={"index": "variable"})

def fig_roc_pr(df, tx):
    """Curvas ROC y precisión-recall clásicas (dos paneles), con AUC en la leyenda y GOES-FDC como punto."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for m in MODELOS:
        fpr, tpr, _ = roc_curve(df.label, df[f"prob_{m}"]); pr, rc, _ = precision_recall_curve(df.label, df[f"prob_{m}"])
        a1.plot(fpr, tpr, color=COLOR_MODELO[m], lw=2, label=f"{NOMBRE_MODELO[m]} (AUC = {roc_auc_score(df.label, df[f'prob_{m}']):.3f})")
        a2.plot(rc, pr, color=COLOR_MODELO[m], lw=2, label=f"{NOMBRE_MODELO[m]} (AP = {average_precision_score(df.label, df[f'prob_{m}']):.3f})")
    mf = metricas(df.label, df.goes_fdc)
    a1.scatter([mf["fpr"]], [mf["recall"]], s=60, color=TINTA, zorder=5, label=tx["fdc"])
    a2.scatter([mf["recall"]], [mf["precision"]], s=60, color=TINTA, zorder=5, label=tx["fdc"])
    a1.plot([0, 1], [0, 1], color=TINTA2, lw=0.8, ls=":"); a2.axhline(df.label.mean(), color=TINTA2, lw=0.8, ls=":")
    es = IDIOMA_FIGURAS == "es"
    a1.set_xlabel("Tasa de falsos positivos" if es else "False positive rate"); a1.set_ylabel("Recall (tasa de verdaderos positivos)" if es else "Recall (true positive rate)")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precisión" if es else "Precision")
    a1.set_title("(a) ROC", loc="left", fontsize=10, color=TINTA); a2.set_title("(b) Precisión–recall" if es else "(b) Precision–recall", loc="left", fontsize=10, color=TINTA)
    for a in (a1, a2):
        a.set_xlim(0, 1); a.set_ylim(0, 1.02); estilo(a); a.legend(frameon=False, fontsize=8, labelcolor=TINTA, loc="lower right" if a is a1 else "lower left")
    fig.tight_layout(); guardar(fig, "fig9_curvas_roc_pr"); plt.close(fig)

def fig_distribuciones(df):
    """Diagramas de caja de las variables principales por clase (fuego, negativo near, negativo far)."""
    import matplotlib.pyplot as plt
    vars_ = [v for v in ["T7", "BTD", "C07", "T14", "LST_BT_DIFF", "Cloud_Probabilities"] if v in df.columns]
    es = IDIOMA_FIGURAS == "es"
    clases = [("pos", "Fuego" if es else "Fire", AZUL), ("near", "Neg. near", NARANJA), ("far", "Neg. far", AGUA)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2)); axs = axs.ravel()
    for ax, v in zip(axs, vars_):
        datos = [df.loc[df.neg_type == c, v].dropna().to_numpy() for c, _, _ in clases]
        bp = ax.boxplot(datos, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=TINTA, lw=1.2),
                        whiskerprops=dict(color=TINTA2), capprops=dict(color=TINTA2))
        for caja, (_, _, col) in zip(bp["boxes"], clases): caja.set_facecolor(col); caja.set_edgecolor("white"); caja.set_alpha(0.9)
        ax.set_xticks([1, 2, 3], [n for _, n, _ in clases], fontsize=8.5); ax.set_title(v, loc="left", fontsize=10, color=TINTA)
        estilo(ax); ax.grid(axis="x", visible=False)
    for ax in axs[len(vars_):]: ax.set_visible(False)
    fig.tight_layout(); guardar(fig, "fig10_distribuciones_por_clase"); plt.close(fig)

def ejecutar():
    import matplotlib; matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    os.makedirs(CARPETA_SALIDA, exist_ok=True); tx = T[IDIOMA_FIGURAS]; t0 = time.time()
    df = cargar(); hojas = {}

    # ---- 1 y 2: métricas globales con IC por bootstrap de fechas ----
    print(f"\n[1-2] Métricas globales con IC 95 % (bootstrap de {N_BOOTSTRAP} remuestreos de fechas)...")
    tablas = [tabla_ic(bootstrap_por_fecha(df, lambda d: metricas(d.label, d.goes_fdc)), "GOES-FDC")]
    for m in MODELOS:
        tablas.append(tabla_ic(bootstrap_por_fecha(df, lambda d, m=m: metricas(d.label, d[f"pred_{m}"], d[f"prob_{m}"])), m.upper()))
    glob = pd.concat(tablas, ignore_index=True); hojas["1_metricas_IC95"] = glob
    # recall a FPR fija con umbral
    fil = []
    for m in MODELOS:
        for f in FPR_OBJETIVO:
            r, th = recall_a_fpr(df.label, df[f"prob_{m}"], f)
            fil.append(dict(modelo=m.upper(), fpr_objetivo=f, recall=r, umbral=th, negativos_permitidos=round(f * (df.label == 0).sum(), 1)))
    mf = metricas(df.label, df.goes_fdc); fil.append(dict(modelo="GOES-FDC", fpr_objetivo=mf["fpr"], recall=mf["recall"], umbral=np.nan,
                                                          negativos_permitidos=int(((df.goes_fdc == 1) & (df.label == 0)).sum())))
    hojas["1_recall_a_FPR_fija"] = pd.DataFrame(fil)
    fig_curva(df, tx)

    # ---- 3: recall por grupos ----
    print("[3] Recall por FRP, día/noche, sensor, mes y detección FDC...")
    pos = df[df.label == 1].copy()
    pos["FRP (MW)"] = pd.cut(pos.frp_mw, [0, 5, 10, 25, 50, np.inf], labels=["<5", "5–10", "10–25", "25–50", ">50"])
    pos["FDC"] = np.where(pos.goes_fdc == 1, "FDC detectó", "FDC omitió")
    pos["Mes"] = pos.mes.map(lambda v: f"{int(v):02d}")
    fil = []
    for grupo, col in (("frp", "FRP (MW)"), ("dia_noche", "periodo"), ("sensor", "sensor"), ("mes", "Mes"), ("fdc", "FDC")):
        for cat, s in pos.groupby(col, observed=True):
            if cat == "" or len(s) == 0: continue
            for serie, pc in ((MODELO_PRINCIPAL, f"pred_{MODELO_PRINCIPAL}"), ("fdc", "goes_fdc")):
                if grupo == "fdc" and serie == "fdc": continue
                est = bootstrap_por_fecha(s, lambda d, pc=pc: {"recall": d[pc].mean()}, n=min(N_BOOTSTRAP, 500))["recall"]
                fil.append(dict(grupo_tipo=grupo, categoria=str(cat), serie=serie, n=len(s), recall=est[0], ic95_inf=est[1], ic95_sup=est[2]))
    rec = pd.DataFrame(fil); hojas["3_recall_por_grupo"] = rec
    neg = df[df.label == 0]
    hojas["3_falsos_pos_por_tipo"] = pd.DataFrame([dict(tipo=t, n=len(s), **{f"fpr_{m}": s[f"pred_{m}"].mean() for m in MODELOS},
                                                         fpr_fdc=s.goes_fdc.mean()) for t, s in neg.groupby("neg_type")])
    fig_grupos(rec, "frp", "fig2_recall_por_FRP", tx, tx["frp"])
    fig_grupos(rec, "dia_noche", "fig3a_recall_dia_noche", tx, tx["dn"])
    fig_grupos(rec, "sensor", "fig3b_recall_por_sensor", tx, tx["sensor"])
    fig_grupos(rec, "mes", "fig3c_recall_por_mes", tx, tx["mes"])

    # ---- 4: importancia por grupos ----
    print("[4] Importancia por grupos de variables (reentrena 5 pliegues; tarda unos minutos)...")
    imp = importancia_grupos(df); hojas["4_importancia_grupos"] = imp; fig_importancia(imp, tx)

    # ---- 5: mapa ----
    print("[5] Mapa de aciertos, omisiones y falsas alarmas...")
    fig_mapa(df, tx)
    m = MODELO_PRINCIPAL; pr = df[f"pred_{m}"]
    hojas["5_errores_por_imagen"] = (df.assign(TP=(df.label == 1) & (pr == 1), FN=(df.label == 1) & (pr == 0), FP=(df.label == 0) & (pr == 1))
                                     .groupby(["image_id", "fecha"])[["TP", "FN", "FP"]].sum().sort_values("FP", ascending=False).reset_index())

    # ---- 6: comparación de métricas, matrices de confusión, correlación, ROC/PR y distribuciones ----
    print("[6] Barras de métricas, matrices de confusión, correlación de Pearson, ROC/PR y distribuciones...")
    tabs = [tabla_ic(bootstrap_por_fecha(df, lambda d, m=m: metricas_completas(d.label, d[f"pred_{m}"], d[f"prob_{m}"])), m) for m in MODELOS]
    tabs.append(tabla_ic(bootstrap_por_fecha(df, lambda d: metricas_completas(d.label, d.goes_fdc)), "fdc"))
    comp = pd.concat(tabs, ignore_index=True); fig_barras_metricas(comp, tx)
    hojas["6_metricas_comparacion"] = comp.assign(modelo=comp.modelo.map(NOMBRE_MODELO))
    hojas["6_matrices_confusion"] = fig_matrices_confusion(df, [(m, df[f"pred_{m}"]) for m in MODELOS] + [("fdc", df.goes_fdc)])
    hojas["6_correlacion_pearson"] = fig_correlacion(df)
    fig_roc_pr(df, tx); fig_distribuciones(df)

    # ---- resumen listo para el texto ----
    g = glob.set_index(["modelo", "metrica"])
    lin = ["RESUMEN PARA EL TEXTO (valor [IC 95 %], bootstrap por fecha)"]
    for mod in ["GOES-FDC"] + [x.upper() for x in MODELOS]:
        partes = []
        for k in ("recall", "precision", "f1", "fpr", "pr_auc"):
            if (mod, k) in g.index:
                v, a, b = g.loc[(mod, k), ["valor", "ic95_inf", "ic95_sup"]]; partes.append(f"{k}={fmt(v, a, b, pct=(k != 'pr_auc'))}")
        lin.append(f"  {mod}: " + " | ".join(partes))
    r = rec[(rec.grupo_tipo == "fdc") & (rec.categoria == "FDC omitió")]
    if len(r): lin.append(f"  {MODELO_PRINCIPAL.upper()} detecta {fmt(*r[['recall', 'ic95_inf', 'ic95_sup']].iloc[0])} % de los incendios que FDC omitió (n={int(r.n.iloc[0]):,})")
    hojas["0_resumen"] = pd.DataFrame({"resumen": lin})
    with pd.ExcelWriter(os.path.join(CARPETA_SALIDA, "tablas_articulo.xlsx")) as w:
        for k in sorted(hojas): hojas[k].to_excel(w, sheet_name=k[:31], index=False)
    print("\n" + "\n".join(lin))
    print("\nRecall a FPR fija:"); print(hojas["1_recall_a_FPR_fija"].round(4).to_string(index=False))
    print("\nImportancia por grupos:"); print(imp[["grupo", "caida_pr_auc", "ic_inf", "ic_sup"]].round(4).to_string(index=False))
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min. Archivos en {CARPETA_SALIDA}")
    return df, hojas

if __name__ == "__main__":
    datos, tablas = ejecutar()