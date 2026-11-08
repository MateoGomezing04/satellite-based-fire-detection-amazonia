# -*- coding: utf-8 -*-
"""
13_sensibilidad_etiquetado.py -- Revisión, ronda 1. Sensibilidad del etiquetado y de la estrategia de entrenamiento.
Responde a R1-C2 (ablación de etiquetado y entrenamiento), R1-C5 (500 m y 5 km), R2-C3 (supuestos del etiquetado)
y R3-C5 (focos por regla, fin de mes, FRP de confirmados y no confirmados).
NO modifica el modelo del artículo ni sus salidas.

PARTE A (sin entrenar; usa puntos_ideam_2024_estado.csv del script 03)
  A1  focos confirmados frente a no confirmados: FRP, día/noche, sensor, confianza (prueba de Mann-Whitney para el FRP)
  A2  rejilla de umbrales: margen de la regla (i) = 0/250/500/1000 m × distancia de la regla (ii) = 0/1/2/5 km
      -> focos confirmados, positivos candidatos, eventos y píxeles de fuego del dataset que se conservan
      + píxeles de fuego confirmados por la regla (i) frente a los confirmados solo por la regla (ii)
        (detección de FDC, T7, BTD, FRP y recall del modelo principal en cada grupo)
  A3  efecto de fin de mes: confirmación según los días que faltan para terminar el mes
      y, si se da ESTADO_VECINOS (script 03 con meses vecinos), cuántos focos se recuperan con el mes anterior/siguiente
  A4  columnas de un shape mensual de área quemada (para saber si trae la fecha de quema)

PARTE B (reentrena HGB y RF con el esquema del artículo: mismos pliegues por fecha, hiperparámetros y umbral de máximo F1)
  Variantes de etiquetado (más estrictas; los positivos que la variante no confirma se QUITAN del entrenamiento):
     principal (500 m, 5 km) = control | solo regla (i) | margen 250 m | regla (ii) a 2 km
  Estrategias de entrenamiento (con las etiquetas del artículo):
     solo negativos cercanos | solo negativos lejanos | proporción 1:1 | sin pesos balanceados
  Todas se evalúan sobre el MISMO conjunto de prueba (el del artículo) y sobre un núcleo común
  (positivos que confirman todas las variantes + todos los negativos). Diferencias: bootstrap pareado por fecha.

PENSADO PARA SPYDER: edite CONFIGURACIÓN y ejecute con F5. Tiempo: HGB 1-3 min por experimento; RF bastante más.
SALIDA (CARPETA_SALIDA): sensibilidad_etiquetado.xlsx
"""
import os, re, sys, glob, time, importlib.util, warnings
import numpy as np
import pandas as pd

# ============================== CONFIGURACIÓN (EDITE AQUÍ) ==============================
SCRIPT_07 = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\07. Entrenamiento del modelo.py"
DATASET = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Extraccion GOES puntos calientes\dataset_goes16_2024.csv"
ESTADO = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Puntos calientes\Puntos calientes\preparados\puntos_ideam_2024_estado.csv"
ESTADO_VECINOS = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Puntos calientes\Puntos calientes\preparados_meses_vecinos\puntos_ideam_2024_estado.csv"  # None si no existe
AREAS_DIR = None     # carpeta de los shapes mensuales de área quemada (la AREAS_DIR del script 03); None = omitir A4
CARPETA_SALIDA = r"C:\Users\User\Desktop\Articulos\Detección Amazonía\Sensibilidad etiquetado"
MODELOS = ["hgb", "rf"]
EJECUTAR_PARTE_B = True      # False = solo la parte A (segundos)
N_BOOTSTRAP = 1000
SEMILLA = 42

BUFFER_PRINCIPAL_M, EVENTO_PRINCIPAL_KM = 500, 5.0
REJILLA_BUFFER_M = [0, 250, 500, 1000]
REJILLA_EVENTO_KM = [0, 1, 2, 5]          # 0 = sin regla (ii)
# variantes de etiquetado para la parte B: (nombre, margen regla i en m, distancia regla ii en km)
VARIANTES = [("principal", 500, 5.0), ("solo_regla_i", 500, 0.0), ("margen_250m", 250, 5.0), ("regla_ii_2km", 500, 2.0)]
# estrategias de entrenamiento para la parte B (con las etiquetas principales)
ESTRATEGIAS = ["solo_negativos_cercanos", "solo_negativos_lejanos", "proporcion_1a1", "sin_pesos"]

# ============================== UTILIDADES ==============================
def importar_07(ruta):
    if not os.path.exists(ruta): sys.exit(f"[DETENIDO] No existe el script 07: {ruta}")
    spec = importlib.util.spec_from_file_location("entrenamiento07", ruta)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def cargar_estado(ruta):
    if not ruta or not os.path.exists(ruta): return None
    e = pd.read_csv(ruta, low_memory=False)
    falt = [c for c in ("hotspot_id", "event_id", "dist_area_quemada_km", "en_area_quemada", "estado", "confirmado_por") if c not in e.columns]
    if falt: sys.exit(f"[DETENIDO] {ruta} no tiene las columnas {falt}; debe ser la salida puntos_ideam_2024_estado.csv del script 03")
    e["_t"] = pd.to_datetime(e.datetime_utc)
    if "retained" not in e.columns: e["retained"] = 1
    return e

def confirmar(e, buffer_m, evento_km):
    """Misma lógica del script 03: (i) a <= buffer del área del mes; (ii) evento con algún punto (i) y a <= evento_km."""
    d = e.dist_area_quemada_km.to_numpy(float)                 # NaN = más de 6 km (límite de búsqueda del script 03)
    evaluado = e.en_area_quemada.notna().to_numpy()
    dentro = evaluado & (np.nan_to_num(d, nan=np.inf) <= buffer_m / 1000)
    ev_conf = pd.Series(dentro).groupby(e.event_id.to_numpy()).transform("any").to_numpy()
    por_evento = evaluado & ev_conf & ~dentro & (np.nan_to_num(d, nan=np.inf) <= evento_km) if evento_km > 0 else np.zeros(len(e), bool)
    return dentro, por_evento

def ids_de(s):
    return [int(x) for x in re.findall(r"\d+", str(s))] if pd.notna(s) else []

def resumen_grupo(x):
    x = pd.Series(x).dropna()
    return dict(n=len(x), mediana=x.median(), q1=x.quantile(.25), q3=x.quantile(.75), media=x.mean())

# ============================== PARTE A ==============================
def parte_a(e, ev, df, prob_control, pred_control):
    from scipy.stats import mannwhitneyu
    hojas = {}
    conf = e.estado == "confirmado"

    # ---- A1: confirmados frente a no confirmados
    filas = []
    for nombre, s in (("confirmados", e[conf]), ("no confirmados", e[~conf]),
                      ("confirmados (confianza no baja)", e[conf & (e.retained == 1)]), ("no confirmados (confianza no baja)", e[~conf & (e.retained == 1)])):
        r = resumen_grupo(s.frp_mw)
        filas.append(dict(grupo=nombre, focos=len(s), frp_mediana=r["mediana"], frp_q1=r["q1"], frp_q3=r["q3"], frp_media=r["media"],
                          pct_noche=100 * (s.day_night.astype(str).str.upper().str[0] == "N").mean(),
                          pct_viirs=100 * (s.sensor == "VIIRS").mean(),
                          pct_confianza_baja=100 * (s.retained == 0).mean()))
    a1 = pd.DataFrame(filas)
    x, y = e.loc[conf, "frp_mw"].dropna(), e.loc[~conf, "frp_mw"].dropna()
    u = mannwhitneyu(x, y, alternative="two-sided")
    a1_test = pd.DataFrame([dict(prueba="Mann-Whitney FRP confirmados vs no confirmados", U=u.statistic, p=u.pvalue,
                                 prob_superioridad=u.statistic / (len(x) * len(y)),
                                 nota="prob_superioridad = P(FRP de un confirmado > FRP de un no confirmado); 0.5 = sin diferencia")])
    noche = e.day_night.astype(str).str.upper().str[0].map({"N": "noche", "D": "día"})
    a1_frp = (e.assign(frp=pd.cut(e.frp_mw, [0, 5, 10, 25, 50, 100, np.inf], labels=["<5", "5-10", "10-25", "25-50", "50-100", ">100"]),
                       periodo=noche, confirmado=conf)
              .groupby(["periodo", "frp"], observed=True).confirmado.agg(focos="size", pct_confirmado=lambda s: 100 * s.mean()).reset_index())
    a1_sensor = e.assign(confirmado=conf, periodo=noche).groupby(["sensor", "periodo"]).confirmado.agg(focos="size", pct_confirmado=lambda s: 100 * s.mean()).reset_index()
    hojas.update(A1_confirmados_vs_no=a1, A1_prueba_FRP=a1_test, A1_confirmacion_por_FRP=a1_frp, A1_por_sensor=a1_sensor)

    # ---- A2: rejilla de umbrales
    ids_pos = df.loc[df.label == 1, "hotspot_ids"].map(ids_de)
    base_i, base_ii = confirmar(e, BUFFER_PRINCIPAL_M, EVENTO_PRINCIPAL_KM); base = set(e.hotspot_id[(base_i | base_ii) & (e.retained == 1)])
    filas = []
    for b in REJILLA_BUFFER_M:
        for d in REJILLA_EVENTO_KM:
            di, dii = confirmar(e, b, d); c = di | dii
            pos_ids = set(e.hotspot_id[c & (e.retained == 1)])
            px = int(ids_pos.map(lambda L: any(i in pos_ids for i in L)).sum())
            filas.append(dict(margen_regla_i_m=b, distancia_regla_ii_km=d, principal=(b == BUFFER_PRINCIPAL_M and d == EVENTO_PRINCIPAL_KM),
                              confirmados=int(c.sum()), por_regla_i=int(di.sum()), por_regla_ii=int(dii.sum()),
                              pct_confirmados=100 * c.mean(), positivos_candidatos=len(pos_ids),
                              eventos_con_confirmado=int(pd.Series(c).groupby(e.event_id.to_numpy()).any().sum()),
                              pixeles_fuego_del_dataset_conservados=px,
                              positivos_nuevos_sin_extraccion=len(pos_ids - base)))
    hojas["A2_rejilla_umbrales"] = pd.DataFrame(filas)
    dist_ii = e.loc[base_ii, "dist_area_quemada_km"]
    hojas["A2_distancia_regla_ii"] = pd.DataFrame([dict(focos_regla_ii=len(dist_ii), **{f"pct_a_<=_{k}_km": 100 * (dist_ii <= k).mean() for k in (1, 2, 3, 4, 5)},
                                                        mediana_km=dist_ii.median())])

    # píxeles de fuego por regla: si la regla (ii) arrastrara falsas alarmas, esos píxeles se parecerían al fondo
    via = e.set_index("hotspot_id").confirmado_por.to_dict()
    def via_pixel(L):
        v = {via.get(i, "") for i in L}
        return "regla (i)" if "punto" in v else ("solo regla (ii)" if "evento" in v else "sin enlace")
    pos = df[df.label == 1].copy(); pos["regla"] = ids_pos.map(via_pixel)
    pos["detecta_modelo"] = pred_control["hgb"][df.label.to_numpy() == 1] if "hgb" in pred_control else np.nan
    grp = pos.groupby("regla").agg(pixeles=("label", "size"), pct_detectado_FDC=("goes_fdc", lambda s: 100 * s.mean()),
                                   T7_mediana=("T7", "median"), BTD_mediana=("BTD", "median"), FRP_mediana=("frp_mw", "median"),
                                   pct_noche=("day_night", lambda s: 100 * (s.astype(str).str.upper().str[0] == "N").mean()),
                                   recall_HGB_principal=("detecta_modelo", "mean")).reset_index()
    neg = df[df.label == 0]
    grp = pd.concat([grp, pd.DataFrame([dict(regla="negativos (referencia)", pixeles=len(neg), pct_detectado_FDC=100 * neg.goes_fdc.mean(),
                                             T7_mediana=neg.T7.median(), BTD_mediana=neg.BTD.median())])], ignore_index=True)
    hojas["A2_pixeles_por_regla"] = grp

    # ---- A3: fin de mes
    dias_fin = (e._t.dt.days_in_month - e._t.dt.day).to_numpy()
    tramo = pd.cut(dias_fin, [-1, 2, 6, 31], labels=["últimos 3 días", "4 a 7 días antes del fin", "resto del mes"])
    a3 = e.assign(tramo=tramo, confirmado=conf).groupby("tramo", observed=True).confirmado.agg(focos="size", pct_confirmado=lambda s: 100 * s.mean()).reset_index()
    a3_mes = e.assign(tramo=tramo, confirmado=conf, mes=e._t.dt.month).query("mes in [1, 2, 3]").groupby(["mes", "tramo"], observed=True) \
              .confirmado.agg(focos="size", pct_confirmado=lambda s: 100 * s.mean()).reset_index()
    hojas.update(A3_fin_de_mes=a3, A3_fin_de_mes_ene_mar=a3_mes)
    if ev is not None:
        c_ev = ev.set_index("hotspot_id").estado.eq("confirmado")
        c_ba = e.set_index("hotspot_id").estado.eq("confirmado")
        comun = c_ba.index.intersection(c_ev.index)
        nuevos = comun[c_ev.loc[comun].to_numpy() & ~c_ba.loc[comun].to_numpy()]
        en = e.set_index("hotspot_id").loc[nuevos]
        df_fin = (en._t.dt.days_in_month - en._t.dt.day)
        ret_n = int(en.retained.eq(1).sum())
        hojas["A3_meses_vecinos"] = pd.DataFrame([
            dict(concepto="confirmados solo con el mes del foco", valor=int(c_ba.sum())),
            dict(concepto="confirmados con mes anterior y siguiente", valor=int(c_ev.sum())),
            dict(concepto="focos recuperados", valor=len(nuevos)),
            dict(concepto="  de ellos con confianza no baja (posibles positivos nuevos)", valor=ret_n),
            dict(concepto="  de ellos en los últimos 7 días del mes", valor=int((df_fin <= 6).sum())),
            dict(concepto="  de ellos en los primeros 7 días del mes", valor=int((en._t.dt.day <= 7).sum())),
            dict(concepto="aumento de focos confirmados (%)", valor=round(100 * len(nuevos) / max(int(c_ba.sum()), 1), 1)),
            dict(concepto="tasa de confirmación con mes del foco (%)", valor=round(100 * c_ba.mean(), 1)),
            dict(concepto="tasa de confirmación con meses vecinos (%)", valor=round(100 * c_ev.mean(), 1))])
    # ---- A4: columnas de un shape mensual
    if AREAS_DIR:
        try:
            import geopandas as gpd
            f = sorted(glob.glob(os.path.join(AREAS_DIR, "*.shp")))[0]; g = gpd.read_file(f, rows=5)
            cand = [c for c in g.columns if re.search(r"burn|date|fecha|dia|doy|julian|dn|band", c, re.I)]
            hojas["A4_columnas_area_quemada"] = pd.DataFrame(dict(archivo=os.path.basename(f), columna=list(g.columns),
                                                                  ejemplo=[str(g[c].iloc[0])[:60] for c in g.columns],
                                                                  posible_fecha=[c in cand for c in g.columns]))
        except Exception as ex:
            hojas["A4_columnas_area_quemada"] = pd.DataFrame([dict(error=str(ex))])
    return hojas

# ============================== PARTE B ==============================
def validar_variante(m07, df, modelo, cv_splits, entrenar, estrategia=None):
    """Igual que validar() del 07, pero solo entrena con las filas 'entrenar' (bool) y predice TODO el pliegue de prueba."""
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = df[m07.VARIABLES], df.label.to_numpy(), df.fecha.to_numpy()
    prob = np.full(len(df), np.nan); pred = np.zeros(len(df), int); ths = []
    rng = np.random.default_rng(SEMILLA)
    def nuevo():
        mdl = m07.crear_modelo(modelo)
        if estrategia == "sin_pesos" and modelo == "rf": mdl.set_params(randomforestclassifier__class_weight=None)
        return mdl
    def ajustar(mdl, idx):
        if estrategia == "sin_pesos" and modelo == "hgb": return mdl.fit(X.iloc[idx], y[idx])
        return m07.ajustar(mdl, modelo, X.iloc[idx], y[idx])
    for k, (tr, te) in enumerate(cv_splits, 1):
        tr = tr[entrenar[tr]]
        if estrategia == "proporcion_1a1":
            p, n = tr[y[tr] == 1], tr[y[tr] == 0]
            tr = np.sort(np.concatenate([p, rng.choice(n, size=min(len(p), len(n)), replace=False)]))
        cv_in = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=m07.SEMILLA + k)
        p_in = np.full(len(tr), np.nan)
        for a, b in cv_in.split(X.iloc[tr], y[tr], g[tr]):
            p_in[b] = ajustar(nuevo(), tr[a]).predict_proba(X.iloc[tr[b]])[:, 1]
        th = m07.umbral_max_f1(y[tr], p_in); ths.append(th)
        prob[te] = ajustar(nuevo(), tr).predict_proba(X.iloc[te])[:, 1]; pred[te] = (prob[te] >= th).astype(int)
    return prob, pred, ths

def metricas_en(m07, df, prob, pred, mask):
    from sklearn.metrics import matthews_corrcoef
    y = df.label.to_numpy()[mask]; m = m07.metricas(y, pred[mask], prob[mask])
    return dict(n_fuegos=int(y.sum()), recall=m["recall"], precision=m["precision"], f1=m["f1"], mcc=matthews_corrcoef(y, pred[mask]),
                pr_auc=m["pr_auc"], fpr=m["tasa_falsos_positivos"])

def diferencia_pareada(df, mask, pa, ya, pb, yb):
    from sklearn.metrics import average_precision_score, f1_score, recall_score
    sub = df[mask]; y = sub.label.to_numpy(); pa, ya, pb, yb = pa[mask], ya[mask], pb[mask], yb[mask]
    neg = y == 0
    def d(f):
        return (average_precision_score(y[f], pa[f]) - average_precision_score(y[f], pb[f]),
                f1_score(y[f], ya[f]) - f1_score(y[f], yb[f]), recall_score(y[f], ya[f]) - recall_score(y[f], yb[f]),
                ya[f][neg[f]].mean() - yb[f][neg[f]].mean())
    base = d(np.arange(len(y))); idx = sub.groupby("fecha").indices; fe = np.array(list(idx)); rng = np.random.default_rng(SEMILLA); R = []
    for _ in range(N_BOOTSTRAP):
        f = np.concatenate([idx[x] for x in rng.choice(fe, len(fe), replace=True)])
        if y[f].min() == y[f].max(): continue
        R.append(d(f))
    R = np.array(R); out = {}
    for j, nm in enumerate(("pr_auc", "f1", "recall", "fpr")):
        out[f"dif_{nm}"] = base[j]; out[f"dif_{nm}_ic"] = f"[{np.quantile(R[:, j], .025):.3f}, {np.quantile(R[:, j], .975):.3f}]"
    return out

# ============================== PRINCIPAL ==============================
def ejecutar():
    warnings.filterwarnings("ignore"); os.makedirs(CARPETA_SALIDA, exist_ok=True)
    m07 = importar_07(SCRIPT_07)
    df = m07.cargar(DATASET)
    falt = [c for c in ("hotspot_ids", "neg_type", "frp_mw", "day_night") if c not in df.columns]
    if falt: sys.exit(f"[DETENIDO] Faltan columnas en el dataset: {falt}")
    e = cargar_estado(ESTADO)
    if e is None: sys.exit(f"[DETENIDO] No existe {ESTADO}")
    ev = cargar_estado(ESTADO_VECINOS)
    print(f"Dataset: {len(df):,} filas, {int(df.label.sum()):,} fuegos | focos en estado: {len(e):,} ({int((e.estado == 'confirmado').sum()):,} confirmados)"
          f" | meses vecinos: {'sí' if ev is not None else 'no'}")

    # positivos de cada variante (por los hotspot_ids del píxel)
    ids_pos = df.hotspot_ids.map(ids_de)
    es_pos = df.label.to_numpy() == 1
    etiqueta = {}
    for nombre, b, dkm in VARIANTES:
        di, dii = confirmar(e, b, dkm); ok = set(e.hotspot_id[(di | dii) & (e.retained == 1)])
        etiqueta[nombre] = es_pos & ids_pos.map(lambda L: any(i in ok for i in L)).to_numpy()
    n_princ = int(etiqueta["principal"].sum())
    print("Píxeles de fuego conservados por variante: " + ", ".join(f"{k} {int(v.sum()):,}" for k, v in etiqueta.items()))
    if n_princ != int(es_pos.sum()):
        print(f"[AVISO] La variante principal conserva {n_princ:,} de {int(es_pos.sum()):,} píxeles de fuego. "
              "Revise que ESTADO sea el archivo con el que se construyó el dataset.")
    nucleo = np.logical_and.reduce([v for v in etiqueta.values()])
    conjuntos = {"prueba_articulo": np.ones(len(df), bool), "prueba_nucleo": nucleo | ~es_pos}

    # ---- parte B
    met, dif, preds = [], [], {}
    pred_control = {}
    if EJECUTAR_PARTE_B:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = list(StratifiedGroupKFold(n_splits=m07.N_PLIEGUES, shuffle=True, random_state=m07.SEMILLA)
                  .split(df[m07.VARIABLES], df.label.to_numpy(), df.fecha.to_numpy()))
        experimentos = [(f"etiq_{n}", etiqueta[n] | ~es_pos, None) for n, _, _ in VARIANTES]
        cerc, lej = (df.neg_type == "near").to_numpy(), (df.neg_type == "far").to_numpy()
        for s in ESTRATEGIAS:
            entr = {"solo_negativos_cercanos": es_pos | cerc, "solo_negativos_lejanos": es_pos | lej}.get(s, np.ones(len(df), bool))
            experimentos.append((f"entr_{s}", entr, s))
        for modelo in MODELOS:
            for nombre, entrenar, estr in experimentos:
                t0 = time.time()
                prob, pred, ths = validar_variante(m07, df, modelo, cv, entrenar, estr)
                preds[(modelo, nombre)] = (prob, pred)
                for cj, mask in conjuntos.items():
                    met.append(dict(modelo=modelo, experimento=nombre, conjunto_prueba=cj, fuegos_entrenamiento=int((entrenar & es_pos).sum()),
                                    negativos_entrenamiento=int((entrenar & ~es_pos).sum()), umbral_medio=float(np.mean(ths)),
                                    **metricas_en(m07, df, prob, pred, mask)))
                r = met[-2]
                print(f"[{modelo}] {nombre:<32} F1 {r['f1']:.3f} | PR-AUC {r['pr_auc']:.3f} | recall {r['recall']:.3f} | FPR {r['fpr']:.4f} | {time.time() - t0:.0f} s")
            pred_control[modelo] = preds[(modelo, "etiq_principal")][1]
            for nombre, _, _ in experimentos[1:]:
                for cj, mask in conjuntos.items():
                    dif.append(dict(modelo=modelo, experimento=nombre, comparado_con="etiq_principal", conjunto_prueba=cj,
                                    **diferencia_pareada(df, mask, *preds[(modelo, nombre)], *preds[(modelo, "etiq_principal")])))
        # recall por regla de confirmación del píxel, para cada variante de etiquetado
        via = e.set_index("hotspot_id").confirmado_por.to_dict()
        regla = ids_pos.map(lambda L: "regla (i)" if "punto" in {via.get(i, "") for i in L} else "solo regla (ii)").to_numpy()
        rr = []
        for (modelo, nombre), (prob, pred) in preds.items():
            for rg in ("regla (i)", "solo regla (ii)"):
                s = es_pos & (regla == rg)
                rr.append(dict(modelo=modelo, experimento=nombre, regla_del_pixel=rg, pixeles=int(s.sum()), recall=float(pred[s].mean())))
        hoja_rr = pd.DataFrame(rr)

    # ---- parte A
    hojas = parte_a(e, ev, df, {}, pred_control)

    salida = os.path.join(CARPETA_SALIDA, "sensibilidad_etiquetado.xlsx")
    with pd.ExcelWriter(salida) as w:
        for k, v in hojas.items(): v.to_excel(w, sheet_name=k[:31], index=False)
        if EJECUTAR_PARTE_B:
            pd.DataFrame(met).to_excel(w, sheet_name="B_metricas", index=False)
            pd.DataFrame(dif).to_excel(w, sheet_name="B_diferencias", index=False)
            hoja_rr.to_excel(w, sheet_name="B_recall_por_regla", index=False)
        pd.DataFrame({"parametro": ["script_07", "dataset", "estado", "estado_vecinos", "variantes", "estrategias", "n_bootstrap", "semilla",
                                    "fuegos_nucleo", "nota"],
                      "valor": [SCRIPT_07, DATASET, ESTADO, ESTADO_VECINOS, str(VARIANTES), str(ESTRATEGIAS), N_BOOTSTRAP, SEMILLA,
                                int(nucleo.sum()),
                                "prueba_articulo: todos los fuegos y negativos del artículo; prueba_nucleo: fuegos confirmados por todas las variantes "
                                "+ todos los negativos. Diferencias: bootstrap pareado por fecha frente a etiq_principal del mismo algoritmo."]}
                     ).to_excel(w, sheet_name="configuracion", index=False)

    pd.set_option("display.width", 240)
    print("\n================ PARTE A ================")
    for k in ("A1_confirmados_vs_no", "A1_prueba_FRP", "A2_rejilla_umbrales", "A2_pixeles_por_regla", "A3_fin_de_mes", "A3_meses_vecinos"):
        if k in hojas: print(f"\n{k}"); print(hojas[k].round(3).to_string(index=False))
    if EJECUTAR_PARTE_B:
        print("\n================ PARTE B (prueba del artículo) ================")
        mm = pd.DataFrame(met); print(mm[mm.conjunto_prueba == "prueba_articulo"].round(3).to_string(index=False))
        print("\nDiferencias frente a la etiqueta principal (bootstrap pareado por fecha):")
        print(pd.DataFrame(dif).round(3).to_string(index=False))
        for modelo, ref in (("hgb", (0.858, 0.932)), ("rf", (0.858, 0.931))):
            r = mm[(mm.modelo == modelo) & (mm.experimento == "etiq_principal") & (mm.conjunto_prueba == "prueba_articulo")]
            if len(r) and (abs(r.f1.iloc[0] - ref[0]) > 0.005 or abs(r.pr_auc.iloc[0] - ref[1]) > 0.005):
                print(f"[AVISO] {modelo} principal NO reproduce el artículo (F1 {ref[0]}, PR-AUC {ref[1]}). Revise antes de interpretar.")
    print(f"\nGuardado: {salida}")
    return hojas

if __name__ == "__main__":
    resultado = ejecutar()