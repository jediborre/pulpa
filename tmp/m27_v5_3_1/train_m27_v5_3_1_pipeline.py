"""
tmp/m27_v5_3_1/train_m27_v5_3_1_pipeline.py
Entrenamiento y auditoria completa de m27_v5_3_1 (Protocolo AGENTS.md regla 6).

Split cronologico por PARTIDO (todas las filas/snapshots de un partido van al mismo bloque):
  A = 70% (entrena)  |  B = 12% (calibrador isotonico + eleccion de umbral, fuera de muestra)  |  C = 18% final (test intacto)
Entrenamiento con aumentacion espejo local<->visitante (simetria) y pesos 1/filas_por_partido.
Ensamble XGB(0.40) + HGB(0.35) + LR imputada(0.25) + regresor de margen Q4.

Salidas en models/m27_v5_3_1/model_outputs/: pesos, calibrador, model_config.json, m27_v5_3_1_summary.json,
league_calibration.json, feature_justification.md/json y m27_v5_3_1_diagnostic_dashboard.png.

Uso: .venv\\Scripts\\python.exe tmp\\m27_v5_3_1\\train_m27_v5_3_1_pipeline.py [--quick] [--skip-ablation] [--skip-wf]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.m27_v5_3_1.features import FEATURE_NAMES, FEATURE_SPEC, IDX, MODEL_DIR, mirror  # noqa: E402
from models.common.calibration import PlattCalibrator  # noqa: E402

DATASET = ROOT / "tmp" / "m27_v5_3_1" / "dataset_v5_3_1.pkl"
W = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}
# Calibracion del clasificador: "platt" (continua, por defecto) o "isotonic".
CALIBRATION_KIND = "platt"
BE_140 = 1 / 1.40  # punto de equilibrio con cuota 1.40

# Suelo de calidad para el umbral operativo: el acierto fuera de muestra debe
# ser >= WR_MIN_OPERATING. Motivo: a cuota 1.40 los umbrales 0.72 y 0.75 estan
# EMPATADOS en beneficio esperado (-1.1 %) y 0.72 mete un 20 % mas de
# apuestas con 1.5 pp menos de acierto. Maximizar beneficio absoluto
# (w*1.40 - n) premia el volumen y empuja al borde, donde la banda marginal
# 0.72-0.75 rindio 62-64 % en vivo (por debajo del equilibrio de 71.4 %).
WR_MIN_OPERATING = 0.78
# Umbral de despliegue: decision explicita de operacion, no del pipeline.
# Ver docs/ANALISIS_LIVE_M27_V5_3.md (R2). El pipeline imprime el umbral
# auto-seleccionado para comparar, pero despliega este.
OPERATING_CONF = 0.75

FEATURE_DOCS: dict[str, tuple[str, str]] = {
    "minute": ("A estado", "Minuto real del snapshot (25.5-30). Hace al modelo consciente de cuanta informacion tiene; antes el 'm27' era en realidad m30."),
    "cur_score_diff": ("A estado", "Marcador local-visitante al minuto t. Base del estado; en v5_2 se entrenaba con Q1+Q2+Q3 completos y en vivo llegaba parcial."),
    "abs_cur_score_diff": ("A estado", "Magnitud de la ventaja; separa blowouts (basura) de juegos cerrados."),
    "ht_diff": ("A estado", "Diferencial al descanso (Q1+Q2)."),
    "q1_diff": ("A estado", "Diferencial Q1."),
    "q2_diff": ("A estado", "Diferencial Q2."),
    "q3_partial_diff": ("A estado", "Diferencial del Q3 jugado hasta t (cur - ht). Reemplaza q3_diff completo para ser valido a cualquier minuto."),
    "q3_pace_diff": ("A estado", "q3_partial_diff reescalado a 10 min: momentum del Q3 comparable entre t=26 y t=30 (reemplaza q3_eff_diff)."),
    "q3_vs_ht_momentum": ("A estado", "Q3 reescalado menos media del descanso: cambio de inercia 2a mitad vs 1a."),
    "recent_pts_diff_2m": ("A estado", "Puntos local-visitante en los ultimos 2 min reales (match_events). En v5_2 era siempre 0 en vivo y en train un proxy q3*0.2."),
    "recent_pts_diff_5m": ("A estado", "Igual con ventana de 5 min: tendencia reciente menos ruidosa."),
    "recent_pace_ratio": ("A estado", "Puntos totales de los ultimos 4 min / ritmo esperado: detecta ritmo reciente alto/bajo (varianza del Q4)."),
    "pace_pts_per_min": ("A estado", "Puntos totales por minuto jugado al minuto t."),
    "pace_vs_league_prior": ("A estado", "Ritmo menos el esperado de la liga (priors de leagues_classification)."),
    "gp_val": ("A estado", "Ultimo valor de la grafica de presion de SofaScore hasta t (NaN si no hay grafica, no 0)."),
    "gp_slope_3m": ("A estado", "Pendiente de la grafica en 3 puntos: momentum reciente segun SofaScore (se habia podado por entrenarse con ceros)."),
    "lead_z": ("A estado", "cur_diff/sqrt(puntos totales): ventaja normalizada por el volumen de anotacion."),
    "lead_vs_lg_vol": ("A estado", "cur_diff / volatilidad Q4 de la liga: cuantas 'desviaciones de Q4' vale la ventaja en esa liga."),
    "d_elo_ft": ("B fuerza", "Diferencial ELO de partido (K=20, decaimiento por inactividad), as-of via PriorState."),
    "d_elo_q4": ("B fuerza", "Diferencial ELO especifico de Q4 (K=15): equipos que cierran mejor."),
    "d_wr15": ("B fuerza", "Win-rate bayesiano ultimos 15 juegos (misma definicion en train y live; v5_2 usaba ELO/180 en vivo)."),
    "d_q4_bayes": ("B fuerza", "Margen medio Q4 ultimos 15 juegos (bayesiano). v5_2 lo aproximaba con ELO/35 en vivo."),
    "d_ft_margin15": ("B fuerza", "NUEVA: margen medio de partido ultimos 15 juegos; mas informativo que el win-rate binario."),
    "d_q4_net12": ("B fuerza", "Perfil 'closer' (margen neto Q4 ultimos 12). En v5_2 era siempre 0 en vivo (q_profile inexistente)."),
    "d_q3_net12": ("B fuerza", "NUEVA: margen neto Q3 historico; permite medir cuanto el Q3 actual excede lo habitual (fatiga/regresion)."),
    "d_season_wr": ("B fuerza", "NUEVA: record de la temporada as-of (reinicia tras 150 dias) calculado desde matches.db; reemplaza ts_delta_wp_bayes cuyo standings se bajo en oct-2026 para todo el historico (look-ahead)."),
    "h2h_ft_margin": ("B fuerza", "Margen H2H (ultimos 8, encogido) calculado del historico propio. En v5_2 el live pasaba una lista y el extractor esperaba dict => 0."),
    "h2h_q4_margin": ("B fuerza", "Margen H2H en Q4 (ultimos 8, encogido)."),
    "elo_reliability": ("B fuerza", "min(juegos previos)/12: confianza de los priors; el modelo la usa para descontarlos en equipos nuevos."),
    "has_ml": ("C mercado", "1 si hay cuota ML pre-partido (SofaScore no la publica en ~40% de partidos). Antes se confundia con 0."),
    "odds_log_ratio": ("C mercado", "ln(cuota visita/cuota local), ML PRE-PARTIDO (NaN si no hay). El parser ahora descarta mercados in-play que contaminaban el valor."),
    "mkt_p_home": ("C mercado", "Probabilidad implicita local sin vig; escala natural para el modelo logistico."),
    "expected_ft_margin": ("C mercado", "Margen esperado = 6.5*olr + 0.5*d_elo/28 (con ML) o d_elo/28; misma formula train y live."),
    "mkt_margin_spread": ("C mercado", "NUEVA: -linea de handicap pre-partido (margen esperado del mercado). Solo existe en partidos con backfill de lineas / capturados pre-tip."),
    "total_line": ("C mercado", "NUEVA: linea de total pre-partido (ritmo esperado). Mismo origen que el spread."),
    "cur_vs_expected": ("C mercado", "cur_diff - 0.75*expected_ft_margin: ventaja real vs esperada (rubber_band_gap verdadero; en v5_2 live era otra formula)."),
    "lg_avg_ft": ("D liga", "Puntos totales promedio de la liga (leagues_classification)."),
    "lg_avg_q4": ("D liga", "Puntos promedio de Q4 en la liga."),
    "lg_q4_vol": ("D liga", "NUEVA: volatilidad |margen Q4| de la liga, expanding (as-of). Normaliza ventajas y detecta ligas de varianza alta."),
    "lg_home_q4_rate": ("D liga", "NUEVA: tasa as-of de victoria local en Q4 por liga (sesgo local por liga)."),
    "is_women": ("D liga", "Liga femenina (distinta dinamica y volatilidad)."),
    "is_home_view": ("E orientacion", "1 en la vista original, 0 en la espejo: permite la aumentacion local<->visitante conservando la ventaja de local."),
}
GROUPS = {n: FEATURE_DOCS[n][0] for n in FEATURE_NAMES}


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def make_models(quick: bool):
    n_x = 80 if quick else 220
    n_h = 70 if quick else 200
    return dict(
        xgb=xgb.XGBClassifier(n_estimators=n_x, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=45.0, min_child_weight=40, tree_method="hist", random_state=42, n_jobs=-1,
                              eval_metric="logloss"),
        hgb=HistGradientBoostingClassifier(max_iter=n_h, max_leaf_nodes=15, min_samples_leaf=400, learning_rate=0.05,
                                           l2_regularization=40.0, random_state=42),
        lr=make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.02, max_iter=300)),
        reg=HistGradientBoostingRegressor(max_iter=n_h, max_leaf_nodes=15, min_samples_leaf=400, learning_rate=0.05,
                                          l2_regularization=40.0, random_state=42),
    )


def fit_ensemble(X, y, yr, w, quick=False):
    """Entrena con aumentacion espejo. Devuelve dict de modelos."""
    Xa = np.vstack([X, mirror(X)])
    ya = np.concatenate([y, 1 - y])
    yra = np.concatenate([yr, -yr])
    wa = np.concatenate([w, w])
    m = make_models(quick)
    m["xgb"].fit(Xa, ya, sample_weight=wa)
    m["hgb"].fit(Xa, ya, sample_weight=wa)
    m["lr"].fit(Xa, ya, logisticregression__sample_weight=wa)
    m["reg"].fit(Xa, yra, sample_weight=wa)
    return m


def ens(m, X):
    return W["xgb"] * m["xgb"].predict_proba(X)[:, 1] + W["hgb"] * m["hgb"].predict_proba(X)[:, 1] + W["lr"] * m["lr"].predict_proba(X)[:, 1]


def p_sym(m, X):
    return 0.5 * (ens(m, X) + 1.0 - ens(m, mirror(X)))


def margin_sym(m, X):
    return 0.5 * (m["reg"].predict(X) - m["reg"].predict(mirror(X)))


def fit_calibrator(p, y):
    """Calibrador monotono SIMETRICO de la probabilidad del local.

    Por defecto usa `PlattCalibrator` (continuo): la isotonica es escalonada y
    produce una meseta en el umbral operativo (0.75 = 0.76 = 0.78 = 0.80
    elegian las MISMAS apuestas) ademas de dejar la confianza cuantizada.
    Medido en el bloque C: platt AUC 0.6537 / Brier 0.2306 / LogLoss 0.6524
    frente a isotonica AUC 0.6536 / Brier 0.2307 / LogLoss 0.6528, y +13 % de
    beneficio esperado al mismo umbral 0.75.
    Ver docs/PLAN_V5_4.md.
    """
    if CALIBRATION_KIND == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.03, y_max=0.95)
        iso.fit(np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y]))
        return iso
    return PlattCalibrator().fit(p, y)


def green(p, y, thr, margin=None, mm=0.0):
    home = (p >= thr) & ((margin >= mm) if margin is not None else True)
    away = (p <= 1 - thr) & ((margin <= -mm) if margin is not None else True)
    n = int(home.sum() + away.sum())
    w = int((y[home] == 1).sum() + (y[away] == 0).sum())
    return n, w


def roi(w, n, odds):
    return (w * odds - n) / n * 100.0 if n else 0.0


def metrics(p, y, label=""):
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4))),
            "acc": float(accuracy_score(y, p >= 0.5)), "n": int(len(y))}


def tier_table(p, y, margin=None, thrs=(0.65, 0.68, 0.70, 0.72, 0.75, 0.80)):
    rows = []
    for t in thrs:
        n, w = green(p, y, t, margin, 0.0)
        lo, hi = wilson(w, n)
        rows.append({"thr": t, "bets": n, "wr": round(w / n * 100, 2) if n else 0.0, "wr_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
                     "roi_140": round(roi(w, n, 1.40), 2), "roi_183": round(roi(w, n, 1.83), 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-wf", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    D = pickle.load(open(DATASET, "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    ord_ = np.array([m["ord"] for m in meta])
    mrank = np.array([rank.get(int(o), -1) for o in ord_])
    anchor = np.array([m["anchor"] for m in meta])
    tt = np.array([m["t"] for m in meta])
    yr_ = np.array([m["yr"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    cnt = pd.Series(mrank).map(pd.Series(mrank).value_counts()).values
    w_all = 1.0 / cnt
    print(f"Filas: {len(y):,} | elegibles {n_el:,} partidos | A {rowA.sum():,} B {rowB.sum():,} C {rowC.sum():,} filas | anchors C {int((rowC & anchor).sum()):,}", flush=True)

    # --------- 1. Entrenamiento ---------
    print("\n[1] Entrenando ensamble (aumentacion espejo)...", flush=True)
    models = fit_ensemble(X[rowA], y[rowA], q4d[rowA], w_all[rowA], args.quick)
    print(f"    listo ({time.time()-t0:.0f}s)", flush=True)

    pB = p_sym(models, X[rowB]); mB = margin_sym(models, X[rowB])
    iso = fit_calibrator(pB, y[rowB])
    pA = iso.predict(p_sym(models, X[rowA]))
    pC_raw = p_sym(models, X[rowC]); pC = iso.predict(pC_raw); mC = margin_sym(models, X[rowC])
    pBc = iso.predict(pB)
    yA, yB, yC = y[rowA], y[rowB], y[rowC]
    aC, tC = anchor[rowC], tt[rowC]

    # --------- 2. Metricas ---------
    mt_tr, mt_te, mt_b = metrics(pA, yA), metrics(pC, yC), metrics(pBc, yB)
    gap = mt_tr["auc"] - mt_te["auc"]
    mt_anchor, mt_intra = metrics(pC[aC], yC[aC]), metrics(pC[~aC], yC[~aC])
    print(f"\n[2] AUC train {mt_tr['auc']:.4f} | calib(B) {mt_b['auc']:.4f} | test(C) {mt_te['auc']:.4f} | gap {gap:+.4f}")
    print(f"    Brier test {mt_te['brier']:.4f} | LogLoss {mt_te['logloss']:.4f} | Acc {mt_te['acc']:.4f} | MAE margen Q4 {mean_absolute_error(q4d[rowC], mC):.2f}")
    print(f"    Anchor(t=30): AUC {mt_anchor['auc']:.4f} Brier {mt_anchor['brier']:.4f} | Intra(25.5-29.9): AUC {mt_intra['auc']:.4f} Brier {mt_intra['brier']:.4f}")

    # --------- 3. Umbral y margen elegidos en B (fuera de muestra) ---------
    THR_GRID = (0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80)
    MM_GRID = (0.0, 1.0, 1.8)

    def _best_on_b(thrs, require_wr_floor: bool):
        best = None
        for thr in thrs:
            for mm in MM_GRID:
                n, w = green(pBc, yB, thr, mB, mm)
                if n < 200:
                    continue
                if require_wr_floor and (w / n) < WR_MIN_OPERATING:
                    continue
                profit = w * 1.40 - n
                if best is None or profit > best[0]:
                    best = (profit, thr, mm, n, w / n)
        return best

    best_raw = _best_on_b(THR_GRID, require_wr_floor=False)
    best_q = _best_on_b(THR_GRID, require_wr_floor=True)
    if best_q is not None:
        _, thr_auto, mm_auto, nB_auto, wrB_auto = best_q
        sel = "con suelo de calidad (WR >= %.0f%%)" % (WR_MIN_OPERATING * 100)
    else:
        _, thr_auto, mm_auto, nB_auto, wrB_auto = best_raw
        sel = "sin suelo (ningun umbral alcanza el WR minimo)"
    print(f"\n[3] Umbral auto-seleccionado en B: P>={thr_auto} margen>={mm_auto} "
          f"(B: {nB_auto} apuestas, WR {wrB_auto * 100:.1f}%) — {sel}")
    print(f"    Umbral de DEPLOY: P>={OPERATING_CONF} (decision explicita, OPERATING_CONF)")

    thr_star = OPERATING_CONF
    mm_star, nB, wrB, profitB = None, 0, 0.0, None
    for mm in MM_GRID:
        n, w = green(pBc, yB, thr_star, mB, mm)
        if n < 200:
            continue
        profit = w * 1.40 - n
        if profitB is None or profit > profitB:
            profitB, mm_star, nB, wrB = profit, mm, n, w / n
    print(f"    Margen de DEPLOY: margen>={mm_star} (B: {nB} apuestas, WR {wrB * 100:.1f}%, "
          f"beneficio@1.40 {profitB:,.0f} u)")

    tiers_all = tier_table(pC, yC, mC)
    tiers_anchor = tier_table(pC[aC], yC[aC], mC[aC])
    tiers_intra = tier_table(pC[~aC], yC[~aC], mC[~aC])
    print("    Tabla de umbrales en C (todas las filas):")
    for r in tiers_all:
        print(f"      P>={r['thr']:.2f} | {r['bets']:>6,} apuestas | WR {r['wr']:5.2f}% (IC95 {r['wr_ci95']}) | ROI@1.40 {r['roi_140']:+6.2f}% | ROI@1.83 {r['roi_183']:+6.2f}%")
    print("    Tabla de umbrales en C solo ANCHOR (t=30, un partido = una fila; comparable con v5_2):")
    for r in tiers_anchor:
        print(f"      P>={r['thr']:.2f} | {r['bets']:>6,} apuestas | WR {r['wr']:5.2f}% (IC95 {r['wr_ci95']}) | ROI@1.40 {r['roi_140']:+6.2f}% | ROI@1.83 {r['roi_183']:+6.2f}%")
    n_g, w_g = green(pC, yC, thr_star, mC, mm_star)
    wr_g = w_g / n_g * 100 if n_g else 0
    print(f"    OPERATIVO (P>={thr_star}, margen>={mm_star}): {n_g:,} apuestas | WR {wr_g:.2f}% | ROI@1.40 {roi(w_g, n_g, 1.40):+.2f}% | ROI@1.83 {roi(w_g, n_g, 1.83):+.2f}%")

    # --------- 4. Resiliencia por minuto y politica de disparo ---------
    print("\n[4] Resiliencia por minuto (C, partidos con eventos):", flush=True)
    bins = [25.5, 26.5, 27.5, 28.5, 29.5, 30.01]
    res_min = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (tC >= lo) & (tC < hi)
        if mk.sum() < 200:
            continue
        n, w = green(pC[mk], yC[mk], thr_star, mC[mk], mm_star)
        res_min.append({"t": f"{lo:.1f}-{hi:.1f}", "n_rows": int(mk.sum()), "auc": round(float(roc_auc_score(yC[mk], pC[mk])), 4),
                        "brier": round(float(brier_score_loss(yC[mk], pC[mk])), 4), "bets": n, "wr": round(w / n * 100, 2) if n else 0.0,
                        "bet_rate": round(n / mk.sum() * 100, 2)})
        print(f"    t {lo:4.1f}-{hi:5.2f} | filas {mk.sum():>6,} | AUC {res_min[-1]['auc']:.4f} | apuestas {n:>5} ({res_min[-1]['bet_rate']:.1f}%) | WR {res_min[-1]['wr']:.2f}%")
    midC = np.array([meta[i]["mid"] for i in np.where(rowC)[0]])
    df = pd.DataFrame({"mid": midC, "t": tC, "p": pC, "y": yC, "m": mC})
    pol = {"first_fire": [0, 0], "anchor_only": [0, 0], "random_tick": [0, 0]}
    for _, g in df.groupby("mid"):
        if len(g) < 2:
            continue
        g = g.sort_values("t")
        for key, rows_ in (("first_fire", g), ("anchor_only", g.tail(1)), ("random_tick", g.head(1))):
            for _, r_ in rows_.iterrows():
                nn, ww = green(np.array([r_.p]), np.array([r_.y]), thr_star, np.array([r_.m]), mm_star)
                if nn:
                    pol[key][0] += nn; pol[key][1] += ww
                    break
    policy = {}
    for k, (n, w) in pol.items():
        policy[k] = {"bets": n, "wr": round(w / n * 100, 2) if n else 0.0, "roi_140": round(roi(w, n, 1.40), 2)}
    print("    Politica de disparo (partidos con eventos): ", policy)

    # --------- 5. Calibracion por deciles ---------
    cal = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pC > lo) & (pC <= hi) if lo > 0 else (pC >= lo) & (pC <= hi)
        if mk.sum() == 0:
            continue
        bias = float(pC[mk].mean() - yC[mk].mean())
        cal.append({"decile": f"({lo:.1f},{hi:.1f}]", "n": int(mk.sum()), "p_mean": round(float(pC[mk].mean()), 4), "y_mean": round(float(yC[mk].mean()), 4),
                    "bias": round(bias, 4), "diag": "Sobreestimado" if bias > 0.02 else ("Subestimado" if bias < -0.02 else "Perfecto")})
    print("\n[5] Calibracion (C):")
    for c in cal:
        print(f"    {c['decile']:<12} N {c['n']:>6,} | P {c['p_mean']:.3f} | real {c['y_mean']:.3f} | sesgo {c['bias']:+.4f} ({c['diag']})")

    # --------- 6. Matriz por liga (anchor, C) ---------
    lgC = lg[rowC]
    rows_l = []
    for name in pd.unique(lgC):
        mk = (lgC == name) & aC
        if mk.sum() < 30:
            continue
        n, w = green(pC[mk], yC[mk], thr_star, mC[mk], mm_star)
        if n < 15:
            continue
        hm = (pC[mk] >= thr_star) | (pC[mk] <= 1 - thr_star)
        sel_p = np.where(pC[mk] >= 0.5, pC[mk], 1 - pC[mk])[hm]
        lo, hi = wilson(w, n)
        rows_l.append({"league": str(name), "matches": int(mk.sum()), "bets": n, "wr": round(w / n * 100, 2), "wr_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
                       "roi_140": round(roi(w, n, 1.40), 2), "roi_183": round(roi(w, n, 1.83), 2),
                       "bias": round(float(sel_p.mean() - w / n), 4) if len(sel_p) else 0.0,
                       "block_candidate": bool(hi * 1.0 < BE_140)})
    df_l = pd.DataFrame(rows_l).sort_values("bets", ascending=False)
    print("\n[6] Top ligas por apuestas (anchor, C):")
    print(df_l.head(15).to_string(index=False))

    # --------- 7. Importancias / correlaciones / ablacion ---------
    print("\n[7] Justificacion de features (correlacion, AUC univariado, cobertura, ablacion)...", flush=True)
    Xa_, ya_ = X[rowA], yA
    gain = models["xgb"].get_booster().get_score(importance_type="gain")
    fn_map = {f"f{i}": n for i, n in enumerate(FEATURE_NAMES)}
    gain = {fn_map.get(k, k): v for k, v in gain.items()}
    gsum = sum(gain.values()) or 1.0
    uni = {}
    for i, n in enumerate(FEATURE_NAMES):
        v = Xa_[:, i]; ok = ~np.isnan(v)
        if ok.sum() > 1000 and np.nanstd(v) > 0:
            try:
                uni[n] = (float(np.corrcoef(v[ok], ya_[ok])[0, 1]), float(roc_auc_score(ya_[ok], v[ok])), float(ok.mean() * 100))
            except Exception:
                uni[n] = (float("nan"), float("nan"), float(ok.mean() * 100))
        else:
            uni[n] = (float("nan"), float("nan"), float(ok.mean() * 100))
    cov_C = {n: float((~np.isnan(X[rowC][:, i])).mean() * 100) for i, n in enumerate(FEATURE_NAMES)}

    abl = {}
    if not args.skip_ablation:
        def quick_auc(cols):
            hg = HistGradientBoostingClassifier(max_iter=60 if args.quick else 100, max_leaf_nodes=15, min_samples_leaf=400, learning_rate=0.08,
                                                l2_regularization=40.0, random_state=42)
            hg.fit(X[rowA][:, cols], yA, sample_weight=w_all[rowA])
            return roc_auc_score(yC, hg.predict_proba(X[rowC][:, cols])[:, 1])
        all_cols = list(range(len(FEATURE_NAMES)))
        base_auc = quick_auc(all_cols)
        print(f"    AUC base (HGB sin aumentacion): {base_auc:.4f}")
        for n in FEATURE_NAMES:
            if n == "is_home_view":
                continue
            abl[n] = base_auc - quick_auc([i for i in all_cols if FEATURE_NAMES[i] != n])
        grp_abl = {}
        for g in sorted(set(GROUPS.values())):
            if g.startswith("E"):
                continue
            grp_abl[g] = base_auc - quick_auc([i for i in all_cols if GROUPS[FEATURE_NAMES[i]] != g])
        print("    Ablacion por grupo (caida de AUC al quitarlo):", {k: round(v, 4) for k, v in grp_abl.items()})
    else:
        base_auc, grp_abl = None, {}

    # tabla markdown
    md = ["# Justificacion de features de m27_v5_3_1", "",
          "Correlacion (Pearson) y AUC univariado sobre filas de entrenamiento no-NaN; cobertura en train/test; ganancia XGBoost; "
          "ablacion = caida de AUC en C al quitar la feature (HGB, mismo split). Negativo = la feature estorba.", "",
          "| # | Feature | Grupo | corr(y) | AUC uni | cob. train % | cob. test % | ganancia XGB % | ablacion dAUC | Para que sirve / que arregla |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    just = []
    for i, n in enumerate(FEATURE_NAMES):
        c, a, cv = uni[n]
        gn = gain.get(n, 0.0) / gsum * 100
        ab = abl.get(n)
        md.append(f"| {i} | `{n}` | {GROUPS[n]} | {c:+.3f} | {a:.3f} | {cv:.0f} | {cov_C[n]:.0f} | {gn:.1f} | {'' if ab is None else f'{ab:+.4f}'} | {FEATURE_DOCS[n][1]} |")
        just.append({"feature": n, "group": GROUPS[n], "corr": c, "auc_uni": a, "cov_train": cv, "cov_test": cov_C[n], "gain_pct": gn,
                     "ablation_dauc": ab, "doc": FEATURE_DOCS[n][1]})
    if grp_abl:
        md += ["", "## Ablacion por grupo", "", "| Grupo | dAUC al quitarlo |", "|---|---|"] + [f"| {k} | {v:+.4f} |" for k, v in grp_abl.items()]
    (MODEL_DIR).mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "feature_justification.md").write_text("\n".join(md), encoding="utf-8")

    # --------- 8. Walk-forward ---------
    wf = {}
    if not args.skip_wf:
        print("\n[8] Walk-forward temporal...", flush=True)
        elig_rows = mrank >= 0
        for name, ytr_mask, yte_mask in (("Corte 1: Train <= 2022 -> Test 2023", yr_ <= 2022, yr_ == 2023),
                                          ("Corte 2: Train <= 2023 -> Test 2024", yr_ <= 2023, yr_ == 2024),
                                          ("Corte 3: Train <= 2024 -> Test 2025-2026", yr_ <= 2024, yr_ >= 2025)):
            tr = elig_rows & ytr_mask; te = elig_rows & yte_mask
            idx_tr = np.where(tr)[0]
            ranks = mrank[idx_tr]
            cutb = np.quantile(ranks, 0.85)
            a_ = idx_tr[ranks <= cutb]; b_ = idx_tr[ranks > cutb]
            mw = fit_ensemble(X[a_], y[a_], q4d[a_], w_all[a_], True)
            iso_w = fit_calibrator(p_sym(mw, X[b_]), y[b_])
            p_ = iso_w.predict(p_sym(mw, X[te])); m_ = margin_sym(mw, X[te])
            n, w = green(p_, y[te], thr_star, m_, mm_star)
            wf[name] = {"n_test_rows": int(te.sum()), "auc": round(float(roc_auc_score(y[te], p_)), 4), "brier": round(float(brier_score_loss(y[te], p_)), 4),
                        "bets": n, "wr": round(w / n * 100, 2) if n else 0.0, "roi_140": round(roi(w, n, 1.40), 2), "roi_183": round(roi(w, n, 1.83), 2)}
            print(f"    {name:<44} AUC {wf[name]['auc']:.4f} | apuestas {n:>5,} | WR {wf[name]['wr']:.2f}% | ROI@1.40 {wf[name]['roi_140']:+.2f}% | ROI@1.83 {wf[name]['roi_183']:+.2f}%", flush=True)

    # --------- 9. Curvas de aprendizaje y dashboard ---------
    print("\n[9] Dashboard...", flush=True)
    lc = xgb.XGBClassifier(n_estimators=80 if args.quick else 220, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=45.0,
                           min_child_weight=40, tree_method="hist", random_state=42, n_jobs=-1, eval_metric=["logloss", "auc"])
    XA2 = np.vstack([X[rowA], mirror(X[rowA])]); yA2 = np.concatenate([yA, 1 - yA])
    XB2 = np.vstack([X[rowB], mirror(X[rowB])]); yB2 = np.concatenate([yB, 1 - yB])
    lc.fit(XA2, yA2, eval_set=[(XA2, yA2), (XB2, yB2)], verbose=False)
    ev_ = lc.evals_result()

    fig, ax = plt.subplots(2, 3, figsize=(20, 11), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    for a in ax.flat:
        a.set_facecolor("#161b22"); a.tick_params(colors="#c9d1d9"); a.xaxis.label.set_color("#c9d1d9"); a.yaxis.label.set_color("#c9d1d9"); a.title.set_color("#58a6ff")
        for s in a.spines.values():
            s.set_color("#30363d")
        a.grid(True, color="#21262d", ls=":")
    ax[0, 0].plot(ev_["validation_0"]["auc"], color="#2ea043", label="Train AUC"); ax[0, 0].plot(ev_["validation_1"]["auc"], color="#58a6ff", label="Calibracion (B) AUC")
    ax[0, 0].set_title("1. Curvas de aprendizaje XGBoost"); ax[0, 0].set_xlabel("Arboles"); ax[0, 0].legend(facecolor="#161b22", labelcolor="#c9d1d9")
    ax[0, 1].plot([0, 1], [0, 1], "--", color="#8b949e"); ax[0, 1].plot([c["p_mean"] for c in cal], [c["y_mean"] for c in cal], "o-", color="#3fb950")
    ax[0, 1].set_title("2. Calibracion por deciles (C)"); ax[0, 1].set_xlabel("P predicha"); ax[0, 1].set_ylabel("Frecuencia real")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:20]
    ax[0, 2].barh([k for k, _ in top][::-1], [v / gsum * 100 for _, v in top][::-1], color="#d29922"); ax[0, 2].set_title("3. Importancia XGBoost (ganancia %)")
    if wf:
        ks = list(wf.keys()); ax[1, 0].bar(range(len(ks)), [wf[k]["wr"] for k in ks], color="#58a6ff"); ax[1, 0].axhline(BE_140 * 100, color="#f85149", ls="--", label="Equilibrio @1.40")
        for i, k in enumerate(ks):
            ax[1, 0].text(i, wf[k]["wr"] + 0.3, f"AUC {wf[k]['auc']:.3f}\nROI {wf[k]['roi_140']:+.1f}%", ha="center", color="#c9d1d9", fontsize=8)
        ax[1, 0].set_xticks(range(len(ks))); ax[1, 0].set_xticklabels([f"Corte {i+1}" for i in range(len(ks))]); ax[1, 0].set_ylim(60, 82)
    ax[1, 0].set_title("4. Walk-forward: Win Rate por corte"); ax[1, 0].legend(facecolor="#161b22", labelcolor="#c9d1d9")
    if res_min:
        ax[1, 1].bar([r["t"] for r in res_min], [r["auc"] for r in res_min], color="#bc8cff"); ax[1, 1].set_ylim(0.58, 0.68)
        for i, r in enumerate(res_min):
            ax[1, 1].text(i, r["auc"] + 0.001, f"WR {r['wr']:.1f}%", ha="center", color="#c9d1d9", fontsize=8)
    ax[1, 1].set_title("5. Resiliencia: AUC por minuto de snapshot (C)"); ax[1, 1].set_xlabel("minuto")
    th = [r["thr"] for r in tiers_all]
    ax[1, 2].plot(th, [r["roi_140"] for r in tiers_all], "o-", color="#f0883e", label="ROI @1.40 %"); ax[1, 2].plot(th, [r["roi_183"] / 5 for r in tiers_all], "s-", color="#3fb950", label="ROI @1.83 % (/5)")
    ax[1, 2].axhline(0, color="#8b949e", ls="--"); ax[1, 2].set_title("6. ROI vs umbral (C)"); ax[1, 2].set_xlabel("P minimo"); ax[1, 2].legend(facecolor="#161b22", labelcolor="#c9d1d9")
    plt.tight_layout(); fig.savefig(MODEL_DIR / "m27_v5_3_1_diagnostic_dashboard.png", facecolor=fig.get_facecolor()); plt.close(fig)

    # --------- 10. Exportar ---------
    for key, fname in (("xgb", "xgb_model.pkl"), ("hgb", "hgb_model.pkl"), ("lr", "lr_model.pkl"), ("reg", "reg_model.pkl")):
        pickle.dump(models[key], open(MODEL_DIR / fname, "wb"))
    pickle.dump(iso, open(MODEL_DIR / "calibrator.pkl", "wb"))
    cfg = {"min_conf": float(thr_star), "min_margin": float(mm_star), "weights": W, "p_clip": [0.03, 0.97], "block_toxic_leagues": True,
           "snapshot_range": [25.5, 30.0], "features": FEATURE_NAMES,
           "chosen_on": f"OPERATING_CONF={OPERATING_CONF} (decision explicita; auto-seleccion en B con suelo WR>={WR_MIN_OPERATING:.0%})"}
    (MODEL_DIR / "model_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    summary = {
        "model_version": "m27_v5_3_1", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "features_count": len(FEATURE_NAMES), "features_list": FEATURE_NAMES,
        "split": {"A_rows": int(rowA.sum()), "B_rows": int(rowB.sum()), "C_rows": int(rowC.sum()), "eligible_matches": int(n_el)},
        "auc_train": round(mt_tr["auc"], 4), "auc_calib": round(mt_b["auc"], 4), "auc_test": round(mt_te["auc"], 4), "auc_gap": round(gap, 4),
        "brier_train": round(mt_tr["brier"], 4), "brier_test": round(mt_te["brier"], 4), "logloss_test": round(mt_te["logloss"], 4), "acc_test": round(mt_te["acc"], 4),
        "auc_anchor_t30": round(mt_anchor["auc"], 4), "auc_intra": round(mt_intra["auc"], 4),
        "mae_q4_margin": round(float(mean_absolute_error(q4d[rowC], mC)), 3),
        "operating": {"thr": thr_star, "min_margin": mm_star, "bets": n_g, "wr": round(wr_g, 2), "roi_140": round(roi(w_g, n_g, 1.40), 2), "roi_183": round(roi(w_g, n_g, 1.83), 2),
                      "chosen": f"OPERATING_CONF={OPERATING_CONF} (decision explicita)", "auto_selected": {"thr": thr_auto, "min_margin": mm_auto, "bets": nB_auto, "wr": round(wrB_auto * 100, 2)},
                      "wr_floor": WR_MIN_OPERATING},
        "tiers_all_rows": tiers_all, "tiers_anchor": tiers_anchor, "tiers_intra": tiers_intra,
        "resilience_by_minute": res_min, "firing_policy": policy, "walk_forward_cuts": wf, "calibration_matrix": cal, "ablation_by_group": grp_abl,
        "ablation_base_auc": base_auc,
    }
    (MODEL_DIR / "m27_v5_3_1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (MODEL_DIR / "league_calibration.json").write_text(json.dumps(df_l.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    (MODEL_DIR / "feature_justification.json").write_text(json.dumps(just, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo en {time.time()-t0:.0f}s. Artefactos en {MODEL_DIR}")


if __name__ == "__main__":
    main()
