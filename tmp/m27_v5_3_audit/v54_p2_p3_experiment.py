# =====================================================================
# v5_4: P2 (DISTRIBUCION DEL MARGEN) y P3 (FOCO EN LA BANDA DE DECISION)
# ---------------------------------------------------------------------
# P2 · La idea estructural que queda. Hoy el modelo tiene DOS cabezas:
#      un clasificador (P(local gana Q4)) y un regresor de margen (Q4 local - Q4
#     Away). Se contradicen en el 4.3 % de las filas (0 de las apuestas).
#      IDEA: modelar la DISTRIBUCION del margen de Q4 (cuantil 0.1..0.9) y
#      derivar de ahi P(margen > 0): un solo objeto da el pick y una
#      confianza continua, y la confianza deja de estar cuantizada.
#
# P3 · El modelo se entrena en TODO el espacio (incluido el ~50 % de filas que
#      nunca se betting) con peso 1/partido. IDEA: reentrenar con pesos que
#      concentren la capacidad en la BANDA DE DECISION (alrededor del umbral
#      operativo), que es donde la confianza decide.
#
# COMPARA (todo ajustado en B, medido en C, anclas t=30):
#   V0  actual (clasificador + regresor de margen, isotonica->Platt)
#   V1  P3: reentrenado con foco en la banda de decision
#   V2  P2: modelo de distribucion del margen -> P(margen > 0)
#   V3  P2+P3: distribucion reentrenada con foco
#   V4  mezcla: clasificador actual + distribucion (promedio de logits)
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_p2_p3_experiment.py
# =====================================================================

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from models.common.calibration import PlattCalibrator  # noqa: E402
from models.m27_v5_3.features import mirror  # noqa: E402
from train_m27_v5_3_pipeline import (  # noqa: E402
    fit_calibrator, fit_ensemble, margin_sym, p_sym,
)

W = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
OUT = ROOT / "tmp" / "m27_v5_3_audit" / "v54_p2_p3_experiment.json"


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def tiers(p, y, thrs=(0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80)):
    out = []
    conf = np.maximum(p, 1 - p)
    pick_home = p >= 0.5
    for t in thrs:
        b = conf >= t
        n = int(b.sum())
        if n == 0:
            continue
        w = int((pick_home[b] == (y[b] == 1)).sum())
        lo, hi = wilson(w, n)
        out.append({"thr": t, "bets": n, "wr": round(w * 100 / n, 2),
                    "wr_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
                    "roi140": round((w * 1.40 - n) / n * 100, 2),
                    "roi183": round((w * 1.83 - n) / n * 100, 2),
                    "u140": round(n * (w * 1.40 - n) / n, 1),
                    "u183": round(n * (w * 1.83 - n) / n, 1)})
    return out


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmo(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_quantile_models(Xtr, ytr, qtr, wtr):
    """Modelos de cuantil del margen de Q4 (local - visitante), con espejo."""
    Xa = np.vstack([Xtr, mirror(Xtr)])
    ya = np.concatenate([ytr, 1 - ytr])
    qa = np.concatenate([qtr, -qtr])
    wa = np.concatenate([wtr, wtr])
    models = []
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                          learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=40.0,
                                          random_state=42)
        m.fit(Xa, ya, sample_weight=wa)
        models.append(m)
    return models


def prob_from_quantiles(models, X):
    """P(margen > 0) a partir de la distribucion predicha (lineal entre cuantiles)."""
    qs = np.column_stack([m.predict(X) for m in models])
    qs.sort(axis=1)
    p = np.zeros(len(X))
    for i in range(len(X)):
        row = qs[i]
        if row[0] > 0:
            p[i] = 1.0
        elif row[-1] < 0:
            p[i] = 0.0
        else:
            #cuantil mas bajo cuyo valor es > 0
            idx = int(np.searchsorted(-row, 0, side="right"))  # primer indice con row > 0
            # Busqueda lineal robusta
            idx = 0
            while idx < len(row) and row[idx] <= 0:
                idx += 1
            if idx == 0:
                p[i] = 1.0
            elif idx >= len(row):
                p[i] = 0.0
            else:
                lo, hi = row[idx - 1], row[idx]
                frac = 0.0 if hi == lo else (0 - lo) / (hi - lo)
                p[i] = (len(row) - idx + frac) / len(row)
    return np.clip(p, 0.001, 0.999)


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    anchor = np.array([m["anchor"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    w_all = 1.0 / np.bincount(inv)[inv]
    idxB = np.where(rowB & anchor)[0]
    idxC = np.where(rowC & anchor)[0]
    maskB = (rowB & anchor)[rowB]

    print("1) V0: modelo actual (referencia)")
    t0 = time.time()
    models = fit_ensemble(X[rowA], y[rowA], q4d[rowA], w_all[rowA], quick=False)
    pB_all = p_sym(models, X[rowB])
    cal = fit_calibrator(pB_all, y[rowB])
    pB = np.clip(cal.predict(pB_all), 0.03, 0.97)[maskB]
    pC = np.clip(cal.predict(p_sym(models, X[idxC])), 0.03, 0.97)
    yB, yC = y[idxB], y[idxC]
    mB = margin_sym(models, X[idxB])
    mC = margin_sym(models, X[idxC])
    print(f"   ({time.time() - t0:.0f}s)  AUC {roc_auc_score(yC, pC):.4f}")

    res = {}

    def evaluate(name, pC_use, extra=None):
        auc = roc_auc_score(yC, pC_use)
        brier = float(np.mean((pC_use - yC) ** 2))
        ll = float(-np.mean(yC * np.log(pC_use) + (1 - yC) * np.log(1 - pC_use)))
        print(f"\n[{name}]  AUC {auc:.4f} | Brier {brier:.4f} | LogLoss {ll:.4f} | "
              f"valores distintos {len(np.unique(np.round(pC_use, 4))):,}")
        for t in tiers(pC_use, yC):
            print(f"    thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
                  f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi140']:>+6.2f}% | u@1.40 {t['u140']:>6.0f} "
                  f"| u@1.83 {t['u183']:>6.0f}")
        res[name] = {"auc": round(auc, 4), "brier": round(brier, 4), "logloss": round(ll, 4),
                     "tiers": tiers(pC_use, yC), **(extra or {})}
        return auc

    evaluate("V0 actual", pC)

    # ------------------------------------------------------------------
    block("2) V1 (P3): REENTRENO CON FOCO EN LA BANDA DE DECISION")
    # pesos: 1/partido * foco(|p_raw - 0.75|) -> se concentra donde decide
    p_raw_A = p_sym(models, X[rowA])
    foco = np.exp(-((np.abs(p_raw_A - 0.75) / 0.12) ** 2))
    w_focus = w_all[rowA] * foco
    print(f"   peso medio del foco: {foco.mean():.3f} (1.0 = sin foco); "
          f"fracción de filas con peso > 0.5: {(foco > 0.5).mean() * 100:.1f} %")
    t0 = time.time()
    m_focus = fit_ensemble(X[rowA], y[rowA], q4d[rowA], w_focus, quick=False)
    pB_f = p_sym(m_focus, X[rowB])
    cal_f = fit_calibrator(pB_f, y[rowB])
    pB_fc = np.clip(cal_f.predict(pB_f), 0.03, 0.97)[maskB]
    pC_f = np.clip(cal_f.predict(p_sym(m_focus, X[idxC])), 0.03, 0.97)
    print(f"   ({time.time() - t0:.0f}s)")
    evaluate("V1 foco en la banda", pC_f)

    # ------------------------------------------------------------------
    block("3) V2 (P2): DISTRIBUCION DEL MARGEN -> P(margen > 0)")
    t0 = time.time()
    qm = fit_quantile_models(X[rowA], y[rowA], q4d[rowA], w_all[rowA])
    pB_q = prob_from_quantiles(qm, X[idxB])
    pC_q = prob_from_quantiles(qm, X[idxC])
    cal_q = PlattCalibrator().fit(pB_q, yB)
    pB_qc = np.clip(cal_q.predict(pB_q), 0.03, 0.97)
    pC_qc = np.clip(cal_q.predict(pC_q), 0.03, 0.97)
    print(f"   ({time.time() - t0:.0f}s)")
    evaluate("V2 distribucion del margen", pC_qc)

    # ------------------------------------------------------------------
    block("4) V3 (P2+P3): DISTRIBUCION REENTRENADA CON FOCO")
    t0 = time.time()
    qm_f = fit_quantile_models(X[rowA], y[rowA], q4d[rowA], w_focus)
    pB_qf = prob_from_quantiles(qm_f, X[idxB])
    pC_qf = prob_from_quantiles(qm_f, X[idxC])
    cal_qf = PlattCalibrator().fit(pB_qf, yB)
    pC_qfc = np.clip(cal_qf.predict(pC_qf), 0.03, 0.97)
    print(f"   ({time.time() - t0:.0f}s)")
    evaluate("V3 distribucion + foco", pC_qfc)

    # ------------------------------------------------------------------
    block("5) V4: MEZCLA del clasificador actual y la distribucion del margen")
    best_w = None
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        mix = sigmo((1 - w) * logit(pC) + w * logit(pC_qc))
        mix = np.clip(mix, 0.03, 0.97)
        auc = roc_auc_score(yC, mix)
        if best_w is None or auc > best_w[1]:
            best_w = (w, auc, mix)
    print(f"   mejor peso de la distribucion: {best_w[0]:.1f} (AUC {best_w[1]:.4f})")
    evaluate("V4 mezcla clasificador+distribucion", best_w[2])

    # ------------------------------------------------------------------
    block("6) ¿CUANTAS APUESTAS DIFIEREN ENTRE EL ACTUAL Y LA DISTRIBUCION?")
    conf_c = np.maximum(pC, 1 - pC)
    conf_q = np.maximum(pC_qc, 1 - pC_qc)
    for t in (0.72, 0.75, 0.80):
        sel_c = conf_c >= t
        sel_q = conf_q >= t
        print(f"   thr {t}: actual {int(sel_c.sum()):,} | distribucion {int(sel_q.sum()):,} | "
              f"coinciden {int((sel_c == sel_q).sum()):,} | solo actual {int((sel_c & ~sel_q).sum()):,} | "
              f"solo distribucion {int((~sel_c & sel_q).sum()):,}")

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT}")


if __name__ == "__main__":
    main()
