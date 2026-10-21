# =====================================================================
# ABLACION "SIN MERCADO" (estudio preparatorio de v5_4)
# ---------------------------------------------------------------------
# Premisa: las cuotas pueden estar manipuladas ("hacer trampa") en mercados
# con poca accion, y ademas no existe el mercado de Q4 (el que seReally
# apuesta) para contrastar. Si el modelo funciona igual SIN las features de
# mercado, la dependencia de un señal manipulable es un riesgo gratis.
#
# Se entrena el mismo ensemble con las features de mercado puestas a NaN
# (como cuando el partido no tiene cuotas) y se compara con el modelo
# desplegado: AUC, WR por umbral y comportamiento por poblacion.
# =====================================================================

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from models.m27_v5_3.features import IDX  # noqa: E402
from train_m27_v5_3_pipeline import (  # noqa: E402
    fit_calibrator, fit_ensemble, margin_sym, p_sym,
)

MKT = ["has_ml", "odds_log_ratio", "mkt_p_home", "expected_ft_margin",
       "mkt_margin_spread", "total_line", "cur_vs_expected"]


def wilson(w, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


import math  # noqa: E402


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    w_all = 1.0 / np.bincount(inv)[inv]

    has_ml = X[:, IDX["has_ml"]] > 0.5
    idxC = np.where(rowC)[0]
    idxB = np.where(rowB)[0]
    maskB = has_ml[idxB]          # el calibrador se ajusta con la poblacion CON mercado
    maskC = has_ml[idxC]

    # ---- variante SIN mercado: las features de mercado a NaN ----
    X_sin = X.copy()
    for f in MKT:
        X_sin[:, IDX[f]] = np.nan

    res = {}
    for nombre, Xt in (("CON mercado (desplegado)", X), ("SIN mercado (NaN)", X_sin)):
        t0 = time.time()
        models = fit_ensemble(Xt[rowA], y[rowA], q4d[rowA], w_all[rowA], quick=False)
        pB_all = p_sym(models, Xt[idxB])
        cal = fit_calibrator(pB_all, y[idxB])
        pB = np.clip(cal.predict(pB_all), 0.03, 0.97)
        pC = np.clip(cal.predict(p_sym(models, Xt[idxC])), 0.03, 0.97)
        yB, yC = y[idxB], y[idxC]
        print(f"\n=== {nombre}  ({time.time() - t0:.0f}s) ===")
        for etiqueta, mask in (("C completo", np.ones(len(yC), bool)),
                               ("C con mercado", maskC), ("C sin mercado", ~maskC)):
            m = mask
            auc = roc_auc_score(yC[m], pC[m])
            wr = (np.where(pC[m] >= 0.5, yC[m] == 1, yC[m] == 0)).mean()
            conf = np.maximum(pC[m], 1 - pC[m])
            brier = float(np.mean((pC[m] - yC[m]) ** 2))
            print(f"   {etiqueta:<16} n={int(m.sum()):>7,} | AUC {auc:.4f} | "
                  f"acierto {wr * 100:.1f}% | Brier {brier:.4f} | conf media {conf.mean():.3f}")
        print(f"   {'umbral':>8}{'apuestas':>10}{'WR':>9}{'IC95':>16}{'u@1.40':>10}{'u@1.83':>10}")
        for t in (0.65, 0.70, 0.72, 0.75, 0.78, 0.80):
            conf_all = np.maximum(pC, 1 - pC)
            b = conf_all >= t
            n = int(b.sum())
            if n == 0:
                continue
            w = int((np.where(pC[b] >= 0.5, yC[b] == 1, yC[b] == 0)).sum())
            lo, hi = wilson(w, n)
            print(f"   {t:>8.2f}{n:>10,}{w / n * 100:>8.1f}%   [{lo * 100:>4.1f}, {hi * 100:>4.1f}]"
                  f"{w * 1.40 - n:>10.0f}{w * 1.83 - n:>10.0f}")
        res[nombre] = {"auc": float(roc_auc_score(yC, pC))}

    print()
    print("=== DIFERENCIA (con mercado - sin mercado) ===")
    a1 = res["CON mercado (desplegado)"]["auc"]
    a2 = res["SIN mercado (NaN)"]["auc"]
    print(f"   AUC: {a1:.4f} vs {a2:.4f} -> {a1 - a2:+.4f}")


if __name__ == "__main__":
    main()
