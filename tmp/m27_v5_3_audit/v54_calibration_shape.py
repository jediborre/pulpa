# =====================================================================
# v5_4: ISOTONICA (escalones) vs PLATT (continua) — ¿CUANTIZAR LA CONFIANZA
#         es lo que impide bajar el umbral sin perder acierto?
# ---------------------------------------------------------------------
# INDICIO: las confianzas live de m27_v5_3 toman solo ~9 valores distintos
#   (0.727, 0.749, 0.751, 0.779, 0.799, 0.801, 0.826, 0.882, 0.915) y el
#   bloque C tiene una MESETA: thr 0.75 -> 2.172 apuestas y thr 0.80 -> 2.171
#   (el mismo conjunto). La isotónica es una funcion escalonada: dentro de
#   un escalon la confianza no distingue nada, asi que el umbral no puede
#   afinar. Una calibracion monotona CONTINUA (Platt sobre el logit de
#   p_raw) devuelve una confianza con resolucion en toda la banda.
#
# COMPARA (calibrado en B, medido en C, anclas t=30):
#   A  isotonica global  (produccion)
#   B  platt global      (continua)
#   C  isotonica + platt (hibrido: isotonica y luego platt sobre el logit)
#   D  platt por poblacion (market presente/ausente)
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_calibration_shape.py
# =====================================================================

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from models.m27_v5_3.features import IDX  # noqa: E402
from train_m27_v5_3_pipeline import fit_calibrator, fit_ensemble, p_sym  # noqa: E402

HAS_ML = IDX["has_ml"]
OUT = ROOT / "tmp" / "m27_v5_3_audit" / "v54_calibration_shape.json"


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


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmo(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_platt(p, y):
    m = LogisticRegression(C=1e6, max_iter=2000)
    m.fit(logit(p).reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def apply_platt(p, a, b):
    return np.clip(sigmo(a * logit(p) + b), 0.03, 0.97)


def tiers(p, y, thrs=(0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.85)):
    out = []
    conf = np.maximum(p, 1 - p)
    pick_home = p >= 0.5
    for t in thrs:
        b = conf >= t
        n = int(b.sum())
        if n == 0:
            out.append({"thr": t, "bets": 0})
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


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, meta = D["X"], D["y"], D["meta"]
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

    print("Entrenando el ensamble (A) como en produccion...")
    models = fit_ensemble(X[rowA], y[rowA], D["q4_diff"][rowA], w_all[rowA], quick=False)
    pB_all = p_sym(models, X[rowB])
    iso = fit_calibrator(pB_all, y[rowB])
    idxB = np.where(rowB & anchor)[0]
    idxC = np.where(rowC & anchor)[0]
    maskB = (rowB & anchor)[rowB]
    pB = pB_all[maskB]
    pC_raw = p_sym(models, X[idxC])
    yB, yC = y[idxB], y[idxC]

    variants = {}

    def add(name, p_cal_B, p_cal_C, meta_extra=None):
        variants[name] = (p_cal_B, p_cal_C, meta_extra or {})

    # A: isotonica global (produccion)
    add("A isotonica (actual)", np.clip(iso.predict(pB), 0.03, 0.97),
        np.clip(iso.predict(pC_raw), 0.03, 0.97))

    # B: platt global sobre p_raw
    a, b = fit_platt(pB_all, y[rowB])
    add("B platt global", apply_platt(pB, a, b), apply_platt(pC_raw, a, b), {"a": a, "b": b})

    # C: hibrido isotonica -> platt
    pB_iso = np.clip(iso.predict(pB), 0.03, 0.97)
    pC_iso = np.clip(iso.predict(pC_raw), 0.03, 0.97)
    a2, b2 = fit_platt(pB_iso, yB)
    add("C hibrido iso+platt", apply_platt(pB_iso, a2, b2), apply_platt(pC_iso, a2, b2), {"a": a2, "b": b2})

    # D: platt por poblacion
    hasB = X[idxB, HAS_ML] > 0.5
    hasC = X[idxC, HAS_ML] > 0.5
    pb = np.empty(len(yB))
    pc = np.empty(len(yC))
    pars = {}
    for gname, selB, selC in (("CON_MERCADO", hasB, hasC), ("SIN_MERCADO", ~hasB, ~hasC)):
        aa, bb = fit_platt(pB[selB], yB[selB])
        pars[gname] = (aa, bb)
        pb[selB] = apply_platt(pB[selB], aa, bb)
        pc[selC] = apply_platt(pC_raw[selC], aa, bb)
    add("D platt por poblacion", pb, pc, pars)

    res = {}
    for name, (pB_u, pC_u, extra) in variants.items():
        block(name)
        print(f"  valores distintos de confianza en C: {len(np.unique(np.round(pC_u, 4))):,}")
        auc = roc_auc_score(yC, pC_u)
        brier = float(np.mean((pC_u - yC) ** 2))
        ll = float(-np.mean(yC * np.log(pC_u) + (1 - yC) * np.log(1 - pC_u)))
        print(f"  AUC {auc:.4f} | Brier {brier:.4f} | LogLoss {ll:.4f}")
        tl = tiers(pC_u, yC)
        for t in tl:
            if not t["bets"]:
                continue
            print(f"    thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
                  f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi140']:>+6.2f}% | ROI183 {t['roi183']:>+6.2f}% "
                  f"| u@1.40 {t['u140']:>6.0f} | u@1.83 {t['u183']:>6.0f}")
        res[name] = {"auc": round(auc, 4), "brier": round(brier, 4), "logloss": round(ll, 4),
                     "n_valores": int(len(np.unique(np.round(pC_u, 4)))), "tiers": tl}

    block("MEJOR UMBRAL POR VARIANTE (beneficio esperado)")
    print(f"  {'variante':<26}{'thr':>6}{'apuestas':>10}{'WR':>8}{'u@1.40':>9}{'u@1.83':>9}")
    for name, r in res.items():
        for t in r["tiers"]:
            if not t["bets"]:
                continue
            if abs(t["thr"] - 0.75) > 1e-9 and abs(t["u140"] - max(x["u140"] for x in r["tiers"] if x["bets"])) > 1e-9:
                continue
            print(f"  {name:<26}{t['thr']:>6.2f}{t['bets']:>10,}{t['wr']:>7.2f}%{t['u140']:>9.0f}{t['u183']:>9.0f}")
            break
    print()
    for name, r in res.items():
        best140 = max((t for t in r["tiers"] if t["bets"]), key=lambda t: t["u140"])
        best183 = max((t for t in r["tiers"] if t["bets"]), key=lambda t: t["u183"])
        print(f"  {name:<26} mejor@1.40: thr {best140['thr']:.2f} ({best140['bets']:,} ap, "
              f"WR {best140['wr']:.1f} %, {best140['u140']:.0f} u) | "
              f"mejor@1.83: thr {best183['thr']:.2f} ({best183['bets']:,} ap, "
              f"WR {best183['wr']:.1f} %, {best183['u183']:.0f} u)")

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT}")


if __name__ == "__main__":
    main()
