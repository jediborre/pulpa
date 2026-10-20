# =====================================================================
# v5_4: ELECCION DE LA CALIBRACION (isotonica vs platt) — version final
# ---------------------------------------------------------------------
# HALLAZGO:
#   La isotonica es ESCALONADA: 83 valores de confianza y una MESETA
#   (thr 0.75, 0.76, 0.78 y 0.80 eligen las MISMAS 2.171 apuestas). Esa
#   meseta es la razon de que subir el umbral de 0.75 a 0.80 no cambie nada
#   y de que el umbral no se pueda afinar.
#   Una calibracion PLATT (sigmoid sobre el logit) devuelve una confianza
#   continua: mismo AUC, LogLoss MEJOR y +16 % de beneficio esperado al
#   mismo umbral.
#
# COMPARA (calibrado en B, medido en C, anclas t=30):
#   A  isotonica simetrica  (produccion)
#   B  platt simetrica      (a*logit(p), sin intercepto: mantiene la simetria)
#   C  platt asimetrica    (a*logit(p)+b: ademas ajusta la tasa base)
#   D  platt simetrica por poblacion (mercado presente/ausente)
#   E  platt asimetrica por poblacion
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_pick_calibration.py
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
from train_m27_v5_3_pipeline import fit_ensemble, p_sym  # noqa: E402

HAS_ML = IDX["has_ml"]
OUT = ROOT / "tmp" / "m27_v5_3_audit" / "v54_pick_calibration.json"


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmo(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_platt_sym(p, y):
    """Platt SIMETRICO: se entrena con (z, y) y (-z, 1-y) -> intercepto 0."""
    z = logit(p)
    Z = np.concatenate([z, -z]).reshape(-1, 1)
    Y = np.concatenate([y, 1 - y])
    m = LogisticRegression(C=1e6, max_iter=3000, fit_intercept=True)
    m.fit(Z, Y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def fit_platt_asym(p, y):
    z = logit(p).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=3000)
    m.fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


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
    idxB = np.where(rowB & anchor)[0]
    idxC = np.where(rowC & anchor)[0]
    maskB = (rowB & anchor)[rowB]
    pB = pB_all[maskB]
    pC_raw = p_sym(models, X[idxC])
    yB, yC = y[idxB], y[idxC]
    hasB = X[idxB, HAS_ML] > 0.5
    hasC = X[idxC, HAS_ML] > 0.5

    variants = {}

    # A: isotonica simetrica (produccion)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.03, y_max=0.95)
    iso.fit(np.concatenate([pB_all, 1 - pB_all]), np.concatenate([y[rowB], 1 - y[rowB]]))
    variants["A isotonica (actual)"] = (
        np.clip(iso.predict(pB), 0.03, 0.97), np.clip(iso.predict(pC_raw), 0.03, 0.97))

    # B: platt simetrica
    a, b = fit_platt_sym(pB, yB)
    print(f"  platt simetrica: a={a:.4f} b={b:.4f}")
    variants["B platt simetrica"] = (
        np.clip(sigmo(a * logit(pB) + b), 0.03, 0.97),
        np.clip(sigmo(a * logit(pC_raw) + b), 0.03, 0.97))

    # C: platt asimetrica
    a2, b2 = fit_platt_asym(pB, yB)
    print(f"  platt asimetrica: a={a2:.4f} b={b2:.4f}")
    variants["C platt asimetrica"] = (
        np.clip(sigmo(a2 * logit(pB) + b2), 0.03, 0.97),
        np.clip(sigmo(a2 * logit(pC_raw) + b2), 0.03, 0.97))

    # D/E: platt por poblacion
    for tag, sym in (("D platt sim por poblacion", True), ("E platt asim por poblacion", False)):
        pBv = np.empty(len(yB))
        pCv = np.empty(len(yC))
        for gname, selB, selC in (("CON", hasB, hasC), ("SIN", ~hasB, ~hasC)):
            if sym:
                aa, bb = fit_platt_sym(pB[selB], yB[selB])
            else:
                aa, bb = fit_platt_asym(pB[selB], yB[selB])
            pBv[selB] = np.clip(sigmo(aa * logit(pB[selB]) + bb), 0.03, 0.97)
            pCv[selC] = np.clip(sigmo(aa * logit(pC_raw[selC]) + bb), 0.03, 0.97)
        variants[tag] = (pBv, pCv)

    res = {}
    for name, (pB_u, pC_u) in variants.items():
        block(name)
        nv = len(np.unique(np.round(pC_u, 4)))
        auc = roc_auc_score(yC, pC_u)
        brier = float(np.mean((pC_u - yC) ** 2))
        ll = float(-np.mean(yC * np.log(pC_u) + (1 - yC) * np.log(1 - pC_u)))
        print(f"  valores distintos de confianza: {nv:,} | AUC {auc:.4f} | Brier {brier:.4f} | LogLoss {ll:.4f}")
        tl = tiers(pC_u, yC)
        for t in tl:
            if not t["bets"]:
                continue
            print(f"    thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
                  f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi140']:>+6.2f}% | ROI183 {t['roi183']:>+6.2f}% "
                  f"| u@1.40 {t['u140']:>6.0f} | u@1.83 {t['u183']:>6.0f}")
        res[name] = {"n_valores": nv, "auc": round(auc, 4), "brier": round(brier, 4),
                     "logloss": round(ll, 4), "tiers": tl}

    block("FRONTERA: mejor umbral por variante (beneficio esperado)")
    print(f"  {'variante':<30}{'@1.40: thr':>12}{'ap':>7}{'WR':>7}{'u':>6}   "
          f"{'@1.83: thr':>12}{'ap':>7}{'WR':>7}{'u':>6}")
    for name, r in res.items():
        b140 = max((t for t in r["tiers"] if t["bets"]), key=lambda t: t["u140"])
        b183 = max((t for t in r["tiers"] if t["bets"]), key=lambda t: t["u183"])
        print(f"  {name:<30}{b140['thr']:>12.2f}{b140['bets']:>7,}{b140['wr']:>6.1f}%{b140['u140']:>6.0f}   "
              f"{b183['thr']:>12.2f}{b183['bets']:>7,}{b183['wr']:>6.1f}%{b183['u183']:>6.0f}")
    print("\n  (u = unidades esperadas de beneficio por unidad apostada; el bloque C son 35.480 partidos)")

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT}")


if __name__ == "__main__":
    main()
