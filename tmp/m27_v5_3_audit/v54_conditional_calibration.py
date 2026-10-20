# =====================================================================
# v5_4 P0: RECALIBRACION CONDICIONAL (post-hoc sobre p_cal)
# ---------------------------------------------------------------------
# HIPOTESIS (medida en el analisis previo):
#   El modelo esta 7 pp sobreconfiado en la poblacion CON mercado
#   pre-partido (dice 81.1 %, acierta 74.1 %) y solo 2 pp en la SIN mercado
#   (dice 85.3 %, acierta 83.3 %). Ademas el sesgo de calibracion varia
#   por categoria de fase (-0.029 a +0.033).
#
# QUE COMPARA (todo ajustado en el bloque B y medido en el bloque C):
#   A  isotonica global (produccion actual)
#   B  Platt por poblacion (2 parametros por grupo, logit(p_cal))
#   C  isotonica por poblacion (con minimo de muestra y mezcla al global)
#   D  Platt por poblacion + isotonica por fase
#   E  politica: solo apuesta en la poblacion SIN mercado
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_conditional_calibration.py
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

from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import IDX, mirror  # noqa: E402
from train_m27_v5_3_pipeline import fit_calibrator, fit_ensemble, p_sym  # noqa: E402

MD = ROOT / "models" / "m27_v5_3" / "model_outputs"
CFG = json.loads((MD / "model_config.json").read_text(encoding="utf-8"))
WEIGHTS = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}
OUT = ROOT / "tmp" / "m27_v5_3_audit" / "v54_conditional_calibration.json"
HAS_ML = IDX["has_ml"]


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
    return 1.0 / (1.0 + np.exp(-z))


def tiers(p, y, thrs=(0.70, 0.72, 0.75, 0.80)):
    out = []
    conf = np.maximum(p, 1 - p)
    pick_home = p >= 0.5
    for t in thrs:
        b = conf >= t
        n = int(b.sum())
        if n == 0:
            out.append({"thr": t, "bets": 0, "wr": 0.0, "roi140": 0.0, "roi183": 0.0, "wr_ci95": [0, 0]})
            continue
        w = int((pick_home[b] == (y[b] == 1)).sum())
        lo, hi = wilson(w, n)
        out.append({"thr": t, "bets": n, "wr": round(w * 100 / n, 2), "wr_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
                    "roi140": round((w * 1.40 - n) / n * 100, 2), "roi183": round((w * 1.83 - n) / n * 100, 2)})
    return out


def fit_platt(p, y):
    z = logit(p).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(z, y)
    a = float(m.coef_[0][0])
    b = float(m.intercept_[0])
    return a, b


def apply_platt(p, a, b):
    return np.clip(sigmo(a * logit(p) + b), 0.03, 0.97)


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, meta = D["X"], D["y"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    anchor = np.array([m["anchor"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    w_all = 1.0 / np.bincount(inv)[inv]

    print("Entrenando el ensamble (A) y el calibrador global (B), como en produccion...")
    models = fit_ensemble(X[rowA], y[rowA], D["q4_diff"][rowA], w_all[rowA], quick=False)
    pB = p_sym(models, X[rowB])
    iso_global = fit_calibrator(pB, y[rowB])

    idxB = np.where(rowB & anchor)[0]
    idxC = np.where(rowC & anchor)[0]
    maskB_anchor = (rowB & anchor)[rowB]     # booleano sobre las filas de B
    pB_cal_all = np.clip(iso_global.predict(pB), 0.03, 0.97)
    pB_cal = pB_cal_all[maskB_anchor]        # solo las anclas de B (mismo universe que C)
    pC_cal = np.clip(iso_global.predict(p_sym(models, X[idxC])), 0.03, 0.97)
    yB, yC = y[idxB], y[idxC]
    hasB = X[idxB, HAS_ML] > 0.5
    hasC = X[idxC, HAS_ML] > 0.5
    catB = np.array([classify_stage(x)[0] for x in lg[idxB]])
    catC = np.array([classify_stage(x)[0] for x in lg[idxC]])

    res = {}

    def evaluate(name: str, pC_use: np.ndarray, mask_extra: np.ndarray | None = None) -> None:
        m = np.ones(len(yC), dtype=bool) if mask_extra is None else mask_extra
        pc = np.clip(pC_use, 0.03, 0.97)
        auc = roc_auc_score(yC[m], pc[m])
        brier = float(np.mean((pc[m] - yC[m]) ** 2))
        ll = float(-np.mean(yC[m] * np.log(pc[m]) + (1 - yC[m]) * np.log(1 - pc[m])))
        tl = tiers(pc, yC)
        print(f"\n[{name}]  AUC {auc:.4f} | Brier {brier:.4f} | LogLoss {ll:.4f}")
        for t in tl:
            if t["bets"] == 0:
                continue
            print(f"    thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
                  f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi140']:>+6.2f}% | ROI183 {t['roi183']:>+6.2f}%")
        res[name] = {"auc": round(auc, 4), "brier": round(brier, 4), "logloss": round(ll, 4),
                     "tiers": tl}

    block("A) ISOTONICA GLOBAL (produccion actual)")
    evaluate("A global (actual)", pC_cal)

    block("B) PLATT POR POBLACION (ajustado en B)")
    platt = {}
    for gname, sel in (("CON_MERCADO", hasB), ("SIN_MERCADO", ~hasB)):
        if sel.sum() < 100:
            platt[gname] = (1.0, 0.0)
            continue
        platt[gname] = fit_platt(pB_cal[sel], yB[sel])
        print(f"  {gname}: n={int(sel.sum()):,} | a={platt[gname][0]:.3f} b={platt[gname][1]:.3f}")
    pB_platt = np.empty(len(yB))
    for gname, sel in (("CON_MERCADO", hasB), ("SIN_MERCADO", ~hasB)):
        a, b = platt[gname]
        pB_platt[sel] = apply_platt(pB_cal[sel], a, b)
    pC_platt = np.empty(len(yC))
    for gname, sel in (("CON_MERCADO", hasC), ("SIN_MERCADO", ~hasC)):
        a, b = platt[gname]
        pC_platt[sel] = apply_platt(pC_cal[sel], a, b)
    evaluate("B platt por poblacion", pC_platt)

    block("C) ISOTONICA POR POBLACION (min 500 muestras, mezcla al global si no)")
    MIN_N = 500
    isos = {}
    for gname, sel in (("CON_MERCADO", hasB), ("SIN_MERCADO", ~hasB)):
        if sel.sum() >= MIN_N:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.03, y_max=0.95)
            iso.fit(pB_cal[sel], yB[sel])
            isos[gname] = iso
            print(f"  {gname}: isotonica propia (n={int(sel.sum()):,})")
        else:
            isos[gname] = None
            print(f"  {gname}: n={int(sel.sum()):,} < {MIN_N} -> se usa la global")
    pC_iso = np.empty(len(yC))
    for gname, sel in (("CON_MERCADO", hasC), ("SIN_MERCADO", ~hasC)):
        if isos[gname] is None:
            pC_iso[sel] = pC_cal[sel]
        else:
            pC_iso[sel] = np.clip(isos[gname].predict(pC_cal[sel]), 0.03, 0.97)
    evaluate("C isotonica por poblacion", pC_iso)

    block("D) PLATT POR POBLACION + ISOTONICA POR FASE")
    isos_fase = {}
    for cat in sorted(set(catB.tolist())):
        sel = catB == cat
        if sel.sum() >= MIN_N:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.03, y_max=0.95)
            iso.fit(pB_platt[sel], yB[sel])
            isos_fase[cat] = iso
    print(f"  categorias con isotonica propia: {len(isos_fase)}")
    pC_d = pC_platt.copy()
    for cat, iso in isos_fase.items():
        sel = catC == cat
        pC_d[sel] = np.clip(iso.predict(pC_platt[sel]), 0.03, 0.97)
    evaluate("D platt poblacion + isotonica fase", pC_d)

    block("E) POLITICA: SOLO APUESTA EN LA POBLACION SIN MERCADO")
    for thr in (0.72, 0.75):
        conf = np.maximum(pC_cal, 1 - pC_cal)
        m = (conf >= thr) & (~hasC)
        pc = np.clip(pC_cal, 0.03, 0.97)
        auc = roc_auc_score(yC[m], pc[m])
        print(f"  thr {thr} solo SIN mercado: n={int(m.sum()):,} | WR "
              f"{int((pc[m] >= 0.5)[0] * 0 + 0) if False else ''}"
              f"{100 * float(((pc[m] >= 0.5) == (yC[m] == 1)).mean()):.2f}% | AUC {auc:.4f}")
        res[f"E sin mercado thr {thr}"] = {"n": int(m.sum()),
                                           "wr": round(100 * float(((pc[m] >= 0.5) == (yC[m] == 1)).mean()), 2),
                                           "auc": round(auc, 4)}

    block("F) ¿LA RECALIBRACION CONDICIONAL MEJORA LA POBLACION CON MERCADO?")
    for name, pC_use in (("A global", pC_cal), ("B platt", pC_platt), ("C isotonica", pC_iso),
                         ("D combinado", pC_d)):
        sel = hasC
        conf = np.maximum(pC_use, 1 - pC_use)
        b = sel & (conf >= 0.75)
        if b.sum() < 20:
            continue
        w = int(((pC_use[b] >= 0.5) == (yC[b] == 1)).sum())
        print(f"  {name:<14} con mercado @0.75: n={int(b.sum()):>4,} | WR {w * 100 / b.sum():>5.1f} % | "
              f"dice {conf[b].mean() * 100:.1f} % | ROI140 {(w * 1.40 - b.sum()) / b.sum() * 100:>+6.1f} %")

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT}")


if __name__ == "__main__":
    main()
