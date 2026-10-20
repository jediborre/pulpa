# =====================================================================
# ABLACION: ¿se cumplio la premisa de EXCLUIR del entrenamiento las fases
#            de torneo con comportamiento propio (eliminatorias, copas,
#            championship round, promotion, formativas)?
# ---------------------------------------------------------------------
# PREMISA ORIGINAL (diseño de m27_v5_3):
#   Esos.matches se DESCARTARIAN del entrenamiento para quitar ruido, y para
#   ellos se planeaba un modelo independiente. En la implementacion real solo
#   se aplico `not is_toxic_league(liga)` (build_dataset.py:130), es decir una
#   lista de 16 patrones escritos a mano, 5 de los cuales no casan con NINGUN
#   nombre real de SofaScore. Las fases especiales nunca se excluyeron.
#
# QUE HACE:
#   Reentrena el ensamble desde el dataset ya construido (sin reextraer
#   features) con tres mascara de entrenamiento:
#     V0: real            (lo que hay hoy)  -> valida la reproduccion
#     V1: sin ELIMINATORIA ni COPA          (formatos especiales)
#     V2: sin las 5 categorias especiales    (todo lo que iria a modelo propio)
#   y evalua SIEMPRE sobre el mismo bloque C, en dos vistas:
#     - C completo  (todo el holdout)
#     - C filtrado  (solo las categorias que el variante conserva)
#   Mide AUC, Brier, LogLoss y la tabla de umbrales (apuestas, WR, ROI).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/phase_exclusion_ablation.py
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

from models.m27_v5_3.features import mirror  # noqa: E402
from train_m27_v5_3_pipeline import (  # noqa: E402
    fit_calibrator, fit_ensemble, green, margin_sym, metrics, p_sym, roi, tier_table,
)

DATASET = ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl"
OUT_JSON = ROOT / "tmp" / "m27_v5_3_audit" / "phase_exclusion_ablation.json"

sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3_audit"))
from phase_backtest import classify_phase  # noqa: E402

ESPECIALES = ("ELIMINATORIA", "COPA")
TODAS_ESPECIALES = ("ELIMINATORIA", "COPA", "SEGUNDA FASE", "PROMOCION", "FORMATIVAS")


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def run_variant(name: str, X, y, q4d, rowA, rowB, rowC, cats, drop_cats, w_all):
    t0 = time.time()
    keepA = rowA & ~np.isin(cats, list(drop_cats))
    keepB = rowB & ~np.isin(cats, list(drop_cats))
    n_drop = int((rowA & np.isin(cats, list(drop_cats))).sum())

    models = fit_ensemble(X[keepA], y[keepA], q4d[keepA], w_all[keepA], quick=False)
    pB = p_sym(models, X[keepB])
    iso = fit_calibrator(pB, y[keepB])
    pC = iso.predict(p_sym(models, X[rowC]))
    mC = margin_sym(models, X[rowC])

    keptC = rowC & ~np.isin(cats, list(drop_cats))
    out = {"variant": name, "drop_cats": list(drop_cats),
           "train_rows": int(keepA.sum()), "dropped_train_rows": n_drop}

    for label, mask in (("C_completo", rowC), ("C_filtrado", keptC)):
        if mask.sum() == 0:
            continue
        mt = metrics(pC[mask], y[mask])
        tiers = tier_table(pC[mask], y[mask], mC[mask], thrs=(0.72, 0.75, 0.80))
        out[label] = {"auc": round(mt["auc"], 4), "brier": round(mt["brier"], 4),
                      "logloss": round(mt["logloss"], 4), "acc": round(mt["acc"], 4),
                      "n": int(mask.sum()), "tiers": tiers}
    print(f"  [{name}] entrenado en {out['train_rows']:,} filas "
          f"(descartadas {n_drop:,}) en {time.time() - t0:.0f}s")
    return out, models, iso


def main() -> None:
    print("Cargando dataset...")
    D = pickle.load(open(DATASET, "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    ord_ = np.array([m["ord"] for m in meta])
    mrank = np.array([rank.get(int(o), -1) for o in ord_])
    anchor = np.array([m["anchor"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    cats = np.array([classify_phase(x)[2] for x in lg])

    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    cnt = np.bincount(inv)[inv]
    w_all = 1.0 / cnt
    print(f"  filas {len(y):,} | A {int(rowA.sum()):,} B {int(rowB.sum()):,} C {int(rowC.sum()):,}")

    block("COMPOSICION DEL ENTRENAMIENTO ACTUAL (V0): categorias de fase")
    for cat in sorted(set(cats.tolist())):
        n_a = int((rowA & (cats == cat)).sum())
        n_c = int((rowC & (cats == cat)).sum())
        n_sp = int((rowA & (cats == cat) & anchor).sum())
        print(f"  {cat:<14} filas train {n_a:>8,} ({n_a * 100 / rowA.sum():>5.1f}%) | "
              f"partidos ancla train {n_sp:>7,} | filas test C {n_c:>8,}")

    results = []
    for name, drop in (("V0 real (sin exclusion adicional)", ()),
                       ("V1 sin ELIMINATORIA ni COPA", ESPECIALES),
                       ("V2 sin las 5 categorias especiales", TODAS_ESPECIALES)):
        out, _, _ = run_variant(name, X, y, q4d, rowA, rowB, rowC, cats, drop, w_all)
        results.append(out)
        for label in ("C_completo", "C_filtrado"):
            if label not in out:
                continue
            d = out[label]
            print(f"    {label:<12} AUC {d['auc']:.4f} | Brier {d['brier']:.4f} | "
                  f"LogLoss {d['logloss']:.4f} | Acc {d['acc'] * 100:.2f}% | n={d['n']:,}")
            for t in d["tiers"]:
                print(f"        thr {t['thr']:.2f} | apuestas {t['bets']:>6,} | "
                      f"WR {t['wr']:>5.2f}% (IC95 {t['wr_ci95']}) | ROI140 {t['roi_140']:>+6.2f}% | "
                      f"ROI183 {t['roi_183']:>+6.2f}%")
        print()

    block("COMPARATIVA")
    base = results[0]
    for r in results[1:]:
        print(f"  {r['variant']}  vs  V0")
        for label in ("C_completo", "C_filtrado"):
            if label not in r or label not in base:
                continue
            a, b = base[label], r[label]
            print(f"    {label:<12} dAUC {b['auc'] - a['auc']:+.4f} | "
                  f"dBrier {b['brier'] - a['brier']:+.4f} | dLogLoss {b['logloss'] - a['logloss']:+.4f}")
            for t0, t1 in zip(a["tiers"], b["tiers"]):
                print(f"        thr {t0['thr']:.2f}: apuestas {t0['bets']:>6,} -> {t1['bets']:>6,} | "
                      f"WR {t0['wr']:>5.2f}% -> {t1['wr']:>5.2f}% | "
                      f"ROI140 {t0['roi_140']:>+6.2f}% -> {t1['roi_140']:>+6.2f}%")

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT_JSON}")


if __name__ == "__main__":
    main()
