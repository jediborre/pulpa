# =====================================================================
# ABLACION DE EXCLUSION DE FASES (m27_v5_3) —Decision del usuario
# ---------------------------------------------------------------------
# DECISIONES A VALIDAR (2026-10-20):
#   Descartar del train y bloquear en inferencia:
#     - AMISTOSOS
#     - SEGUNDA FASE + PROMOCION + PRELIMINAR + PUESTO
#   Mantener (no se descartan):
#     - ELIMINATORIA, COPA  (el backtest dice que rinden bien: 84.3 % / 82.8 %)
#   Pendiente de medicion (el usuario pidió medir antes de decidir):
#     - R1 JUVENILES (is_youth=1)  -> 851 ligas / 7.743 partidos en el train
#     - R2 NCAA (is_college=1)     ->  53 ligas / 2.914 partidos en el train
#
# QUE HACE:
#   Reentrena el ensamble desde el dataset YA construido (no reextrae
#   features) con cuatro mascaras de entrenamiento y evalua SIEMPRE sobre el
#   mismo bloque C (test fuera de muestra):
#     V0  real (lo que hay hoy)
#     V1  sin AMISTOSOS                       (decision del usuario)
#     V2  sin AMISTOSOS + 2ªFASE+PROMO+PRELIM+PUESTO   (decision del usuario)
#     V3  V2 + sin JUVENILES + sin NCAA      (lo que hay que medir)
#   Mide AUC, Brier, LogLoss y la tabla de umbrales, en dos vistas:
#     - C completo  (todo el holdout, comparable entre variantes)
#     - C filtrado  (solo las categorias que cada variante conserva)
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

from models.common.competition_stage import classify_stage  # noqa: E402
from train_m27_v5_3_pipeline import (  # noqa: E402
    fit_calibrator, fit_ensemble, green, margin_sym, metrics, p_sym, roi, tier_table,
)

DATASET = ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl"
OUT_JSON = ROOT / "tmp" / "m27_v5_3_audit" / "phase_exclusion_ablation.json"

VARIANTES = [
    ("V0 real (sin exclusion adicional)", ()),
    ("V1 sin AMISTOSOS", ("AMISTOSOS",)),
    ("V2 sin AMISTOSOS+2FASE+PROMO+PRELIM+PUESTO",
     ("AMISTOSOS", "SEGUNDA_FASE", "PROMOCION", "PRELIMINAR", "PUESTO")),
    ("V3 V2 + sin JUVENILES + sin NCAA",
     ("AMISTOSOS", "SEGUNDA_FASE", "PROMOCION", "PRELIMINAR", "PUESTO", "__JUVENIL__", "__NCAA__")),
]


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def main() -> None:
    import sqlite3
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    lc_flags = {}
    for lg, you, col in conn.execute(
            "SELECT league, is_youth, is_college FROM leagues_classification"):
        lc_flags[lg] = (bool(you), bool(col))
    conn.close()

    print("Cargando dataset...")
    D = pickle.load(open(DATASET, "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    ord_ = np.array([m["ord"] for m in meta])
    mrank = np.array([rank.get(int(o), -1) for o in ord_])
    anchor = np.array([m["anchor"] for m in meta])
    lg = np.array([m["lg"] for m in meta])

    cats = np.array([classify_stage(x)[0] for x in lg])
    is_youth = np.array([lc_flags.get(x, (False, False))[0] for x in lg])
    is_college = np.array([lc_flags.get(x, (False, False))[1] for x in lg])

    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    cnt = np.bincount(inv)[inv]
    w_all = 1.0 / cnt
    print(f"  filas {len(y):,} | A {int(rowA.sum()):,} B {int(rowB.sum()):,} C {int(rowC.sum()):,}")

    def mask_drop(drop) -> np.ndarray:
        m = np.zeros(len(y), dtype=bool)
        for c in drop:
            if c == "__JUVENIL__":
                m |= is_youth
            elif c == "__NCAA__":
                m |= is_college
            else:
                m |= (cats == c)
        return m

    block("COMPOSICION DEL ENTRENAMIENTO ACTUAL (V0) por categoria de fase")
    for cat in sorted(set(cats.tolist())):
        n_a = int((rowA & (cats == cat)).sum())
        n_c = int((rowC & (cats == cat)).sum())
        print(f"  {cat:<16} filas train {n_a:>8,} ({n_a * 100 / rowA.sum():>5.1f}%) | test C {n_c:>8,}")
    print(f"  {'JUVENIL (is_youth)':<16} filas train {int((rowA & is_youth).sum()):>8,} "
          f"({(rowA & is_youth).sum() * 100 / rowA.sum():>5.1f}%) | test C {int((rowC & is_youth).sum()):>8,}")
    print(f"  {'NCAA (is_college)':<16} filas train {int((rowA & is_college).sum()):>8,} "
          f"({(rowA & is_college).sum() * 100 / rowA.sum():>5.1f}%) | test C {int((rowC & is_college).sum()):>8,}")

    results = []
    for nombre, drop in VARIANTES:
        t0 = time.time()
        dm = mask_drop(drop)
        keepA, keepB = rowA & ~dm, rowB & ~dm
        models = fit_ensemble(X[keepA], y[keepA], q4d[keepA], w_all[keepA], quick=False)
        pB = p_sym(models, X[keepB])
        iso = fit_calibrator(pB, y[keepB])
        pC = iso.predict(p_sym(models, X[rowC]))
        mC = margin_sym(models, X[rowC])

        out = {"variant": nombre, "drop": list(drop),
               "train_rows": int(keepA.sum()), "dropped_train_rows": int((rowA & dm).sum())}
        yC = y[rowC]
        for label, mask in (("C_completo", rowC), ("C_filtrado", rowC & ~dm)):
            if mask.sum() == 0:
                continue
            mc = mask[rowC]
            mt = metrics(pC[mc], yC[mc])
            tiers = tier_table(pC[mc], yC[mc], mC[mc], thrs=(0.72, 0.75, 0.80))
            out[label] = {"auc": round(mt["auc"], 4), "brier": round(mt["brier"], 4),
                          "logloss": round(mt["logloss"], 4), "acc": round(mt["acc"], 4),
                          "n": int(mask.sum()), "tiers": tiers}
        results.append(out)
        print(f"\n[{nombre}]  train {out['train_rows']:,} filas "
              f"(descartadas {out['dropped_train_rows']:,}) en {time.time() - t0:.0f}s")
        for label in ("C_completo", "C_filtrado"):
            if label not in out:
                continue
            d = out[label]
            print(f"    {label:<12} AUC {d['auc']:.4f} | Brier {d['brier']:.4f} | "
                  f"LogLoss {d['logloss']:.4f} | Acc {d['acc'] * 100:.2f}% | n={d['n']:,}")
            for t in d["tiers"]:
                print(f"        thr {t['thr']:.2f} | apuestas {t['bets']:>6,} | "
                      f"WR {t['wr']:>5.2f}% (IC95 {t['wr_ci95']}) | "
                      f"ROI140 {t['roi_140']:>+6.2f}% | ROI183 {t['roi_183']:>+6.2f}%")

    block("COMPARATIVA CONTRA V0")
    base = results[0]
    for r in results[1:]:
        print(f"  {r['variant']}")
        for label in ("C_completo", "C_filtrado"):
            if label not in r or label not in base:
                continue
            a, b = base[label], r[label]
            print(f"    {label:<12} dAUC {b['auc'] - a['auc']:+.4f} | "
                  f"dBrier {b['brier'] - a['brier']:+.4f} | "
                  f"dLogLoss {b['logloss'] - a['logloss']:+.4f} | "
                  f"dAcc {(b['acc'] - a['acc']) * 100:+.2f} pp")
            for t0, t1 in zip(a["tiers"], b["tiers"]):
                print(f"        thr {t0['thr']:.2f}: apuestas {t0['bets']:>6,} -> {t1['bets']:>6,} | "
                      f"WR {t0['wr']:>5.2f}% -> {t1['wr']:>5.2f}% | "
                      f"ROI140 {t0['roi_140']:>+6.2f}% -> {t1['roi_140']:>+6.2f}% | "
                      f"ROI183 {t0['roi_183']:>+6.2f}% -> {t1['roi_183']:>+6.2f}%")

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT_JSON}")


if __name__ == "__main__":
    main()
