# =====================================================================
# v5_4: UMBRAL POR POBLACION (con mercado pre-partido vs sin mercado)
# ---------------------------------------------------------------------
# MEDIDO:
#   - El modelo ya internaliza el mercado (corr(p_model, p_mercado) = 0.79-0.86)
#     -> anadir el mercado como feature no aporta (mejor mezcla: peso 0.0).
#   - Donde HAY mercado pre-partido, el modelo acierta 74.2 % (apenas sobre el
#     equilibrio de 1.40 = 71.4 %); donde NO hay mercado, acierta 83.7 %.
#   - Las features de estado no aportan (dAUC -0.0002) y el prior contextual
#     por liga empeora (dAUC -0.0164).
#
# ESTE SCRIPT DICE: como cambia el ROI y el volumen si el umbral depende de
# si el partido tiene o no mercado pre-partido.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_threshold_by_population.py
# =====================================================================

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from train_m27_v5_3_pipeline import fit_calibrator, fit_ensemble, p_sym  # noqa: E402


def dec(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, meta = D["X"], D["y"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    anchor = np.array([m["anchor"] for m in meta])
    mid = np.array([m["mid"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    w_all = 1.0 / np.bincount(inv)[inv]

    models = fit_ensemble(X[rowA], y[rowA], D["q4_diff"][rowA], w_all[rowA], quick=False)
    pB = p_sym(models, X[rowB])
    iso = fit_calibrator(pB, y[rowB])
    idxC = np.where(rowC & anchor)[0]
    pC = iso.predict(p_sym(models, X[idxC]))
    yC = y[idxC]
    midC = mid[idxC]

    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row
    con_ml = {}
    for r in conn.execute(
        "SELECT match_id, home_value, away_value FROM match_odds "
        "WHERE odds_type='Home/Away' AND market_period='Match'"
    ):
        h, a = dec(r["home_value"]), dec(r["away_value"])
        if h and a and h > 1 and a > 1:
            con_ml[r["match_id"]] = (1 / h) / ((1 / h) + (1 / a))
    conn.close()

    hay = np.array([m in con_ml for m in midC])
    conf = np.maximum(pC, 1 - pC)
    pick_home = pC >= 0.5

    def stats(mask):
        n = int(mask.sum())
        if n == 0:
            return None
        w = int(((pick_home[mask]) == (yC[mask] == 1)).sum())
        return {"n": n, "wr": w * 100 / n, "roi140": (w * 1.40 - n) / n * 100,
                "roi183": (w * 1.83 - n) / n * 100}

    print("=" * 96)
    print("1) TABLA DE UMBRALES POR POBLACION (anclas t=30, bloque C)")
    print("=" * 96)
    for thr in (0.70, 0.72, 0.75, 0.78, 0.80, 0.85):
        base = conf >= thr
        s_hay = stats(base & hay)
        s_no = stats(base & ~hay)
        s_all = stats(base)
        if not s_all:
            continue
        print(f"\n  thr {thr:.2f}  (todo: {s_all['n']:,} apuestas, WR {s_all['wr']:.1f} %, "
              f"ROI140 {s_all['roi140']:+.1f} %)")
        for tag, s in (("con mercado", s_hay), ("sin mercado", s_no)):
            if not s:
                continue
            print(f"     {tag:<14} n={s['n']:>6,} | WR {s['wr']:>5.1f} % | "
                  f"ROI140 {s['roi140']:>+6.1f} % | ROI183 {s['roi183']:>+6.1f} %")

    print()
    print("=" * 96)
    print("2) CONTRAFACTUAL: DISTINTAS POLITICAS")
    print("=" * 96)
    politicas = {
        "A: thr 0.75 para todos (actual)": (conf >= 0.75),
        "B: thr 0.75 solo SIN mercado": (conf >= 0.75) & ~hay,
        "C: thr 0.80 con mercado / 0.75 sin": ((conf >= 0.80) & hay) | ((conf >= 0.75) & ~hay),
        "D: thr 0.85 con mercado / 0.75 sin": ((conf >= 0.85) & hay) | ((conf >= 0.75) & ~hay),
        "E: thr 0.80 para todos": (conf >= 0.80),
    }
    for nombre, mask in politicas.items():
        s = stats(mask)
        if not s:
            continue
        print(f"  {nombre:<42} n={s['n']:>6,} | WR {s['wr']:>5.1f} % | "
              f"ROI140 {s['roi140']:>+6.1f} % | ROI183 {s['roi183']:>+6.1f} % | "
              f"beneficio esperado @1.40 = {s['n'] * s['roi140'] / 100:>7,.0f} u")

    print()
    print("=" * 96)
    print("3) ¿DONDE ESTA EL EDGE? (probabilidad del mercado vs acierto real)")
    print("=" * 96)
    p_mkt = np.array([con_ml.get(m, np.nan) for m in midC])
    for tag, mask in (("con mercado", hay), ("sin mercado", ~hay)):
        sel = mask & (conf >= 0.75)
        if sel.sum() < 30:
            continue
        w = int(((pick_home[sel]) == (yC[sel] == 1)).sum())
        print(f"  {tag:<14} n={int(sel.sum()):>6,} | WR {w * 100 / sel.sum():>5.1f} % | "
              f"p_model medio {conf[sel].mean():.3f} | "
              f"p_mercado medio {np.nanmean(p_mkt[sel]):.3f}")
    print("\n  (en la poblacion CON mercado, el mercado y el modelo coinciden: no hay edge)")
    print("   (en la poblacion SIN mercado, el modelo va solo: ahi esta el edge)")


if __name__ == "__main__":
    main()
