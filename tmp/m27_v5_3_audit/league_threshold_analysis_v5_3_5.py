import json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3_5"))

from train_m27_v5_3_5_pipeline import fit_ensemble, p_sym, fit_calibrator, margin_sym, green, wilson, roi
from models.m27_v5_3_5.features import mirror

DATASET = ROOT / "tmp" / "m27_v5_3_5" / "dataset_v5_3_5.pkl"
OUT = Path(__file__).resolve().parent / "league_threshold_analysis_v5_3_5.json"
THRS = (0.65, 0.68, 0.70, 0.72, 0.75, 0.80)
BE_140 = 1 / 1.40

def main():
    t0 = time.time()
    D = pickle.load(open(DATASET, "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    ord_ = np.array([m["ord"] for m in meta])
    mrank = np.array([rank.get(int(o), -1) for o in ord_])
    anchor = np.array([m["anchor"] for m in meta])
    tt = np.array([m["t"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    cnt = pd.Series(mrank).map(pd.Series(mrank).value_counts()).values
    w_all = 1.0 / cnt
    print(f"Filas {len(y):,} | A {rowA.sum():,} B {rowB.sum():,} C {rowC.sum():,}", flush=True)

    print("[1] Entrenando ensamble...", flush=True)
    models = fit_ensemble(X[rowA], y[rowA], q4d[rowA], w_all[rowA], quick=False)
    print(f"    listo ({time.time()-t0:.0f}s)", flush=True)

    pB = p_sym(models, X[rowB])
    iso = fit_calibrator(pB, y[rowB])
    pC = iso.predict(p_sym(models, X[rowC]))
    mC = margin_sym(models, X[rowC])
    yC = y[rowC]
    lgC = lg[rowC]
    aC = anchor[rowC]

    idxC = np.where(aC)[0]
    pCa = pC[idxC]; mCa = mC[idxC]; yCa = yC[idxC]; lgCa = lgC[idxC]

    aA = anchor[rowA]; lgA = lg[rowA]
    train_counts = {}
    for name in pd.unique(lgA[aA]):
        train_counts[str(name)] = int((lgA[aA] == name).sum())

    leagues_out = []
    for name in pd.unique(lgCa):
        name = str(name)
        mk = (lgCa == name)
        n_rows = int(mk.sum())
        if n_rows < 30:
            continue
        p = pCa[mk]; yv = yCa[mk]; mv = mCa[mk]
        tiers = []
        for thr in THRS:
            nn, ww = green(p, yv, thr, mv, 0.0)
            if nn < 15:
                tiers.append({"thr": thr, "bets": nn, "wr": None, "wr_ci95": None, "roi_140": None, "roi_183": None})
                continue
            lo, hi = wilson(ww, nn)
            tiers.append({"thr": thr, "bets": int(nn), "wr": round(ww/nn*100, 2),
                          "wr_ci95": [round(lo*100, 1), round(hi*100, 1)],
                          "roi_140": round(roi(ww, nn, 1.40), 2), "roi_183": round(roi(ww, nn, 1.83), 2),
                          "block_candidate": bool(hi < BE_140)})
        hm = (p >= 0.75) | (p <= 0.25)
        sel_p = np.where(p >= 0.5, p, 1 - p)[hm]
        bias = round(float(sel_p.mean() - ((yv[p>=0.75]==1).sum() + (yv[p<=0.25]==0).sum())/max(hm.sum(),1)), 4) if hm.sum() else 0.0
        leagues_out.append({"league": name, "matches_holdout": n_rows, "matches_train": train_counts.get(name, 0), "tiers": tiers, "bias_at_075": bias})

    def bets75(L):
        for t in L["tiers"]:
            if t["thr"] == 0.75: return t.get("bets") or 0
        return 0
    leagues_out.sort(key=bets75, reverse=True)

    result = {"model": "m27_v5_3_5", "split": "C (holdout), anclas t=30", "thresholds": list(THRS),
              "n_leagues": len(leagues_out), "leagues": leagues_out, "elapsed_s": round(time.time()-t0, 1)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo: {len(leagues_out)} ligas -> {OUT} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
