# =====================================================================
# v5_4: ¿EL MERCADO APORTA AL MODELO PARA PREDECIR EL GANADOR DE Q4?
# ---------------------------------------------------------------------
# CONTEXTO MEDIDO:
#   - Los cuartos REALES no predicen el Q4: corr(Q1 real, Q4 real) = -0.008.
#   - El mercado de "1st quarter winner" SI lo predice: corr = +0.173.
#   - Las features de estado (patron de cuartos, ventaja extrema) no aportan
#     nada al modelo (dAUC -0.0002): el modelo ya tiene la ventaja, el Elo
#     y el mercado del partido.
#   - El prior contextual por liga (F1) EMPEORA (dAUC -0.0164).
#   => la unica direccion con informacion nueva es el MERCADO.
#
# QUE HACE (sobre los 13.979 partidos de LMB con mercado de 1st quarter):
#   1. AUC del modelo (p_cal) para predecir el ganador de Q4.
#   2. AUC del mercado de 1st quarter (prob implicita) para lo mismo.
#   3. AUC de la MEZCLA (logit del modelo + peso del logit del mercado).
#   4. Lo mismo con el mercado del PARTIDO (ML full time) como referencia.
#   5. El WR a umbral 0.75 con y sin mercado.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_market_vs_model.py
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

from sklearn.metrics import roc_auc_score  # noqa: E402
from models.m27_v5_3.features import mirror  # noqa: E402
from train_m27_v5_3_pipeline import fit_calibrator, fit_ensemble, p_sym  # noqa: E402

W = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}
THR = 0.75


def dec(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def implied(h, a):
    if not h or not a or h <= 1 or a <= 1:
        return None
    ph = (1 / h) / ((1 / h) + (1 / a))
    return ph


def logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


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

    print("Entrenando el modelo desplegado (para tener p_cal en el bloque C)...")
    models = fit_ensemble(X[rowA], y[rowA], D["q4_diff"][rowA], w_all[rowA], quick=False)
    pB = p_sym(models, X[rowB])
    iso = fit_calibrator(pB, y[rowB])
    idxC = np.where(rowC)[0]
    pC = iso.predict(p_sym(models, X[rowC]))
    yC = y[rowC]
    aC = anchor[rowC]
    midC = mid[idxC]

    # ------------------------------------------------------------------
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row
    print("Cargando mercados historicos...")
    odds = {}
    for r in conn.execute(
            "SELECT match_id, market_period, home_value, away_value FROM match_odds "
            "WHERE odds_type='Home/Away' AND market_period IN ('Match','1st quarter')"):
        odds.setdefault(r["match_id"], {})[r["market_period"]] = (r["home_value"], r["away_value"])
    conn.close()
    print(f"  partidos con algun mercado: {len(odds):,}")

    def collect(period_key):
        idxs, pmkt, pmatch = [], [], []
        for j, m in enumerate(midC):
            if not aC[j]:
                continue
            o = odds.get(m)
            if not o or period_key not in o:
                continue
            h, a = dec(o[period_key][0]), dec(o[period_key][1])
            p = implied(h, a)
            if p is None:
                continue
            if "Match" in o:
                hm, am = dec(o["Match"][0]), dec(o["Match"][1])
                pm = implied(hm, am)
            else:
                pm = None
            idxs.append(j)
            pmkt.append(p)
            pmatch.append(pm)
        return np.array(idxs), np.array(pmkt), pmatch

    for period_key, label in (("1st quarter", "MERCADO DE 1st QUARTER (proxy del Q4)"),
                              ("Match", "MERCADO DEL PARTIDO (ML full time)")):
        idxs, pmkt, pmatch = collect(period_key)
        if len(idxs) < 100:
            print(f"\n{label}: sin datos ({len(idxs)})")
            continue
        yy = yC[idxs]
        pmodel = pC[idxs]
        print()
        print("=" * 92)
        print(f"{label}  (n={len(idxs):,}, anclas t=30)")
        print("=" * 92)
        auc_m = roc_auc_score(yy, pmodel)
        auc_k = roc_auc_score(yy, pmkt)
        print(f"  AUC del MODELO (p_cal)            : {auc_m:.4f}")
        print(f"  AUC del MERCADO                    : {auc_k:.4f}")
        if pmatch and all(v is not None for v in pmatch):
            pm = np.array(pmatch)
            print(f"  AUC del mercado del PARTIDO        : {roc_auc_score(yy, pm):.4f}")
        # mezclas
        lm = np.array([logit(v) for v in pmodel])
        lk = np.array([logit(v) for v in pmkt])
        mejor = None
        for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            z = lm + w * lk
            auc = roc_auc_score(yy, z)
            if mejor is None or auc > mejor[1]:
                mejor = (w, auc)
            print(f"  mezcla modelo + {w:.1f}*logit(mercado): AUC {auc:.4f}")
        print(f"  -> mejor mezcla: peso {mejor[0]:.1f} con AUC {mejor[1]:.4f} "
              f"(modelo solo {auc_m:.4f}, mercado solo {auc_k:.4f})")
        # WR a umbral con/sin mercado
        for tag, sel in (("con mercado", np.ones(len(idxs), dtype=bool)),):
            p = pmodel[sel]
            yv = yy[sel]
            conf = np.maximum(p, 1 - p)
            b = conf >= THR
            if b.sum():
                w = int(((p[b] >= 0.5) == (yv[b] == 1)).sum())
                print(f"  modelo @thr {THR}: apuestas {int(b.sum()):,} | WR {w * 100 / b.sum():.1f} %")
        # ¿que pasa si el modelo NO ve el mercado (p_c sin el)?
        print(f"  El mercado aporta informacion INDEPENDIENTE? "
              f"corr(p_model, p_mercado) = {np.corrcoef(pmodel, pmkt)[0, 1]:+.4f}")

    # ------------------------------------------------------------------
    # ¿Como se comporta el modelo donde NO hay mercado? (el 62 % de las apuestas)
    print()
    print("=" * 92)
    print("EL MODELO DONDE NO HAY MERCADO (suena a donde esta el edge)")
    print("=" * 92)
    conf = np.maximum(pC, 1 - pC)
    bet = conf >= THR
    for tag, mask in (("con mercado pre-partido", np.array([bool(odds.get(m, {}).get("Match")) for m in midC])),
                      ("sin mercado pre-partido", np.array([not odds.get(m, {}).get("Match") for m in midC]))):
        sel = bet & mask
        if sel.sum() < 30:
            print(f"  {tag}: n={int(sel.sum())} (poco)")
            continue
        p = pC[sel]
        yv = yC[sel]
        w = int(((p >= 0.5) == (yv == 1)).sum())
        print(f"  {tag:<24} apuestas {int(sel.sum()):>6,} | WR {w * 100 / sel.sum():>5.1f} % | "
              f"prob media {conf[sel].mean():.3f}")


if __name__ == "__main__":
    main()
