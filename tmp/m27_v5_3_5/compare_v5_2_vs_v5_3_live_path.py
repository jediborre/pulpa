"""
tmp/m27_v5_3_5/compare_v5_2_vs_v5_3_live_path.py
Comparativo cuantitativo v5_2 vs v5_3 sobre los MISMOS partidos del holdout y por el camino LIVE de cada modelo
(load_match + payload como lo entrega monitor_v3: odds como lista, h2h/team_strength como lista, sin cuotas in-play).
Es lo que cada modelo habria visto en el monitor, no el backtest ideal.

Notas de justicia: v5_2 usa su estado de produccion (incluye los resultados de estos partidos en sus ELOs => ventaja para v5_2);
v5_3 usa state_at_test_start.pkl (estado obsoleto desde el inicio del holdout => desventaja para v5_3).
Se evalua al minuto 27 y al 30, y se reportan AUC / Brier / tabla de umbrales (P calibrada del modelo, sin reglas extra).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.m27_v5_3_5.features as F  # noqa: E402
from models.common.data_loader import get_db_connection, load_match  # noqa: E402
from models.m27_v5_2.predict import predict as predict_v52  # noqa: E402
from models.m27_v5_3_5.predict import predict as predict_v53  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
D = pickle.load(open(ROOT / "tmp" / "m27_v5_3_5" / "dataset_v5_3.pkl", "rb"))
meta, y_all, cut = D["meta"], D["y"], D["cut_ord"]
with open(ROOT / "tmp" / "m27_v5_3_5" / "state_at_test_start.pkl", "rb") as f:
    F._STATE_CACHE = pickle.load(f)  # estado obsoleto = produccion sin refresco

rng = np.random.default_rng(11)
cand = [i for i, m in enumerate(meta) if m["ord"] >= cut and m["anchor"] and m["has_ev"] and not m["toxic"]]
rng.shuffle(cand)
cand = cand[:N]
conn = get_db_connection()

out = {27.0: {"v52": [], "v53": [], "y": [], "m53": []}, 30.0: {"v52": [], "v53": [], "y": [], "m53": []}}
t0 = time.time()
for k, i in enumerate(cand):
    mid = meta[i]["mid"]
    md = load_match(conn, mid)
    if not md:
        continue
    md["odds"] = [dict(zip(("odds_type", "market_name", "market_period", "home_value", "away_value", "line"), r)) for r in conn.execute(
        "SELECT odds_type, market_name, market_period, home_value, away_value, line FROM match_odds WHERE match_id=? ORDER BY id", (mid,))]
    md["h2h"] = []  # el monitor entrega una lista de partidos H2H
    md["team_strength"] = [dict(team_id=0, team_name=r[0], wins=r[1], losses=r[2]) for r in conn.execute(
        "SELECT team_name, wins, losses FROM team_strength WHERE match_id=?", (mid,))]
    for snap in (27.0, 30.0):
        r2 = predict_v52(mid, match_data=md, snapshot_minute=snap, conn=conn)
        r3 = predict_v53(mid, match_data=md, snapshot_minute=snap, conn=conn)
        if not (r2.available and r3.available):
            continue
        o = out[snap]
        o["v52"].append(r2.p_home_win); o["v53"].append(r3.p_home_win); o["y"].append(y_all[i])
        o["m53"].append((r3.raw_json or {}).get("expected_q4_margin", 0.0))
    if (k + 1) % 500 == 0:
        print(f"  {k+1}/{len(cand)} ({time.time()-t0:.0f}s)", flush=True)


def tiers(p, y):
    p, y = np.array(p), np.array(y)
    rows = []
    for t in (0.65, 0.68, 0.70, 0.72, 0.75, 0.80):
        h, a = p >= t, p <= 1 - t
        n = int(h.sum() + a.sum()); w = int((y[h] == 1).sum() + (y[a] == 0).sum())
        rows.append((t, n, w / n * 100 if n else 0.0, (w * 1.4 - n) / n * 100 if n else 0.0, (w * 1.83 - n) / n * 100 if n else 0.0))
    return rows


for snap, o in out.items():
    y = np.array(o["y"])
    print(f"\n==== Snapshot min {snap:.0f} | {len(y):,} partidos del holdout (camino LIVE) ====")
    for name in ("v52", "v53"):
        p = np.array(o[name])
        print(f"{name}: AUC {roc_auc_score(y, p):.4f} | Brier {brier_score_loss(y, p):.4f} | acc {np.mean((p >= .5) == y):.4f}")
        for t, n, wr, r140, r183 in tiers(p, y):
            print(f"    P>={t:.2f}: {n:>5,} apuestas ({n/len(y)*100:4.1f}%) | WR {wr:5.2f}% | ROI@1.40 {r140:+6.2f}% | ROI@1.83 {r183:+6.2f}%")
