# =====================================================================
# COBERTURA de las features de linea (spread / total) en el dataset actual
#   -> si el backfill de lineas ya avanzo, un reentrenamiento podria
#      activar mkt_margin_spread y total_line (hoy inertes).
# =====================================================================

import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import IDX  # noqa: E402

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, meta = D["X"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
rowA = (mrank >= 0) & (mrank < int(len(elig) * 0.70))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
print("DATOS EN LA BASE (match_odds):")
for r in conn.execute(
        "SELECT odds_type, market_period, COUNT(*) n, "
        "SUM(CASE WHEN line IS NOT NULL THEN 1 ELSE 0 END) con_linea "
        "FROM match_odds GROUP BY odds_type, market_period ORDER BY n DESC LIMIT 8"):
    print(f"   {r['odds_type']:<14} {r['market_period']:<14} n={r['n']:>8,} con linea={r['con_linea']:>8,}")

print()
print("COBERTURA EN EL DATASET (bloque de train):")
for feat in ("mkt_margin_spread", "total_line", "expected_ft_margin", "mkt_p_home",
             "odds_log_ratio", "has_ml"):
    i = IDX[feat]
    col = X[rowA, i]
    nan = int(np.isnan(col).sum())
    n = len(col)
    print(f"   {feat:<20} nulos {nan:>8,} / {n:>8,} = {nan * 100 / n:>5.1f} % | "
          f"con dato {n - nan:>8,} ({100 - nan * 100 / n:.1f} %)")

print()
print("PARTIDOS CON LINEA en match_odds (por anio):")
for r in conn.execute(
        "SELECT substr(m.date,1,4) yr, COUNT(DISTINCT o.match_id) n "
        "FROM match_odds o JOIN matches m USING(match_id) "
        "WHERE o.line IS NOT NULL GROUP BY yr ORDER BY yr DESC LIMIT 8"):
    print(f"   {r['yr']}: {r['n']:,}")
conn.close()
