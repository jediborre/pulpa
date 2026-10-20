# =====================================================================
# ¿POR QUE LAS LINEAS NO LLEGAN A LAS FEATURES?
#  (match_odds las tiene; el dataset no las ve)
# =====================================================================

import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import IDX, extract_market  # noqa: E402

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, meta = D["X"], D["meta"]
# indice mid -> filas
por_mid = {}
for i, m in enumerate(meta):
    por_mid.setdefault(str(m["mid"]), []).append(i)

print("1) ¿LOS PARTIDOS CON LINEA ESTAN EN EL DATASET?")
con_linea = [r["match_id"] for r in conn.execute(
    "SELECT DISTINCT match_id FROM match_odds WHERE odds_type='Point spread' AND line IS NOT NULL")]
en_ds = [m for m in con_linea if m in por_mid]
print(f"   partidos con linea en match_odds : {len(con_linea):,}")
print(f"   de ellos, presentes en el dataset: {len(en_ds):,} ({len(en_ds) * 100 / len(con_linea):.1f} %)")

print()
print("2) ¿LAS FILAS DE ESOS PARTIDOS TIENEN LA FEATURE DE LINEA?")
col_sp = X[:, IDX["mkt_margin_spread"]]
col_tl = X[:, IDX["total_line"]]
n_sp = n_tl = n_filas = 0
for m in en_ds[:3000]:
    for i in por_mid[m]:
        n_filas += 1
        if not np.isnan(col_sp[i]):
            n_sp += 1
        if not np.isnan(col_tl[i]):
            n_tl += 1
print(f"   filas revisadas: {n_filas:,} | con spread: {n_sp:,} ({n_sp * 100 / max(n_filas, 1):.1f} %) "
      f"| con total: {n_tl:,} ({n_tl * 100 / max(n_filas, 1):.1f} %)")

print()
print("3) ¿QUE HAY EN match_odds PARA UNO DE ESOS PARTIDOS?")
if en_ds:
    mid = en_ds[0]
    for r in conn.execute(
            "SELECT id, odds_type, market_name, market_period, home_value, away_value, line "
            "FROM match_odds WHERE match_id=? ORDER BY id", (mid,)):
        print("   ", dict(r))

print()
print("4) ¿extract_market LOS LEE? (prueba directa con las filas de la BD)")
if en_ds:
    mid = en_ds[0]
    rows = [{"odds_type": r["odds_type"], "market_period": r["market_period"],
             "home_value": r["home_value"], "away_value": r["away_value"], "line": r["line"]}
            for r in conn.execute(
                "SELECT odds_type, market_period, home_value, away_value, line FROM match_odds "
                "WHERE match_id=? ORDER BY id", (mid,))]
    print(f"   extract_market -> {extract_market(rows)}")

print()
print("5) TIPOS DE MERCADO QUE HAY PARA LOS PARTIDOS CON LINEA:")
for r in conn.execute(
        "SELECT odds_type, market_period, COUNT(*) n, "
        "SUM(CASE WHEN line IS NOT NULL THEN 1 ELSE 0 END) con_linea "
        "FROM match_odds WHERE match_id IN (SELECT match_id FROM match_odds "
        "WHERE odds_type='Point spread' AND line IS NOT NULL LIMIT 5000) "
        "GROUP BY odds_type, market_period ORDER BY n DESC LIMIT 10"):
    print("   ", dict(r))
conn.close()
