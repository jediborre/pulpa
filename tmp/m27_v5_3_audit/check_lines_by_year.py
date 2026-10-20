# =====================================================================
# DISTRIBUCION DE LAS LINEAS POR AÑO (y por bloque del split)
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

from models.m27_v5_3.features import IDX  # noqa: E402

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, meta = D["X"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
n_el = len(elig)
a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
rowA = (mrank >= 0) & (mrank < a_end)
rowB = (mrank >= a_end) & (mrank < b_end)
rowC = mrank >= b_end
years = np.array([int(str(m.get("dord") or m.get("date") or "0")[:4]) for m in meta])
col_sp = X[:, IDX["mkt_margin_spread"]]
has_sp = ~np.isnan(col_sp)

print("COBERTURA DE LA LINEA DE SPREAD POR AÑO (todas las filas del dataset):")
print(f"  {'año':<7}{'filas':>9}{'con linea':>12}{'%':>8}")
for yr in sorted(set(years.tolist())):
    m = years == yr
    print(f"  {yr:<7}{int(m.sum()):>9,}{int((m & has_sp).sum()):>12,}"
          f"{(m & has_sp).sum() * 100 / max(m.sum(), 1):>7.1f}%")

print()
print("COBERTURA POR BLOQUE DEL SPLIT:")
for nombre, mask in (("A (train)", rowA), ("B (calib)", rowB), ("C (test)", rowC)):
    n = int(mask.sum())
    c = int((mask & has_sp).sum())
    print(f"  {nombre:<12} filas {n:>8,} | con linea {c:>7,} ({c * 100 / n:>5.1f} %)")

print()
print("PARTIDOS ELEGIBLES SIN LINEA, POR AÑO (cuanto falta por descargar):")
sin = conn.execute("""
    SELECT substr(m.date,1,4) yr, COUNT(*) n
    FROM matches m
    WHERE m.status_type='finished' AND m.date >= '2015-01-01'
      AND m.match_id NOT IN (SELECT match_id FROM match_odds WHERE line IS NOT NULL)
    GROUP BY yr ORDER BY yr
""").fetchall()
total = 0
for r in sin:
    print(f"   {r['yr']}: {r['n']:,}")
    total += r["n"]
print(f"   TOTAL sin linea (candidatos a descargar): {total:,}")
conn.close()
