# =====================================================================
# ESTADO DEL BACKFILL DE LINEAS
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

cols = [r["name"] for r in conn.execute("PRAGMA table_info(odds_backfill_state)")]
print("columnas de odds_backfill_state:", cols)
for r in conn.execute("SELECT * FROM odds_backfill_state LIMIT 5"):
    print("  ", dict(r))
print()
print("conteo por estado:")
for r in conn.execute("SELECT mode, status, COUNT(*) n FROM odds_backfill_state GROUP BY mode, status"):
    print("  ", dict(r))
print()
print("partidos con linea de spread guardada, por anio del partido:")
for r in conn.execute(
        "SELECT substr(m.date,1,4) yr, COUNT(DISTINCT o.match_id) n FROM match_odds o "
        "JOIN matches m USING(match_id) WHERE o.odds_type='Point spread' AND o.line IS NOT NULL "
        "GROUP BY yr ORDER BY yr"):
    print(f"   {r['yr']}: {r['n']:,}")
conn.close()
