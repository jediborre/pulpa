# =====================================================================
# Reinicia el estado del backfill (para reprocesar con la version nueva)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
n = conn.execute("SELECT COUNT(*) FROM team_stats_backfill_state").fetchone()[0]
conn.execute("DELETE FROM team_stats_backfill_state")
conn.commit()
print(f"estado reiniciado ({n} filas borradas)")
conn.close()
