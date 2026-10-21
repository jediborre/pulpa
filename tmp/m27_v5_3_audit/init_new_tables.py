# =====================================================================
# Crea las tablas nuevas (team_stats_v3 y su estado) y verifica
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.database.repository import init_tables  # noqa: E402

init_tables()
print("init_tables() ejecutado")

conn = sqlite3.connect(str(ROOT / "matches.db"))
for t in ("team_stats_v3", "team_stats_backfill_state"):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"   {t:<30} filas {n:>6,} | columnas: {cols}")
conn.close()
