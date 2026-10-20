# =====================================================================
# Estado de las liquidaciones pendientes tras el fix del guard
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("apuestas pendientes:")
for b in conn.execute("SELECT id, match_id, model_version, result FROM bet_monitor_log_v3 "
                       "WHERE result='pending' ORDER BY id"):
    print("   ", dict(b))
print()
print("ultimas liquidaciones:")
for b in conn.execute("SELECT id, match_id, model_version, signal_type, result "
                       "FROM bet_monitor_log_v3 WHERE result IN ('win','loss') ORDER BY id DESC LIMIT 8"):
    print("   ", dict(b))
conn.close()
