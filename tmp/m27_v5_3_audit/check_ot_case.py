# =====================================================================
# ¿El guard de Q4 parcial se confunde con la PROLONGACION?
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

r = conn.execute(
    "SELECT match_id, home_team, away_team, home_score, away_score, status_description "
    "FROM matches WHERE match_id=17060600").fetchone()
print("match:", dict(r) if r else None)
print("legacy (formato largo):")
for q in conn.execute("SELECT quarter, home, away FROM quarter_scores WHERE match_id=17060600"):
    print("   ", dict(q))
v = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id=17060600").fetchone()
print("v2 (formato ancho):", dict(v) if v else None)
print()
n_ot = conn.execute(
    "SELECT COUNT(DISTINCT match_id) FROM quarter_scores WHERE quarter='OT1'").fetchone()[0]
n_ot_v2 = conn.execute(
    "SELECT COUNT(*) FROM quarter_scores_v2 WHERE ot_home IS NOT NULL OR ot_away IS NOT NULL").fetchone()[0]
print(f"partidos con prolongation en la legacy : {n_ot}")
print(f"de ellos, con OT ya informado en v2  : {n_ot_v2}  <-- la migracion no copio el OT")
print()
print("apuestas pendientes de liquidar:")
for b in conn.execute(
        "SELECT id, match_id, model_version, signal_type, picked_side, result, created_at "
        "FROM bet_monitor_log_v3 WHERE result='pending' ORDER BY id"):
    print("   ", dict(b))
conn.close()
