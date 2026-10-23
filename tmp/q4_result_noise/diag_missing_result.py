# Diagnóstico temporal: apuesta de m27_v5_3 enviada (Vasto Verde Blumenau vs
# AJAB Jaraguá) sin tarjeta de resultado FT. Revisa schedule, log de apuesta y
# liquidación de la apuesta en matches.db.
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DB = Path(__file__).resolve().parents[2] / "matches.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

print("--- schedule ---")
rows = conn.execute(
    "SELECT * FROM bet_monitor_schedule_v3 WHERE home_team LIKE '%Vasto Verde%' "
    "OR away_team LIKE '%Jaragu%' ORDER BY scheduled_utc_ts DESC LIMIT 5"
).fetchall()
for r in rows:
    d = dict(r)
    print({k: d.get(k) for k in ("match_id", "home_team", "away_team", "status", "skip_reason",
                                 "final_fetched", "signal", "model", "scheduled_utc_ts")})

if rows:
    mid = rows[0]["match_id"]
    print(f"\n--- bet_monitor_log_v3 ({mid}) ---")
    for r in conn.execute("SELECT * FROM bet_monitor_log_v3 WHERE match_id = ?", (mid,)):
        d = dict(r)
        print({k: d.get(k) for k in ("id", "model_version", "signal_type", "picked_side", "confidence",
                                     "result", "actual_home_score", "actual_away_score", "created_at")})

    print(f"\n--- model_inference_v3 ({mid}) ---")
    for r in conn.execute(
        "SELECT model_version, COUNT(*) n, MIN(game_minute), MAX(game_minute), MAX(signal) "
        "FROM model_inference_v3 WHERE match_id = ? GROUP BY model_version", (mid,)
    ):
        print(dict(r))

    print(f"\n--- quarter_scores_v2 ({mid}) ---")
    for r in conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (mid,)):
        print(dict(r))

    print(f"\n--- bet_monitor_log_v3 pending global (muestra) ---")
    for r in conn.execute(
        "SELECT match_id, model_version, signal_type, result, created_at FROM bet_monitor_log_v3 "
        "WHERE result IN ('pending','') ORDER BY id DESC LIMIT 10"
    ):
        print(dict(r))
conn.close()
