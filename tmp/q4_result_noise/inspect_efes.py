# Inspección temporal: verificar si el partido Anadolu Efes vs Besiktas tiene
# log de m27_v5_3 (visible) ademas del m27_v5_2 (sombra), para explicar por que
# el bot envio "RESULTADO Q4" con el modelo en modo sombra.
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).resolve().parents[2] / "matches.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT match_id, home_team, away_team, scheduled_utc_ts FROM bet_monitor_schedule_v3 "
    "WHERE home_team LIKE '%Efes%' ORDER BY scheduled_utc_ts DESC LIMIT 5"
).fetchall()
for r in rows:
    print(dict(r))

if rows:
    mid = rows[0]["match_id"]
    print("\n--- bet_monitor_log_v3 ---")
    logs = conn.execute(
        "SELECT id, model_version, signal_type, picked_side, confidence, result, "
        "inference_minute, target_quarter FROM bet_monitor_log_v3 WHERE match_id = ?",
        (mid,),
    ).fetchall()
    for log in logs:
        print(dict(log))
    print("\n--- model_inference_v3 (count por modelo) ---")
    for r in conn.execute(
        "SELECT model_version, COUNT(*) n, MIN(inference_minute), MAX(inference_minute) "
        "FROM model_inference_v3 WHERE match_id = ? GROUP BY model_version",
        (mid,),
    ):
        print(dict(r))
conn.close()
