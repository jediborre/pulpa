# =====================================================================
# ESTADO DEL MONITOR TRAS EL REINICIO (verificacion de salud)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]

conn = sqlite3.connect(str(ROOT / "matches.db"))
print("model_inference_v3 filas:", conn.execute(
    "SELECT COUNT(*) FROM model_inference_v3").fetchone()[0])
print("por modelo:", conn.execute(
    "SELECT model_version, COUNT(*), COUNT(DISTINCT match_id) FROM model_inference_v3 "
    "GROUP BY model_version").fetchall())
print()
print("ultimas filas de bet_monitor_log_v3:")
for r in conn.execute("SELECT model_version, signal_type, created_at FROM bet_monitor_log_v3 "
                       "ORDER BY created_at DESC LIMIT 8"):
    print("   ", r)
print()
print("inferencias por minuto (muestra):")
for r in conn.execute(
    "SELECT model_version, game_minute, p_cal, margin, signal, confidence, block_reason "
    "FROM model_inference_v3 ORDER BY id DESC LIMIT 10"
):
    print("   ", r)
conn.close()
