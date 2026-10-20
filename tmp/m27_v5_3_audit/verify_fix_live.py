# =====================================================================
# VERIFICA QUE EL FIX ESTA VIVO:
#   (a) cuando se mando el aviso del partido Falco (antes o despues del fix)
#   (b) como queda el predictions dict REAL del evaluador para un partido
#       con un modelo que apostó y otro que todavia no ha confirmado
#   (c) si el alert de apuesta incluye hoy a los modelos que no apostaron
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) CUANDO SE MANDO EL AVISO DE FALCO (17230818):")
for r in conn.execute(
        "SELECT model_version, signal_type, created_at FROM bet_monitor_log_v3 "
        "WHERE match_id=17230818 ORDER BY id"):
    print(f"    {r['model_version']:<12} {r['signal_type']:<12} {r['created_at']}")
print("   (el reinicio con el fix fue a las 2026-10-20 12:25)")

print()
print("2) APUESTAS REGISTRADAS DESDE EL ULTIMO REINICIO (12:25):")
for r in conn.execute(
        "SELECT model_version, signal_type, picked_side, confidence, created_at, match_id "
        "FROM bet_monitor_log_v3 WHERE created_at > '2026-10-20T12:25' ORDER BY id"):
    print(f"    {r['model_version']:<12} {r['signal_type']:<12} {str(r['picked_side']):<5} "
          f"conf={r['confidence']} {r['created_at'][:19]} {r['match_id']}")

print()
print("3) ULTIMOS PARTIDOS CON APUESTA: ¿estan todos los modelos en el log?")
for mid, in conn.execute(
        "SELECT DISTINCT match_id FROM bet_monitor_log_v3 "
        "WHERE created_at > '2026-10-20T12:25' AND signal_type LIKE '%BET%' LIMIT 5"):
    filas = [dict(r) for r in conn.execute(
        "SELECT model_version, signal_type, confidence FROM bet_monitor_log_v3 "
        "WHERE match_id=? ORDER BY id", (mid,))]
    print(f"    {mid}: " + " | ".join(
        f"{f['model_version']}={f['signal_type']}({f['confidence']})" for f in filas))

print()
print("4) SERIE POR MINUTO (model_inference_v3) de los ultimos partidos:")
for r in conn.execute(
        "SELECT match_id, model_version, game_minute, p_cal, signal FROM model_inference_v3 "
        "ORDER BY id DESC LIMIT 10"):
    print("   ", dict(r))
conn.close()
