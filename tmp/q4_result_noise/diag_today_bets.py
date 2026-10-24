# Diagnóstico temporal: apuestas BET reales de hoy (m27_v5_3) y si su alerta
# de Telegram se llegó a emitir. Sirve para separar "el monitor no evaluó" de
# "Telegram rechazó el envío (502 Bad Gateway)".
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row

today = datetime.now().strftime("%Y-%m-%d")
print(f"hoy = {today}\n")
rows = conn.execute("""
    SELECT l.match_id, l.model_version, l.signal_type, l.picked_side, l.confidence,
           l.result, l.created_at, s.home_team, s.away_team, s.league
    FROM bet_monitor_log_v3 l
    LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
    WHERE date(l.created_at) = ?
      AND l.model_version = 'm27_v5_3'
      AND l.signal_type LIKE 'BET%' AND l.signal_type NOT LIKE '%NO_BET%'
    ORDER BY l.created_at DESC
""", (today,)).fetchall()
for r in rows:
    print(f"{r['created_at']} {r['match_id']} {r['home_team']} vs {r['away_team']} | "
          f"{r['signal_type']} {r['confidence']:.3f} -> {r['result']} | {r['league']}")
print(f"\nBET reales hoy: {len(rows)}")

print("\n--- envios Telegram de hoy (por tipo) ---")
import re
log = ROOT / "logs" / f"monitor_v3_{today}.log"
counts = {}
for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
    for key in ("[BROADCAST-IMG]", "[BROADCAST]", "[FAIL]", "Corte de red en polling"):
        if key in line:
            counts[key] = counts.get(key, 0) + 1
print(counts)
conn.close()
