# =====================================================================
# ESTADO FINAL: monitor, tabla por minuto y viabilidad del calculo de EV
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("=== TABLA POR MINUTO (model_inference_v3) ===")
print("filas:", conn.execute("SELECT COUNT(*) FROM model_inference_v3").fetchone()[0])
for r in conn.execute("SELECT model_version, COUNT(*) n, COUNT(DISTINCT match_id) m, "
                      "MIN(game_minute) mn, MAX(game_minute) mx FROM model_inference_v3 "
                      "GROUP BY model_version"):
    print("  ", dict(r))

print()
print("=== live_snapshots_v3 (telemetria del minuto) ===")
print("filas:", conn.execute("SELECT COUNT(*) FROM live_snapshots_v3").fetchone()[0])
r = conn.execute("SELECT * FROM live_snapshots_v3 ORDER BY id DESC LIMIT 1").fetchone()
if r:
    print("  ultima fila: match", r["match_id"], "min", r["game_minute"])
    try:
        odds = json.loads(r["odds_live_json"] or "{}")
        print("  claves de odds_live:", list(odds)[:8])
        print("  muestra:", json.dumps(odds, ensure_ascii=False)[:600])
    except Exception as exc:
        print("  odds_live no parseable:", exc)
else:
    print("  (vacia)")

print()
print("=== ULTIMAS APUESTAS EN EL LOG ===")
for r in conn.execute("SELECT model_version, signal_type, created_at, league FROM bet_monitor_log_v3 "
                      "ORDER BY created_at DESC LIMIT 6"):
    print("  ", dict(r))
conn.close()
