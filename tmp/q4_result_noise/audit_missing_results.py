# Auditoría temporal: partidos con apuesta REAL de m27_v5_3 cuyo cierre quedó
# marcado con skip_reason (no_graph / final_fetch_failed / ...). En esos casos
# el código antiguo hacía return antes de enviar la tarjeta de resultado, así
# que el usuario no recibió el ✅/❌.
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
conn = sqlite3.connect(Path(__file__).resolve().parents[2] / "matches.db")
conn.row_factory = sqlite3.Row

print("--- cierres con skip_reason y apuesta REAL de m27_v5_3 (v3) ---")
rows = conn.execute("""
    SELECT s.match_id, s.skip_reason, s.final_fetched, s.home_team, s.away_team,
           l.signal_type, l.result, l.confidence, l.created_at
    FROM bet_monitor_schedule_v3 s
    JOIN bet_monitor_log_v3 l ON l.match_id = s.match_id
    WHERE l.model_version = 'm27_v5_3'
      AND l.signal_type LIKE 'BET%' AND l.signal_type NOT LIKE '%NO_BET%'
      AND COALESCE(s.skip_reason, '') != ''
    ORDER BY l.created_at DESC
""").fetchall()
for r in rows:
    print(f"{r['match_id']} [{r['skip_reason']}] final_fetched={r['final_fetched']} "
          f"{r['home_team']} vs {r['away_team']} | {r['signal_type']} -> {r['result']} "
          f"({r['confidence']:.2f}) @ {r['created_at']}")

print(f"\ntotal afectados: {len(rows)}")

print("\n--- desglose por skip_reason ---")
for r in conn.execute("""
    SELECT COALESCE(s.skip_reason,'(ninguno)') sr, COUNT(*) n
    FROM bet_monitor_schedule_v3 s
    JOIN bet_monitor_log_v3 l ON l.match_id = s.match_id
    WHERE l.model_version = 'm27_v5_3'
      AND l.signal_type LIKE 'BET%' AND l.signal_type NOT LIKE '%NO_BET%'
    GROUP BY sr ORDER BY n DESC
"""):
    print(f"{r['sr']}: {r['n']}")
conn.close()
