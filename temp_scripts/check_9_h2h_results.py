import sqlite3
from pathlib import Path

conn = sqlite3.connect("match/matches.db")
conn.row_factory = sqlite3.Row

# Get matches evaluated by m27_v3 that have COMPLETE H2H
rows = conn.execute("""
    SELECT l.match_id, l.signal_type, l.picked_side, l.confidence, l.result,
           s.home_team, s.away_team, s.league, s.event_date,
           (SELECT COUNT(*) FROM match_h2h h
            WHERE h.match_id = l.match_id
              AND h.q1_home IS NOT NULL AND h.q1_away IS NOT NULL) as h2h_complete,
           (SELECT COUNT(*) FROM match_h2h h WHERE h.match_id = l.match_id) as h2h_total
    FROM bet_monitor_log_v2 l
    JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
    WHERE l.model_version = 'm27_v3'
      AND l.result IN ('win', 'hit', 'loss', 'miss')
      AND EXISTS (
          SELECT 1 FROM match_h2h h
          WHERE h.match_id = l.match_id
            AND h.q1_home IS NOT NULL AND h.q1_away IS NOT NULL
      )
    ORDER BY s.event_date ASC
""").fetchall()

print(f"Partidos con H2H COMPLETO evaluados por m27_v3: {len(rows)}")
print()

wins = losses = 0
for r in rows:
    res = "W" if r["result"] in ("win", "hit") else "L"
    if res == "W":
        wins += 1
    else:
        losses += 1
    ht = r["home_team"][:22]
    at = r["away_team"][:22]
    lg = r["league"].split(",")[0][:25]
    conf = r["confidence"]
    if conf <= 1.0:
        conf *= 100
    print(f"  {r['event_date']} {ht} vs {at}  {lg}")
    print(f"    pick={r['picked_side']} conf={conf:.0f}% h2h={r['h2h_complete']} filas -> {res}")

total = wins + losses
print(f"\n  RESUMEN: W={wins} L={losses} WR={wins/total*100:.1f}% (n={total})")

conn.close()
