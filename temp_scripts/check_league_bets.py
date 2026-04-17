import sqlite3, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'match'))
conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), '..', 'match', 'matches.db'))
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT m.league, COUNT(*) as cnt
    FROM eval_match_results e
    JOIN matches m ON m.match_id = e.match_id
    WHERE e.event_date >= DATE('now', '-60 days')
      AND (
        e.q3_signal__bot_hybrid_f1 IN ('BET','BET_HOME','BET_AWAY')
        OR e.q4_signal__bot_hybrid_f1 IN ('BET','BET_HOME','BET_AWAY')
      )
    GROUP BY m.league ORDER BY cnt DESC LIMIT 40
""").fetchall()
conn.close()
print(f"{'Bets':>5}  Liga")
print("-" * 60)
for r in rows:
    print(f"{r['cnt']:>5}  {r['league']}")
# distribution
buckets = {"1-4": 0, "5-9": 0, "10-19": 0, "20-49": 0, "50+": 0}
all_rows = conn.execute("""...""") if False else []
