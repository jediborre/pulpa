import sqlite3, os
db = os.path.join(os.path.dirname(__file__), '..', 'match', 'matches.db')
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT m.league, COUNT(*) as cnt "
    "FROM eval_match_results e JOIN matches m ON m.match_id = e.match_id "
    "WHERE e.event_date >= DATE('now', '-60 days') "
    "AND (e.q3_signal__bot_hybrid_f1 IN ('BET','BET_HOME','BET_AWAY') "
    "  OR e.q4_signal__bot_hybrid_f1 IN ('BET','BET_HOME','BET_AWAY')) "
    "GROUP BY m.league ORDER BY cnt DESC"
).fetchall()
conn.close()

total_leagues = len(rows)
buckets = {"1-4": [], "5-9": [], "10-19": [], "20-49": [], "50+": []}
for r in rows:
    c = r['cnt']
    if c >= 50: buckets["50+"].append((c, r['league']))
    elif c >= 20: buckets["20-49"].append((c, r['league']))
    elif c >= 10: buckets["10-19"].append((c, r['league']))
    elif c >= 5: buckets["5-9"].append((c, r['league']))
    else: buckets["1-4"].append((c, r['league']))

print(f"Total ligas con BETs: {total_leagues}\n")
for bucket, items in buckets.items():
    print(f"  [{bucket} bets] {len(items)} ligas:", ", ".join(f"{n}:{lg}" for n, lg in items[:8]))
print()
print("Top 20:")
for r in rows[:20]:
    print(f"  {r['cnt']:>4}  {r['league']}")
