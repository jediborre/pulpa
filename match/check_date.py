import sqlite3
import json

DATE = "2026-03-30"
conn = sqlite3.connect("matches.db")
conn.row_factory = sqlite3.Row

# 1. eval_match_results
n = conn.execute(f"SELECT COUNT(*) as n FROM eval_match_results WHERE event_date='{DATE}'").fetchone()["n"]
print(f"eval_match_results [{DATE}]: {n}")

# 2. tabla matches con columna "date"
n2 = conn.execute(f"SELECT COUNT(*) as n FROM matches WHERE date='{DATE}'").fetchone()["n"]
print(f"matches (date col) [{DATE}]: {n2}")

# 3. discovered_ft_matches
r3 = conn.execute(f"SELECT COUNT(*) as n, SUM(CASE WHEN processed=1 THEN 1 ELSE 0 END) as proc FROM discovered_ft_matches WHERE event_date='{DATE}'").fetchone()
print(f"discovered_ft_matches total={r3['n']} processed={r3['proc']}")

# 4. Ligas de los matches guardados
print(f"\nLigas en tabla matches para {DATE}:")
league_counts = {}
rows = conn.execute(f"SELECT match_id, league FROM matches WHERE date='{DATE}'").fetchall()
for row in rows:
    lg = row["league"] or "unknown"
    league_counts[lg] = league_counts.get(lg, 0) + 1

for lg, cnt in sorted(league_counts.items(), key=lambda x: -x[1]):
    print(f"  {cnt:3d}  {lg}")

# 5. Matches con cuartos completos (Q1+Q2+Q3+Q4 en quarter_scores)
print(f"\nMatches con 4 cuartos en quarter_scores [{DATE}]:")
cur5 = conn.execute(
    f"""
    SELECT m.match_id, m.home_team, m.away_team, m.league, COUNT(qs.quarter) as qcount
    FROM matches m
    JOIN quarter_scores qs ON m.match_id = qs.match_id
    WHERE m.date = '{DATE}'
    GROUP BY m.match_id
    HAVING COUNT(qs.quarter) >= 4
    ORDER BY m.league, m.home_team
    """
)
complete_rows = cur5.fetchall()
print(f"  Total: {len(complete_rows)}")
for r in complete_rows:
    print(f"    {r['league']:30s}  {r['home_team']} vs {r['away_team']}")

conn.close()
