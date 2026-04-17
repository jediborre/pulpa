"""Quick diagnostic: show V13 raw confidence distribution and what gates fire."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'match'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'match', 'training'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'match'))

import sqlite3
from pathlib import Path
from collections import Counter

db_path = Path("matches.db")
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

# Load April matches
rows = conn.execute(
    "SELECT m.match_id, m.home_team, m.away_team, m.league "
    "FROM matches m "
    "WHERE m.date BETWEEN '2026-04-01' AND '2026-04-15' "
    "ORDER BY m.date LIMIT 80"
).fetchall()
conn.close()

# Import v13 gates directly
from training.v13.infer_match_v13 import _is_women_league, _penalty_boost, _count_league_bets
from training.v13 import config

block_women = []
block_penalty = []
pass_gate = []
seen_leagues = set()

for row in rows:
    league = row["league"] or ""
    if league in seen_leagues:
        continue
    seen_leagues.add(league)

    if _is_women_league(league):
        block_women.append(league)
        continue

    boost = _penalty_boost(league)
    if boost > 0:
        block_penalty.append((league, boost))
        continue

    pass_gate.append(league)

print(f"\n=== GATE 1 (mujeres bloqueadas): {len(block_women)} ligas ===")
for lg in block_women[:15]:
    print(f"  {lg}")

print(f"\n=== GATE 3a (penalización): {len(block_penalty)} ligas ===")
for lg, boost in block_penalty[:20]:
    print(f"  +{boost:.2f}  {lg}")

print(f"\n=== PASAN AMBOS GATES: {len(pass_gate)} ligas ===")
for lg in pass_gate[:20]:
    print(f"  {lg}")

print(f"\nTotal ligas únicas evaluadas: {len(seen_leagues)}")
