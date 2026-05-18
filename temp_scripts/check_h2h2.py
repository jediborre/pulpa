import sys
sys.path.insert(0, 'match/training')
from infer_match import db_mod, DB_PATH
conn = db_mod.get_conn(str(DB_PATH))
db_mod.init_db(conn)

# Find a pair with H2H history (at least 2 matches, one before the other)
rows = conn.execute("""
    SELECT a.match_id, a.home_team, a.away_team, a.date
    FROM matches a
    WHERE EXISTS (
        SELECT 1 FROM matches b
        WHERE b.date < a.date
        AND ((b.home_team = a.home_team AND b.away_team = a.away_team)
             OR (b.home_team = a.away_team AND b.away_team = a.home_team))
    )
    LIMIT 5
""").fetchall()
for r in rows:
    print(f'{r[0]}: {r[1]} vs {r[2]} ({r[3]})')

conn.close()
