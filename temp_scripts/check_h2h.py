import sys
sys.path.insert(0, 'match/training')
from infer_match import db_mod, DB_PATH
conn = db_mod.get_conn(str(DB_PATH))
db_mod.init_db(conn)

rows = conn.execute("""
    SELECT match_id, date FROM matches
    WHERE (home_team = 'Pari Nizhny Novgorod' AND away_team = 'Enisey Krasnoyarsk')
       OR (home_team = 'Enisey Krasnoyarsk' AND away_team = 'Pari Nizhny Novgorod')
    ORDER BY date
""").fetchall()
print(f'H2H match count: {len(rows)}')
for r in rows:
    print(f'  {r[0]} ({r[1]})')

conn.close()
