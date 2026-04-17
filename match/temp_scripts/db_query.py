import sqlite3
from pathlib import Path

DB_PATH = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Total matches
cursor.execute("SELECT COUNT(*) FROM matches")
total = cursor.fetchone()[0]

# Completed matches
cursor.execute("SELECT COUNT(*) FROM matches WHERE status_type = 'finished'")
completed = cursor.fetchone()[0]

# Distinct leagues
cursor.execute("SELECT DISTINCT league FROM matches WHERE status_type = 'finished'")
leagues = len(cursor.fetchall())

# Date range
cursor.execute("SELECT MIN(date), MAX(date) FROM matches")
date_range = cursor.fetchone()

# Gender distribution (infer from league name)
cursor.execute("""
    SELECT 
        CASE 
            WHEN LOWER(league) LIKE '%women%' OR LOWER(league) LIKE '%femenino%' THEN 'women'
            ELSE 'men'
        END as gender,
        COUNT(*) 
    FROM matches 
    WHERE status_type = 'finished'
    GROUP BY gender
""")
gender_dist = cursor.fetchall()

# League distribution
cursor.execute("""
    SELECT league, COUNT(*) as cnt 
    FROM matches 
    WHERE status_type = 'finished'
    GROUP BY league 
    ORDER BY cnt DESC 
    LIMIT 20
""")
top_leagues = cursor.fetchall()

# Matches with scores
cursor.execute("""
    SELECT COUNT(*) FROM matches 
    WHERE status_type = 'finished'
    AND home_score IS NOT NULL
""")
with_scores = cursor.fetchone()[0]

conn.close()

print(f"Total matches: {total}")
print(f"Completed matches: {completed}")
print(f"Completed with scores: {with_scores}")
print(f"Distinct leagues: {leagues}")
print(f"\nDate range: {date_range[0]} to {date_range[1]}")
print(f"\nGender distribution:")
for gender, cnt in gender_dist:
    print(f"  {gender}: {cnt}")
print(f"\nTop 20 leagues:")
for league, cnt in top_leagues:
    print(f"  {league}: {cnt}")
