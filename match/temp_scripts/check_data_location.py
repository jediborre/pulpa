import sqlite3
from pathlib import Path

DB_PATH = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Check quarter_scores
cursor.execute("SELECT COUNT(*) FROM quarter_scores")
print(f"quarter_scores count: {cursor.fetchone()[0]}")

cursor.execute("SELECT * FROM quarter_scores LIMIT 5")
rows = cursor.fetchall()
print(f"Sample quarter_scores: {rows}")

# Check eval_match_results for quarter data
cursor.execute("""
    SELECT 
        match_id, 
        q3_home_score, q3_away_score, 
        q4_home_score, q4_away_score
    FROM eval_match_results 
    WHERE q3_home_score IS NOT NULL 
    LIMIT 10
""")
eval_rows = cursor.fetchall()
print(f"\neval_match_results with Q3/Q4: {len(eval_rows)}")
for r in eval_rows[:3]:
    print(f"  {r}")

# Check graph_points
cursor.execute("SELECT COUNT(*) FROM graph_points")
print(f"\ngraph_points count: {cursor.fetchone()[0]}")

cursor.execute("SELECT * FROM graph_points LIMIT 3")
gp_rows = cursor.fetchall()
print(f"Sample graph_points: {gp_rows}")

conn.close()
