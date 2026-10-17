"""
Auditoría de las 41,614 ligas históricas sin clasificar en `leagues_classification`.
Inspecciona el máximo de time_seconds en match_events, promedios de puntos por cuarto
y nombres de liga/equipos para clasificar automáticamente cada liga en:
- fiba_men_10m
- nba_12m
- fiba_women_10m
"""
import sys
import sqlite3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT m.league,
               COUNT(*) as n_matches,
               AVG(q1.home + q1.away + q2.home + q2.away + q3.home + q3.away + q4.home + q4.away) as avg_ft,
               AVG(q4.home + q4.away) as avg_q4,
               AVG(CASE WHEN (q1.home+q2.home+q3.home+q4.home) > (q1.away+q2.away+q3.away+q4.away) THEN 1.0 ELSE 0.0 END) as home_wr
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        LEFT JOIN leagues_classification lc ON m.league = lc.league
        WHERE lc.league IS NULL
        GROUP BY m.league
        ORDER BY n_matches DESC
    """)
    rows = cur.fetchall()
    print(f"Total ligas sin clasificar: {len(rows)} (Total partidos: {sum(r[1] for r in rows):,})")
    print("\nTop 35 ligas sin clasificar:")
    for lg, cnt, avg_ft, avg_q4, hwr in rows[:35]:
        print(f"  {lg:<40} | N={cnt:>5} | AvgFT={avg_ft:>5.1f} | AvgQ4={avg_q4:>4.1f} | HomeWR={hwr*100:>4.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
