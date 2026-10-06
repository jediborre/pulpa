"""
Script para extraer el reporte final de la tabla leagues_classification mejorada.
"""
import sqlite3
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    stage_dist = cur.execute("""
        SELECT stage_detail, COUNT(*) as leagues, SUM(match_count) as matches
        FROM leagues_classification
        GROUP BY stage_detail
        ORDER BY matches DESC
    """).fetchall()

    gender_dist = cur.execute("""
        SELECT gender, COUNT(*) as leagues, SUM(match_count) as matches,
               AVG(avg_total_points) as avg_pts, AVG(home_win_pct) as win_pct,
               AVG(scoring_pace_per_minute) as pace
        FROM leagues_classification
        GROUP BY gender
    """).fetchall()

    final_sample = cur.execute("""
        SELECT league, clean_name, stage_detail, match_count, avg_total_points, blowout_rate, clutch_rate
        FROM leagues_classification
        WHERE is_final = 1
        ORDER BY match_count DESC
        LIMIT 10
    """).fetchall()

    women_sample = cur.execute("""
        SELECT league, clean_name, match_count, avg_total_points, scoring_pace_per_minute
        FROM leagues_classification
        WHERE is_women = 1 AND league NOT LIKE '%women%' AND league NOT LIKE '%femen%'
        ORDER BY match_count DESC
        LIMIT 15
    """).fetchall()

    confed_dist = cur.execute("""
        SELECT confederation, COUNT(*) as leagues, SUM(match_count) as matches,
               AVG(avg_total_points) as avg_pts
        FROM leagues_classification
        GROUP BY confederation
        ORDER BY matches DESC
    """).fetchall()

    out = {
        "stage_dist": [dict(r) for r in stage_dist],
        "gender_dist": [dict(r) for r in gender_dist],
        "final_sample": [dict(r) for r in final_sample],
        "women_sample": [dict(r) for r in women_sample],
        "confed_dist": [dict(r) for r in confed_dist]
    }

    Path("tmp/league_classification/final_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("Reporte generado con exito.")

if __name__ == "__main__":
    main()
