"""
Script para verificar y auditar las estadísticas de la tabla leagues_classification.
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

    total_leagues = cur.execute("SELECT COUNT(*) FROM leagues_classification").fetchone()[0]
    total_matches = cur.execute("SELECT SUM(match_count) FROM leagues_classification").fetchone()[0]

    # Distribución por género
    gender_dist = cur.execute("""
        SELECT gender, COUNT(*) as leagues, SUM(match_count) as matches,
               AVG(avg_total_points) as avg_pts, AVG(home_win_pct) as win_pct
        FROM leagues_classification
        GROUP BY gender
    """).fetchall()

    # Distribución por categoría de edad
    youth_dist = cur.execute("""
        SELECT is_youth, COUNT(*) as leagues, SUM(match_count) as matches
        FROM leagues_classification
        GROUP BY is_youth
    """).fetchall()

    # Distribución por college
    college_dist = cur.execute("""
        SELECT is_college, COUNT(*) as leagues, SUM(match_count) as matches
        FROM leagues_classification
        GROUP BY is_college
    """).fetchall()

    # Distribución por playoffs
    playoff_dist = cur.execute("""
        SELECT is_playoffs, COUNT(*) as leagues, SUM(match_count) as matches
        FROM leagues_classification
        GROUP BY is_playoffs
    """).fetchall()

    # Distribución por tipo de competición
    comp_dist = cur.execute("""
        SELECT competition_type, COUNT(*) as leagues, SUM(match_count) as matches
        FROM leagues_classification
        GROUP BY competition_type
        ORDER BY matches DESC
    """).fetchall()

    # Distribución por tier
    tier_dist = cur.execute("""
        SELECT tier_level, COUNT(*) as leagues, SUM(match_count) as matches,
               AVG(avg_total_points) as avg_pts
        FROM leagues_classification
        GROUP BY tier_level
        ORDER BY matches DESC
    """).fetchall()

    # Distribución por duración de cuarto
    duration_dist = cur.execute("""
        SELECT quarter_duration_minutes, COUNT(*) as leagues, SUM(match_count) as matches,
               AVG(avg_total_points) as avg_pts
        FROM leagues_classification
        GROUP BY quarter_duration_minutes
    """).fetchall()

    # Top 20 ligas con sus clasificaciones completas
    sample_top = cur.execute("""
        SELECT league, clean_name, stage, gender, is_youth, is_college, is_playoffs,
               tier_level, quarter_duration_minutes, match_count, avg_total_points,
               home_win_pct, ot_rate, avg_q4_total_points, pbp_coverage_pct, graph_coverage_pct
        FROM leagues_classification
        ORDER BY match_count DESC
        LIMIT 20
    """).fetchall()

    audit = {
        "total_leagues": total_leagues,
        "total_matches": total_matches,
        "gender_dist": [dict(r) for r in gender_dist],
        "youth_dist": [dict(r) for r in youth_dist],
        "college_dist": [dict(r) for r in college_dist],
        "playoff_dist": [dict(r) for r in playoff_dist],
        "comp_dist": [dict(r) for r in comp_dist],
        "tier_dist": [dict(r) for r in tier_dist],
        "duration_dist": [dict(r) for r in duration_dist],
        "sample_top": [dict(r) for r in sample_top]
    }

    Path("tmp/league_classification/classification_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("Auditoría guardada exitosamente en classification_audit.json")

if __name__ == "__main__":
    main()
