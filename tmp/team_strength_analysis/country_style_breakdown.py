"""
Script para comparar estilos tácticos y fortaleza de plantillas entre países/regiones en matches.db.
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

    query = """
    SELECT 
        lc.country_or_region,
        COUNT(DISTINCT m.match_id) as sample_matches,
        AVG(CASE WHEN l.is_starter = 1 THEN ps.sofascore_rating END) as starter_rating,
        AVG(CASE WHEN l.is_starter = 0 THEN ps.sofascore_rating END) as bench_rating,
        SUM(CASE WHEN l.is_starter = 0 THEN ps.points ELSE 0 END) * 100.0 / 
            NULLIF(SUM(ps.points), 0) as bench_pts_share,
        SUM(ps.three_attempted) * 100.0 / 
            NULLIF(SUM(ps.field_goals_attempted), 0) as three_point_rate,
        SUM(ps.assists) * 1.0 / 
            NULLIF(SUM(ps.turnovers), 0) as ast_to_tov_ratio,
        AVG(ps.fouls) * 5.0 as avg_starter_fouls
    FROM matches m
    JOIN leagues_classification lc ON m.league = lc.league
    JOIN player_stats ps ON m.match_id = ps.match_id
    JOIN lineups l ON ps.id = l.id
    WHERE m.status_type = 'finished'
      AND lc.country_or_region != 'Other'
    GROUP BY lc.country_or_region
    HAVING sample_matches >= 200
    ORDER BY sample_matches DESC;
    """

    rows = cur.execute(query).fetchall()
    print(f"{'PAÍS / REGIÓN':<16} | {'PARTIDOS':<8} | {'STARTER':<7} | {'BENCH':<6} | {'BENCH PTS%':<11} | {'3P RATE%':<9} | {'AST/TO':<7}")
    print("-" * 80)
    
    country_styles = []
    for r in rows:
        st_r = round(r["starter_rating"], 2) if r["starter_rating"] else 0
        bn_r = round(r["bench_rating"], 2) if r["bench_rating"] else 0
        b_pts = round(r["bench_pts_share"], 1) if r["bench_pts_share"] else 0
        tpr = round(r["three_point_rate"], 1) if r["three_point_rate"] else 0
        ast_to = round(r["ast_to_tov_ratio"], 2) if r["ast_to_tov_ratio"] else 0
        
        country_styles.append({
            "country": r["country_or_region"],
            "matches": r["sample_matches"],
            "starter_rating": st_r,
            "bench_rating": bn_r,
            "bench_pts_share": b_pts,
            "three_point_rate": tpr,
            "ast_to_tov_ratio": ast_to
        })
        print(f"{r['country_or_region']:<16} | {r['sample_matches']:<8} | {st_r:<7} | {bn_r:<6} | {b_pts:<11}% | {tpr:<9}% | {ast_to:<7}")

    Path("tmp/team_strength_analysis/country_style_results.json").write_text(
        json.dumps(country_styles, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
