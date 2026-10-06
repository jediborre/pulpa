"""
Script de prueba y diseño de agregación de player_stats y lineups por equipo.
Analiza rendimiento de consultas y calcula métricas clave para team_strength_matrix.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Probar una consulta de agregación sobre una muestra de partidos
    print("Probando unión entre player_stats, lineups y matches...")
    query = """
    SELECT 
        CASE WHEN ps.team = 'home' THEN m.home_team ELSE m.away_team END as team_name,
        m.league,
        COUNT(DISTINCT ps.match_id) as games_sample,
        AVG(ps.sofascore_rating) as avg_rating,
        SUM(ps.points) as total_pts,
        SUM(CASE WHEN l.is_starter = 1 THEN ps.points ELSE 0 END) as starter_pts,
        SUM(CASE WHEN l.is_starter = 0 THEN ps.points ELSE 0 END) as bench_pts,
        SUM(ps.field_goals_made) as fgm,
        SUM(ps.field_goals_attempted) as fga,
        SUM(ps.three_made) as tpm,
        SUM(ps.three_attempted) as tpa,
        SUM(ps.free_throws_made) as ftm,
        SUM(ps.free_throws_attempted) as fta,
        SUM(ps.rebounds) as reb,
        SUM(ps.assists) as ast,
        SUM(ps.turnovers) as tov,
        SUM(ps.fouls) as pf
    FROM player_stats ps
    JOIN lineups l ON ps.id = l.id
    JOIN matches m ON ps.match_id = m.match_id
    WHERE m.status_type = 'finished'
    GROUP BY team_name
    HAVING games_sample >= 20
    ORDER BY games_sample DESC
    LIMIT 10;
    """
    
    rows = cur.execute(query).fetchall()
    print(f"Top 10 equipos agregados probados exitosamente:")
    for r in rows:
        pts = r["total_pts"] or 1
        bench_pct = (r["bench_pts"] * 100.0 / pts) if pts > 0 else 0
        efg = ((r["fgm"] + 0.5 * r["tpm"]) * 100.0 / r["fga"]) if r["fga"] and r["fga"] > 0 else 0
        ast_tov = (r["ast"] / r["tov"]) if r["tov"] and r["tov"] > 0 else 0
        print(f"  - {r['team_name'][:30]:<30} | {r['games_sample']} part | Rating: {r['avg_rating']:.2f} | Bench%: {bench_pct:.1f}% | eFG%: {efg:.1f}% | AST/TO: {ast_tov:.2f}")

if __name__ == "__main__":
    main()
