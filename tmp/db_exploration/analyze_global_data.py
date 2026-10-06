"""
Script para generar estadísticas avanzadas globales de matches.db para EXPLORACION_GLOBAL_DATOS.md
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

    report = {}

    # 1. Temporal por mes
    months_query = cur.execute("""
        SELECT substr(date, 1, 7) as ym, count(*) as cnt
        FROM matches
        WHERE date IS NOT NULL AND date != ''
        GROUP BY ym
        ORDER BY ym
    """).fetchall()
    report["by_month"] = [{"month": r["ym"], "matches": r["cnt"]} for r in months_query]

    # 2. Métricas de tanteo
    score_stats = cur.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(home_score) as avg_home,
            AVG(away_score) as avg_away,
            AVG(home_score + away_score) as avg_total,
            MIN(home_score + away_score) as min_total,
            MAX(home_score + away_score) as max_total,
            SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
            SUM(CASE WHEN away_score > home_score THEN 1 ELSE 0 END) as away_wins,
            SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as draws
        FROM matches
        WHERE status_type = 'finished' AND home_score IS NOT NULL AND away_score IS NOT NULL
    """).fetchone()
    report["scores"] = dict(score_stats)

    # 3. Distribución de volumen de ligas
    league_dist = cur.execute("""
        SELECT 
            CASE 
                WHEN cnt >= 500 THEN 'Gigante (>=500)'
                WHEN cnt >= 200 THEN 'Muy Grande (200-499)'
                WHEN cnt >= 100 THEN 'Grande (100-199)'
                WHEN cnt >= 50 THEN 'Mediana (50-99)'
                WHEN cnt >= 20 THEN 'Pequeña (20-49)'
                ELSE 'Micro (<20)'
            END as tier,
            COUNT(*) as leagues_in_tier,
            SUM(cnt) as matches_in_tier
        FROM (
            SELECT league, COUNT(*) as cnt
            FROM matches
            GROUP BY league
        )
        GROUP BY tier
        ORDER BY matches_in_tier DESC
    """).fetchall()
    report["league_distribution"] = [dict(r) for r in league_dist]

    # 4. Top 50 Ligas
    top50 = cur.execute("""
        SELECT league, COUNT(*) as cnt,
               AVG(home_score + away_score) as avg_total_pts,
               SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as home_win_pct
        FROM matches
        WHERE league IS NOT NULL AND status_type = 'finished'
        GROUP BY league
        ORDER BY cnt DESC
        LIMIT 50
    """).fetchall()
    report["top_50_leagues"] = [dict(r) for r in top50]

    # 5. Promedio de granularidad en tablas secundarias
    pbp_stats = cur.execute("""
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT match_id) as matches_with_pbp,
            COUNT(*) * 1.0 / COUNT(DISTINCT match_id) as avg_events_per_match
        FROM play_by_play
    """).fetchone()
    report["pbp_stats"] = dict(pbp_stats)

    graph_stats = cur.execute("""
        SELECT 
            COUNT(*) as total_points,
            COUNT(DISTINCT match_id) as matches_with_graph,
            COUNT(*) * 1.0 / COUNT(DISTINCT match_id) as avg_points_per_match
        FROM graph_points
    """).fetchone()
    report["graph_stats"] = dict(graph_stats)

    team_stats_metrics = cur.execute("""
        SELECT DISTINCT stat_key, stat_name, group_name
        FROM team_statistics
        ORDER BY group_name, stat_key
        LIMIT 40
    """).fetchall()
    report["team_stats_sample_metrics"] = [dict(r) for r in team_stats_metrics]

    # Guardar
    Path("tmp/db_exploration/global_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("global_stats.json generado con exito.")

if __name__ == "__main__":
    main()
