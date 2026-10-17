"""
Diagnóstico de Overfitting y Covariate Shift entre Cortes Temporales (2015-2022 vs 2023 vs 2024 vs 2025 vs 2026)
y en `nba_12m`.

Investiga:
1. Qué features cambian de cobertura/distribución por año (causando que entrenar en <=2022 y probar en 2023/2024 tenga gap).
2. Por qué `box_2pt_diff_q1_q3` en `team_statistics` degradó la calibración en Test (y al quitarlo subió el WR@0.70 a 85.88%).
3. Por qué `nba_12m` tuvo Gap=+0.1949 (repetición de snapshots x4 en dataset pequeño + deriva de ELO sin reset de temporada + profundidad excesiva)
   y cómo corregirlo en la Iteración V4.4 / V5.
"""
import sys
import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("1. Cobertura de tablas por Año (2018 a 2026):")
    cur.execute("""
        SELECT SUBSTR(m.date, 1, 4) as yr,
               COUNT(*) as n_matches,
               AVG(CASE WHEN EXISTS(SELECT 1 FROM match_odds mo WHERE mo.match_id = m.match_id) THEN 100.0 ELSE 0.0 END) as pct_odds,
               AVG(CASE WHEN EXISTS(SELECT 1 FROM team_statistics ts WHERE ts.match_id = m.match_id AND ts.period='Q3') THEN 100.0 ELSE 0.0 END) as pct_box_q3,
               AVG(CASE WHEN EXISTS(SELECT 1 FROM match_events me WHERE me.match_id = m.match_id) THEN 100.0 ELSE 0.0 END) as pct_events,
               AVG(CASE WHEN EXISTS(SELECT 1 FROM graph_points gp WHERE gp.match_id = m.match_id) THEN 100.0 ELSE 0.0 END) as pct_gp,
               AVG(CASE WHEN EXISTS(SELECT 1 FROM player_stats ps WHERE ps.match_id = m.match_id) THEN 100.0 ELSE 0.0 END) as pct_players,
               AVG(CASE WHEN EXISTS(SELECT 1 FROM team_strength tst WHERE tst.match_id = m.match_id) THEN 100.0 ELSE 0.0 END) as pct_ts
        FROM matches m
        WHERE m.date >= '2018-01-01'
        GROUP BY yr
        ORDER BY yr ASC
    """)
    print(f"{'Año':<6} | {'Partidos':>8} | {'%Odds':>7} | {'%BoxQ3':>7} | {'%Events':>7} | {'%Graph':>7} | {'%Players':>8} | {'%Strength':>9}")
    print("-" * 78)
    for row in cur.fetchall():
        yr, n, po, pb, pe, pg, pp, pts = row
        print(f"{yr:<6} | {n:>8,} | {po:>6.1f}% | {pb:>6.1f}% | {pe:>6.1f}% | {pg:>6.1f}% | {pp:>7.1f}% | {pts:>8.1f}%")

    print("\n2. Inspeccionando `team_statistics` Q1..Q3 vs `quarter_scores` Q1..Q3:")
    # ¿Por qué `box_2pt_diff_q1_q3` tuvo problemas? ¡Veamos si _parse_period_stats usó 600s o 720s o si en NBA/FIBA está desfasado!
    cur.execute("""
        SELECT m.match_id, m.league,
               (q1.home + q2.home + q3.home) - (q1.away + q2.away + q3.away) as qs_q3_diff,
               (q4.home - q4.away) as qs_q4_diff,
               ts.home_value - ts.away_value as ts_pts_q1_q3_diff
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        JOIN (
            SELECT match_id, SUM(home_value) as home_value, SUM(away_value) as away_value
            FROM team_statistics
            WHERE period IN ('Q1', 'Q2', 'Q3') AND stat_key = 'points'
            GROUP BY match_id
        ) ts ON m.match_id = ts.match_id
        LIMIT 5000
    """)
    rows = cur.fetchall()
    if rows:
        qs_d = np.array([r[2] for r in rows], dtype=float)
        q4_d = np.array([r[3] for r in rows], dtype=float)
        ts_d = np.array([r[4] for r in rows], dtype=float)
        exact_match = np.mean(np.abs(qs_d - ts_d) <= 1) * 100.0
        print(f"   Muestra N={len(rows)}: Coincidencia exacta entre suma Q1..Q3 de team_statistics y quarter_scores: {exact_match:.1f}%")
    else:
        print("   No hay stat_key='points' en Q1..Q3; verificando qué stat_keys existen en Q1..Q3 por año...")

    cur.execute("""
        SELECT SUBSTR(m.date, 1, 4) as yr, ts.stat_key, COUNT(*)
        FROM team_statistics ts
        JOIN matches m ON m.match_id = ts.match_id
        WHERE ts.period = 'Q3' AND ts.stat_key IN ('twoPointsMade', 'threePointsMade', 'fouls', 'points')
        GROUP BY yr, ts.stat_key
        ORDER BY yr, ts.stat_key
    """)
    for yr, sk, cnt in cur.fetchall():
        print(f"   Año {yr} | stat_key={sk:<18} | N={cnt:,}")

    conn.close()


if __name__ == "__main__":
    main()
