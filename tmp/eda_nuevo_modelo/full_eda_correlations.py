"""
Script: full_eda_correlations.py
Ubicación: tmp/eda_nuevo_modelo/full_eda_correlations.py
Propósito: Realizar un EDA profundo sobre los 227,028 partidos de matches.db,
           auditar la calidad de las tablas enriquecidas (match_odds, match_events, team_statistics,
           team_strength, lineups, player_stats, match_h2h), verificar posibles fugas de datos (data leakage)
           en propuestas previas, y calcular correlaciones reales (Pearson, Spearman y AUC univariado)
           con el resultado de Q4 (ganador y margen) y del cierre de partido (FT desde min 27).
Hipótesis/Problema: Identificar cuáles son las features verdaderamente predictivas y libres de leakage
                    para el nuevo modelo, evaluando tanto FIBA 10m (snap 27) como NBA 12m (snap 34).
"""
import sys
import sqlite3
import math
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 100:
        return np.nan, len(x)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=== 1. AUDITORÍA DE TABLAS NUEVAS Y MERCADOS (ODDS, EVENTS, TEAM_STATS, H2H) ===", flush=True)
    cur.execute("SELECT odds_type, market_name, market_period, COUNT(*), COUNT(DISTINCT match_id) FROM match_odds GROUP BY odds_type, market_name, market_period ORDER BY COUNT(*) DESC LIMIT 15")
    print("Mercados en match_odds:", flush=True)
    for r in cur.fetchall():
        print(f"  {r[0]} | {r[1]} | {r[2]}: {r[3]:,} filas ({r[4]:,} partidos)", flush=True)

    cur.execute("SELECT incident_type, subtype, COUNT(*) FROM match_events GROUP BY incident_type, subtype ORDER BY COUNT(*) DESC LIMIT 15")
    print("\nTipos de incidentes en match_events:", flush=True)
    for r in cur.fetchall():
        print(f"  {r[0]} / {r[1]}: {r[2]:,} eventos", flush=True)

    cur.execute("SELECT period, COUNT(DISTINCT match_id), COUNT(*) FROM team_statistics GROUP BY period ORDER BY COUNT(*) DESC")
    print("\nPeríodos en team_statistics:", flush=True)
    for r in cur.fetchall():
        print(f"  period={r[0]}: {r[1]:,} partidos ({r[2]:,} filas)", flush=True)

    print("\n=== 2. AUDITORÍA DE LIGAS NO CLASIFICADAS (UNCLASSIFIED) ===", flush=True)
    cur.execute("""
        SELECT m.league, COUNT(*) as cnt, MIN(m.date), MAX(m.date)
        FROM matches m
        LEFT JOIN leagues_classification lc ON m.league = lc.league
        WHERE m.status_type = 'finished' AND lc.league IS NULL
        GROUP BY m.league
        ORDER BY cnt DESC
        LIMIT 20
    """)
    print("Top 20 ligas sin entrada en leagues_classification:", flush=True)
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]:,} partidos [{r[2]} .. {r[3]}]", flush=True)

    print("\n=== 3. VERIFICACIÓN DE DATA LEAKAGE EN H2H Y PLAYER_STATS ===", flush=True)
    # Check match_h2h dates vs match date
    cur.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN h.date < m.date THEN 1 ELSE 0 END) as strictly_past,
               SUM(CASE WHEN h.date = m.date THEN 1 ELSE 0 END) as same_day,
               SUM(CASE WHEN h.date > m.date THEN 1 ELSE 0 END) as future_leak
        FROM match_h2h h
        JOIN matches m ON h.match_id = m.match_id
    """)
    r = cur.fetchone()
    print(f"match_h2h vs matches.date: Total={r[0]:,} | Pasado estricto (<)={r[1]:,} | Mismo día (=)={r[2]:,} | Futuro (> LEAK!)={r[3]:,}", flush=True)

    conn.close()

if __name__ == "__main__":
    main()
