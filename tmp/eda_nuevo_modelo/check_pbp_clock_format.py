"""
Script: check_pbp_clock_format.py
Ubicación: tmp/eda_nuevo_modelo/check_pbp_clock_format.py
Propósito: Inspeccionar el formato exacto de play_by_play (quarter, seq, time, points, home_score, away_score)
           y graph_points (seq, minute, value) en matches.db tanto en partidos antiguos (2018-2024) como recientes (2025-2026),
           y en ligas FIBA 10m vs NBA 12m.
Hipótesis/Problema: Verificar cómo se codifica el tiempo en play_by_play y por qué recent_3m_pts_diff mostró
                    correlación alta o cómo opera la escala de minutos 10m vs 12m.
"""
import sys
import sqlite3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for label, wh in [
        ("Euroleague 2026 (10m)", "m.league LIKE '%Euroleague%' AND m.date >= '2026-01-01'"),
        ("Euroleague 2024 (10m)", "m.league LIKE '%Euroleague%' AND m.date BETWEEN '2024-01-01' AND '2024-06-01'"),
        ("NBA 2026 (12m)", "m.league = 'NBA' AND m.date >= '2026-01-01'"),
    ]:
        cur.execute(f"""
            SELECT m.match_id, m.date, m.league, m.home_team, m.away_team, m.home_score, m.away_score
            FROM matches m
            WHERE {wh} AND EXISTS (SELECT 1 FROM play_by_play p WHERE p.match_id = m.match_id)
            LIMIT 1
        """)
        m = cur.fetchone()
        if not m:
            continue
        mid = m[0]
        print(f"\n=== {label}: {m} ===")
        cur.execute("SELECT quarter, home, away FROM quarter_scores WHERE match_id = ?", (mid,))
        print("Quarter scores:", cur.fetchall())
        cur.execute("SELECT quarter, MIN(seq), MAX(seq), MIN(time), MAX(time), COUNT(*) FROM play_by_play WHERE match_id = ? GROUP BY quarter", (mid,))
        print("PBP summary by quarter:", cur.fetchall())
        cur.execute("SELECT quarter, seq, time, team, points, home_score, away_score FROM play_by_play WHERE match_id = ? AND quarter IN ('Q3', 'Q4') ORDER BY id ASC LIMIT 15", (mid,))
        for r in cur.fetchall():
            print("  PBP sample:", r)
        cur.execute("SELECT MIN(minute), MAX(minute), COUNT(*) FROM graph_points WHERE match_id = ?", (mid,))
        print("Graph points summary (min, max, cnt):", cur.fetchone())

    conn.close()

if __name__ == "__main__":
    main()
