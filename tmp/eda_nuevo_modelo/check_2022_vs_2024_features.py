"""
Inspecciona por año (2015..2026) las medias, desviaciones estándar y correlaciones con Q4 Win
de cada feature individual para descubrir exactamente por qué Corte 1 (<=2022 -> 2023) y Corte 2 (<=2023 -> 2024)
tuvieron caída de AUC y por qué nba_12m tuvo overfitting.
"""
import sys
import sqlite3
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT SUBSTR(m.date, 1, 4) as yr,
               COUNT(*) as n,
               AVG(CASE WHEN q4.home > q4.away THEN 1.0 ELSE 0.0 END) as q4_hwr,
               AVG((q1.home+q2.home+q3.home) - (q1.away+q2.away+q3.away)) as avg_q3_diff,
               AVG(CASE WHEN ((q1.home+q2.home+q3.home) > (q1.away+q2.away+q3.away)) = (q4.home > q4.away) THEN 1.0 ELSE 0.0 END) as leader_wins_q4,
               AVG(CASE WHEN me_cnt.cnt > 0 THEN 1.0 ELSE 0.0 END) as has_me
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        LEFT JOIN (SELECT match_id, COUNT(*) as cnt FROM match_events GROUP BY match_id) me_cnt ON m.match_id = me_cnt.match_id
        WHERE q4.home != q4.away
        GROUP BY yr
        ORDER BY yr
    """)
    print(f"{'Año':<6} | {'N':>7} | {'Q4 HomeWR':>9} | {'Avg Q3 Diff':>11} | {'LeaderQ3 Wins Q4':>16} | {'Has match_events':>16}")
    print("-" * 78)
    for r in cur.fetchall():
        print(f"{r[0]:<6} | {r[1]:>7,} | {r[2]*100:>8.2f}% | {r[3]:>+11.2f} | {r[4]*100:>15.2f}% | {r[5]*100:>15.1f}%")

    # ¡Verifiquemos qué hay en team_strength por año!
    print("\nInspeccionando correlación de team_strength (wins/losses) con Q4 Win por año:")
    cur.execute("""
        SELECT SUBSTR(m.date, 1, 4) as yr,
               m.match_id, m.home_team, m.away_team,
               (q1.home+q2.home+q3.home+q4.home) - (q1.away+q2.away+q3.away+q4.away) as ft_diff,
               (q4.home - q4.away) as q4_diff,
               tsh.wins, tsh.losses, tsa.wins, tsa.losses
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        JOIN team_strength tsh ON m.match_id = tsh.match_id AND m.home_team = tsh.team_name
        JOIN team_strength tsa ON m.match_id = tsa.match_id AND m.away_team = tsa.team_name
        WHERE q4.home != q4.away
    """)
    by_yr = {}
    for yr, mid, ht, at, ftd, q4d, hw, hl, aw, al in cur.fetchall():
        h_wp = (hw or 0) / max(1, (hw or 0) + (hl or 0))
        a_wp = (aw or 0) / max(1, (aw or 0) + (al or 0))
        by_yr.setdefault(yr, []).append((h_wp - a_wp, ftd, q4d))

    for yr in sorted(by_yr.keys()):
        arr = np.array(by_yr[yr], dtype=float)
        if len(arr) < 100:
            continue
        r_ft = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
        r_q4 = np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]
        print(f"  Año {yr}: N={len(arr):>6,} | r(ts_wp_diff, FT_margin) = {r_ft:+.4f} | r(ts_wp_diff, Q4_margin) = {r_q4:+.4f}")

    conn.close()


if __name__ == "__main__":
    main()
