"""
Script: check_team_stats_keys.py
Ubicación: tmp/eda_nuevo_modelo/check_team_stats_keys.py
Propósito: Listar todos los stat_key reales en team_statistics para períodos Q1, Q2, Q3 (1Q, 2Q, 3Q)
           y medir su correlación real (sin leakage, usando únicamente Q1+Q2 o Q1+Q2+Q3 cerrados) con Q4.
Hipótesis/Problema: En full_leakfree_eda.py buscamos 'rebounds' en vez de los nombres exactos de SofaScore
                    (ej. totalRebounds, offensiveRebounds, fouls, freeThrowsScored, etc.).
"""
import sys
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT stat_key, stat_name, COUNT(*) as cnt
        FROM team_statistics
        WHERE period IN ('Q1', '1Q')
        GROUP BY stat_key
        ORDER BY cnt DESC
        LIMIT 30
    """)
    print("=== Top stat_key en team_statistics (Q1 / 1Q) ===", flush=True)
    for r in cur.fetchall():
        print(f"  {r[0]:<30} | {r[1]:<30} | {r[2]:,}", flush=True)

    # Correlación de estadísticas de 1ª mitad (Q1+Q2) y fin de Q3 (Q1+Q2+Q3) con Q4 margin
    cur.execute("""
        SELECT m.match_id, q4.home - q4.away as q4_margin
        FROM matches m
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        WHERE m.date >= '2024-01-01' AND m.status_type = 'finished' AND q4.home != q4.away
    """)
    q4_map = {r[0]: float(r[1]) for r in cur.fetchall()}

    cur.execute("""
        SELECT ts.match_id, ts.stat_key,
               SUM(CASE WHEN ts.period IN ('Q1','Q2','1Q','2Q') THEN (ts.home_value - ts.away_value) ELSE 0 END) as ht_diff,
               SUM(CASE WHEN ts.period IN ('Q3','3Q') THEN (ts.home_value - ts.away_value) ELSE 0 END) as q3_diff
        FROM team_statistics ts
        JOIN matches m ON ts.match_id = m.match_id
        WHERE m.date >= '2024-01-01' AND ts.period IN ('Q1','Q2','Q3','1Q','2Q','3Q')
        GROUP BY ts.match_id, ts.stat_key
    """)
    stat_ht = defaultdict(list)
    stat_3q = defaultdict(list)
    for mid, sk, htd, q3d in cur.fetchall():
        q4m = q4_map.get(mid)
        if q4m is None or htd is None:
            continue
        stat_ht[sk].append((float(htd), q4m))
        stat_3q[sk].append((float(htd + (q3d or 0)), q4m))

    print("\n=== Correlación de Box Score acumulado (Q1+Q2 y Q1..Q3) con Q4 Margin ===", flush=True)
    rows = []
    for sk in stat_ht:
        pairs_ht = stat_ht[sk]
        pairs_3q = stat_3q[sk]
        if len(pairs_ht) < 1000:
            continue
        x1 = np.array([p[0] for p in pairs_ht])
        y1 = np.array([p[1] for p in pairs_ht])
        x3 = np.array([p[0] for p in pairs_3q])
        r_ht = float(np.corrcoef(x1, y1)[0, 1]) if np.std(x1) > 1e-9 else 0.0
        r_3q = float(np.corrcoef(x3, y1)[0, 1]) if np.std(x3) > 1e-9 else 0.0
        rows.append((sk, len(pairs_ht), r_ht, r_3q))
    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    for sk, n, r_ht, r_3q in rows:
        print(f"  {sk:<32} | N={n:>7,} | r(HT -> Q4)={r_ht:+.4f} | r(Q1..Q3 -> Q4)={r_3q:+.4f}", flush=True)

    conn.close()

if __name__ == "__main__":
    main()
