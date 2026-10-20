# =====================================================================
# ¿EL MERCADO DE "1st quarter winner" PREDICE EL GANADOR DE Q4?
# ---------------------------------------------------------------------
# La API no publica el mercado de "4th quarter" en pre-partido, pero si
# el de "1st quarter winner" (15.546 partidos, todos de LMB). Si la
# probabilidad implicita de ese mercado (cercana a un cuarto) informa sobre
# quien gana un cuarto, entonces un mercado de Q4 seria oro como feature.
# Esta es la prueba de viabilidad del proxy.
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


def dec(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def corr(a, b):
    n = len(a)
    if n < 10:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


print("1) ¿QUE HAY EN match_odds PARA EL MERCADO POR PERIODO? (columnas disponibles)")
cols = [r["name"] for r in conn.execute("PRAGMA table_info(match_odds)")]
print("   columnas:", cols)
for r in conn.execute(
    "SELECT odds_type, market_period, market_name, COUNT(*) n FROM match_odds "
    "WHERE market_period='1st quarter' GROUP BY odds_type, market_period, market_name"
):
    print(f"   {r['odds_type']:<14} {r['market_period']:<14} {r['market_name']:<22} {r['n']:>7,}")

print()
print("2) ¿CASA EL MERCADO '1st quarter' CON EL GANADOR REAL DE Q1 Y DE Q4?")
rows = [dict(r) for r in conn.execute(
    """
    SELECT o.match_id, o.home_value, o.away_value, o.odds_type,
           q.q1_home, q.q1_away, q.q4_home, q.q4_away
    FROM match_odds o JOIN quarter_scores_v2 q ON q.match_id = o.match_id
    WHERE o.market_period = '1st quarter' AND o.market_name = '1st quarter winner'
      AND q.q1_home IS NOT NULL AND q.q4_home IS NOT NULL
      AND q.q1_home <> q.q1_away AND q.q4_home <> q.q4_away
    """)]
print(f"   partidos con mercado de 1st quarter y cuartos reales: {len(rows):,}")

p_q1_mkt, y_q1, p_q4, y_q4 = [], [], [], []
for r in rows:
    h, a = dec(r["home_value"]), dec(r["away_value"])
    if not h or not a or h <= 1 or a <= 1:
        continue
    p_home_q1 = (1 / h) / ((1 / h) + (1 / a))
    p_q1_mkt.append(p_home_q1)
    y_q1.append(1.0 if r["q1_home"] > r["q1_away"] else 0.0)
    y_q4.append(1.0 if r["q4_home"] > r["q4_away"] else 0.0)
    p_q4.append(p_home_q1)

print(f"   n utilizables: {len(p_q1_mkt):,}")
if len(p_q1_mkt) > 50:
    c1 = corr(p_q1_mkt, y_q1)
    c4 = corr(p_q1_mkt, y_q4)
    print(f"   corr(mercado 1st quarter, resultado real de Q1) = {c1:+.4f}")
    print(f"   corr(mercado 1st quarter, resultado real de Q4) = {c4:+.4f}")
    print()
    # ¿El mercado del 1st quarter predeciria el Q4 mejor que el propio Q1?
    q1_real = y_q1
    print(f"   ¿mercado 1st quarter vs Q1 real?         {c1:+.4f}")
    print(f"   ¿Q1 real vs Q4 real?                     {corr(q1_real, y_q4):+.4f}  "
          f"(la correlacion 'natural' entre cuartos)")
    print(f"   ¿mercado 1st quarter vs Q4 real?         {c4:+.4f}")
    print()
    # Calibracion: como de bien calibrado esta el mercado para el Q4
    import numpy as np
    p = np.array(p_q4)
    yy = np.array(y_q4)
    for lo, hi in ((0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        print(f"   p_mercado {lo:.2f}-{hi:.2f}: n={int(m.sum()):>6,}  "
              f"P(local gana Q4) real = {yy[m].mean() * 100:>5.1f} %  "
              f"(mercado decia {p[m].mean() * 100:.1f} %)")
conn.close()
