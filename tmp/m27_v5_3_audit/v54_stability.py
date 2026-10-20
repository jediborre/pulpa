# =====================================================================
# v5_4: ¿los efectos son estables en el tiempo y sobreviven al control
#        por liga? (si no, son ruido y no sirven como feature)
# =====================================================================

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

SQL = """
SELECT m.match_id, m.league, m.date,
  MAX(CASE WHEN q.quarter='Q1' THEN q.home END) q1h, MAX(CASE WHEN q.quarter='Q1' THEN q.away END) q1a,
  MAX(CASE WHEN q.quarter='Q2' THEN q.home END) q2h, MAX(CASE WHEN q.quarter='Q2' THEN q.away END) q2a,
  MAX(CASE WHEN q.quarter='Q3' THEN q.home END) q3h, MAX(CASE WHEN q.quarter='Q3' THEN q.away END) q3a,
  MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4h, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4a
FROM matches m JOIN quarter_scores q ON q.match_id = m.match_id
WHERE m.status_type='finished' AND m.date IS NOT NULL
GROUP BY m.match_id
"""
rows = []
for r in conn.execute(SQL):
    d = dict(r)
    if any(d[k] is None for k in ("q1h", "q1a", "q2h", "q2a", "q3h", "q3a", "q4h", "q4a")):
        continue
    if d["q1h"] == d["q1a"] or d["q2h"] == d["q2a"] or d["q3h"] == d["q3a"] or d["q4h"] == d["q4a"]:
        continue
    d["year"] = int(str(d["date"])[:4])
    d["lead3"] = d["q1h"] + d["q2h"] + d["q3h"] - d["q1a"] - d["q2a"] - d["q3a"]
    d["home_q4"] = 1 if d["q4h"] > d["q4a"] else 0
    d["sweep"] = 1 if (d["q1h"] > d["q1a"] and d["q2h"] > d["q2a"] and d["q3h"] > d["q3a"]) else (
        -1 if (d["q1h"] < d["q1a"] and d["q2h"] < d["q2a"] and d["q3h"] < d["q3a"]) else 0)
    d["extreme"] = 1 if d["lead3"] >= 22 else (-1 if d["lead3"] <= -21 else 0)
    rows.append(d)
print(f"muestra: {len(rows):,} partidos (2015-2026)")


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


def rate(sel, key):
    if not sel:
        return None
    return sum(r["home_q4"] for r in sel) / len(sel)


block("1. ESTABILIDAD EN EL TIEMPO: efecto barrido (HHH vs AAA)")
print(f"  {'periodo':<16}{'n HHH':>9}{'P(local gana Q4) HHH':>24}{'n AAA':>9}{'P(local gana Q4) AAA':>24}{'base':>8}")
for name, pred in (("2015-2018", lambda r: r["year"] <= 2018),
                   ("2019-2022", lambda r: 2019 <= r["year"] <= 2022),
                   ("2023-2024", lambda r: 2023 <= r["year"] <= 2024),
                   ("2025-2026", lambda r: r["year"] >= 2025)):
    sel = [r for r in rows if pred(r)]
    hhh = [r for r in sel if r["sweep"] == 1]
    aaa = [r for r in sel if r["sweep"] == -1]
    b = rate(sel, "home_q4")
    print(f"  {name:<16}{len(hhh):>9,}{rate(hhh, 'home_q4') * 100:>23.1f}%{len(aaa):>9,}"
          f"{rate(aaa, 'home_q4') * 100:>23.1f}%{b * 100:>7.1f}%")

block("2. ESTABILIDAD: efecto ventaja extrema al final de Q3")
print(f"  {'periodo':<16}{'n local+22':>13}{'P(local Q4)':>14}{'n local-21':>13}{'P(local Q4)':>14}{'base':>8}")
for name, pred in (("2015-2018", lambda r: r["year"] <= 2018),
                   ("2019-2022", lambda r: 2019 <= r["year"] <= 2022),
                   ("2023-2024", lambda r: 2023 <= r["year"] <= 2024),
                   ("2025-2026", lambda r: r["year"] >= 2025)):
    sel = [r for r in rows if pred(r)]
    up = [r for r in sel if r["extreme"] == 1]
    dn = [r for r in sel if r["extreme"] == -1]
    print(f"  {name:<16}{len(up):>13,}{rate(up, 'home_q4') * 100:>13.1f}%{len(dn):>13,}"
          f"{rate(dn, 'home_q4') * 100:>13.1f}%{rate(sel, 'home_q4') * 100:>7.1f}%")

block("3. ¿SOBREVIVE AL CONTROL POR LIGA? (efecto dentro de cada liga)")
by_league = defaultdict(list)
for r in rows:
    by_league[r["league"]].append(r)
sizes = []
for lg, sel in by_league.items():
    hhh = [r for r in sel if r["sweep"] == 1]
    aaa = [r for r in sel if r["sweep"] == -1]
    if len(hhh) < 60 or len(aaa) < 60:
        continue
    sizes.append((rate(hhh, "home_q4") - rate(sel, "home_q4"),
                  rate(sel, "home_q4") - rate(aaa, "home_q4"),
                  lg, len(sel), rate(sel, "home_q4")))
sizes.sort(reverse=True)
print(f"  ligas con muestra suficiente: {len(sizes)}")
if sizes:
    up = [s[0] for s in sizes]
    dn = [s[1] for s in sizes]
    print(f"  efecto dentro de la liga (HHH - base): mediana {sorted(up)[len(up) // 2] * 100:+.1f} pp, "
          f"positivo en {sum(1 for x in up if x > 0)}/{len(up)} ligas")
    print(f"  efecto dentro de la liga (base - AAA): mediana {sorted(dn)[len(dn) // 2] * 100:+.1f} pp, "
          f"positivo en {sum(1 for x in dn if x > 0)}/{len(dn)} ligas")
    print(f"\n  las 12 ligas con MAS efecto (barrido local):")
    for d, _, lg, n, b in sizes[:12]:
        print(f"    {lg[:52]:<54}n={n:>6,}  base={b * 100:>5.1f}%  efecto HHH={d * 100:>+5.1f} pp")
    print(f"\n  las 12 ligas con MENOS efecto (o contrario):")
    for d, _, lg, n, b in sizes[-12:]:
        print(f"    {lg[:52]:<54}n={n:>6,}  base={b * 100:>5.1f}%  efecto HHH={d * 100:>+5.1f} pp")

block("4. ¿Y LA VARIANTE POR LIGA (prior contextual del Q4 segun ventaja)?")
print("  Si el efecto varies por liga, la feature util es 'PRIOR DEL Q4 DADO EL")
print("  ESTADO' (una tabla por liga x bucket de ventaja), no un flag global.")
print("  Fraccion de ligas donde el efecto del barrido cambia de signo: "
      f"{sum(1 for d, _, _, _, _ in sizes if d < 0)}/{len(sizes)}")
conn.close()
