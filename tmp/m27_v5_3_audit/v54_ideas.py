# =====================================================================
# v5_4: cobertura REAL de cuartos y medición de las ideas
#   - La tabla legacy `quarter_scores` (250k+) es la que usa el entrenamiento.
#   - `quarter_scores_v2` (la "canónica" que escribe el monitor) está casi vacía.
# =====================================================================

import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


def one(sql, args=()):
    return conn.execute(sql, args).fetchone()[0]


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


print("COBERTURA DE LAS DOS TABLAS DE CUARTOS")
print(f"  quarter_scores   (legacy, la que usa el train): {one('SELECT COUNT(*) FROM quarter_scores'):,} filas")
print(f"  quarter_scores_v2 (canónica, la que escribe el monitor): {one('SELECT COUNT(*) FROM quarter_scores_v2'):,} filas")
print(f"  coincidencias entre ambas: {one('SELECT COUNT(*) FROM quarter_scores_v2 v JOIN quarter_scores q ON q.match_id=v.match_id'):,}")

sql_full = """
SELECT m.match_id, m.league, m.date,
  MAX(CASE WHEN q.quarter='Q1' THEN q.home END) q1h, MAX(CASE WHEN q.quarter='Q1' THEN q.away END) q1a,
  MAX(CASE WHEN q.quarter='Q2' THEN q.home END) q2h, MAX(CASE WHEN q.quarter='Q2' THEN q.away END) q2a,
  MAX(CASE WHEN q.quarter='Q3' THEN q.home END) q3h, MAX(CASE WHEN q.quarter='Q3' THEN q.away END) q3a,
  MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4h, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4a
FROM matches m JOIN quarter_scores q ON q.match_id = m.match_id
WHERE m.status_type='finished' AND m.date IS NOT NULL
GROUP BY m.match_id
"""
rows = [dict(r) for r in conn.execute(sql_full)]
print(f"\n  partidos terminados con alguna fila de cuartos: {len(rows):,}")
full = [r for r in rows if all(r[k] is not None for k in ("q1h", "q1a", "q2h", "q2a", "q3h", "q3a", "q4h", "q4a"))]
print(f"  con los 4 cuartos completos: {len(full):,}")
notie = [r for r in full if r["q1h"] != r["q1a"] and r["q2h"] != r["q2a"]
         and r["q3h"] != r["q3a"] and r["q4h"] != r["q4a"]]
print(f"  con los 4 cuartos completos y SIN empates: {len(notie):,}")

block("1. PATRÓN DE GANADORES DE Q1-Q3 -> QUIÉN GANA Q4")
pat = defaultdict(Counter)
for r in notie:
    p = "".join("H" if r[f"q{i}h"] > r[f"q{i}a"] else "A" for i in (1, 2, 3))
    pat[p]["H" if r["q4h"] > r["q4a"] else "A"] += 1
tot = len(notie)
print(f"  {'patron Q1-Q3':<14}{'n':>9}{'P(local gana Q4)':>18}{'margen medio Q4':>17}")
for p, c in sorted(pat.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(c.values())
    sel = [r for r in notie if "".join("H" if r[f"q{i}h"] > r[f"q{i}a"] else "A" for i in (1, 2, 3)) == p]
    m4 = sum(r["q4h"] - r["q4a"] for r in sel) / len(sel)
    print(f"  {p:<14}{n:>9,}{c['H'] * 100 / n:>17.1f}%{m4:>17.2f}")
base = sum(c["H"] for c in pat.values()) / tot
print(f"\n  base (local gana Q4): {base * 100:.1f}%")

for name, cond in (
    ("local barrete 3-0", lambda r: all(r[f"q{i}h"] > r[f"q{i}a"] for i in (1, 2, 3))),
    ("visitante barrete 0-3", lambda r: all(r[f"q{i}h"] < r[f"q{i}a"] for i in (1, 2, 3))),
    ("mixto (1-2 o 2-1)", lambda r: sum(1 for i in (1, 2, 3) if r[f"q{i}h"] > r[f"q{i}a"]) in (1, 2)),
):
    sel = [r for r in notie if cond(r)]
    if not sel:
        continue
    ph = sum(1 for r in sel if r["q4h"] > r["q4a"]) / len(sel)
    m4 = sum(r["q4h"] - r["q4a"] for r in sel) / len(sel)
    print(f"  {name:<24} n={len(sel):>7,}  P(local gana Q4)={ph * 100:>5.1f}%  margen medio Q4={m4:>+6.2f}")

block("2. EFECTO 'BASURA': MARGEN DE Q4 SEGÚN LA VENTAJA AL FINAL DE Q3")
buckets = [(-999, -21), (-21, -15), (-15, -9), (-9, -3), (-3, 4), (4, 10),
           (10, 16), (16, 22), (22, 999)]
print(f"  {'ventaja local al final de Q3':<32}{'n':>9}{'margen medio Q4':>17}{'P(local gana Q4)':>19}")
for lo, hi in buckets:
    sel = [r for r in notie if lo <= (r["q1h"] + r["q2h"] + r["q3h"] - r["q1a"] - r["q2a"] - r["q3a"]) < hi]
    if len(sel) < 100:
        continue
    m4 = sum(r["q4h"] - r["q4a"] for r in sel) / len(sel)
    ph = sum(1 for r in sel if r["q4h"] > r["q4a"]) / len(sel)
    label = ("<= -21" if hi == -21 else ">= 22" if lo == 22 else f"{lo}..{hi - 1}")
    print(f"  {label:<32}{len(sel):>9,}{m4:>17.2f}{ph * 100:>18.1f}%")

block("3. ¿QUÉ PARTE DE ESTO CAPTURA YA EL MODELO? (features de diferencias)")
ph4 = [1.0 if r["q4h"] > r["q4a"] else 0.0 for r in notie]
feats = {
    "q1_diff": [r["q1h"] - r["q1a"] for r in notie],
    "cur_diff_Q3": [r["q1h"] + r["q2h"] + r["q3h"] - r["q1a"] - r["q2a"] - r["q3a"] for r in notie],
    "n_quarters_local": [sum(1 for i in (1, 2, 3) if r[f"q{i}h"] > r[f"q{i}a"]) for r in notie],
    "sweep_local": [1.0 if all(r[f"q{i}h"] > r[f"q{i}a"] for i in (1, 2, 3)) else 0.0 for r in notie],
    "local_lost_any": [1.0 if any(r[f"q{i}h"] < r[f"q{i}a"] for i in (1, 2, 3)) else 0.0 for r in notie],
}


def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


print(f"  {'feature candidata':<22}{'corr con P(local gana Q4)':>28}")
for k, v in feats.items():
    print(f"  {k:<22}{corr(v, ph4):>28.4f}")
print()
print("  (P(local gana Q4) global: " + f"{sum(ph4) / len(ph4) * 100:.1f}%)")
conn.close()
