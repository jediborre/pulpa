# =====================================================================
# RESUMEN COMPLETO DE m27_v5_3 EN VIVO (todos los dias)
# =====================================================================

import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
HOY = "2026-10-21"


def wilson(w, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def linea_bets(n, w, extra=""):
    if n == 0:
        return f"      sin apuestas{extra}"
    lo, hi = wilson(w, n)
    return (f"      {n:>3} apuestas | {w} ganadas | WR {w * 100 / n:>5.1f}% "
            f"(IC95 {lo * 100:>4.1f}-{hi * 100:>4.1f}) | "
            f"u@1.40 {w * 1.40 - n:>+5.0f} | u@1.83 {w * 1.83 - n:>+5.0f}{extra}")


print("=" * 96)
print("1) APUESTAS REALES POR DIA (signal_type BET, resueltas)")
print("=" * 96)
for r in conn.execute("""
    SELECT substr(created_at,1,10) d, COUNT(*) n,
           SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w,
           SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) l,
           AVG(confidence) c
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')
    GROUP BY d ORDER BY d"""):
    lo, hi = wilson(r["w"], r["n"])
    print(f"   {r['d']}  {linea_bets(r['n'], r['w'], ' | conf media %.3f' % r['c'])}")

tot = conn.execute("""
    SELECT COUNT(*) n, SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')""").fetchone()
lo, hi = wilson(tot["w"], tot["n"])
print()
print(f"   ACUMULADO {linea_bets(tot['n'], tot['w'])}")
print(f"   esperado por el backtest a 0.75: WR 80.7 % | z = "
      f"{(tot['w'] / tot['n'] - 0.807) / math.sqrt(0.807 * 0.193 / tot['n']):+.2f}")

print()
print("=" * 96)
print("2) POR BANDA DE CONFIANZA (acumulado)")
print("=" * 96)
for r in conn.execute("""
    SELECT CASE WHEN confidence<0.80 THEN '0.75-0.80'
                WHEN confidence<0.85 THEN '0.80-0.85' ELSE '0.85+' END banda,
           COUNT(*) n, SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w,
           AVG(confidence) c
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')
    GROUP BY banda ORDER BY banda"""):
    print(f"   {r['banda']:<12}{linea_bets(r['n'], r['w'], ' | conf media %.3f' % r['c'])}")

print()
print("=" * 96)
print("3) SERIE POR MINUTO (model_inference_v3, con el Q4 real)")
print("=" * 96)
filas = conn.execute("""
    SELECT i.match_id, i.game_minute, i.pick, i.confidence, i.p_cal,
           q.q4_home, q.q4_away
    FROM model_inference_v3 i JOIN quarter_scores_v2 q
      ON CAST(q.match_id AS INTEGER)=i.match_id
    WHERE i.model_version='m27_v5_3' AND q.q4_home IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()
por_min = defaultdict(list)
for r in filas:
    dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
           else (r["q4_away"] - r["q4_home"]))
    por_min[r["game_minute"]].append((dif, r["confidence"], r["p_cal"]))
for mn in sorted(por_min):
    sel = por_min[mn]
    w = sum(1 for d, _c, _p in sel if d > 0)
    n = sum(1 for d, _c, _p in sel if d != 0)
    conf = np.mean([c for _d, c, _p in sel])
    pcal = np.mean([p for _d, _c, p in sel])
    if n == 0:
        continue
    print(f"   min {mn:>5.1f} | {linea_bets(n, w, f' | conf {conf:.3f} | p_cal {pcal:.3f}')}")

print()
print("=" * 96)
print("4) CURVA DE UMBRAL VIVA (todas las predicciones, una por partido)")
print("=" * 96)
mejor = {}
for r in filas:
    m = str(r["match_id"])
    if m not in mejor or r["game_minute"] > mejor[m][0]:
        mejor[m] = (r["game_minute"], r)
print(f"   partidos evaluados: {len(mejor)}")
for t in (0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80):
    sel = [r for _m, (_g, r) in mejor.items()
           if r["pick"] in ("HOME", "AWAY") and r["confidence"] >= t]
    w = n = 0
    for r in sel:
        dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
               else (r["q4_away"] - r["q4_home"]))
        if dif != 0:
            n += 1
            w += dif > 0
    if n == 0:
        print(f"   {t:.2f} | sin predicciones")
        continue
    lo, hi = wilson(w, n)
    print(f"   {t:.2f} | {n:>4} | WR {w * 100 / n:>5.1f}% (IC95 {lo * 100:>4.1f}-{hi * 100:>4.1f})"
          f" | u@1.40 {w * 1.40 - n:>+5.0f} | u@1.83 {w * 1.83 - n:>+5.0f}"
          f" | equilibrio 1.40 {'OK' if w / n >= 1 / 1.40 else 'NO'}")

print()
print("=" * 96)
print("5) POR LIGA (apuestas reales, acumulado)")
print("=" * 96)
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}
por_liga = defaultdict(lambda: [0, 0])
for r in conn.execute("""
    SELECT match_id, result FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')"""):
    lg = liga_de.get(str(r["match_id"]), "?")
    por_liga[lg][0] += 1
    por_liga[lg][1] += r["result"] == "win"
for lg, (n, w) in sorted(por_liga.items(), key=lambda kv: -kv[1][0]):
    print(f"   {str(lg)[:48]:<50}{linea_bets(n, w)}")

print()
print("=" * 96)
print("6) COMPARATIVA CON m27_v5_2 EN LOS MISMOS PARTIDOS")
print("=" * 96)
bets = {}
for r in conn.execute("""
    SELECT match_id, model_version, result, confidence, picked_side, signal_type
    FROM bet_monitor_log_v3
    WHERE model_version IN ('m27_v5_3','m27_v5_2') AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')"""):
    bets.setdefault(str(r["match_id"]), {})[r["model_version"]] = r
comun = [(mid, d) for mid, d in bets.items()
         if "m27_v5_3" in d and "m27_v5_2" in d]
if comun:
    w3 = sum(1 for _m, d in comun if d["m27_v5_3"]["result"] == "win")
    w2 = sum(1 for _m, d in comun if d["m27_v5_2"]["result"] == "win")
    print(f"   partidos con apuesta de los dos: {len(comun)}")
    print(f"      m27_v5_3: {w3}/{len(comun)} = {w3 * 100 / len(comun):.1f} %")
    print(f"      m27_v5_2: {w2}/{len(comun)} = {w2 * 100 / len(comun):.1f} %")
    mismos = sum(1 for _m, d in comun
                 if (d["m27_v5_3"]["picked_side"] == d["m27_v5_2"]["picked_side"]))
    print(f"      eligieron el mismo lado: {mismos}/{len(comun)}")
else:
    print("   sin partidos comunes con apuesta de los dos")

print()
print("=" * 96)
print("7) DIAS EN LOS QUE EL MODELO VIO PARTIDOS (cobertura)")
print("=" * 96)
for r in conn.execute("""
    SELECT substr(created_at,1,10) d, COUNT(DISTINCT match_id) partidos,
           COUNT(*) inferencias
    FROM model_inference_v3 WHERE model_version='m27_v5_3'
    GROUP BY d ORDER BY d"""):
    print(f"   {r['d']}: {r['partidos']:>3} partidos | {r['inferencias']:>4} inferencias")
conn.close()
