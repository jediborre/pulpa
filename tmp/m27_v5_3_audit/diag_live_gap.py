# =====================================================================
# ¿ES UN PROBLEMA DE m27_v5_3 O DEL ENTRENAMIENTO EN VIVO?
#   1) WR vivo acumulado por modelo (todas las apuestas resueltas)
#   2) WR vivo por dia de todos los modelos
#   3) Las ligas de las apuestas de hoy: WR historico del modelo en esa liga
#   4) Comparacion con la expectativa del backtest (0.75 -> 80.7 %)
# =====================================================================

import json
import math
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


def wilson(w, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


print("1) WR VIVO ACUMULADO POR MODELO (apuestas resueltas win/loss):")
for r in conn.execute("""
    SELECT model_version,
           SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w,
           COUNT(*) n
    FROM bet_monitor_log_v3
    WHERE signal_type LIKE 'BET%' AND result IN ('win','loss')
    GROUP BY model_version ORDER BY n DESC"""):
    lo, hi = wilson(r["w"], r["n"])
    print(f"   {r['model_version']:<12} n={r['n']:>4} | ganadas {r['w']:>3} | "
          f"WR {r['w'] * 100 / r['n']:>5.1f}% | IC95 [{lo * 100:>4.1f}, {hi * 100:>4.1f}]")

print()
print("2) WR VIVO POR DIA Y MODELO:")
for r in conn.execute("""
    SELECT substr(created_at,1,10) d, model_version,
           SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w, COUNT(*) n
    FROM bet_monitor_log_v3
    WHERE signal_type LIKE 'BET%' AND result IN ('win','loss')
    GROUP BY d, model_version ORDER BY d, n DESC"""):
    print(f"   {r['d']} | {r['model_version']:<12} n={r['n']:>3} | WR {r['w'] * 100 / r['n']:>5.1f}%")

print()
print("3) HOY: WR por modelo (apuestas resueltas):")
for r in conn.execute("""
    SELECT model_version, SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w, COUNT(*) n
    FROM bet_monitor_log_v3
    WHERE signal_type LIKE 'BET%' AND result IN ('win','loss')
      AND substr(created_at,1,10)='2026-10-20'
    GROUP BY model_version ORDER BY n DESC"""):
    lo, hi = wilson(r["w"], r["n"])
    print(f"   {r['model_version']:<12} n={r['n']:>3} | WR {r['w'] * 100 / r['n']:>5.1f}% "
          f"| IC95 [{lo * 100:>4.1f}, {hi * 100:>4.1f}]")

print()
print("4) LAS LIGAS DE LAS APUESTAS DE HOY SEGUN LA CALIBRACION DEL ENTRENAMIENTO:")
cal_path = ROOT / "models" / "m27_v5_3" / "model_outputs" / "league_calibration.json"
cal = {}
if cal_path.exists():
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    for c in data:
        cal[c.get("league", "")] = c
for r in conn.execute("""
    SELECT b.match_id, b.picked_side, b.confidence, b.result, b.created_at, m.league
    FROM bet_monitor_log_v3 b LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
    WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
      AND substr(b.created_at,1,10)='2026-10-20' ORDER BY b.id"""):
    lg = r["league"] or ""
    c = cal.get(lg)
    if c:
        print(f"   {r['created_at'][11:19]} | {lg[:34]:<34} | {r['picked_side']:<5} "
              f"conf {r['confidence']:.3f} | {r['result']:<5} | "
              f"WR historico de la liga {c.get('wr')}% (n={c.get('n')}) "
              f"CI {c.get('wr_ci95')}")
    else:
        print(f"   {r['created_at'][11:19]} | {lg[:34]:<34} | {r['picked_side']:<5} "
              f"conf {r['confidence']:.3f} | {r['result']:<5} | liga NO en la calibracion")

print()
print("5) CONFianza de las apuestas vivas vs la que el backtest predice para ese WR:")
tot_w = sum(r["w"] for r in conn.execute("""
    SELECT SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%' AND result IN ('win','loss')"""))
tot_n = conn.execute("""
    SELECT COUNT(*) n FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%' AND result IN ('win','loss')""").fetchone()["n"]
conf_med = conn.execute("""
    SELECT AVG(confidence) c FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%' AND result IN ('win','loss')""").fetchone()["c"]
print(f"   confianza media de las {tot_n} apuestas vivas: {conf_med:.4f}")
print(f"   WR realizado: {tot_w * 100 / tot_n:.1f}% | WR esperado por el backtest a esa confianza: ~80-86%")
print(f"   diferencia: {tot_w * 100 / tot_n - conf_med * 100:+.1f} pp")
conn.close()
