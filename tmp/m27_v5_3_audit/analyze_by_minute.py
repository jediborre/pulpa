# =====================================================================
# ANALISIS POR MINUTO: ¿en que minuto es mas fiable el modelo?
#   Se une model_inference_v3 con el resultado real del Q4 y se calcula,
#   por minuto: acierto del pick, calibracion (p_cal vs real) y lo que
#   habria pasado con el umbral operativo 0.75.
# =====================================================================

import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

UMBRAL = 0.75

print("SERIE POR MINUTO (con el resultado real del Q4)")
print()
for modelo in ("m27_v5_3", "m27_v5_2", "m27_v3", "v6_2"):
    filas = conn.execute("""
        SELECT i.game_minute, i.p_home, i.p_away, i.p_cal, i.confidence, i.pick,
               i.margin, q.q4_home, q.q4_away
        FROM model_inference_v3 i
        JOIN quarter_scores_v2 q ON CAST(q.match_id AS INTEGER) = i.match_id
        WHERE i.model_version = ?
        ORDER BY i.game_minute""", (modelo,)).fetchall()
    if not filas:
        continue
    print(f"═══ {modelo}  ({len(filas)} filas con Q4) ═══")
    print(f"   {'min':>5}{'n':>6}{'WR pick':>9}{'p_cal med':>11}{'gap':>8}"
          f"{'conf med':>10}{'>=umbral':>10}{'WR umbral':>11}{'Brier':>9}")
    minutos = sorted({r["game_minute"] for r in filas})
    for mn in minutos:
        sel = [r for r in filas if r["game_minute"] == mn]
        w = n = 0
        pcal = []
        confs = []
        brier = []
        umbral_rows = []
        for r in sel:
            if r["q4_home"] is None or r["q4_away"] is None:
                continue
            pick = str(r["pick"] or "").upper()
            if pick not in ("HOME", "AWAY"):
                continue
            dif = (r["q4_home"] - r["q4_away"]) if pick == "HOME" else (r["q4_away"] - r["q4_home"])
            if dif == 0:
                continue  # push: no computa acierto
            gano = dif > 0
            w += gano
            n += 1
            pc = float(r["p_cal"] or 0.5)
            pcal.append(pc if pick == "HOME" else 1 - pc)
            confs.append(float(r["confidence"] or 0))
            brier.append((pc - (1.0 if gano else 0.0)) ** 2)
            if float(r["confidence"] or 0) >= UMBRAL:
                umbral_rows.append(gano)
        if n == 0:
            print(f"   {mn:>5.1f}{0:>6}   sin datos")
            continue
        confs_arr = np.array(confs)
        pcal_arr = np.array(pcal)
        sel_umb = confs_arr >= UMBRAL
        wr_u = (f"{np.mean(umbral_rows) * 100:>10.1f}%" if umbral_rows else f"{'-':>10}")
        gap = (pcal_arr.mean() - w / n) * 100
        print(f"   {mn:>5.1f}{n:>6}{w / n * 100:>8.1f}%{pcal_arr.mean():>11.4f}"
              f"{gap:>+7.1f}pp{confs_arr.mean():>10.3f}{int(sel_umb.sum()):>10}"
              f"{wr_u}{np.mean(brier):>9.4f}")
    print()

print("═══ ¿CUANTO CAMBIA LA PREDICCION DENTRO DE LA VENTANA? (m27_v5_3) ═══")
for r in conn.execute("""
    SELECT match_id, COUNT(DISTINCT ROUND(p_cal, 4)) variantes,
           MIN(p_cal) pmin, MAX(p_cal) pmax, MIN(pick) pk
    FROM model_inference_v3 WHERE model_version='m27_v5_3'
    GROUP BY match_id"""):
    pass
res = conn.execute("""
    SELECT match_id, COUNT(DISTINCT ROUND(p_cal, 4)) variantes,
           MIN(p_cal) pmin, MAX(p_cal) pmax,
           COUNT(DISTINCT pick) picks
    FROM model_inference_v3 WHERE model_version='m27_v5_3' GROUP BY match_id""").fetchall()
n_multi = sum(1 for r in res if r["variantes"] > 1)
n_pick = sum(1 for r in res if r["picks"] > 1)
delta = [(r["pmax"] - r["pmin"]) for r in res]
print(f"   partidos con mas de un valor de p_cal: {n_multi}/{len(res)}")
print(f"   partidos donde el PICK cambia entre minutos: {n_pick}/{len(res)}")
print(f"   recorrido tipico de p_cal dentro de la ventana: {np.median(delta):.4f} "
      f"(mediana) | max {np.max(delta):.4f}")

print()
print("═══ DISPONIBILIDAD: ¿a que minuto se decide la apuesta? ═══")
for r in conn.execute("""
    SELECT i.game_minute, COUNT(*) n,
           SUM(CASE WHEN i.signal LIKE 'BET%' AND i.signal NOT LIKE 'NO_BET%' THEN 1 ELSE 0 END) bets,
           SUM(CASE WHEN i.signal='PASS' THEN 1 ELSE 0 END) pass
    FROM model_inference_v3 i WHERE i.model_version='m27_v5_3'
    GROUP BY i.game_minute ORDER BY i.game_minute"""):
    print(f"   min {r['game_minute']:>5.1f} | filas {r['n']:>4} | señales BET {r['bets']:>3} | "
          f"PASS {r['pass']:>4}")
conn.close()
