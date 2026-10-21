# =====================================================================
# INVENTARIO DE LA SERIE POR MINUTO (model_inference_v3)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) TOTAL Y RANGO:")
for r in conn.execute("""
    SELECT COUNT(*) n, COUNT(DISTINCT match_id) partidos,
           COUNT(DISTINCT model_version) modelos,
           MIN(game_minute) min_min, MAX(game_minute) max_min,
           MIN(created_at) desde, MAX(created_at) hasta
    FROM model_inference_v3"""):
    print("  ", dict(r))

print()
print("2) FILAS POR MODELO:")
for r in conn.execute("""
    SELECT model_version, COUNT(*) n, COUNT(DISTINCT match_id) partidos,
           MIN(game_minute) min_min, MAX(game_minute) max_min
    FROM model_inference_v3 GROUP BY model_version"""):
    print("  ", dict(r))

print()
print("3) FILAS POR MINUTO (game_minute):")
for r in conn.execute("""
    SELECT game_minute, COUNT(*) n, COUNT(DISTINCT match_id) partidos,
           COUNT(DISTINCT model_version) modelos
    FROM model_inference_v3 GROUP BY game_minute ORDER BY game_minute"""):
    print(f"   min {r['game_minute']:>5.1f} | {r['n']:>4} filas | {r['partidos']:>3} partidos | "
          f"{r['modelos']} modelos")

print()
print("4) ¿QUE PARTIDOS Y HAY RESULTADO DE Q4 PARA PODER EVALUAR?")
n_con_q4 = conn.execute("""
    SELECT COUNT(DISTINCT i.match_id) n FROM model_inference_v3 i
    JOIN quarter_scores_v2 q ON CAST(q.match_id AS INTEGER) = i.match_id
    WHERE (q.q4_home IS NOT NULL OR q.q4_away IS NOT NULL)""").fetchone()["n"]
n_total = conn.execute("SELECT COUNT(DISTINCT match_id) n FROM model_inference_v3").fetchone()["n"]
print(f"   partidos en la serie: {n_total} | con Q4 en quarter_scores_v2: {n_con_q4}")

print()
print("5) SERIE DE UN PARTIDO (ejemplo):")
mid = conn.execute("""
    SELECT match_id FROM model_inference_v3
    GROUP BY match_id HAVING COUNT(*) >= 3 ORDER BY match_id LIMIT 1""").fetchone()
if mid:
    for r in conn.execute("""
        SELECT * FROM model_inference_v3 WHERE match_id=? ORDER BY model_version, game_minute""",
            (mid["match_id"],)):
        print(f"   {r['model_version']:<12} min {r['game_minute']:>5.1f} | "
              f"p_cal {r['p_cal']:.4f} | conf {r['confidence']:.4f} | margen {r['margin']} | "
              f"pick {r['pick']} | signal {r['signal']} | snap {r['snapshot_minute']} | "
              f"block {r['block_reason']!r}")
    q = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id=?",
                     (mid["match_id"],)).fetchone()
    if q:
        print(f"   Q4 real: {q['q4_home']}-{q['q4_away']}")

print()
print("6) ¿SE PUEDE EVALUAR (picks distintos por minuto en el mismo partido)?")
for r in conn.execute("""
    SELECT model_version, COUNT(*) n FROM (
        SELECT model_version, match_id, COUNT(DISTINCT pick) p
        FROM model_inference_v3 GROUP BY model_version, match_id)
    WHERE p > 1 GROUP BY model_version"""):
    print("  ", dict(r))
conn.close()
