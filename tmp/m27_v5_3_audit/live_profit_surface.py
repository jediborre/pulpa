# =====================================================================
# CURVA DE WR VIVO POR UMBRAL (no hace falta esperar a las apuestas)
#   Sale de model_inference_v3 (una fila por partido y minuto) unida al
#   resultado real del Q4. Se toma UNA fila por partido (el ultimo minuto
#   valido <= 30) para no contar dos veces el mismo partido.
# =====================================================================

import sqlite3
import sys
from pathlib import Path

import numpy as np

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
    return (max(0.0, c - h), min(1.0, c + h))


import math  # noqa: E402

for MODELO in ("m27_v5_3", "m27_v5_2"):
    print("=" * 96)
    print(f"SUPERFICIE DE BENEFICIO VIVA — {MODELO}")
    print("=" * 96)
    # una fila por partido: el ultimo minuto <= 30
    filas = conn.execute("""
        SELECT i.match_id, i.game_minute, i.p_cal, i.pick, i.confidence, i.margin,
               q.q4_home, q.q4_away
        FROM model_inference_v3 i
        JOIN quarter_scores_v2 q ON CAST(q.match_id AS INTEGER) = i.match_id
        WHERE i.model_version = ? AND i.game_minute <= 30
          AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL
        ORDER BY i.match_id, i.game_minute""", (MODELO,)).fetchall()
    por_partido = {}
    for r in filas:
        por_partido[r["match_id"]] = r  # el ultimo minuto <= 30 gana
    print(f"   partidos evaluables: {len(por_partido)}")
    print()
    print(f"   {'umbral':>8}{'apuestas':>10}{'WR':>9}{'IC95':>18}{'u@1.40':>10}"
          f"{'ROI140':>9}{'u@1.83':>10}{'ROI183':>9}")
    for t in (0.65, 0.68, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.85):
        sel = [r for r in por_partido.values()
               if r["pick"] in ("HOME", "AWAY") and r["confidence"] >= t]
        w = 0
        for r in sel:
            dif = (r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME" else (r["q4_away"] - r["q4_home"])
            if dif != 0:
                w += dif > 0
        n = len(sel)
        if n == 0:
            print(f"   {t:>8.2f}{0:>10}")
            continue
        wr = w / n
        lo, hi = wilson(w, n)
        u140 = w * 1.40 - n
        u183 = w * 1.83 - n
        print(f"   {t:>8.2f}{n:>10}{wr * 100:>8.1f}%   [{lo * 100:>4.1f}, {hi * 100:>4.1f}]"
              f"{u140:>10.0f}{u140 / n * 100:>8.2f}%{u183:>10.0f}{u183 / n * 100:>8.2f}%")

    print()
    print("   COMPARADO CON EL BACKTEST (bloque C, ancla t=30):")
    back = {0.70: (5458, 76.00), 0.72: (4207, 77.99), 0.75: (2720, 80.77),
            0.80: (1300, 86.15)}
    for t, (n_b, wr_b) in back.items():
        sel = [r for r in por_partido.values()
               if r["pick"] in ("HOME", "AWAY") and r["confidence"] >= t]
        w = sum(1 for r in sel
                if ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
                    else (r["q4_away"] - r["q4_home"])) != 0
                and ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
                     else (r["q4_away"] - r["q4_home"])) > 0)
        wr_v = (w / len(sel) * 100) if sel else 0
        print(f"      thr {t:.2f}: backtest n={n_b:>6,} WR {wr_b:>5.1f}% | "
              f"VIVO n={len(sel):>4} WR {wr_v:>5.1f}% | gap {wr_v - wr_b:>+5.1f} pp")
    print()
conn.close()
