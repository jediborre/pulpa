# =====================================================================
# SUPERFICIE DE BENEFICIO POR UMBRAL Y CUOTA (m27_v5_3, test C)
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   El pipeline eligio min_conf=0.72 maximizando BENEFICIO ABSOLUTO a cuota
#   1.40 en el bloque B (train_m27_v5_3_pipeline.py:236):
#       profit = w * 1.40 - n
#   Eso premia el volumen: un umbral mas bajo mete mas apuestas y gana mas
#   unidades esperadas, aunque el acierto caiga. Como la cuota REAL del
#   mercado Q4 todavia no se registra, no se sabe si 1.40 es el supuesto
#   correcto. Este script muestra como se mueve el umbral optimo segun la
#   cuota, y si la eleccion de 0.72 es realmente superior a 0.75 o solo
#   esta empatada dentro del ruido.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/profit_surface.py
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json"
ODDS = (1.40, 1.83)


def block(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    op = s["operating"]
    print(f"Umbral operativo en produccion: min_conf={op['thr']}  min_margin={op['min_margin']}")
    print(f"  (elegido en B maximizando profit = w*1.40 - n, pipeline linea 236)")

    for key, label in (("tiers_all_rows", "TODAS las filas de C (multi-snapshot)"),
                       ("tiers_anchor", "SOLO ANCLA t=30 (comparable con v5_2)")):
        tiers = s[key]
        block(f"SUPERFICIE — {label}")
        for odds in ODDS:
            print(f"\n  Cuota {odds:.2f}  (equilibrio WR = {100 / odds:.1f}%)")
            print(f"  {'thr':>5} {'apuestas':>9} {'WR':>7} {'ROI':>8} "
                  f"{'beneficio esp.':>15} {'vs 0.72':>9} {'volumen':>9}")
            base = None
            rows = []
            for t in tiers:
                n, wr = t["bets"], t["wr"] / 100.0
                profit = n * (odds * wr - 1.0)
                rows.append((t["thr"], n, wr, profit, t["roi_140"], t["roi_183"]))
            for thr, n, wr, profit, r140, r183 in rows:
                if base is None and abs(thr - op["thr"]) < 1e-9:
                    base = profit
            for thr, n, wr, profit, r140, r183 in rows:
                delta = "" if base is None else f"{(profit / base - 1) * 100:+8.1f}%"
                mark = "  <-- operativo" if abs(thr - op["thr"]) < 1e-9 else ""
                print(f"  {thr:>5.2f} {n:>9,} {wr * 100:>6.2f}% "
                      f"{r140:>+7.2f}% {profit:>15,.0f} {delta:>9} {n:>9,}{mark}")
            best = max(rows, key=lambda r: r[3])
            best75 = [r for r in rows if abs(r[0] - 0.75) < 1e-9]
            if best75:
                b75 = best75[0]
                print(f"\n  -> optimo por beneficio absoluto a {odds:.2f}: thr={best[0]:.2f} "
                      f"({best[3]:,.0f} u, {best[1]:,} apuestas, WR {best[2] * 100:.2f}%)")
                print(f"  -> thr=0.75: {b75[3]:,.0f} u ({b75[3] / best[3] * 100 - 100:+.1f}% vs optimo), "
                      f"{b75[1]:,} apuestas ({b75[1] / best[1] * 100 - 100:+.1f}% volumen), "
                      f"WR {b75[2] * 100:.2f}% vs {best[2] * 100:.2f}%")

    block("LECTURA")
    print("  1) El umbral optimo depende de la cuota. A 1.40 manda 0.72-0.75; a 1.83")
    print("     mandan los umbrales bajos (mas volumen). Elegir 0.72 sin conocer la cuota")
    print("     real del Q4 es apostar a que el mercado paga ~1.40.")
    print("  2) 0.72 y 0.75 estan empatados en beneficio esperado a 1.40 (diferencia <2 %),")
    print("     pero 0.75 hace un 20 % menos apuestas con 1.5 pp mas de acierto: menos")
    print("     varianza y menos exposicion a los errores de las bandas marginales.")
    print("  3) El umbral NO se puede elegir de forma permanente hasta registrar la cuota")
    print("     real del Q4 (ya se guarda odds_live en live_snapshots_v3) y pasar a EV.")


if __name__ == "__main__":
    main()
