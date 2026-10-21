# =====================================================================
# HERRAMIENTA: curva de WR vivo por umbral (capa de analisis)
# ---------------------------------------------------------------------
# Uso:
#   .venv\Scripts\python.exe tools\live_threshold_curve.py
#   .venv\Scripts\python.exe tools\live_threshold_curve.py --json
#   .venv\Scripts\python.exe tools\live_threshold_curve.py --model m27_v5_2
#
# Lee la serie por minuto (model_inference_v3) y la une con el resultado real
# del Q4 (quarter_scores_v2) para calcular el WR y el beneficio en vivo para
# CADA umbral. No hace falta que el modelo haya apostado en ese umbral.
# =====================================================================

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitor_v3.database.connection import get_db_connection  # noqa: E402
from monitor_v3.models.live_threshold_curve import (  # noqa: E402
    DEFAULT_ODDS, DEFAULT_THRESHOLDS, compute_curve, fetch_live_predictions,
)

# Referencia del backtest (bloque C, ancla t=30) del modelo desplegado, para
# contrastar la curva viva. Se lee del resumen que escribe el pipeline.
BACKTEST_REF = {
    "m27_v5_3": {0.65: (9725, 71.58), 0.68: (6964, 74.28), 0.70: (5458, 76.00),
                 0.72: (4207, 77.99), 0.75: (2720, 80.77), 0.78: (1793, 83.77),
                 0.80: (1300, 86.15)},
    "m27_v5_2": {0.65: (9725, 71.58), 0.68: (6964, 74.28), 0.70: (5458, 76.00),
                 0.72: (4207, 77.99), 0.75: (2720, 80.77), 0.78: (1793, 83.77),
                 0.80: (1300, 86.15)},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m27_v5_3")
    ap.add_argument("--json", action="store_true", help="salida en JSON")
    ap.add_argument("--odds", default="1.40,1.83")
    args = ap.parse_args()

    odds = tuple(float(x) for x in args.odds.split(","))
    conn = get_db_connection()
    try:
        preds = fetch_live_predictions(conn, args.model)
    finally:
        conn.close()

    data = compute_curve(preds, thresholds=DEFAULT_THRESHOLDS, odds=odds)
    data["model"] = args.model

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

    ref = BACKTEST_REF.get(args.model, {})
    print("=" * 100)
    print(f"CURVA DE WR VIVO — {args.model}   (partidos evaluados: {data['partidos_evaluados']})")
    print("=" * 100)
    print(f"{'umbral':>8}{'n':>7}{'WR':>9}{'IC95':>16}"
          + "".join(f"{'u@' + format(o, '.2f'):>10}" for o in odds)
          + f"{'WR backtest':>14}{'gap':>9}")
    for fila in data["curva"]:
        t = fila["threshold"]
        if not fila.get("n"):
            print(f"{t:>8.2f}{0:>7}   sin predicciones")
            continue
        lo, hi = fila["wr_ci95"]
        n_b, wr_b = ref.get(t, (0, 0.0))
        gap = f"{fila['wr'] * 100 - wr_b:>+8.1f}pp" if n_b else f"{'-':>9}"
        print(f"{t:>8.2f}{fila['n']:>7}{fila['wr'] * 100:>8.1f}%"
              f"   [{lo * 100:>4.0f},{hi * 100:>4.0f}]"
              + "".join(f"{fila.get(f'u_{o:.2f}', 0):>10.0f}" for o in odds)
              + f"{(str(wr_b) + ' %') if n_b else '-':>14}{gap:>9}")

    print()
    for o in odds:
        be = 1.0 / o
        linea = ""
        for fila in data["curva"]:
            if fila.get("n") and fila.get(f"breakeven_{o:.2f}") is not None:
                if fila["wr"] >= be:
                    linea = f"el primer umbral con WR >= equilibrio es {fila['threshold']:.2f}"
                    break
        print(f"   equilibrio @ {o:.2f} = {be * 100:.1f} % de acierto -> {linea or 'ningun umbral llega'}")
    print()
    print("   (La APUESTA sigue en 0.75: esto es solo la capa de analisis.)")


if __name__ == "__main__":
    main()
