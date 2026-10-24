# =====================================================================
# CURVA DE WR VIVO POR UMBRAL (capa de analisis, no de apuesta)
# ---------------------------------------------------------------------
# La serie por minuto (model_inference_v3) guarda CADA prediccion del
# modelo dentro de su ventana valida, no solo las que Betty. Con eso se
# puede calcular el WR y el beneficio en vivo para CUALQUIER umbral, sin
# esperar a que se acumulen apuestas reales.
#
# Motivo (2026-10-20): a 0.75 el modelo dispara ~6 apuestas/dia, asi que
# llegar a 200 resueltas lleva un mes. Con la curva completa el veredicto
# llega en dias. La APUESTA sigue en 0.75: esto es solo analisis.
#
# Regla de equilibrio por cuota: WR_equilibrio = 1 - 1/cuota
#   1.40 -> 71.4 %   |   1.83 -> 54.6 %
# =====================================================================

from __future__ import annotations

import math
from typing import Any

# Umbrales que se evalúan en la curva. 0.65 da ~20 predicciones/dia (3x mas
# muestra que 0.75) y es donde empieza a haber señal estadistica.
DEFAULT_THRESHOLDS = (0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80)
DEFAULT_ODDS = (1.40, 1.83)


def _wilson(wins: int, n: int) -> tuple[float, float]:
    """IC95 del acierto (proporcion de successes)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _outcome(pick: str, q4_home: int, q4_away: int) -> int:
    """1 si el pick gano el Q4, 0 si perdio, None si empate (push)."""
    if pick == "HOME":
        dif = q4_home - q4_away
    elif pick == "AWAY":
        dif = q4_away - q4_home
    else:
        return None
    if dif == 0:
        return None
    return 1 if dif > 0 else 0


def fetch_live_predictions(conn, model_version: str, max_minute: float = 30.0) -> list[dict[str, Any]]:
    """Una fila por partido: la del ULTIMO minuto <= max_minute (la decision final).

    Se une con quarter_scores_v2 para traer el resultado real del Q4.
    """
    rows = conn.execute(
        """
        SELECT i.match_id, i.game_minute, i.pick, i.confidence, i.p_cal, i.margin,
               i.signal, i.block_reason,
               q.q4_home, q.q4_away
        FROM model_inference_v3 i
        JOIN quarter_scores_v2 q ON q.match_id = i.match_id
        WHERE i.model_version = ?
          AND i.game_minute <= ?
          AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL
        ORDER BY i.match_id, i.game_minute
        """,
        (model_version, max_minute),
    ).fetchall()
    por_partido: dict[str, dict[str, Any]] = {}
    for r in rows:
        por_partido[str(r["match_id"])] = dict(r)
    return list(por_partido.values())


def compute_curve(
    preds: list[dict[str, Any]],
    thresholds=DEFAULT_THRESHOLDS,
    odds=DEFAULT_ODDS,
) -> dict[str, Any]:
    """Curva de WR/beneficio por umbral sobre las predicciones vivas."""
    filas: list[dict[str, Any]] = []
    for t in thresholds:
        sel = [p for p in preds
               if p["pick"] in ("HOME", "AWAY") and (p["confidence"] or 0) >= t]
        wins = 0
        n = 0
        for p in sel:
            out = _outcome(p["pick"], p["q4_home"], p["q4_away"])
            if out is None:
                continue
            wins += out
            n += 1
        if n == 0:
            filas.append({"threshold": t, "n": 0})
            continue
        lo, hi = _wilson(wins, n)
        fila = {
            "threshold": t,
            "n": n,
            "wins": wins,
            "wr": wins / n,
            "wr_ci95": (lo, hi),
            "avg_conf": sum(p["confidence"] or 0 for p in sel) / len(sel),
        }
        for o in odds:
            u = wins * o - n
            fila[f"u_{o:.2f}"] = u
            fila[f"roi_{o:.2f}"] = u / n
            # equilibrio: WR*(o-1) - (1-WR) = 0  ->  WR = 1/o
            fila[f"breakeven_{o:.2f}"] = 1.0 / o
        filas.append(fila)
    return {
        "model": "m27_v5_3",
        "partidos_evaluados": len(preds),
        "curva": filas,
        "odds": list(odds),
    }


def _row_text(fila: dict[str, Any], odds=DEFAULT_ODDS) -> str:
    if not fila.get("n"):
        return f"{fila['threshold']:.2f} | sin predicciones"
    lo, hi = fila["wr_ci95"]
    partes = [f"{fila['threshold']:.2f}", f"{fila['n']:>4}", f"{fila['wr'] * 100:>5.1f}%"]
    for o in odds:
        partes.append(f"{fila.get(f'u_{o:.2f}', 0):>6.0f}u")
    partes.append(f"[{lo * 100:.0f}, {hi * 100:.0f}]")
    return " | ".join(partes)


def render_curve_text(data: dict[str, Any], odds=DEFAULT_ODDS) -> str:
    """Texto de la curva para el resumen de Telegram."""
    lineas = [
        "<b>Curva de WR vivo</b> (predicciones por minuto, no solo apuestas)",
        f"Partidos evaluados: <b>{data['partidos_evaluados']}</b>",
        "umbral | n | WR | " + " | ".join(f"u@{o:.2f}" for o in odds) + " | IC95",
    ]
    for fila in data["curva"]:
        if not fila.get("n"):
            continue
        lo, hi = fila["wr_ci95"]
        lineas.append(
            f"  {fila['threshold']:.2f} | {fila['n']:>4} | {fila['wr'] * 100:>5.1f}% | "
            + " | ".join(f"{fila.get(f'u_{o:.2f}', 0):>6.0f}u" for o in odds)
            + f" | [{lo * 100:.0f}, {hi * 100:.0f}]"
        )
    for o in odds:
        be = 1.0 / o
        lineas.append(f"Equilibrio @ {o:.2f}: <b>{be * 100:.1f}%</b> de acierto")
    return "\n".join(lineas)
