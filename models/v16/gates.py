"""
gates.py - Sistema de gates modular para decidir BET vs NO_BET.

Cada gate es una funcion pura (inputs -> PASS/FAIL + reason) que se ejecuta
en orden. El primer FAIL bloquea la apuesta. Cada resultado queda en el
debug para que el usuario pueda analizar que gates estan siendo demasiado
estrictos / demasiado laxos.

Un gate tiene la forma:
  def gate_xxx(ctx: GateContext) -> GateResult

Los gates se pueden activar/desactivar por liga via league_overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from training.v16 import config, league_overrides


# ============================================================================
# Contexto y resultado
# ============================================================================

@dataclass
class GateContext:
    """Toda la info que los gates pueden necesitar."""
    target: str                      # 'q3' | 'q4'
    league: str
    model_available: bool
    league_overrides: dict           # overrides aplicables

    proba_home: float                # probabilidad calibrada de home
    threshold: float                 # threshold aprendido para la liga

    gp_count: int
    pbp_count: int
    match_minute: float              # snapshot_minute

    pred_total: float | None         # prediccion de regresor (total)
    pred_home: float | None
    pred_away: float | None
    reg_mae_total: float | None      # MAE de la regresion en val

    current_run_pts: float
    volatility_swings: int
    league_samples_history: int      # partidos historicos de la liga en DB


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


# ============================================================================
# Gates
# ============================================================================

def gate_model_exists(ctx: GateContext) -> GateResult:
    if not ctx.model_available:
        return GateResult(
            "model_exists", False,
            "No hay modelo entrenado para esta liga. Sin fallback global.",
        )
    return GateResult("model_exists", True)


def gate_league_force_nobet(ctx: GateContext) -> GateResult:
    over = ctx.league_overrides
    target_key = f"force_nobet_{ctx.target}"
    if over.get("force_nobet") or over.get(target_key):
        return GateResult(
            "league_force_nobet", False,
            f"Kill-switch manual activo para liga {ctx.league} / {ctx.target}",
        )
    return GateResult("league_force_nobet", True)


def gate_data_quality(ctx: GateContext) -> GateResult:
    over = ctx.league_overrides
    if ctx.target == "q3":
        min_gp = over.get("min_gp_q3", config.MIN_GP_Q3)
        min_pbp = over.get("min_pbp_q3", config.MIN_PBP_Q3)
    else:
        min_gp = over.get("min_gp_q4", config.MIN_GP_Q4)
        min_pbp = over.get("min_pbp_q4", config.MIN_PBP_Q4)
    if ctx.gp_count < min_gp or ctx.pbp_count < min_pbp:
        return GateResult(
            "data_quality", False,
            f"Datos insuficientes (gp={ctx.gp_count}/min{min_gp}, "
            f"pbp={ctx.pbp_count}/min{min_pbp}).",
            details={"min_gp": min_gp, "min_pbp": min_pbp},
        )
    return GateResult("data_quality", True, details={
        "min_gp": min_gp, "min_pbp": min_pbp,
    })


def gate_league_history(ctx: GateContext) -> GateResult:
    if ctx.league_samples_history < config.LEAGUE_MIN_HISTORY_FOR_INFERENCE:
        return GateResult(
            "league_history", False,
            f"Historico de liga insuficiente ({ctx.league_samples_history} < "
            f"{config.LEAGUE_MIN_HISTORY_FOR_INFERENCE}).",
        )
    return GateResult("league_history", True)


def gate_confidence(ctx: GateContext) -> GateResult:
    conf = max(ctx.proba_home, 1 - ctx.proba_home)
    # Piso puede venir del league override
    key = "min_confidence_q3" if ctx.target == "q3" else "min_confidence_q4"
    min_conf = ctx.league_overrides.get(key, ctx.threshold)
    effective = max(min_conf, ctx.threshold)
    if conf < effective:
        return GateResult(
            "confidence", False,
            f"Confianza {conf:.3f} < threshold {effective:.3f}.",
            details={"confidence": conf, "threshold": effective},
        )
    return GateResult("confidence", True, details={
        "confidence": conf, "threshold": effective,
    })


def gate_volatility(ctx: GateContext) -> GateResult:
    if ctx.volatility_swings > config.MAX_VOLATILITY_SWINGS:
        return GateResult(
            "volatility", False,
            f"Partido demasiado volatil (swings={ctx.volatility_swings} > "
            f"{config.MAX_VOLATILITY_SWINGS}).",
            details={"swings": ctx.volatility_swings},
        )
    return GateResult("volatility", True, details={"swings": ctx.volatility_swings})


def gate_current_run(ctx: GateContext) -> GateResult:
    if ctx.current_run_pts > config.MAX_CURRENT_RUN_PTS:
        return GateResult(
            "current_run", False,
            f"Racha actual extrema ({ctx.current_run_pts} pts). Posible estado "
            "transitorio no representativo.",
            details={"current_run": ctx.current_run_pts},
        )
    return GateResult("current_run", True)


def gate_regression_confirms(ctx: GateContext) -> GateResult:
    """
    Usa la regresion como confirmacion: si la prediccion de scores apunta al
    contrario que la clasificacion con un spread significativo, bloquea.
    """
    enabled = ctx.league_overrides.get(
        "enable_regression_filter", config.ENABLE_REGRESSION_CONFIRMATION,
    )
    if not enabled:
        return GateResult("regression_confirm", True, details={"skipped": True})
    if ctx.pred_home is None or ctx.pred_away is None:
        return GateResult("regression_confirm", True, details={"missing": True})
    if ctx.reg_mae_total is not None and ctx.reg_mae_total > config.REG_MAX_MAE_ACCEPTABLE:
        return GateResult("regression_confirm", True, details={
            "skipped": True, "reason": "reg_mae_too_high",
            "mae": ctx.reg_mae_total,
        })
    # Direccion del clasificador
    clf_pick_home = ctx.proba_home >= 0.5
    reg_pick_home = ctx.pred_home > ctx.pred_away
    if clf_pick_home != reg_pick_home:
        spread = abs(ctx.pred_home - ctx.pred_away)
        if spread >= config.REG_DISAGREEMENT_BLOCK_PTS:
            return GateResult(
                "regression_confirm", False,
                f"Regresion predice sentido opuesto (spread {spread:.1f}).",
                details={
                    "clf_pick": "home" if clf_pick_home else "away",
                    "reg_pick": "home" if reg_pick_home else "away",
                    "spread": spread,
                },
            )
    return GateResult("regression_confirm", True, details={
        "clf_pick": "home" if clf_pick_home else "away",
        "reg_pick": "home" if reg_pick_home else "away",
    })


# ============================================================================
# Pipeline
# ============================================================================

# Orden de ejecucion (mas cheap / mas estructurales primero)
DEFAULT_PIPELINE: tuple[Callable[[GateContext], GateResult], ...] = (
    gate_model_exists,
    gate_league_force_nobet,
    gate_league_history,
    gate_data_quality,
    gate_volatility,
    gate_current_run,
    gate_confidence,
    gate_regression_confirms,
)


def run_gates(ctx: GateContext) -> tuple[bool, list[GateResult]]:
    """Ejecuta la pipeline. Devuelve (passed_all, lista detallada)."""
    results: list[GateResult] = []
    for g in DEFAULT_PIPELINE:
        r = g(ctx)
        results.append(r)
        if not r.passed:
            # Cortamos: no tiene sentido seguir corriendo gates si ya fallo uno
            # estructural. Pero si queremos el reporte completo, podemos seguir.
            # Aqui preferimos ver TODOS los gates posteriores como "not_run".
            for remaining in DEFAULT_PIPELINE[len(results):]:
                results.append(GateResult(remaining.__name__.replace("gate_", ""), False,
                                          "not_run (prev gate failed)",
                                          details={"short_circuited": True}))
            return False, results
    return True, results
