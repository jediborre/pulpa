"""
league_overrides.py - Parametros tuneables POR LIGA (autogenerado).

Generado desde training_summary_v17.json por _auto_overrides.py.
Editar manualmente si hace falta; la proxima corrida lo sobreescribira.

Regla de bloqueo automatico:
  - val_roi < 0  OR  val_hit < 0.72  OR  gap > 0.3
  - (gap > 0.25 AND val_hit < 0.75)
"""
from __future__ import annotations
from typing import TypedDict


class LeagueOverride(TypedDict, total=False):
    min_confidence_q3: float
    min_confidence_q4: float
    min_samples_train: int
    min_gp_q3: int
    min_gp_q4: int
    min_pbp_q3: int
    min_pbp_q4: int
    enable_regression_filter: bool
    force_nobet: bool
    force_nobet_q3: bool
    force_nobet_q4: bool
    notes: str


LEAGUE_OVERRIDES: dict[str, LeagueOverride] = {
    "Argentina Liga Nacional": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.113; val_hit=0.633<0.72; gap=+0.362>0.3; q4: val_roi=-0.024; val_hit=0.697<0.72; gap=+0.452>0.3',
    },
    "B1 League": {
        "force_nobet_q4": True,
        "notes": 'q4 blocked: val_roi=-0.168; val_hit=0.595<0.72; q3 ok ROI=0.14545454545454536',
    },
    "B2 League": {
        "force_nobet": True,
        "notes": "manual holdout trim: april holdout q3 ROI=-0.16 on 15 bets; q4 ROI=-0.80 on 7 bets",
    },
    "BNXT League": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.020; val_hit=0.700<0.72; gap=+0.552>0.3; q4: gap=+0.395>0.3',
    },
    "Brazil NBB": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.038; val_hit=0.688<0.72; gap=+0.406>0.3; q4: val_roi=-0.209; val_hit=0.565<0.72; gap=+0.414>0.3',
    },
    "CIBACOPA , Primera Vuelta": {
        "force_nobet_q3": True,
        "notes": 'q3 blocked: None; q4 ok ROI=0.11999999999999993',
    },
    "China CBA": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.448; val_hit=0.394<0.72; gap=+0.580>0.3; q4: val_hit=0.720<0.72; gap=+0.351>0.3',
    },
    "Euroleague": {
        "force_nobet": True,
        "notes": 'q3: gap=+0.366>0.3; q4: gap=+0.329>0.3',
    },
    "France Pro A": {
        "force_nobet": True,
        "notes": 'q3: (no model); q4: gap=+0.373>0.3',
    },
    "Hungary NB 1.A": {
        "force_nobet": True,
        "notes": 'q3: gap=+0.333>0.3; q4: val_roi=-0.058; val_hit=0.673<0.72; gap=+0.467>0.3',
    },
    "Israeli National League Basketball": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.243; val_hit=0.541<0.72; gap=+0.415>0.3; q4: val_roi=-0.347; val_hit=0.467<0.72; gap=+0.418>0.3',
    },
    "LF Challenge, Regular Season": {
        "force_nobet_q3": True,
        "notes": 'q3 blocked: val_hit=0.719<0.72; gap=+0.351>0.3; q4 ok ROI=0.1729729729729729',
    },
    "Liga ACB": {
        "force_nobet": True,
        "notes": "manual holdout trim: april holdout q3 ROI=-0.40 on 7 bets; q4 ROI=-1.00 on 1 bet",
    },
    "Liga Nationala": {
        "force_nobet": True,
        "notes": 'q3: (no model); q4: val_roi=-0.054; val_hit=0.676<0.72; gap=+0.405>0.3',
    },
    "Meridianbet KLS": {
        "force_nobet_q3": True,
        "notes": 'q3 blocked: val_roi=-0.109; val_hit=0.636<0.72; gap=+0.472>0.3; q4 ok ROI=0.23529411764705876',
    },
    "Korean Basketball League": {
        "force_nobet_q3": True,
        "notes": "manual holdout trim: april holdout q3 ROI=-0.30 on 4 bets",
    },
    "NBA": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.116; val_hit=0.632<0.72; q4: val_roi=-0.475; val_hit=0.375<0.72',
    },
    "NBL Men , Regular Season": {
        "force_nobet": True,
        "notes": 'q3: (no model); q4: gap=+0.322>0.3',
    },
    "Poland 1st Division Basketball": {
        "force_nobet": True,
        "notes": 'q3: (no model); q4: val_roi=-0.146; val_hit=0.610<0.72; gap=+0.452>0.3',
    },
    "Polish Basketball League": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.287; val_hit=0.509<0.72; gap=+0.502>0.3; q4: val_roi=-0.382; val_hit=0.441<0.72; gap=+0.706>0.3',
    },
    "Primera FEB": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.219; val_hit=0.558<0.72; gap=+0.510>0.3; q4: gap=+0.421>0.3',
    },
    "Segunda FEB, Group East": {
        "force_nobet": True,
        "notes": "manual holdout trim: q3 blocked in training; april holdout q4 ROI=-1.00 on 2 bets",
    },
    "Segunda FEB, Group West": {
        "force_nobet": True,
        "notes": 'q3: (no model); q4: val_roi=-0.481; val_hit=0.370<0.72; gap=+0.613>0.3',
    },
    "Superliga": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.160; val_hit=0.600<0.72; gap=+0.526>0.3; q4: gap=+0.430>0.3',
    },
    "Swedish Basketball Superettan": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.067; val_hit=0.667<0.72; gap=+0.322>0.3; q4: val_roi=-0.289; val_hit=0.508<0.72; gap=+0.456>0.3',
    },
    "The Basketball League, Regular Season": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.210; val_hit=0.564<0.72; gap=+0.421>0.3; q4: gap=+0.363>0.3',
    },
    "Turkish Basketball Super League": {
        "force_nobet": True,
        "notes": 'q3: (no model); q4: gap=+0.385>0.3',
    },
    "U21 Espoirs Elite": {
        "force_nobet_q3": True,
        "notes": 'q3 blocked: val_roi=-0.067; val_hit=0.667<0.72; gap=+0.659>0.3; q4 ok ROI=0.3999999999999999',
    },
    "Serie B, Group B": {
        "force_nobet_q4": True,
        "notes": "manual holdout trim: april holdout q3 ROI=+0.05 on 4 bets; q4 ROI=-0.20 on 7 bets",
    },
    "\u00c9lite 2": {
        "force_nobet": True,
        "notes": 'q3: val_roi=-0.346; val_hit=0.467<0.72; gap=+0.616>0.3; q4: val_roi=-0.038; val_hit=0.688<0.72; gap=+0.411>0.3',
    },
}


def get_league_override(league: str) -> LeagueOverride:
    return LEAGUE_OVERRIDES.get(league, {})


def list_tuned_leagues() -> list[str]:
    return sorted(LEAGUE_OVERRIDES.keys())

