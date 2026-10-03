"""
league_overrides.py - Parametros tuneables POR LIGA.

Este archivo es el knob principal para exploracion manual:
editalo para endurecer/relajar restricciones en ligas especificas despues
de analizar los reportes de backtest.

Cada entrada es opcional; si una liga no aparece, se usan los defaults de
config.py. El nombre de la liga debe matchear EXACTAMENTE al de la columna
matches.league en la DB.

Ejemplo de overrides posibles:
    "NBA": {
        "min_confidence_q3": 0.70,        # piso mas bajo en NBA
        "min_confidence_q4": 0.68,
        "min_samples_train": 500,         # exigir mas datos
        "enable_regression_filter": True,
        "force_nobet": False,
    },
    "B2 League": {
        "force_nobet": True,              # ligas con outliers sistematicos
    },

Campos soportados (todos opcionales):
    min_confidence_q3 : float   - threshold minimo para Q3 en esta liga
    min_confidence_q4 : float   - threshold minimo para Q4
    min_samples_train : int     - overrides config.LEAGUE_MIN_SAMPLES_TRAIN
    min_gp_q3 / min_gp_q4 : int - overrides data-quality gates
    min_pbp_q3 / min_pbp_q4 : int
    enable_regression_filter : bool - usar regresion como confirmacion
    force_nobet : bool          - bloquea todo (kill-switch)
    notes : str                 - documentacion para vos mismo
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
    force_nobet: bool        # bloquea liga entera
    force_nobet_q3: bool     # bloquea solo Q3 en esta liga
    force_nobet_q4: bool     # bloquea solo Q4 en esta liga
    notes: str


# Overrides DERIVADOS DEL BACKTEST DE PRODUCCION (val_roi < 0 -> force_nobet)
# Reentrenamiento del modelo:  train=72 val=15 cal=3 holdout=0 min=200 active=14
# Fecha de calibracion: ultimo entrenamiento. Revisar cada semana.
LEAGUE_OVERRIDES: dict[str, LeagueOverride] = {
    # Liga con ROI negativo EN AMBOS TARGETS -> bloqueo total
    "Brazil NBB": {
        "force_nobet": True,
        "notes": "val_roi q3=-100% q4=-49%. Modelo no generaliza.",
    },
    "Israeli National League Basketball": {
        "force_nobet": True,
        "notes": "val_roi q3=-20% q4=-5.5%. Data ruidosa.",
    },

    # Ligas con un target rentable y otro no -> bloqueo por target
    "NBA": {
        "force_nobet_q4": True,
        "notes": "NBA q3 val_roi +3.2% (732%) ok. q4 val_roi -27.6%, hit 51.7%. "
                 "Typical late-game variance NBA, bloqueamos q4.",
    },
    "China CBA": {
        "force_nobet_q3": True,
        "notes": "q3 val_roi -36%, q4 val_roi +7.4%. Q4 unico rentable.",
    },
    "Argentina Liga Nacional": {
        "force_nobet_q4": True,
        "notes": "q3 val_roi +4.4%, q4 val_roi -20.8%.",
    },
    "B2 League": {
        "force_nobet_q4": True,
        "notes": "q3 val_roi +15.3% (fuerte), q4 val_roi -11.3%.",
    },
    "Superliga": {
        "force_nobet_q3": True,
        "notes": "q3 val_roi -12.8%, q4 val_roi +7.7%.",
    },
    "\u00c9lite 2": {  # Elite 2 (con acento)
        "force_nobet_q3": True,
        "notes": "q3 val_roi -36%, q4 val_roi +12.3%.",
    },
}


def get_league_override(league: str) -> LeagueOverride:
    """Devuelve overrides de la liga o dict vacio si no hay."""
    return LEAGUE_OVERRIDES.get(league, {})


def list_tuned_leagues() -> list[str]:
    """Ligas con overrides activos (utilidad para reportes)."""
    return sorted(LEAGUE_OVERRIDES.keys())
