"""Esquema de datos canónico para resultados de predicción e inferencia.

Define la estructura universal que todos los modelos del sistema (m27_v3, v6_2, etc.)
deben retornar para desacoplar el motor de ML de los monitores y base de datos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PredictionResult:
    """Resultado canónico de inferencia de un modelo para un cuarto determinado."""
    available: bool
    model_version: str
    target: str  # "q3" o "q4"
    signal: str  # "BET", "NO_BET", "LATE_BET", "FILTERED", "PASS", "TIE", etc.
    pick: str  # "HOME", "AWAY", "NONE"
    confidence: float  # [0.0, 1.0] o porcentaje equivalente
    p_home_win: float  # [0.0, 1.0]
    p_away_win: float  # [0.0, 1.0]
    reason: str | None = None
    snapshot_minute: int | None = None
    raw_json: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado a un diccionario serializable."""
        data = asdict(self)
        if self.raw_json is not None:
            data["raw_json"] = self.raw_json
        return data

    @classmethod
    def unavailable(cls, model_version: str, target: str, reason: str) -> PredictionResult:
        """Crea una respuesta canónica para modelo no disponible o datos insuficientes."""
        return cls(
            available=False,
            model_version=model_version,
            target=target,
            signal="NO_BET",
            pick="NONE",
            confidence=0.0,
            p_home_win=0.5,
            p_away_win=0.5,
            reason=reason,
            snapshot_minute=None,
            raw_json=None,
        )
