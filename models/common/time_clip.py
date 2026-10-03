"""Truncado temporal estricto para ventanas de inferencia y entrenamiento.

Garantiza que la ventana de datos vista en inferencia sea idéntica a la vista
durante el entrenamiento (evitando data leakage o desfases de tiempo).
- Para Q3: graph_points <= 24, play_by_play {Q1, Q2}.
- Para Q4: graph_points <= 36 (o minuto del snapshot como 27 o 30), play_by_play {Q1, Q2, Q3}.
"""

from __future__ import annotations

from typing import Any


def clip_match_data_for_target(
    match_data: dict[str, Any],
    target: str,
    max_minute: int | None = None,
) -> dict[str, Any]:
    """Retorna una copia superficial de match_data truncada a la ventana requerida."""
    target_lower = target.lower()
    if max_minute is None:
        max_minute = 24 if target_lower == "q3" else 36

    pbp_quarters = {"Q1", "Q2"} if target_lower == "q3" else {"Q1", "Q2", "Q3"}

    clipped = dict(match_data)
    clipped["graph_points"] = [
        p for p in match_data.get("graph_points", [])
        if int(p.get("minute", 0)) <= max_minute
    ]

    orig_pbp = match_data.get("play_by_play", {}) or {}
    clipped["play_by_play"] = {
        q: plays for q, plays in orig_pbp.items() if q in pbp_quarters
    }
    return clipped


def clip_match_data_to_minute(
    match_data: dict[str, Any],
    cutoff_minute: int,
) -> dict[str, Any]:
    """Trunca graph_points y PBP exactamente hasta un minuto de snapshot dado."""
    clipped = dict(match_data)
    clipped["graph_points"] = [
        p for p in match_data.get("graph_points", [])
        if int(p.get("minute", 0)) <= cutoff_minute
    ]
    return clipped
