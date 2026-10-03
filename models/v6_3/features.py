"""Extracción de features para el modelo v6_3 con snapshots dinámicos (m27, m30)."""

from __future__ import annotations

import sqlite3
from typing import Any

from models.v6_2.features import extract_features as extract_v6_2_features


def extract_features(
    match_data: dict[str, Any],
    conn: sqlite3.Connection,
    target: str = "q4",
) -> dict[str, Any]:
    """Extrae features utilizando el pipeline de v6_2 adaptado a v6_3."""
    return extract_v6_2_features(match_data, conn, target=target)
