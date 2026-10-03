"""Cargador de datos canónico para el subsistema de modelos.

Provee conexión garantizada a matches.db en la raíz del proyecto y funciones
para cargar partidos, calcular historiales de equipos (prior win rate) y métricas H2H.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Ruta canónica a la base de datos en la raíz del proyecto (Regla 3 de AGENTS.md)
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Obtiene una conexión SQLite a matches.db con modo WAL y timeout configurado."""
    target_path = str(db_path or DB_PATH)
    conn = sqlite3.connect(target_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_match(conn: sqlite3.Connection, match_id: str) -> dict[str, Any] | None:
    """Reconstruye el diccionario canónico de un partido desde la base de datos."""
    row = conn.execute(
        "SELECT * FROM matches WHERE match_id = ?", (str(match_id),)
    ).fetchone()
    if not row:
        return None

    quarters: dict[str, dict[str, int | None]] = {}
    for qr in conn.execute(
        "SELECT quarter, home, away FROM quarter_scores WHERE match_id = ? ORDER BY quarter",
        (str(match_id),),
    ):
        quarters[qr["quarter"]] = {"home": qr["home"], "away": qr["away"]}

    pbp: dict[str, list[dict[str, Any]]] = {}
    for pr in conn.execute(
        "SELECT quarter, time, player, points, team, home_score, away_score "
        "FROM play_by_play WHERE match_id = ? ORDER BY quarter, seq",
        (str(match_id),),
    ):
        pbp.setdefault(pr["quarter"], []).append({
            "time": pr["time"],
            "player": pr["player"],
            "points": pr["points"],
            "team": pr["team"],
            "home_score": pr["home_score"],
            "away_score": pr["away_score"],
        })

    graph_points: list[dict[str, int]] = []
    for gr in conn.execute(
        "SELECT minute, value FROM graph_points WHERE match_id = ? ORDER BY seq",
        (str(match_id),),
    ):
        graph_points.append({"minute": gr["minute"], "value": gr["value"]})

    row_keys = row.keys()
    out: dict[str, Any] = {
        "match_id": str(match_id),
        "match": {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_slug": row["home_slug"] or "unknown",
            "away_slug": row["away_slug"] or "unknown",
            "event_slug": row["event_slug"] or "unknown",
            "custom_id": row["custom_id"] or "",
            "status_type": row["status_type"] or "",
            "status_description": row["status_description"] or "",
            "home_team_id": row["home_team_id"] if "home_team_id" in row_keys else None,
            "away_team_id": row["away_team_id"] if "away_team_id" in row_keys else None,
            "home_rating": row["home_rating"] if "home_rating" in row_keys else None,
            "away_rating": row["away_rating"] if "away_rating" in row_keys else None,
            "date": row["date"],
            "time": row["time"],
            "venue": row["venue"],
            "league": row["league"],
            "home_record": row["home_record"],
            "away_record": row["away_record"],
        },
        "score": {
            "home": row["home_score"],
            "away": row["away_score"],
            "quarters": quarters,
        },
        "graph_points": graph_points,
        "play_by_play": pbp,
    }
    return out


def compute_team_prior_wr(
    conn: sqlite3.Connection,
    team_name: str,
    match_date: str,
    match_time: str = "23:59",
    window: int = 12,
) -> float:
    """Calcula el porcentaje de victorias históricas previas de un equipo."""
    rows = conn.execute(
        """
        SELECT home_team, away_team, home_score, away_score, date, time
        FROM matches
        WHERE (home_team = ? OR away_team = ?)
          AND (date < ? OR (date = ? AND time < ?))
        ORDER BY date DESC, time DESC
        LIMIT ?
        """,
        (team_name, team_name, match_date, match_date, match_time, window),
    ).fetchall()

    if not rows:
        return 0.0

    wins = 0
    total = 0
    for row in rows:
        hs = row[2]
        as_ = row[3]
        if hs is None or as_ is None:
            continue
        total += 1
        if row[0] == team_name:
            wins += int(hs > as_)
        else:
            wins += int(as_ > hs)

    return _safe_rate(wins, total)
