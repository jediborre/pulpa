"""Cálculo y extracción de features Head-to-Head (H2H) histórico.

Utilizado primordialmente por la familia m27_v3 para capturar dominancia previa,
dinámica de Q1 y resultados recientes entre dos equipos contendientes.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any


def _parse_date(dt_str: str) -> str:
    """Extrae la fecha limpia 'YYYY-MM-DD'."""
    return dt_str.split(" ")[0] if dt_str else ""


def compute_h2h_for_match(
    conn: sqlite3.Connection,
    home_team: str,
    away_team: str,
    match_date: str,
) -> dict[str, Any]:
    """Calcula las features H2H entre home_team y away_team antes de match_date."""
    clean_date = _parse_date(match_date)
    pair = tuple(sorted([home_team, away_team]))

    qs_rows = conn.execute(
        """
        SELECT m.date, qs.quarter, qs.home, qs.away
        FROM quarter_scores qs
        JOIN matches m ON m.match_id = qs.match_id
        WHERE (m.home_team = ? AND m.away_team = ?)
           OR (m.home_team = ? AND m.away_team = ?)
        ORDER BY m.date ASC
        """,
        (*pair, *pair),
    ).fetchall()

    past_games: list[dict[str, Any]] = []
    for dt, qtr, hs, aw in qs_rows:
        if hs is None or aw is None:
            continue
        g = past_games[-1] if past_games and past_games[-1].get("_date") == dt else None
        if g is None:
            g = {"_date": dt, "home_team": None, "away_team": None, "quarters": {}}
            past_games.append(g)
        g["quarters"][qtr] = (int(hs), int(aw))

    # Poblar home/away de cada partido
    for g in past_games:
        row = conn.execute(
            """
            SELECT home_team, away_team FROM matches WHERE date = ? AND (
                (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)
            )
            """,
            (g["_date"], *pair, *pair),
        ).fetchone()
        if row:
            g["home_team"], g["away_team"] = row[0], row[1]

    # Filtrar solo partidos anteriores a match_date
    past = [g for g in past_games if _parse_date(str(g["_date"])) < clean_date]
    if not past:
        return {
            "h2h_available": 0,
            "h2h_avg_q1_diff": 0.0,
            "h2h_recent3_home_won": 0.0,
            "h2h_last_home_won": 0,
        }

    q1_diffs: list[int] = []
    n = len(past)
    recent_3 = past[-3:] if n >= 3 else past

    for g in past:
        qs = g.get("quarters", {})
        is_home = g["home_team"] == home_team
        sign = 1 if is_home else -1
        q1 = qs.get("Q1")
        if q1:
            q1_diffs.append((q1[0] - q1[1]) * sign)

    home_won_recent_3 = sum(
        1 for g in recent_3
        if ((sum(v[0] for v in g["quarters"].values()) > sum(v[1] for v in g["quarters"].values())) == (g["home_team"] == home_team))
    )

    last_g = past[-1]
    is_home_last = last_g["home_team"] == home_team
    total_h_last = sum(v[0] for v in last_g["quarters"].values())
    total_a_last = sum(v[1] for v in last_g["quarters"].values())
    last_home_won = int((total_h_last > total_a_last) == is_home_last)

    return {
        "h2h_available": 1,
        "h2h_avg_q1_diff": round(sum(q1_diffs) / len(q1_diffs), 3) if q1_diffs else 0.0,
        "h2h_recent3_home_won": round(home_won_recent_3 / len(recent_3), 3) if recent_3 else 0.0,
        "h2h_last_home_won": last_home_won,
    }
