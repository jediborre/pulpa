"""Utilidades de procesamiento de Play-By-Play (PBP), reloj y puntos de presión.

Funciones compartidas por los extractores de features e inferencia en vivo
para cálculo de posesiones, rachas anotadoras, reloj y estimación de cuartos.
"""

from __future__ import annotations

from typing import Any


def safe_rate(num: float, den: float) -> float:
    """División segura evitando ZeroDivisionError."""
    return float(num / den) if den else 0.0


def clock_to_seconds(clock: str) -> int | None:
    """Convierte un reloj 'MM:SS' a segundos totales restantes."""
    if not clock or ":" not in clock:
        return None
    try:
        mm, ss = clock.split(":", 1)
        return int(mm) * 60 + int(ss)
    except ValueError:
        return None


def quarter_index(label: str) -> int | None:
    """Retorna el número de cuarto (1..4) o None si no es un cuarto estándar."""
    if label.startswith("Q"):
        try:
            return int(label[1:])
        except ValueError:
            return None
    return None


def quarter_points(data: dict[str, Any], quarter: str) -> tuple[int | None, int | None]:
    """Obtiene los puntos anotados por home y away en un cuarto específico."""
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def margin_bin(value: int) -> str:
    """Clasifica una diferencia de marcador en cubetas categóricas."""
    abs_val = abs(int(value))
    if abs_val <= 3:
        return "01_03"
    if abs_val <= 7:
        return "04_07"
    return "08_plus"


def winner_tag(home: int, away: int) -> str:
    """Etiqueta el ganador de un parcial ('home', 'away' o 'tied')."""
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "tied"


def infer_gender(league: str, home_team: str, away_team: str) -> str:
    """Infiere si el partido pertenece a liga femenina o masculina/abierta."""
    text = f"{league} {home_team} {away_team}".lower()
    markers = [
        "women", "woman", "female", "femen", "fem.", " ladies ",
        "(w)", " w ", "wnba", "girls",
    ]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


def count_sign_swings(values: list[int]) -> int:
    """Cuenta el número de cambios de signo en la curva de puntos de presión."""
    swings = 0
    prev_sign = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign
    return swings


def graph_stats_upto(graph_points: list[dict[str, Any]], max_minute: int) -> dict[str, Any]:
    """Calcula estadísticas agregadas de la gráfica de presión hasta un minuto dado."""
    points = [p for p in graph_points if int(p.get("minute", 0)) <= max_minute]
    values = [int(p.get("value", 0)) for p in points]
    if not values:
        return {
            "gp_count": 0,
            "gp_last": 0,
            "gp_peak_home": 0,
            "gp_peak_away": 0,
            "gp_area_home": 0,
            "gp_area_away": 0,
            "gp_area_diff": 0,
            "gp_mean_abs": 0.0,
            "gp_swings": 0,
            "gp_slope_3m": 0,
            "gp_slope_5m": 0,
        }

    home_vals = [v for v in values if v > 0]
    away_vals = [abs(v) for v in values if v < 0]
    area_home = sum(home_vals)
    area_away = sum(away_vals)

    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else values[-1] - values[0]

    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_peak_home": max(home_vals) if home_vals else 0,
        "gp_peak_away": max(away_vals) if away_vals else 0,
        "gp_area_home": area_home,
        "gp_area_away": area_away,
        "gp_area_diff": area_home - area_away,
        "gp_mean_abs": round(sum(abs(v) for v in values) / len(values), 3),
        "gp_swings": count_sign_swings(values),
        "gp_slope_3m": slope_3m,
        "gp_slope_5m": slope_5m,
    }


def pbp_events_upto(data: dict[str, Any], cutoff_minute: int) -> list[dict[str, Any]]:
    """Filtra y normaliza los eventos PBP jugados hasta el minuto cutoff."""
    pbp = data.get("play_by_play", {})
    events = []

    for quarter_label, plays in pbp.items():
        q_idx = quarter_index(quarter_label)
        if q_idx is None or q_idx < 1 or q_idx > 4:
            continue

        q_start = (q_idx - 1) * 12.0
        for play in plays:
            rem_sec = clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None:
                continue

            elapsed_in_q = max(0.0, 12.0 - (rem_sec / 60.0))
            event_min = q_start + elapsed_in_q

            if event_min <= cutoff_minute:
                ev = dict(play)
                ev["_global_min"] = event_min
                events.append(ev)

    events.sort(key=lambda e: float(e.get("_global_min", 0.0)))
    return events


def max_scoring_run(events: list[dict[str, Any]], team_name: str) -> int:
    """Encuentra la racha máxima ininterrumpida de anotación de un equipo."""
    best = 0
    run = 0
    for event in events:
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
            if run > best:
                best = run
        elif team in ("home", "away"):
            run = 0
    return best


def current_scoring_run(events: list[dict[str, Any]], team_name: str) -> int:
    """Encuentra la racha actual (desde el final hacia atrás) de un equipo."""
    run = 0
    for event in reversed(events):
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
        else:
            break
    return run


def recent_window_features(
    data: dict[str, Any],
    cutoff_minute: float,
    window_minutes: float,
    prefix: str = "recent",
) -> dict[str, Any]:
    """Extrae features de la ventana reciente inmediata anterior al minuto cutoff."""
    events_upto = pbp_events_upto(data, int(cutoff_minute))
    start_min = max(0.0, cutoff_minute - window_minutes)
    events = [
        e for e in events_upto
        if float(e.get("_global_min", 0.0)) >= start_min
    ]

    home_points = 0
    away_points = 0
    home_events = 0
    away_events = 0
    last_scoring = "none"

    for event in events:
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == "home" and pts > 0:
            home_points += pts
            home_events += 1
            last_scoring = "home"
        elif team == "away" and pts > 0:
            away_points += pts
            away_events += 1
            last_scoring = "away"

    scoring_events = home_events + away_events
    h_run = max_scoring_run(events, "home")
    a_run = max_scoring_run(events, "away")

    return {
        f"{prefix}_home_points": home_points,
        f"{prefix}_away_points": away_points,
        f"{prefix}_points_diff": home_points - away_points,
        f"{prefix}_home_event_share": safe_rate(home_events, scoring_events),
        f"{prefix}_home_max_run": h_run,
        f"{prefix}_away_max_run": a_run,
        f"{prefix}_run_diff": h_run - a_run,
        f"{prefix}_last_scoring_home": int(last_scoring == "home"),
        f"{prefix}_last_scoring_away": int(last_scoring == "away"),
    }


def score_upto(data: dict[str, Any], cutoff_minute: int) -> tuple[int, int]:
    """Estima el marcador global acumulado hasta el minuto de corte."""
    events = pbp_events_upto(data, cutoff_minute)
    home = 0
    away = 0
    for event in events:
        hs = event.get("home_score")
        as_ = event.get("away_score")
        if hs is not None and as_ is not None:
            home = int(hs)
            away = int(as_)
    return home, away


def infer_regulation_quarter_minutes(match_data: dict[str, Any]) -> float:
    """Infiere si la liga juega cuartos reglamentarios de 10 o 12 minutos."""
    pbp = match_data.get("play_by_play", {}) or {}
    best_clock_seconds = 0
    for quarter_label, plays in pbp.items():
        q_idx = quarter_index(str(quarter_label))
        if q_idx is None or q_idx < 1 or q_idx > 4:
            continue
        for play in plays or []:
            rem_sec = clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None or rem_sec > (12 * 60):
                continue
            if rem_sec > best_clock_seconds:
                best_clock_seconds = rem_sec

    if best_clock_seconds >= (11 * 60):
        return 12.0
    if best_clock_seconds >= (9 * 60):
        return 10.0

    quarters = (match_data.get("score") or {}).get("quarters") or {}
    n_played_quarters = sum(
        1 for q in ("Q1", "Q2", "Q3", "Q4")
        if isinstance(quarters.get(q), dict)
        and quarters[q].get("home") is not None
        and quarters[q].get("away") is not None
    )
    gp_minutes = [
        int(point.get("minute", 0))
        for point in (match_data.get("graph_points") or [])
        if point.get("minute") is not None
    ]
    if gp_minutes and n_played_quarters:
        approx_q_minutes = max(gp_minutes) / float(n_played_quarters)
        return min((10.0, 12.0), key=lambda val: abs(val - approx_q_minutes))

    league = str((match_data.get("match") or {}).get("league") or "")
    if "nba" in league.lower():
        return 12.0
    return 10.0


def infer_minute_from_pbp(match_data: dict[str, Any]) -> int | None:
    """Estima el minuto global jugado del partido a partir del último evento PBP."""
    pbp = match_data.get("play_by_play", {}) or {}
    latest_min = 0.0

    for q_label, plays in pbp.items():
        q_idx = quarter_index(str(q_label))
        if q_idx is None or q_idx < 1 or q_idx > 4:
            continue
        q_start = (q_idx - 1) * 12.0
        for play in plays or []:
            rem_sec = clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None:
                continue
            elapsed_q = max(0.0, 12.0 - (rem_sec / 60.0))
            tot_min = q_start + elapsed_q
            if tot_min > latest_min:
                latest_min = tot_min

    return int(round(latest_min)) if latest_min > 0 else None
