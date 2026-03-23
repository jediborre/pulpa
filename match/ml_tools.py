"""Utilities for ML feature export and graph reconstruction.

This module consumes normalized match dicts returned by db.get_match() and
provides:
1) Feature engineering for training datasets.
2) Seaborn-based visualization of SofaScore graph_points.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import pstdev


def _graph_quarter(minute: int) -> str:
    if minute <= 12:
        return "Q1"
    if minute <= 24:
        return "Q2"
    if minute <= 36:
        return "Q3"
    return "Q4"


def _count_sign_swings(values: list[int]) -> int:
    swings = 0
    prev_sign = 0
    for v in values:
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign
    return swings


def build_feature_row(match_data: dict) -> dict:
    """Build one flat feature row for one match.

    Output is intentionally model-agnostic:
    - target label: home_win
    - static context: league, teams, date/time
    - pressure signal from graph_points
    - coarse dynamics from play_by_play
    """
    match_id = match_data["match_id"]
    m = match_data["match"]
    s = match_data["score"]
    pbp = match_data.get("play_by_play", {})
    graph_points = match_data.get("graph_points", [])

    values = [int(p["value"]) for p in graph_points]
    minutes = [int(p["minute"]) for p in graph_points]

    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]

    home_area_total = sum(max(v, 0) for v in values)
    away_area_total = sum(max(-v, 0) for v in values)
    mean_abs_pressure = (
        sum(abs(v) for v in values) / len(values) if values else 0.0
    )

    q_home_area = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    q_away_area = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for p in graph_points:
        q = _graph_quarter(int(p["minute"]))
        v = int(p["value"])
        q_home_area[q] += max(v, 0)
        q_away_area[q] += max(-v, 0)

    home_plays = 0
    away_plays = 0
    home_3pt = 0
    away_3pt = 0
    for plays in pbp.values():
        for play in plays:
            is_home = play.get("team") == "home"
            pts = int(play.get("points", 0))
            if is_home:
                home_plays += 1
                if pts == 3:
                    home_3pt += 1
            else:
                away_plays += 1
                if pts == 3:
                    away_3pt += 1

    home_score = int(s["home"])
    away_score = int(s["away"])

    row = {
        "match_id": match_id,
        "date": m.get("date", ""),
        "time_utc": m.get("time", ""),
        "league": m.get("league", ""),
        "home_team": m.get("home_team", ""),
        "away_team": m.get("away_team", ""),
        "home_score": home_score,
        "away_score": away_score,
        "score_diff_home": home_score - away_score,
        "home_win": 1 if home_score > away_score else 0,
        "graph_points_count": len(graph_points),
        "graph_minute_first": minutes[0] if minutes else None,
        "graph_minute_last": minutes[-1] if minutes else None,
        "graph_value_first": values[0] if values else 0,
        "graph_value_last": values[-1] if values else 0,
        "graph_peak_home": max(values) if values else 0,
        "graph_peak_away": abs(min(values)) if values else 0,
        "graph_home_area_total": home_area_total,
        "graph_away_area_total": away_area_total,
        "graph_area_diff_home": home_area_total - away_area_total,
        "graph_mean_abs_pressure": round(mean_abs_pressure, 4),
        "graph_swing_count": _count_sign_swings(values),
        "graph_volatility": round(pstdev(diffs), 6) if len(diffs) > 1 else 0.0,
        "q1_home_area": q_home_area["Q1"],
        "q2_home_area": q_home_area["Q2"],
        "q3_home_area": q_home_area["Q3"],
        "q4_home_area": q_home_area["Q4"],
        "q1_away_area": q_away_area["Q1"],
        "q2_away_area": q_away_area["Q2"],
        "q3_away_area": q_away_area["Q3"],
        "q4_away_area": q_away_area["Q4"],
        "pbp_home_scoring_plays": home_plays,
        "pbp_away_scoring_plays": away_plays,
        "pbp_home_3pt_made": home_3pt,
        "pbp_away_3pt_made": away_3pt,
        "pbp_scoring_plays_total": home_plays + away_plays,
    }
    return row


def build_feature_rows_by_quarter(match_data: dict) -> list[dict]:
    """Build one feature row per quarter for in-game modeling."""
    base = build_feature_row(match_data)
    quarters = match_data.get("score", {}).get("quarters", {})
    graph_points = match_data.get("graph_points", [])
    pbp = match_data.get("play_by_play", {})

    rows: list[dict] = []
    order = ["Q1", "Q2", "Q3", "Q4"]
    quarter_end_minute = {"Q1": 12, "Q2": 24, "Q3": 36, "Q4": 48}

    cum_home = 0
    cum_away = 0
    for q in order:
        q_score = quarters.get(q)
        if not q_score:
            continue

        q_home = int(q_score.get("home", 0))
        q_away = int(q_score.get("away", 0))
        cum_home += q_home
        cum_away += q_away

        q_points = [
            p
            for p in graph_points
            if int(p.get("minute", 0)) <= quarter_end_minute[q]
        ]
        q_values = [int(p["value"]) for p in q_points]

        q_pbp = []
        for key in order:
            if key == q:
                q_pbp.extend(pbp.get(key, []))
                break
            q_pbp.extend(pbp.get(key, []))

        home_plays = sum(1 for p in q_pbp if p.get("team") == "home")
        away_plays = sum(1 for p in q_pbp if p.get("team") == "away")

        row = {
            "match_id": base["match_id"],
            "quarter": q,
            "quarter_index": order.index(q) + 1,
            "league": base["league"],
            "home_team": base["home_team"],
            "away_team": base["away_team"],
            "q_home_score": q_home,
            "q_away_score": q_away,
            "q_score_diff_home": q_home - q_away,
            "cum_home_score": cum_home,
            "cum_away_score": cum_away,
            "cum_score_diff_home": cum_home - cum_away,
            "graph_points_count_to_q": len(q_points),
            "graph_value_last_to_q": q_values[-1] if q_values else 0,
            "graph_peak_home_to_q": max(q_values) if q_values else 0,
            "graph_peak_away_to_q": abs(min(q_values)) if q_values else 0,
            "graph_swing_count_to_q": _count_sign_swings(q_values),
            "pbp_home_scoring_plays_to_q": home_plays,
            "pbp_away_scoring_plays_to_q": away_plays,
            "pbp_scoring_plays_total_to_q": home_plays + away_plays,
        }
        rows.append(row)

    return rows


def export_feature_rows(rows: list[dict], out_path: str, fmt: str) -> str:
    """Write rows to disk in csv or jsonl format."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(path)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return str(path)

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def plot_graph(match_data: dict, out_path: str) -> str:
    """Reconstruct SofaScore pressure graph using seaborn + matplotlib.

    Positive values are interpreted as home-team pressure advantage,
    negative values as away-team pressure advantage.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    graph_points = match_data.get("graph_points", [])
    if not graph_points:
        raise ValueError("Match has no graph_points to plot")

    minutes = [int(p["minute"]) for p in graph_points]
    values = [int(p["value"]) for p in graph_points]

    m = match_data["match"]
    s = match_data["score"]

    sns.set_theme(style="darkgrid", context="notebook")
    fig, ax = plt.subplots(figsize=(11, 4.5))

    sns.lineplot(x=minutes, y=values, linewidth=2.2, color="#2F80ED", ax=ax)
    ax.fill_between(minutes, values, 0, where=[v >= 0 for v in values],
                    color="#2ECC71", alpha=0.35, interpolate=True)
    ax.fill_between(minutes, values, 0, where=[v < 0 for v in values],
                    color="#5B7CFA", alpha=0.35, interpolate=True)

    for x in (12, 24, 36):
        ax.axvline(x=x, color="#777777", linestyle="--", linewidth=1)

    ymax = max(values) if values else 1
    ymin = min(values) if values else -1
    y_text = ymax - (ymax - ymin) * 0.08
    for label, x in (("Q1", 6), ("Q2", 18), ("Q3", 30), ("Q4", 42)):
        ax.text(x, y_text, label, ha="center", va="center", fontsize=9,
                color="#555555")

    ax.axhline(y=0, color="#333333", linewidth=1)
    ax.set_xlim(min(minutes), max(minutes))
    ax.set_xlabel("Game minute")
    ax.set_ylabel("Pressure / momentum value")
    quarters = s.get("quarters", {})
    q_order = ["Q1", "Q2", "Q3", "Q4"]
    q_parts = []
    for q in q_order:
        if q in quarters:
            q_parts.append(f"{q} {quarters[q]['home']}-{quarters[q]['away']}")

    score_line = " | ".join(q_parts)
    if score_line:
        score_line = score_line + f" | FT {s['home']}-{s['away']}"
    else:
        score_line = f"FT {s['home']}-{s['away']}"

    ax.set_title(f"{m['home_team']} vs {m['away_team']}\n{score_line}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)
