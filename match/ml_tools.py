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
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    graph_points = match_data.get("graph_points", [])
    if not graph_points:
        raise ValueError("Match has no graph_points to plot")

    minutes = [int(p["minute"]) for p in graph_points]
    values = [int(p["value"]) for p in graph_points]

    m = match_data["match"]
    s = match_data["score"]

    status_type = str(m.get("status_type", "") or "").strip().lower()
    quarters = s.get("quarters", {}) if isinstance(s, dict) else {}

    # Count periods that have actual final scores.
    n_played_quarters = sum(
        1 for q in ("Q1", "Q2", "Q3", "Q4")
        if isinstance(quarters.get(q), dict)
        and quarters[q].get("home") is not None
        and quarters[q].get("away") is not None
    )
    finished = status_type == "finished"
    # A finished match with ≤2 scored periods is a 2-half game (e.g. NCAA).
    if finished and 0 < n_played_quarters <= 2:
        n_total_quarters = n_played_quarters
        # Infer half duration from graph data (NCAA=20 min, etc.).
        approx_qd = (max(minutes) if minutes else 40) / n_total_quarters
        quarter_duration = min(
            (10, 12, 15, 20, 25), key=lambda d: abs(approx_qd - d)
        )
    else:
        n_total_quarters = 4
        quarter_duration = 12
    x_end = float(n_total_quarters * quarter_duration)

    last_q = f"Q{n_total_quarters}"
    last_q_data = quarters.get(last_q)
    last_q_complete = (
        isinstance(last_q_data, dict)
        and last_q_data.get("home") is not None
        and last_q_data.get("away") is not None
    )
    extended_to_ft = False
    extend_start_minute = 0
    extend_start_value = 0
    # Some FT matches have graph_points ending before x_end.
    # Extend the last value to FT so the chart does not look truncated.
    if minutes and max(minutes) < x_end and finished:
        extend_start_minute = int(minutes[-1])
        extend_start_value = int(values[-1])
        extended_to_ft = True
        minutes = [*minutes, int(x_end)]
        values = [*values, values[-1]]

    figure_bg = "#0C141B"
    axes_bg = "#111A22"
    quarter_bg_a = "#132031"
    quarter_bg_b = "#101B29"
    future_bg = "#2A323B"
    top_zone_bg = "#234A2F"
    bottom_zone_bg = "#343C6B"
    text_color = "#E2E8F0"
    muted_text_color = "#8FA0B4"
    positive_fill = "#4CD05E"
    negative_fill = "#7882EB"
    line_color = "#4FD3FF"
    baseline_color = "#8FA4BD"
    halftime_color = "#FF4D57"

    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(figure_bg)
    ax.set_facecolor(axes_bg)
    ax.set_aspect("auto")

    ymax = max(values) if values else 1
    ymin = min(values) if values else -1
    peak_abs = max(abs(ymax), abs(ymin), 1)
    y_pad = max(2, int(peak_abs * 0.12))

    max_minute = max(minutes)
    x_start = 0.0

    for i in range(n_total_quarters):
        q_start = i * quarter_duration
        q_end = (i + 1) * quarter_duration
        bg = quarter_bg_a if i % 2 == 0 else quarter_bg_b
        if q_start >= max_minute:
            bg = future_bg
        ax.axvspan(q_start, q_end, color=bg, alpha=0.95, zorder=0)

    # Split background in upper/lower panels to resemble SofaScore visual language.
    ax.axhspan(0, peak_abs + y_pad, color=top_zone_bg, alpha=0.16, zorder=0.4)
    ax.axhspan(-peak_abs - y_pad, 0, color=bottom_zone_bg, alpha=0.20, zorder=0.4)

    for i in range(1, n_total_quarters):
        ax.axvline(x=i * quarter_duration, color="#2B3C4F", linewidth=1.0, zorder=1)

    if extended_to_ft and len(minutes) >= 2:
        # Real feed segment
        ax.plot(minutes[:-1], values[:-1], linewidth=2.2, color=line_color, zorder=4)
        # Extrapolated segment to FT
        ax.plot(
            [extend_start_minute, int(x_end)],
            [extend_start_value, extend_start_value],
            linewidth=2.2,
            color=line_color,
            linestyle="--",
            alpha=0.95,
            zorder=5,
        )
    else:
        ax.plot(minutes, values, linewidth=2.2, color=line_color, zorder=4)
    ax.fill_between(
        minutes,
        values,
        0,
        where=[v >= 0 for v in values],
        color=positive_fill,
        alpha=0.88,
        interpolate=True,
        zorder=3,
    )
    ax.fill_between(
        minutes,
        values,
        0,
        where=[v < 0 for v in values],
        color=negative_fill,
        alpha=0.88,
        interpolate=True,
        zorder=3,
    )

    y_text = ymax - (ymax - ymin) * 0.1
    for i in range(n_total_quarters):
        mid_x = (i + 0.5) * quarter_duration
        ax.text(
            mid_x,
            y_text,
            f"Q{i + 1}",
            ha="center",
            va="center",
            fontsize=11,
            color=muted_text_color,
            zorder=5,
        )

    ax.axhline(y=0, color=baseline_color, linewidth=1.2, alpha=0.8, zorder=2)
    ax.set_xlim(x_start, x_end)

    ax.set_ylim(-peak_abs - y_pad, peak_abs + y_pad)

    ax.set_xticks([0, int(x_end)])
    ax.set_xticklabels(["0:00", "FT"], color=muted_text_color, fontsize=10)
    ax.yaxis.tick_right()
    ax.set_yticks([peak_abs, 0, -peak_abs])
    ax.set_yticklabels(
        [str(peak_abs), "0", str(peak_abs)],
        color=muted_text_color,
        fontsize=10,
    )
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    halftime_x = x_end / 2
    ax.axvline(halftime_x, color=halftime_color, linewidth=1.4, zorder=6)
    ax.plot(
        halftime_x,
        peak_abs + y_pad * 0.05,
        marker="o",
        markersize=5,
        markerfacecolor=axes_bg,
        markeredgewidth=1.4,
        markeredgecolor=halftime_color,
        zorder=7,
        clip_on=False,
    )
    ax.text(
        halftime_x,
        peak_abs + y_pad * 0.45,
        "Halftime",
        ha="center",
        va="bottom",
        color=halftime_color,
        fontsize=11,
        fontweight="bold",
        zorder=7,
        clip_on=False,
    )

    # Live time marker: red dot + wall-clock label at the last data point
    if not finished and minutes and values:
        import datetime as _dt
        live_x = minutes[-1]
        live_y = values[-1]
        live_time_str = _dt.datetime.now().strftime("%H:%M")
        ax.axvline(live_x, color=halftime_color, linewidth=1.4, zorder=6)
        ax.plot(
            live_x,
            peak_abs + y_pad * 0.05,
            marker="o",
            markersize=5,
            markerfacecolor=axes_bg,
            markeredgewidth=1.4,
            markeredgecolor=halftime_color,
            zorder=8,
            clip_on=False,
        )
        ax.text(
            live_x,
            peak_abs + y_pad * 0.45,
            live_time_str,
            ha="center",
            va="bottom",
            color=halftime_color,
            fontsize=11,
            fontweight="bold",
            zorder=8,
            clip_on=False,
        )

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

    ax.set_title(
        f"{m['home_team']} vs {m['away_team']}\n{score_line}",
        color=text_color,
        pad=18,
        fontsize=18,
        fontweight="semibold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=150,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(fig)
    return str(out)
