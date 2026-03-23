"""EDA for basketball match dataset stored in SQLite.

Outputs are written to training/eda_outputs/:
- eda_summary.json
- league_summary.csv
- gender_summary.csv
- coverage_summary.csv
- halftime_baseline.csv
- prior_strength_baseline.csv
- charts (png)
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "eda_outputs"


@dataclass
class MatchRow:
    match_id: str
    date: str
    time: str
    league: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int


def _to_dt(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M")


def _infer_gender(league: str, home_team: str, away_team: str) -> str:
    text = f"{league} {home_team} {away_team}".lower()
    markers = [
        "women", "woman", "female", "femen", "fem.", " ladies ",
        "(w)", " w ", "wnba", "girls",
    ]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


def _fetch_matches(conn: sqlite3.Connection) -> list[MatchRow]:
    rows = conn.execute(
        """
        SELECT
            match_id, date, time, league,
            home_team, away_team, home_score, away_score
        FROM matches
        ORDER BY date, time, match_id
        """
    ).fetchall()
    return [
        MatchRow(
            match_id=str(r[0]),
            date=r[1],
            time=r[2],
            league=r[3] or "",
            home_team=r[4] or "",
            away_team=r[5] or "",
            home_score=int(r[6] or 0),
            away_score=int(r[7] or 0),
        )
        for r in rows
    ]


def _fetch_quarter_scores(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, tuple[int, int]]]:
    out: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    rows = conn.execute(
        "SELECT match_id, quarter, home, away FROM quarter_scores"
    ).fetchall()
    for match_id, quarter, home, away in rows:
        out[str(match_id)][str(quarter)] = (int(home or 0), int(away or 0))
    return out


def _count_by_match(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT match_id, COUNT(*) FROM {table} GROUP BY match_id"
    ).fetchall()
    return {str(match_id): int(cnt) for match_id, cnt in rows}


def _safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def _write_csv(
    path: Path,
    rows: Iterable[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _make_chart_bar(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run_eda(db_path: Path = DB_PATH, out_dir: Path = OUT_DIR) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    matches = _fetch_matches(conn)
    quarter_scores = _fetch_quarter_scores(conn)
    pbp_counts = _count_by_match(conn, "play_by_play")
    graph_counts = _count_by_match(conn, "graph_points")

    n_matches = len(matches)
    if n_matches == 0:
        raise RuntimeError("No matches found in DB.")

    dates = [m.date for m in matches]
    leagues = Counter(m.league or "UNKNOWN" for m in matches)
    genders = Counter(
        _infer_gender(m.league, m.home_team, m.away_team)
        for m in matches
    )

    # Coverage summary
    cov_pbp = sum(1 for m in matches if pbp_counts.get(m.match_id, 0) > 0)
    cov_graph = sum(1 for m in matches if graph_counts.get(m.match_id, 0) > 0)
    cov_q4 = sum(
        1 for m in matches if "Q4" in quarter_scores.get(m.match_id, {})
    )
    cov_ot = sum(
        1
        for m in matches
        if any(k.startswith("OT") for k in quarter_scores.get(m.match_id, {}))
    )

    # Halftime baseline (predict winner by halftime leader)
    ht_total = 0
    ht_non_tie = 0
    ht_correct = 0
    for m in matches:
        q = quarter_scores.get(m.match_id, {})
        if "Q1" not in q or "Q2" not in q:
            continue
        ht_total += 1
        ht_home = q["Q1"][0] + q["Q2"][0]
        ht_away = q["Q1"][1] + q["Q2"][1]
        if ht_home == ht_away:
            continue
        ht_non_tie += 1
        pred_home_win = ht_home > ht_away
        true_home_win = m.home_score > m.away_score
        if pred_home_win == true_home_win:
            ht_correct += 1

    halftime_rows = [
        {
            "matches_with_q1_q2": ht_total,
            "matches_non_tie_ht": ht_non_tie,
            "ht_leader_accuracy": _safe_rate(ht_correct, ht_non_tie),
        }
    ]

    # Prior-strength baseline from short history (last 10 games per team)
    team_history: dict[str, list[int]] = defaultdict(list)
    by_time = sorted(matches, key=lambda m: _to_dt(m.date, m.time))
    ps_total = 0
    ps_usable = 0
    ps_correct = 0

    for m in by_time:
        ps_total += 1
        h_hist = team_history[m.home_team][-10:]
        a_hist = team_history[m.away_team][-10:]
        if len(h_hist) < 3 or len(a_hist) < 3:
            h_win = 1 if m.home_score > m.away_score else 0
            a_win = 1 - h_win
            team_history[m.home_team].append(h_win)
            team_history[m.away_team].append(a_win)
            continue

        h_rate = sum(h_hist) / len(h_hist)
        a_rate = sum(a_hist) / len(a_hist)
        if h_rate == a_rate:
            h_win = 1 if m.home_score > m.away_score else 0
            a_win = 1 - h_win
            team_history[m.home_team].append(h_win)
            team_history[m.away_team].append(a_win)
            continue

        ps_usable += 1
        pred_home_win = h_rate > a_rate
        true_home_win = m.home_score > m.away_score
        if pred_home_win == true_home_win:
            ps_correct += 1

        h_win = 1 if true_home_win else 0
        a_win = 1 - h_win
        team_history[m.home_team].append(h_win)
        team_history[m.away_team].append(a_win)

    prior_strength_rows = [
        {
            "matches_total": ps_total,
            "matches_usable": ps_usable,
            "prior_strength_accuracy": _safe_rate(ps_correct, ps_usable),
            "history_window_games": 10,
        }
    ]

    # CSV outputs
    league_rows = [
        {"league": league, "matches": cnt, "share": _safe_rate(cnt, n_matches)}
        for league, cnt in leagues.most_common()
    ]
    gender_rows = [
        {"bucket": bucket, "matches": cnt, "share": _safe_rate(cnt, n_matches)}
        for bucket, cnt in genders.most_common()
    ]
    coverage_rows = [
        {
            "metric": "matches_total",
            "value": n_matches,
        },
        {
            "metric": "coverage_play_by_play",
            "value": cov_pbp,
        },
        {
            "metric": "coverage_graph_points",
            "value": cov_graph,
        },
        {
            "metric": "coverage_has_q4",
            "value": cov_q4,
        },
        {
            "metric": "coverage_has_overtime",
            "value": cov_ot,
        },
    ]

    _write_csv(
        out_dir / "league_summary.csv",
        league_rows,
        ["league", "matches", "share"],
    )
    _write_csv(
        out_dir / "gender_summary.csv",
        gender_rows,
        ["bucket", "matches", "share"],
    )
    _write_csv(
        out_dir / "coverage_summary.csv",
        coverage_rows,
        ["metric", "value"],
    )
    _write_csv(
        out_dir / "halftime_baseline.csv",
        halftime_rows,
        ["matches_with_q1_q2", "matches_non_tie_ht", "ht_leader_accuracy"],
    )
    _write_csv(
        out_dir / "prior_strength_baseline.csv",
        prior_strength_rows,
        [
            "matches_total",
            "matches_usable",
            "prior_strength_accuracy",
            "history_window_games",
        ],
    )

    # Charts
    top_leagues = league_rows[:12]
    _make_chart_bar(
        out_dir / "top_leagues.png",
        [r["league"] for r in top_leagues],
        [r["matches"] for r in top_leagues],
        "Top leagues by match count",
    )
    _make_chart_bar(
        out_dir / "gender_mix.png",
        [r["bucket"] for r in gender_rows],
        [r["matches"] for r in gender_rows],
        "Gender mix (heuristic)",
    )

    summary = {
        "db_path": str(db_path),
        "n_matches": n_matches,
        "date_min": min(dates),
        "date_max": max(dates),
        "n_leagues": len(leagues),
        "gender_mix": dict(genders),
        "coverage": {
            "play_by_play": {
                "count": cov_pbp,
                "rate": _safe_rate(cov_pbp, n_matches),
            },
            "graph_points": {
                "count": cov_graph,
                "rate": _safe_rate(cov_graph, n_matches),
            },
            "has_q4": {
                "count": cov_q4,
                "rate": _safe_rate(cov_q4, n_matches),
            },
            "has_overtime": {
                "count": cov_ot,
                "rate": _safe_rate(cov_ot, n_matches),
            },
        },
        "baselines": {
            "halftime_leader_accuracy": _safe_rate(ht_correct, ht_non_tie),
            "halftime_sample_non_tie": ht_non_tie,
            "prior_strength_accuracy": _safe_rate(ps_correct, ps_usable),
            "prior_strength_sample": ps_usable,
        },
        "notes": [
            "Gender bucket is heuristic based on league/team text markers.",
            "Prior-strength baseline uses only short history "
            "in current DB window.",
            "For production modeling, use time-based train/validation split.",
        ],
    }

    with (out_dir / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    conn.close()
    return summary


def main() -> None:
    summary = run_eda()
    print("[eda] done")
    print(
        f"[eda] matches={summary['n_matches']} "
        f"leagues={summary['n_leagues']}"
    )
    print(
        "[eda] coverage "
        f"pbp={summary['coverage']['play_by_play']['rate']:.2%} "
        f"graph={summary['coverage']['graph_points']['rate']:.2%} "
        f"q4={summary['coverage']['has_q4']['rate']:.2%}"
    )
    print(
        "[eda] baselines "
        f"halftime={summary['baselines']['halftime_leader_accuracy']:.2%} "
        f"prior_strength={summary['baselines']['prior_strength_accuracy']:.2%}"
    )
    print(f"[eda] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
