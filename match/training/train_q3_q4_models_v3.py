"""Train V3 dynamic live models for Q3/Q4 based on game minute snapshots.

V3 focus:
- Q3 winner model at halftime snapshot (minute 24).
- Q4 winner models at minute snapshots 24, 30, and 36.
- Two base models per snapshot: logistic regression + gradient boosting.
- Ensemble probability as mean(logreg, gb).
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = importlib.import_module("db")

DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs_v3"

TOP_LEAGUES = 20
TOP_TEAMS = 120
TEAM_HISTORY_WINDOW = 12


@dataclass
class SampleRow:
    match_id: str
    dt: datetime
    target: str
    snapshot_minute: int
    features: dict
    y: int


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


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


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _is_complete_for_v3(data: dict) -> bool:
    quarters = data.get("score", {}).get("quarters", {})
    required = {"Q1", "Q2", "Q3", "Q4"}
    if not required.issubset(quarters.keys()):
        return False
    if not data.get("graph_points"):
        return False
    if not data.get("play_by_play"):
        return False
    return True


def _collect_top_buckets(conn) -> tuple[set[str], set[str]]:
    league_rows = conn.execute(
        "SELECT league, COUNT(*) AS n "
        "FROM matches GROUP BY league ORDER BY n DESC"
    ).fetchall()
    top_league_set = {
        str(row[0]) if row[0] else ""
        for row in league_rows[:TOP_LEAGUES]
    }

    team_counter: Counter[str] = Counter()
    team_rows = conn.execute(
        "SELECT home_team, away_team FROM matches"
    ).fetchall()
    for home_team, away_team in team_rows:
        if home_team:
            team_counter[str(home_team)] += 1
        if away_team:
            team_counter[str(away_team)] += 1

    top_team_set = {team for team, _ in team_counter.most_common(TOP_TEAMS)}
    return top_league_set, top_team_set


def _bucket(value: str, top_set: set[str], prefix: str) -> str:
    if value in top_set:
        return value
    return f"{prefix}_OTHER"


def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
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
        }

    area_home = sum(max(v, 0) for v in values)
    area_away = sum(max(-v, 0) for v in values)
    mean_abs = sum(abs(v) for v in values) / len(values)

    swings = 0
    prev_sign = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign

    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_peak_home": max(values),
        "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home,
        "gp_area_away": area_away,
        "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs,
        "gp_swings": swings,
    }


def _quarter_index(label: str) -> int | None:
    if label.startswith("Q"):
        try:
            return int(label[1:])
        except ValueError:
            return None
    return None


def _clock_to_seconds(clock: str) -> int | None:
    if not clock or ":" not in clock:
        return None
    try:
        mm, ss = clock.split(":", 1)
        return int(mm) * 60 + int(ss)
    except ValueError:
        return None


def _pbp_events_upto(data: dict, cutoff_minute: int) -> list[dict]:
    pbp = data.get("play_by_play", {})
    events = []

    for quarter_label, plays in pbp.items():
        q_idx = _quarter_index(quarter_label)
        if q_idx is None or q_idx < 1 or q_idx > 4:
            continue

        q_start = (q_idx - 1) * 12.0
        for play in plays:
            rem_sec = _clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None:
                continue
            elapsed_in_q = 12.0 - (rem_sec / 60.0)
            global_min = q_start + elapsed_in_q
            if global_min <= cutoff_minute + 1e-9:
                e = dict(play)
                e["_global_min"] = global_min
                events.append(e)

    events.sort(key=lambda e: float(e.get("_global_min", 0.0)))
    return events


def _pbp_stats_upto(data: dict, cutoff_minute: int) -> dict:
    events = _pbp_events_upto(data, cutoff_minute)
    home_plays = 0
    away_plays = 0
    home_3pt = 0
    away_3pt = 0

    for event in events:
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == "home":
            home_plays += 1
            if pts == 3:
                home_3pt += 1
        elif team == "away":
            away_plays += 1
            if pts == 3:
                away_3pt += 1

    total_plays = home_plays + away_plays
    total_3pt = home_3pt + away_3pt
    return {
        "pbp_home_plays": home_plays,
        "pbp_away_plays": away_plays,
        "pbp_plays_diff": home_plays - away_plays,
        "pbp_home_3pt": home_3pt,
        "pbp_away_3pt": away_3pt,
        "pbp_3pt_diff": home_3pt - away_3pt,
        "pbp_home_plays_share": _safe_rate(home_plays, total_plays),
        "pbp_home_3pt_share": _safe_rate(home_3pt, total_3pt),
    }


def _score_upto(data: dict, cutoff_minute: int) -> tuple[int, int]:
    events = _pbp_events_upto(data, cutoff_minute)
    if not events:
        return 0, 0

    home = 0
    away = 0
    for event in events:
        hs = event.get("home_score")
        as_ = event.get("away_score")
        if hs is not None and as_ is not None:
            home = int(hs)
            away = int(as_)
    return home, away


def _team_prior_wr(
    conn,
    team_name: str,
    match_date: str,
    match_time: str,
    window: int,
) -> float:
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


def _build_features(
    conn,
    match_data: dict,
    top_leagues: set[str],
    top_teams: set[str],
    cutoff_minute: int,
    target: str,
) -> dict:
    m = match_data["match"]

    q1h, q1a = _quarter_points(match_data, "Q1")
    q2h, q2a = _quarter_points(match_data, "Q2")
    q3h, q3a = _quarter_points(match_data, "Q3")

    home_team = m.get("home_team", "")
    away_team = m.get("away_team", "")
    league = m.get("league", "")

    home_prior_wr = _team_prior_wr(
        conn,
        home_team,
        m.get("date", ""),
        m.get("time", ""),
        TEAM_HISTORY_WINDOW,
    )
    away_prior_wr = _team_prior_wr(
        conn,
        away_team,
        m.get("date", ""),
        m.get("time", ""),
        TEAM_HISTORY_WINDOW,
    )

    est_home, est_away = _score_upto(match_data, cutoff_minute)

    base = {
        "league": league,
        "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
        "gender_bucket": _infer_gender(league, home_team, away_team),
        "home_team_bucket": _bucket(home_team, top_teams, "TEAM"),
        "away_team_bucket": _bucket(away_team, top_teams, "TEAM"),
        "home_prior_wr": home_prior_wr,
        "away_prior_wr": away_prior_wr,
        "prior_wr_diff": home_prior_wr - away_prior_wr,
        "prior_wr_sum": home_prior_wr + away_prior_wr,
        "q1_diff": (q1h or 0) - (q1a or 0),
        "q2_diff": (q2h or 0) - (q2a or 0),
        "cutoff_minute": cutoff_minute,
        "score_est_home": est_home,
        "score_est_away": est_away,
        "score_est_diff": est_home - est_away,
    }

    ht_home = (q1h or 0) + (q2h or 0)
    ht_away = (q1a or 0) + (q2a or 0)

    if target == "q3":
        out = dict(base)
        out.update({
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_diff": ht_home - ht_away,
            "ht_total": ht_home + ht_away,
        })
        out.update(_graph_stats_upto(match_data.get("graph_points", []), 24))
        out.update(_pbp_stats_upto(match_data, 24))
        return out

    out = dict(base)
    out.update({
        "q3_diff": (q3h or 0) - (q3a or 0),
        "q3_total": (q3h or 0) + (q3a or 0),
        "score_3q_home": ht_home + (q3h or 0),
        "score_3q_away": ht_away + (q3a or 0),
        "score_3q_diff": (ht_home + (q3h or 0)) - (ht_away + (q3a or 0)),
    })
    out.update(
        _graph_stats_upto(match_data.get("graph_points", []), cutoff_minute)
    )
    out.update(_pbp_stats_upto(match_data, cutoff_minute))
    return out


def _metric_row(
    target: str,
    snapshot: int,
    model: str,
    n_total: int,
    n_train: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    row = {
        "target": target,
        "snapshot_minute": snapshot,
        "model": model,
        "samples_total": n_total,
        "samples_train": n_train,
        "samples_test": n_test,
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds)), 6),
        "precision": round(float(precision_score(y_true, preds)), 6),
        "recall": round(float(recall_score(y_true, preds)), 6),
        "log_loss": round(float(log_loss(y_true, probs)), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }
    try:
        row["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 6)
    except ValueError:
        row["roc_auc"] = None
    return row


def _train_group(
    rows: list[SampleRow],
    target: str,
    snapshot: int,
) -> list[dict]:
    rows = sorted(rows, key=lambda r: r.dt)
    x_dict = [r.features for r in rows]
    y = [r.y for r in rows]

    n_total = len(rows)
    if n_total < 200:
        return []

    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_dict)
    x_train = x_all[:n_train]
    x_test = x_all[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    models = {
        "logreg": LogisticRegression(
            solver="liblinear",
            max_iter=5000,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
    }

    metrics_rows = []
    probs_map: dict[str, list[float]] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        probs_map[name] = list(probs)

        metrics_rows.append(
            _metric_row(
                target,
                snapshot,
                name,
                n_total,
                n_train,
                n_test,
                y_test,
                list(probs),
            )
        )

        artifact = {
            "version": "v3",
            "target": target,
            "snapshot_minute": snapshot,
            "model_name": name,
            "vectorizer": vec,
            "model": model,
            "feature_count": len(vec.feature_names_),
            "trained_rows": n_total,
        }
        path = OUT_DIR / f"{target}_m{snapshot}_{name}.joblib"
        joblib.dump(artifact, path)

    ensemble_probs = []
    for i in range(len(y_test)):
        ensemble_probs.append(
            (probs_map["logreg"][i] + probs_map["gb"][i]) / 2.0
        )

    metrics_rows.append(
        _metric_row(
            target,
            snapshot,
            "ensemble_avg_prob",
            n_total,
            n_train,
            n_test,
            y_test,
            ensemble_probs,
        )
    )

    return metrics_rows


def _build_samples(db_path: Path) -> list[SampleRow]:
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)

    top_leagues, top_teams = _collect_top_buckets(conn)

    matches = conn.execute(
        "SELECT match_id, date, time "
        "FROM matches ORDER BY date, time, match_id"
    ).fetchall()

    samples: list[SampleRow] = []
    for row in matches:
        match_id = str(row["match_id"])
        dt = datetime.strptime(
            f"{row['date']} {row['time']}",
            "%Y-%m-%d %H:%M",
        )
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_for_v3(data):
            continue

        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        if None in (q3h, q3a, q4h, q4a):
            continue

        y_q3 = None if q3h == q3a else int(q3h > q3a)
        y_q4 = None if q4h == q4a else int(q4h > q4a)

        if y_q3 is not None:
            f_q3_m24 = _build_features(
                conn,
                data,
                top_leagues,
                top_teams,
                cutoff_minute=24,
                target="q3",
            )
            samples.append(
                SampleRow(
                    match_id=match_id,
                    dt=dt,
                    target="q3",
                    snapshot_minute=24,
                    features=f_q3_m24,
                    y=y_q3,
                )
            )

        if y_q4 is not None:
            for cutoff in (24, 30, 36):
                f_q4 = _build_features(
                    conn,
                    data,
                    top_leagues,
                    top_teams,
                    cutoff_minute=cutoff,
                    target="q4",
                )
                samples.append(
                    SampleRow(
                        match_id=match_id,
                        dt=dt,
                        target="q4",
                        snapshot_minute=cutoff,
                        features=f_q4,
                        y=y_q4,
                    )
                )

    conn.close()
    return samples


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = _build_samples(DB_PATH)
    metrics: list[dict] = []

    for target, snaps in (("q3", [24]), ("q4", [24, 30, 36])):
        for snap in snaps:
            group = [
                s for s in samples
                if s.target == target and s.snapshot_minute == snap
            ]
            rows = _train_group(group, target=target, snapshot=snap)
            metrics.extend(rows)

    _write_csv(OUT_DIR / "v3_metrics.csv", metrics)

    summary = {
        "version": "v3",
        "targets": {
            "q3": {"snapshots": [24]},
            "q4": {"snapshots": [24, 30, 36]},
        },
        "models": ["logreg", "gb", "ensemble_avg_prob"],
        "rows_metrics": len(metrics),
    }
    with (OUT_DIR / "v3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[train-v3] done")
    print(f"[train-v3] samples={len(samples)}")
    print(f"[train-v3] metrics_rows={len(metrics)}")
    print(f"[train-v3] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
