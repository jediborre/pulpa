"""Train V4 multi-model predictors for quarter winners (Q3, Q4).

V4 keeps V2 features and adds advanced pressure/comeback features based on
global score context and play-by-play pace proxies.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
OUT_DIR = ROOT / "training" / "model_outputs_v4"
TOP_LEAGUES = 20
TOP_TEAMS = 120
TEAM_HISTORY_WINDOW = 12


@dataclass
class MatchSample:
    match_id: str
    dt: datetime
    features_q3: dict
    target_q3: int | None
    features_q4: dict
    target_q4: int | None


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _count_sign_swings(values: list[int]) -> int:
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
    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_peak_home": max(values),
        "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home,
        "gp_area_away": area_away,
        "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs,
        "gp_swings": _count_sign_swings(values),
    }


def _pbp_stats_upto(pbp: dict, quarters: list[str]) -> dict:
    home_plays = 0
    away_plays = 0
    home_3pt = 0
    away_3pt = 0
    for quarter in quarters:
        for play in pbp.get(quarter, []):
            team = play.get("team")
            pts = int(play.get("points", 0))
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


def _pbp_events_upto_minute(pbp: dict, cutoff_minute: float) -> list[dict]:
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


def _max_scoring_run(events: list[dict], team_name: str) -> int:
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


def _pbp_recent_window_features(
    pbp: dict,
    cutoff_minute: float,
    window_minutes: float,
) -> dict:
    events_upto = _pbp_events_upto_minute(pbp, cutoff_minute)
    start_min = max(0.0, cutoff_minute - window_minutes)
    events = [
        e
        for e in events_upto
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
    home_run = _max_scoring_run(events, "home")
    away_run = _max_scoring_run(events, "away")

    return {
        "clutch_window_minutes": window_minutes,
        "clutch_scoring_events": scoring_events,
        "clutch_home_points": home_points,
        "clutch_away_points": away_points,
        "clutch_points_diff": home_points - away_points,
        "clutch_home_event_share": _safe_rate(home_events, scoring_events),
        "clutch_home_max_run_pts": home_run,
        "clutch_away_max_run_pts": away_run,
        "clutch_run_diff": home_run - away_run,
        "clutch_last_scoring_home": int(last_scoring == "home"),
        "clutch_last_scoring_away": int(last_scoring == "away"),
    }


def _score_pressure_features(
    *,
    score_home: int,
    score_away: int,
    pbp_home_plays: int,
    pbp_away_plays: int,
    pbp_home_3pt: int,
    pbp_away_3pt: int,
    elapsed_minutes: float,
    minutes_left: float,
) -> dict:
    diff = score_home - score_away
    abs_diff = abs(diff)

    if diff > 0:
        trailing_side = "away"
        trailing_score = score_away
        trailing_plays = pbp_away_plays
        trailing_3pt = pbp_away_3pt
        leading_score = score_home
    elif diff < 0:
        trailing_side = "home"
        trailing_score = score_home
        trailing_plays = pbp_home_plays
        trailing_3pt = pbp_home_3pt
        leading_score = score_away
    else:
        trailing_side = "tied"
        trailing_score = max(score_home, score_away)
        trailing_plays = max(pbp_home_plays, pbp_away_plays)
        trailing_3pt = max(pbp_home_3pt, pbp_away_3pt)
        leading_score = trailing_score

    points_to_tie = abs_diff
    points_to_lead = abs_diff + (0 if trailing_side == "tied" else 1)

    total_plays = pbp_home_plays + pbp_away_plays
    pace_events_per_min = _safe_rate(total_plays, elapsed_minutes)
    trailing_play_share = _safe_rate(trailing_plays, total_plays)
    trailing_plays_per_min = _safe_rate(trailing_plays, elapsed_minutes)

    trailing_points_per_min = _safe_rate(trailing_score, elapsed_minutes)
    leading_points_per_min = _safe_rate(leading_score, elapsed_minutes)
    trailing_points_per_play = _safe_rate(trailing_score, trailing_plays)

    required_ppm_tie = _safe_rate(points_to_tie, minutes_left)
    required_ppm_lead = _safe_rate(points_to_lead, minutes_left)

    exp_total_events_left = pace_events_per_min * minutes_left
    exp_trailing_events_left = exp_total_events_left * trailing_play_share
    req_pts_per_trailing_event = _safe_rate(
        points_to_tie,
        exp_trailing_events_left,
    )

    pressure_ratio_tie = _safe_rate(required_ppm_tie, trailing_points_per_min)
    pressure_ratio_lead = _safe_rate(
        required_ppm_lead,
        trailing_points_per_min,
    )
    scoring_gap_per_min = trailing_points_per_min - leading_points_per_min
    urgency_index = _safe_rate(points_to_lead, minutes_left) * (
        1.0 + max(0.0, -scoring_gap_per_min)
    )

    return {
        "global_diff": diff,
        "global_abs_diff": abs_diff,
        "is_tied": int(diff == 0),
        "trailing_is_home": int(trailing_side == "home"),
        "trailing_is_away": int(trailing_side == "away"),
        "trailing_points_to_tie": points_to_tie,
        "trailing_points_to_lead": points_to_lead,
        "remaining_minutes_target": minutes_left,
        "required_ppm_tie": required_ppm_tie,
        "required_ppm_lead": required_ppm_lead,
        "trailing_points_per_min": trailing_points_per_min,
        "leading_points_per_min": leading_points_per_min,
        "trailing_points_per_play": trailing_points_per_play,
        "trailing_play_share": trailing_play_share,
        "trailing_plays_per_min": trailing_plays_per_min,
        "req_pts_per_trailing_event": req_pts_per_trailing_event,
        "pressure_ratio_tie": pressure_ratio_tie,
        "pressure_ratio_lead": pressure_ratio_lead,
        "scoring_gap_per_min": scoring_gap_per_min,
        "urgency_index": urgency_index,
        "trailing_3pt_rate": _safe_rate(trailing_3pt, trailing_plays),
    }


def _is_complete_match(data: dict) -> bool:
    quarters = data.get("score", {}).get("quarters", {})
    required = {"Q1", "Q2", "Q3", "Q4"}
    if not required.issubset(quarters.keys()):
        return False
    if not data.get("graph_points"):
        return False
    if not data.get("play_by_play"):
        return False
    return True


def _collect_top_buckets(conn, top_leagues: int, top_teams: int):
    league_rows = conn.execute(
        "SELECT league, COUNT(*) AS n "
        "FROM matches GROUP BY league ORDER BY n DESC"
    ).fetchall()
    team_counter: Counter[str] = Counter()
    team_rows = conn.execute(
        "SELECT home_team, away_team FROM matches"
    ).fetchall()
    for home_team, away_team in team_rows:
        if home_team:
            team_counter[str(home_team)] += 1
        if away_team:
            team_counter[str(away_team)] += 1

    top_league_set = {
        str(r[0]) if r[0] else ""
        for r in league_rows[:top_leagues]
    }
    top_team_set = {team for team, _ in team_counter.most_common(top_teams)}
    return top_league_set, top_team_set


def _bucket(value: str, top_set: set[str], prefix: str) -> str:
    if value in top_set:
        return value
    return f"{prefix}_OTHER"


def _build_samples(db_path: Path) -> list[MatchSample]:
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)

    top_leagues, top_teams = _collect_top_buckets(
        conn,
        top_leagues=TOP_LEAGUES,
        top_teams=TOP_TEAMS,
    )

    rows = conn.execute(
        "SELECT match_id, date, time FROM matches "
        "ORDER BY date, time, match_id"
    ).fetchall()

    team_history: dict[str, list[int]] = defaultdict(list)
    samples: list[MatchSample] = []

    for row in rows:
        match_id = str(row["match_id"])
        dt = datetime.strptime(
            f"{row['date']} {row['time']}",
            "%Y-%m-%d %H:%M",
        )
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_match(data):
            continue

        match_info = data["match"]
        score = data["score"]
        pbp = data.get("play_by_play", {})
        graph_points = data.get("graph_points", [])

        q1h, q1a = _quarter_points(data, "Q1")
        q2h, q2a = _quarter_points(data, "Q2")
        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue

        home_team = match_info.get("home_team", "")
        away_team = match_info.get("away_team", "")
        league = match_info.get("league", "")

        home_hist = team_history[home_team][-TEAM_HISTORY_WINDOW:]
        away_hist = team_history[away_team][-TEAM_HISTORY_WINDOW:]
        home_prior_wr = _safe_rate(sum(home_hist), len(home_hist))
        away_prior_wr = _safe_rate(sum(away_hist), len(away_hist))

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
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
        }

        ht_home = q1h + q2h
        ht_away = q1a + q2a

        features_q3 = dict(base)
        features_q3.update({
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_diff": ht_home - ht_away,
            "ht_total": ht_home + ht_away,
        })
        q3_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2"])
        features_q3.update(_graph_stats_upto(graph_points, 24))
        features_q3.update(q3_pbp_stats)
        features_q3.update(
            _score_pressure_features(
                score_home=ht_home,
                score_away=ht_away,
                pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
                pbp_home_3pt=q3_pbp_stats["pbp_home_3pt"],
                pbp_away_3pt=q3_pbp_stats["pbp_away_3pt"],
                elapsed_minutes=24.0,
                minutes_left=12.0,
            )
        )
        features_q3.update(
            _pbp_recent_window_features(
                pbp,
                cutoff_minute=24.0,
                window_minutes=6.0,
            )
        )

        features_q4 = dict(base)
        score_3q_home = ht_home + q3h
        score_3q_away = ht_away + q3a
        features_q4.update({
            "q3_diff": q3h - q3a,
            "q3_total": q3h + q3a,
            "score_3q_home": score_3q_home,
            "score_3q_away": score_3q_away,
            "score_3q_diff": score_3q_home - score_3q_away,
        })
        q4_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])
        features_q4.update(_graph_stats_upto(graph_points, 36))
        features_q4.update(q4_pbp_stats)
        features_q4.update(
            _score_pressure_features(
                score_home=score_3q_home,
                score_away=score_3q_away,
                pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
                pbp_home_3pt=q4_pbp_stats["pbp_home_3pt"],
                pbp_away_3pt=q4_pbp_stats["pbp_away_3pt"],
                elapsed_minutes=36.0,
                minutes_left=12.0,
            )
        )
        features_q4.update(
            _pbp_recent_window_features(
                pbp,
                cutoff_minute=36.0,
                window_minutes=6.0,
            )
        )

        target_q3 = None if q3h == q3a else int(q3h > q3a)
        target_q4 = None if q4h == q4a else int(q4h > q4a)

        samples.append(
            MatchSample(
                match_id=match_id,
                dt=dt,
                features_q3=features_q3,
                target_q3=target_q3,
                features_q4=features_q4,
                target_q4=target_q4,
            )
        )

        home_win = int(score["home"] > score["away"])
        away_win = 1 - home_win
        team_history[home_team].append(home_win)
        team_history[away_team].append(away_win)

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


def _metric_row(
    target: str,
    model_name: str,
    n_total: int,
    n_train: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]

    metric = {
        "target": target,
        "model": model_name,
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
        metric["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 6)
    except ValueError:
        metric["roc_auc"] = None

    return metric


def _train_target(samples: list[MatchSample], target_name: str) -> dict:
    if target_name == "q3":
        target_rows = [s for s in samples if s.target_q3 is not None]
        target_rows = sorted(target_rows, key=lambda item: item.dt)
        x_dict = [s.features_q3 for s in target_rows]
        y = [int(s.target_q3) for s in target_rows]
    else:
        target_rows = [s for s in samples if s.target_q4 is not None]
        target_rows = sorted(target_rows, key=lambda item: item.dt)
        x_dict = [s.features_q4 for s in target_rows]
        y = [int(s.target_q4) for s in target_rows]

    n_total = len(target_rows)
    if n_total < 200:
        raise RuntimeError(
            f"Not enough rows for {target_name}. Need >=200, got {n_total}."
        )

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
        "rf": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
    }

    metrics_rows = []
    proba_map = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        proba_map[model_name] = list(probs)

        metrics_rows.append(
            _metric_row(
                target=target_name,
                model_name=model_name,
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                y_true=y_test,
                probs=list(probs),
            )
        )

        artifact = {
            "version": "v4",
            "target": target_name,
            "model_name": model_name,
            "vectorizer": vec,
            "model": model,
            "trained_rows": n_total,
            "feature_count": len(vec.feature_names_),
        }
        joblib.dump(artifact, OUT_DIR / f"{target_name}_{model_name}.joblib")

    ensemble_probs = []
    for idx in range(len(y_test)):
        vals = [proba_map[name][idx] for name in models]
        ensemble_probs.append(sum(vals) / len(vals))

    metrics_rows.append(
        _metric_row(
            target=target_name,
            model_name="ensemble_avg_prob",
            n_total=n_total,
            n_train=n_train,
            n_test=n_test,
            y_true=y_test,
            probs=ensemble_probs,
        )
    )

    agreement = sum(
        1
        for idx in range(len(y_test))
        if len(
            {
                int(proba_map["logreg"][idx] >= 0.5),
                int(proba_map["rf"][idx] >= 0.5),
                int(proba_map["gb"][idx] >= 0.5),
            }
        )
        == 1
    )

    consensus = {
        "version": "v4",
        "target": target_name,
        "n_test": len(y_test),
        "agreement_rate_all_models": round(agreement / len(y_test), 6),
    }

    return {
        "metrics": metrics_rows,
        "consensus": consensus,
        "n_rows": n_total,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = _build_samples(DB_PATH)

    q3_rows = []
    q4_rows = []
    for sample in samples:
        if sample.target_q3 is not None:
            row = {
                "match_id": sample.match_id,
                "datetime": sample.dt.isoformat(),
            }
            row.update(sample.features_q3)
            row["target_q3_home_win"] = sample.target_q3
            q3_rows.append(row)

        if sample.target_q4 is not None:
            row = {
                "match_id": sample.match_id,
                "datetime": sample.dt.isoformat(),
            }
            row.update(sample.features_q4)
            row["target_q4_home_win"] = sample.target_q4
            q4_rows.append(row)

    _write_csv(OUT_DIR / "q3_dataset.csv", q3_rows)
    _write_csv(OUT_DIR / "q4_dataset.csv", q4_rows)

    q3_result = _train_target(samples, "q3")
    q4_result = _train_target(samples, "q4")

    _write_csv(OUT_DIR / "q3_metrics.csv", q3_result["metrics"])
    _write_csv(OUT_DIR / "q4_metrics.csv", q4_result["metrics"])

    with (OUT_DIR / "q3_consensus.json").open("w", encoding="utf-8") as f:
        json.dump(q3_result["consensus"], f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "q4_consensus.json").open("w", encoding="utf-8") as f:
        json.dump(q4_result["consensus"], f, indent=2, ensure_ascii=False)

    print("[train-v4] done")
    print(f"[train-v4] samples_complete={len(samples)}")
    print(f"[train-v4] q3_rows={q3_result['n_rows']}")
    print(f"[train-v4] q4_rows={q4_result['n_rows']}")
    print(f"[train-v4] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
