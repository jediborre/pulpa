"""Train V9 Optimized Model - Simplified for speed.

V9 Principles:
- NO deep learning 
- Simple ensemble (LogReg + GB)
- New momentum/racha features
"""

from __future__ import annotations

import csv
import importlib
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = importlib.import_module("db")

DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs_v9"
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


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _monte_carlo_win_prob(
    score_home: int, score_away: int,
    pbp_home_plays: int, pbp_away_plays: int,
    pbp_home_pts: float, pbp_away_pts: float,
    elapsed_minutes: float, minutes_left: float,
    num_sims: int = 500,
) -> dict:
    home_ppm = _safe_rate(pbp_home_pts, elapsed_minutes)
    away_ppm = _safe_rate(pbp_away_pts, elapsed_minutes)
    
    if home_ppm <= 0 and away_ppm <= 0:
        if score_home == score_away:
            return {"mc_home_win_prob": 0.5}
        return {"mc_home_win_prob": 1.0 if score_home > score_away else 0.0}

    var_home = max(0.01, home_ppm * 1.3)
    var_away = max(0.01, away_ppm * 1.3)

    sim_home = np.random.normal(score_home + home_ppm * minutes_left, np.sqrt(var_home * minutes_left), num_sims)
    sim_away = np.random.normal(score_away + away_ppm * minutes_left, np.sqrt(var_away * minutes_left), num_sims)

    home_wins = np.sum(sim_home > sim_away)
    ties = np.sum(np.abs(sim_home - sim_away) < 0.5)
    
    return {"mc_home_win_prob": float(home_wins + 0.5 * ties) / float(num_sims)}


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
    markers = ["women", "woman", "female", "femen", "fem.", " ladies ", "(w)", " w ", "wnba", "girls"]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
    points = [p for p in graph_points if int(p.get("minute", 0)) <= max_minute]
    values = [int(p.get("value", 0)) for p in points]
    if not values:
        return {
            "gp_count": 0, "gp_last": 0, "gp_peak_home": 0, "gp_peak_away": 0,
            "gp_area_home": 0, "gp_area_away": 0, "gp_area_diff": 0,
            "gp_mean_abs": 0.0, "gp_swings": 0, "gp_slope_3m": 0, "gp_slope_5m": 0,
        }

    area_home = sum(max(v, 0) for v in values)
    area_away = sum(max(-v, 0) for v in values)
    mean_abs = sum(abs(v) for v in values) / len(values)
    
    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else (values[-1] - values[0])

    return {
        "gp_count": len(values), "gp_last": values[-1],
        "gp_peak_home": max(values), "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home, "gp_area_away": area_away, "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs, "gp_swings": _count_sign_swings(values),
        "gp_slope_3m": slope_3m, "gp_slope_5m": slope_5m,
    }


def _pbp_stats_upto(pbp: dict, quarters: list[str]) -> dict:
    home_plays, away_plays, home_3pt, away_3pt, home_pts, away_pts = 0, 0, 0, 0, 0, 0
    for quarter in quarters:
        for play in pbp.get(quarter, []):
            team, pts = play.get("team"), int(play.get("points", 0))
            if team == "home":
                home_plays += 1
                home_pts += pts
                if pts == 3: home_3pt += 1
            elif team == "away":
                away_plays += 1
                away_pts += pts
                if pts == 3: away_3pt += 1

    total_plays = home_plays + away_plays
    total_3pt = home_3pt + away_3pt
    return {
        "pbp_home_pts_per_play": _safe_rate(home_pts, home_plays),
        "pbp_away_pts_per_play": _safe_rate(away_pts, away_plays),
        "pbp_pts_per_play_diff": _safe_rate(home_pts, home_plays) - _safe_rate(away_pts, away_plays),
        "pbp_home_plays": home_plays, "pbp_away_plays": away_plays,
        "pbp_plays_diff": home_plays - away_plays, "pbp_home_3pt": home_3pt,
        "pbp_away_3pt": away_3pt, "pbp_3pt_diff": home_3pt - away_3pt,
        "pbp_home_plays_share": _safe_rate(home_plays, total_plays),
        "pbp_home_3pt_share": _safe_rate(home_3pt, total_3pt),
    }


def _quarter_index(label: str) -> int | None:
    if label.startswith("Q"):
        try: return int(label[1:])
        except ValueError: return None
    return None


def _clock_to_seconds(clock: str) -> int | None:
    if not clock or ":" not in clock: return None
    try:
        mm, ss = clock.split(":", 1)
        return int(mm) * 60 + int(ss)
    except ValueError: return None


def _pbp_events_upto_minute(pbp: dict, cutoff_minute: float) -> list[dict]:
    events = []
    for quarter_label, plays in pbp.items():
        q_idx = _quarter_index(quarter_label)
        if q_idx is None or q_idx < 1 or q_idx > 4: continue
        q_start = (q_idx - 1) * 12.0
        for play in plays:
            rem_sec = _clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None: continue
            elapsed_in_q = 12.0 - (rem_sec / 60.0)
            global_min = q_start + elapsed_in_q
            if global_min <= cutoff_minute + 1e-9:
                e = dict(play)
                e["_global_min"] = global_min
                events.append(e)
    events.sort(key=lambda e: float(e.get("_global_min", 0.0)))
    return events


def _max_scoring_run(events: list[dict], team_name: str) -> int:
    best, run = 0, 0
    for event in events:
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
            if run > best: best = run
        elif team in ("home", "away"):
            run = 0
    return best


def _current_scoring_run(events: list[dict], team_name: str) -> int:
    run = 0
    for event in reversed(events):
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
        else:
            break
    return run


def _pbp_recent_window_features(pbp: dict, cutoff_minute: float, window_minutes: float) -> dict:
    events_upto = _pbp_events_upto_minute(pbp, cutoff_minute)
    start_min = max(0.0, cutoff_minute - window_minutes)
    events = [e for e in events_upto if float(e.get("_global_min", 0.0)) >= start_min]

    home_points, away_points, home_events, away_events, last_scoring = 0, 0, 0, 0, "none"
    for event in events:
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == "home" and pts > 0:
            home_points += pts
            home_events += 1
            last_scoring = "home"
        elif team == "away" and pts > 0:
            away_points += pts
            away_events += 1
            last_scoring = "away"

    scoring_events = home_events + away_events
    return {
        "clutch_window_minutes": window_minutes,
        "clutch_scoring_events": scoring_events,
        "clutch_home_points": home_points,
        "clutch_away_points": away_points,
        "clutch_points_diff": home_points - away_points,
        "clutch_home_event_share": _safe_rate(home_events, scoring_events),
        "clutch_home_max_run_pts": _max_scoring_run(events, "home"),
        "clutch_away_max_run_pts": _max_scoring_run(events, "away"),
        "clutch_run_diff": _max_scoring_run(events, "home") - _max_scoring_run(events, "away"),
        "clutch_last_scoring_home": int(last_scoring == "home"),
        "clutch_last_scoring_away": int(last_scoring == "away"),
    }


def _score_pressure_features(*, score_home: int, score_away: int, pbp_home_plays: int, pbp_away_plays: int,
                             pbp_home_3pt: int, pbp_away_3pt: int, elapsed_minutes: float, minutes_left: float) -> dict:
    diff = score_home - score_away
    abs_diff = abs(diff)
    if diff > 0:
        trailing_side, trailing_score, trailing_plays, trailing_3pt, leading_score = "away", score_away, pbp_away_plays, pbp_away_3pt, score_home
    elif diff < 0:
        trailing_side, trailing_score, trailing_plays, trailing_3pt, leading_score = "home", score_home, pbp_home_plays, pbp_home_3pt, score_away
    else:
        trailing_side, trailing_score, trailing_plays, trailing_3pt, leading_score = "tied", max(score_home, score_away), max(pbp_home_plays, pbp_away_plays), max(pbp_home_3pt, pbp_away_3pt), max(score_home, score_away)

    pts_to_tie, pts_to_lead = abs_diff, abs_diff + (0 if trailing_side == "tied" else 1)
    tot_plays = pbp_home_plays + pbp_away_plays
    trailing_ps = _safe_rate(trailing_plays, tot_plays)
    t_ppm = _safe_rate(trailing_score, elapsed_minutes)
    return {
        "global_diff": diff, "global_abs_diff": abs_diff, "is_tied": int(diff == 0),
        "trailing_is_home": int(trailing_side == "home"), "trailing_is_away": int(trailing_side == "away"),
        "trailing_points_to_tie": pts_to_tie, "trailing_points_to_lead": pts_to_lead,
        "remaining_minutes_target": minutes_left,
        "required_ppm_tie": _safe_rate(pts_to_tie, minutes_left),
        "required_ppm_lead": _safe_rate(pts_to_lead, minutes_left),
        "trailing_points_per_min": t_ppm, "leading_points_per_min": _safe_rate(leading_score, elapsed_minutes),
        "trailing_points_per_play": _safe_rate(trailing_score, trailing_plays),
        "trailing_play_share": trailing_ps, "trailing_plays_per_min": _safe_rate(trailing_plays, elapsed_minutes),
        "req_pts_per_trailing_event": _safe_rate(pts_to_tie, _safe_rate(tot_plays, elapsed_minutes) * minutes_left * trailing_ps),
        "pressure_ratio_tie": _safe_rate(_safe_rate(pts_to_tie, minutes_left), t_ppm),
        "pressure_ratio_lead": _safe_rate(_safe_rate(pts_to_lead, minutes_left), t_ppm),
        "scoring_gap_per_min": t_ppm - _safe_rate(leading_score, elapsed_minutes),
        "urgency_index": _safe_rate(pts_to_lead, minutes_left) * (1.0 + max(0.0, - (t_ppm - _safe_rate(leading_score, elapsed_minutes)))),
        "trailing_3pt_rate": _safe_rate(trailing_3pt, trailing_plays),
    }


def _is_complete_match(data: dict) -> bool:
    quarters = data.get("score", {}).get("quarters", {})
    required = {"Q1", "Q2", "Q3", "Q4"}
    if not required.issubset(quarters.keys()): return False
    gp = data.get("graph_points", [])
    if not gp or len(gp) < 20: return False
    pbp = data.get("play_by_play", {})
    if not pbp or not required.issubset(pbp.keys()): return False
    for q in required:
        if len(pbp[q]) == 0: return False
    return True


def _bucket(value: str, top_set: set[str], prefix: str) -> str:
    return value if value in top_set else f"{prefix}_OTHER"


def _build_samples(db_path: Path) -> list[MatchSample]:
    print("[v9] Loading DB...")
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)

    league_rows = conn.execute("SELECT league, COUNT(*) AS n FROM matches GROUP BY league ORDER BY n DESC").fetchall()
    team_counter: Counter[str] = Counter()
    for ht, at in conn.execute("SELECT home_team, away_team FROM matches").fetchall():
        if ht: team_counter[str(ht)] += 1
        if at: team_counter[str(at)] += 1

    top_leagues = {str(r[0]) if r[0] else "" for r in league_rows[:TOP_LEAGUES]}
    top_teams = {team for team, _ in team_counter.most_common(TOP_TEAMS)}

    rows = conn.execute("SELECT match_id, date, time FROM matches ORDER BY date, time, match_id").fetchall()
    team_history: dict[str, list[int]] = defaultdict(list)
    samples: list[MatchSample] = []
    
    print(f"[v9] Processing {len(rows)} matches...")

    for idx, row in enumerate(rows):
        if idx % 1000 == 0:
            print(f"[v9] Progress: {idx}/{len(rows)}")
            
        match_id = str(row["match_id"])
        dt = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M")
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_match(data): continue

        m, score, pbp, gp = data["match"], data["score"], data.get("play_by_play", {}), data.get("graph_points", [])
        q1h, q1a = _quarter_points(data, "Q1")
        q2h, q2a = _quarter_points(data, "Q2")
        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a): continue

        ht, at, league = m.get("home_team", ""), m.get("away_team", ""), m.get("league", "")
        home_hist, away_hist = team_history[ht][-TEAM_HISTORY_WINDOW:], team_history[at][-TEAM_HISTORY_WINDOW:]
        home_prior_wr, away_prior_wr = _safe_rate(sum(home_hist), len(home_hist)), _safe_rate(sum(away_hist), len(away_hist))

        base = {
            "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
            "gender_bucket": _infer_gender(league, ht, at),
            "home_team_bucket": _bucket(ht, top_teams, "TEAM"),
            "away_team_bucket": _bucket(at, top_teams, "TEAM"),
            "home_prior_wr": home_prior_wr,
            "away_prior_wr": away_prior_wr,
            "prior_wr_diff": home_prior_wr - away_prior_wr,
            "prior_wr_sum": home_prior_wr + away_prior_wr,
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
        }

        ht_home, ht_away = q1h + q2h, q1a + q2a
        q3_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2"])
        
        f_q3 = dict(base, ht_home=ht_home, ht_away=ht_away, ht_diff=ht_home - ht_away)
        f_q3.update(_graph_stats_upto(gp, 24))
        f_q3.update(q3_pbp_stats)
        f_q3.update(_score_pressure_features(score_home=ht_home, score_away=ht_away, 
                                            pbp_home_plays=q3_pbp_stats["pbp_home_plays"], 
                                            pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
                                            pbp_home_3pt=q3_pbp_stats["pbp_home_3pt"],
                                            pbp_away_3pt=q3_pbp_stats["pbp_away_3pt"],
                                            elapsed_minutes=24.0, minutes_left=12.0))
        f_q3.update(_pbp_recent_window_features(pbp, 24.0, 6.0))
        
        q3_mc = _monte_carlo_win_prob(
            score_home=ht_home, score_away=ht_away,
            pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
            pbp_home_pts=q3_pbp_stats["pbp_home_pts_per_play"] * q3_pbp_stats["pbp_home_plays"],
            pbp_away_pts=q3_pbp_stats["pbp_away_pts_per_play"] * q3_pbp_stats["pbp_away_plays"],
            elapsed_minutes=24.0, minutes_left=12.0
        )
        f_q3.update(q3_mc)
        
        q3_events = _pbp_events_upto_minute(pbp, 24.0)
        f_q3["current_run_home"] = _current_scoring_run(q3_events, "home")
        f_q3["current_run_away"] = _current_scoring_run(q3_events, "away")
        f_q3["max_run_home"] = _max_scoring_run(q3_events, "home")
        f_q3["max_run_away"] = _max_scoring_run(q3_events, "away")
        f_q3["run_diff"] = _max_scoring_run(q3_events, "home") - _max_scoring_run(q3_events, "away")

        score_3q_home, score_3q_away = ht_home + q3h, ht_away + q3a
        q4_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])
        
        f_q4 = dict(base, q3_diff=q3h - q3a, score_3q_home=score_3q_home, score_3q_away=score_3q_away, score_3q_diff=score_3q_home - score_3q_away)
        f_q4.update(_graph_stats_upto(gp, 36))
        f_q4.update(q4_pbp_stats)
        f_q4.update(_score_pressure_features(score_home=score_3q_home, score_away=score_3q_away,
                                            pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
                                            pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
                                            pbp_home_3pt=q4_pbp_stats["pbp_home_3pt"],
                                            pbp_away_3pt=q4_pbp_stats["pbp_away_3pt"],
                                            elapsed_minutes=36.0, minutes_left=12.0))
        f_q4.update(_pbp_recent_window_features(pbp, 36.0, 6.0))
        
        q4_mc = _monte_carlo_win_prob(
            score_home=score_3q_home, score_away=score_3q_away,
            pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
            pbp_home_pts=q4_pbp_stats["pbp_home_pts_per_play"] * q4_pbp_stats["pbp_home_plays"],
            pbp_away_pts=q4_pbp_stats["pbp_away_pts_per_play"] * q4_pbp_stats["pbp_away_plays"],
            elapsed_minutes=36.0, minutes_left=12.0
        )
        f_q4.update(q4_mc)
        
        q4_events = _pbp_events_upto_minute(pbp, 36.0)
        f_q4["current_run_home"] = _current_scoring_run(q4_events, "home")
        f_q4["current_run_away"] = _current_scoring_run(q4_events, "away")
        f_q4["max_run_home"] = _max_scoring_run(q4_events, "home")
        f_q4["max_run_away"] = _max_scoring_run(q4_events, "away")
        f_q4["run_diff"] = _max_scoring_run(q4_events, "home") - _max_scoring_run(q4_events, "away")

        samples.append(MatchSample(match_id, dt, f_q3, None if q3h == q3a else int(q3h > q3a), f_q4, None if q4h == q4a else int(q4h > q4a)))
        team_history[ht].append(int(score["home"] > score["away"]))
        team_history[at].append(int(score["away"] > score["home"]))

    conn.close()
    print(f"[v9] Built {len(samples)} samples")
    return samples


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(target: str, model_name: str, n_total: int, n_train: int, n_test: int, y_true: list[int], probs: list[float]) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    m = {
        "target": target, "model": model_name, "samples_total": n_total,
        "samples_train": n_train, "samples_test": n_test,
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds)), 6),
        "precision": round(float(precision_score(y_true, preds)), 6),
        "recall": round(float(recall_score(y_true, preds)), 6),
        "log_loss": round(float(log_loss(y_true, probs)), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }
    try: m["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 6)
    except ValueError: m["roc_auc"] = None
    return m


def _train_target(samples: list[MatchSample], target_name: str) -> dict:
    print(f"[v9] Training {target_name}...")
    
    if target_name == "q3":
        target_rows = sorted([s for s in samples if s.target_q3 is not None], key=lambda i: i.dt)
        x_dict = [s.features_q3 for s in target_rows]
        y = [int(s.target_q3) for s in target_rows]
    else:
        target_rows = sorted([s for s in samples if s.target_q4 is not None], key=lambda i: i.dt)
        x_dict = [s.features_q4 for s in target_rows]
        y = [int(s.target_q4) for s in target_rows]

    n_total = len(target_rows)
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_dict)

    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = np.array(y[:n_train]), np.array(y[n_train:])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    metrics_rows = []
    models_trained = {}

    print(f"[{target_name}] Training on {n_train} samples with {x_train.shape[1]} features")

    # 1. LogReg - fastest
    print(f"[{target_name}] LogReg...")
    logreg = LogisticRegression(C=0.5, max_iter=500, random_state=42, solver="lbfgs")
    logreg.fit(x_train_scaled, y_train)
    probs = logreg.predict_proba(x_test_scaled)[:, 1]
    metrics_rows.append(_metric_row(target_name, "logreg", n_total, n_train, n_test, y_test.tolist(), probs.tolist()))
    models_trained["logreg"] = logreg

    # 2. GradientBoosting only (fast)
    print(f"[{target_name}] GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.15,
        min_samples_split=30, min_samples_leaf=15, random_state=42
    )
    gb.fit(x_train, y_train)
    probs = gb.predict_proba(x_test)[:, 1]
    metrics_rows.append(_metric_row(target_name, "gb", n_total, n_train, n_test, y_test.tolist(), probs.tolist()))
    models_trained["gb"] = gb

    # 3. Ensemble
    probs_lr = models_trained["logreg"].predict_proba(x_test_scaled)[:, 1]
    probs_gb = models_trained["gb"].predict_proba(x_test)[:, 1]
    
    # Simple average
    avg_probs = (probs_lr + probs_gb) / 2.0
    metrics_rows.append(_metric_row(target_name, "ensemble_avg_prob", n_total, n_train, n_test, y_test.tolist(), avg_probs.tolist()))
    
    # Weighted (more weight to GB like V6 for Q3)
    if target_name == "q3":
        weighted_probs = 0.35 * probs_lr + 0.65 * probs_gb
    else:
        weighted_probs = 0.50 * probs_lr + 0.50 * probs_gb
    metrics_rows.append(_metric_row(target_name, "ensemble_weighted", n_total, n_train, n_test, y_test.tolist(), weighted_probs.tolist()))

    # Save models
    for name, model in models_trained.items():
        artifact = {"version": "v9", "target": target_name, "model_name": name, "vectorizer": vec, "scaler": scaler, "model": model}
        joblib.dump(artifact, OUT_DIR / f"{target_name}_{name}.joblib")

    joblib.dump({"version": "v9", "target": target_name, "models": models_trained, "weights": weights}, OUT_DIR / f"{target_name}_ensemble.joblib")

    return {"metrics": metrics_rows, "n_rows": n_total}


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    samples = _build_samples(DB_PATH)
    
    res_q3 = _train_target(samples, "q3")
    _write_csv(OUT_DIR / "q3_metrics.csv", res_q3["metrics"])

    res_q4 = _train_target(samples, "q4")
    _write_csv(OUT_DIR / "q4_metrics.csv", res_q4["metrics"])
    
    elapsed = time.time() - start
    print(f"[train-v9] done in {elapsed:.1f}s")
    print(f"[train-v9] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()