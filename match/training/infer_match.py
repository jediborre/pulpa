"""Infer Q3/Q4 winners for one match_id using best V1/V2 models.

If match is missing in DB, this script can fetch it from SofaScore using
existing scraper and then save it into SQLite before inference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = importlib.import_module("db")
scraper_mod = importlib.import_module("scraper")

COMPARE_JSON = (
    ROOT / "training" / "model_comparison" / "version_comparison.json"
)
MODEL_DIR_V1 = ROOT / "training" / "model_outputs"
MODEL_DIR_V2 = ROOT / "training" / "model_outputs_v2"
MODEL_DIR_V3 = ROOT / "training" / "model_outputs_v3"
MODEL_DIR_V4 = ROOT / "training" / "model_outputs_v4"
GATE_CONFIG = ROOT / "training" / "model_outputs_v2" / "gate_config.json"
DB_PATH = ROOT / "matches.db"
_GATE_CACHE: dict | None | bool = None


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


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


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


def _pbp_stats_upto_minute(data: dict, cutoff_minute: int) -> dict:
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
    data: dict,
    cutoff_minute: float,
    window_minutes: float,
) -> dict:
    events_upto = _pbp_events_upto(data, int(cutoff_minute))
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


def _get_top_buckets(conn, top_leagues: int = 20, top_teams: int = 120):
    league_rows = conn.execute(
        "SELECT league, COUNT(*) AS n "
        "FROM matches GROUP BY league ORDER BY n DESC"
    ).fetchall()
    top_leagues_set = {
        str(row[0]) if row[0] else ""
        for row in league_rows[:top_leagues]
    }

    team_counter = {}
    rows = conn.execute("SELECT home_team, away_team FROM matches").fetchall()
    for home_team, away_team in rows:
        if home_team:
            team_counter[str(home_team)] = (
                team_counter.get(str(home_team), 0) + 1
            )
        if away_team:
            team_counter[str(away_team)] = (
                team_counter.get(str(away_team), 0) + 1
            )

    sorted_teams = sorted(
        team_counter.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_teams_set = {team for team, _ in sorted_teams[:top_teams]}
    return top_leagues_set, top_teams_set


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


def _pick_entry(compare_blob: dict, target: str, metric: str) -> dict | None:
    target_blob = compare_blob.get(target, {})
    key_map = {
        "accuracy": "best_accuracy",
        "f1": "best_f1",
        "log_loss": "best_log_loss",
    }
    key = key_map[metric]
    entry = target_blob.get(key)
    return entry if isinstance(entry, dict) else None


def _load_match(
    conn,
    match_id: str,
    fetch_missing: bool,
    refresh: bool = False,
) -> tuple[dict | None, bool]:
    data = db_mod.get_match(conn, match_id)
    if data is not None and not refresh:
        return data, False

    if not fetch_missing:
        # refresh requested but fetch disabled: return existing or None
        return data, False

    scraped = scraper_mod.fetch_match_by_id(match_id)
    db_mod.save_match(conn, match_id, scraped)
    data = db_mod.get_match(conn, match_id)
    return data, True


def _actual_quarter_outcome(data: dict, quarter: str) -> str | None:
    h, a = _quarter_points(data, quarter)
    if h is None or a is None:
        return None
    if h == a:
        return "push"
    return "home" if h > a else "away"


def _parse_live_clock(desc: str) -> str | None:
    m = re.search(r"(\d{1,2}:\d{2})", desc or "")
    if not m:
        return None
    return m.group(1)


def _infer_state(
    match_data: dict,
    event_snapshot: dict | None,
) -> dict:
    score = match_data.get("score", {})
    quarters = score.get("quarters", {})
    graph_points = match_data.get("graph_points", [])

    status_type = ""
    status_description = ""
    home_score = score.get("home", 0)
    away_score = score.get("away", 0)
    if event_snapshot:
        status_type = str(event_snapshot.get("status_type", "") or "")
        status_description = str(
            event_snapshot.get("status_description", "") or ""
        )
        home_score = int(event_snapshot.get("home_score", home_score) or 0)
        away_score = int(event_snapshot.get("away_score", away_score) or 0)

    state = "unknown"
    if status_type == "finished":
        state = "finished"
    elif status_type in ("inprogress", "inProgress"):
        state = "live"
    elif status_type in ("notstarted", "notStarted"):
        state = "not_started"
    elif "Q" in status_description or "OT" in status_description:
        state = "live"
    elif "Q4" in quarters and len(quarters) >= 4:
        state = "finished"

    clock_text = _parse_live_clock(status_description)
    minute_est = None
    if graph_points:
        minute_est = int(graph_points[-1].get("minute", 0))

    return {
        "state": state,
        "status_type": status_type,
        "status_description": status_description,
        "clock": clock_text,
        "minute_estimate": minute_est,
        "home_score": home_score,
        "away_score": away_score,
    }


def _required_ok(data: dict, target: str) -> tuple[bool, str]:
    score = data.get("score", {})
    quarters = score.get("quarters", {})
    graph_points = data.get("graph_points", [])
    pbp = data.get("play_by_play", {})

    if "Q1" not in quarters or "Q2" not in quarters:
        return False, "Missing Q1/Q2 scores"
    if target == "q4" and "Q3" not in quarters:
        return False, "Missing Q3 score"
    if not graph_points:
        return False, "Missing graph_points"
    if not pbp:
        return False, "Missing play_by_play"
    return True, "ok"


def _build_features(
    conn,
    match_data: dict,
    version: str,
    target: str,
) -> dict:
    m = match_data["match"]
    pbp = match_data.get("play_by_play", {})
    gp = match_data.get("graph_points", [])

    q1h, q1a = _quarter_points(match_data, "Q1")
    q2h, q2a = _quarter_points(match_data, "Q2")
    q3h, q3a = _quarter_points(match_data, "Q3")

    home_team = m.get("home_team", "")
    away_team = m.get("away_team", "")
    league = m.get("league", "")
    match_date = m.get("date", "")
    match_time = m.get("time", "")

    window = 12 if version in ("v2", "v4") else 10
    home_prior_wr = _team_prior_wr(
        conn,
        home_team,
        match_date,
        match_time,
        window=window,
    )
    away_prior_wr = _team_prior_wr(
        conn,
        away_team,
        match_date,
        match_time,
        window=window,
    )

    base = {
        "league": league,
        "gender_bucket": _infer_gender(league, home_team, away_team),
        "home_prior_wr": home_prior_wr,
        "away_prior_wr": away_prior_wr,
        "prior_wr_diff": home_prior_wr - away_prior_wr,
        "q1_diff": (q1h or 0) - (q1a or 0),
        "q2_diff": (q2h or 0) - (q2a or 0),
    }

    if version in ("v2", "v4"):
        top_leagues, top_teams = _get_top_buckets(conn)

        def bucket(value: str, top_set: set[str], prefix: str) -> str:
            if value in top_set:
                return value
            return f"{prefix}_OTHER"

        base["league_bucket"] = bucket(league, top_leagues, "LEAGUE")
        base["home_team_bucket"] = bucket(home_team, top_teams, "TEAM")
        base["away_team_bucket"] = bucket(away_team, top_teams, "TEAM")
        base["prior_wr_sum"] = home_prior_wr + away_prior_wr

    ht_home = (q1h or 0) + (q2h or 0)
    ht_away = (q1a or 0) + (q2a or 0)

    if target == "q3":
        feat = dict(base)
        feat.update({
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_diff": ht_home - ht_away,
        })
        if version in ("v2", "v4"):
            feat["ht_total"] = ht_home + ht_away
        feat.update(_graph_stats_upto(gp, 24))
        q3_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2"])
        feat.update(q3_pbp_stats)
        if version == "v4":
            feat.update(
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
            feat.update(
                _pbp_recent_window_features(
                    match_data,
                    cutoff_minute=24.0,
                    window_minutes=6.0,
                )
            )
        return feat

    feat = dict(base)
    feat.update({
        "q3_diff": (q3h or 0) - (q3a or 0),
        "score_3q_home": ht_home + (q3h or 0),
        "score_3q_away": ht_away + (q3a or 0),
        "score_3q_diff": (ht_home + (q3h or 0)) - (ht_away + (q3a or 0)),
    })
    if version in ("v2", "v4"):
        feat["q3_total"] = (q3h or 0) + (q3a or 0)
    feat.update(_graph_stats_upto(gp, 36))
    q4_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])
    feat.update(q4_pbp_stats)
    if version == "v4":
        feat.update(
            _score_pressure_features(
                score_home=ht_home + (q3h or 0),
                score_away=ht_away + (q3a or 0),
                pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
                pbp_home_3pt=q4_pbp_stats["pbp_home_3pt"],
                pbp_away_3pt=q4_pbp_stats["pbp_away_3pt"],
                elapsed_minutes=36.0,
                minutes_left=12.0,
            )
        )
        feat.update(
            _pbp_recent_window_features(
                match_data,
                cutoff_minute=36.0,
                window_minutes=6.0,
            )
        )
    return feat


def _predict_prob(
    version: str,
    target: str,
    model_name: str,
    features: dict,
) -> float:
    if version == "v4":
        model_dir = MODEL_DIR_V4
    elif version == "v2":
        model_dir = MODEL_DIR_V2
    else:
        model_dir = MODEL_DIR_V1

    def single(name: str) -> float:
        path = model_dir / f"{target}_{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        artifact = joblib.load(path)
        vec = artifact["vectorizer"]
        model = artifact["model"]
        x = vec.transform([features])
        return float(model.predict_proba(x)[0][1])

    if model_name == "ensemble_avg_prob":
        probs = [single("logreg"), single("rf"), single("gb")]
        return float(sum(probs) / len(probs))

    return single(model_name)


def _v3_available() -> bool:
    return MODEL_DIR_V3.exists()


def _predict_prob_v3(target: str, snapshot: int, features: dict) -> dict:
    def single(name: str) -> float:
        path = MODEL_DIR_V3 / f"{target}_m{snapshot}_{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        artifact = joblib.load(path)
        vec = artifact["vectorizer"]
        model = artifact["model"]
        x = vec.transform([features])
        return float(model.predict_proba(x)[0][1])

    p_logreg = single("logreg")
    p_gb = single("gb")
    p_ens = (p_logreg + p_gb) / 2.0
    return {
        "logreg": p_logreg,
        "gb": p_gb,
        "ensemble_avg_prob": p_ens,
    }


def _build_features_v3(
    conn,
    match_data: dict,
    target: str,
    snapshot_minute: int,
) -> dict:
    m = match_data["match"]

    q1h, q1a = _quarter_points(match_data, "Q1")
    q2h, q2a = _quarter_points(match_data, "Q2")
    q3h, q3a = _quarter_points(match_data, "Q3")

    home_team = m.get("home_team", "")
    away_team = m.get("away_team", "")
    league = m.get("league", "")

    top_leagues, top_teams = _get_top_buckets(conn)

    def bucket(value: str, top_set: set[str], prefix: str) -> str:
        if value in top_set:
            return value
        return f"{prefix}_OTHER"

    home_prior_wr = _team_prior_wr(
        conn,
        home_team,
        m.get("date", ""),
        m.get("time", ""),
        window=12,
    )
    away_prior_wr = _team_prior_wr(
        conn,
        away_team,
        m.get("date", ""),
        m.get("time", ""),
        window=12,
    )

    est_home, est_away = _score_upto(match_data, snapshot_minute)

    base = {
        "league": league,
        "league_bucket": bucket(league, top_leagues, "LEAGUE"),
        "gender_bucket": _infer_gender(league, home_team, away_team),
        "home_team_bucket": bucket(home_team, top_teams, "TEAM"),
        "away_team_bucket": bucket(away_team, top_teams, "TEAM"),
        "home_prior_wr": home_prior_wr,
        "away_prior_wr": away_prior_wr,
        "prior_wr_diff": home_prior_wr - away_prior_wr,
        "prior_wr_sum": home_prior_wr + away_prior_wr,
        "q1_diff": (q1h or 0) - (q1a or 0),
        "q2_diff": (q2h or 0) - (q2a or 0),
        "cutoff_minute": snapshot_minute,
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
        out.update(_pbp_stats_upto_minute(match_data, 24))
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
        _graph_stats_upto(match_data.get("graph_points", []), snapshot_minute)
    )
    out.update(_pbp_stats_upto_minute(match_data, snapshot_minute))
    return out


def _select_q4_snapshot(minute_est: int | None) -> int | None:
    if minute_est is None:
        return None
    if minute_est < 24:
        return None
    if minute_est < 30:
        return 24
    if minute_est < 36:
        return 30
    return 36


def _signal_thresholds(
    target: str,
    snapshot_minute: int | None,
) -> tuple[float, float]:
    """Return (lean_threshold, bet_threshold) for confidence signal."""
    if target == "q3":
        return 0.10, 0.18

    # q4 thresholds vary with available context depth.
    snap = snapshot_minute or 36
    if snap <= 24:
        return 0.14, 0.22
    if snap <= 30:
        return 0.11, 0.18
    return 0.08, 0.14


def _load_gate_config() -> dict | None:
    global _GATE_CACHE
    if _GATE_CACHE is not None:
        return None if _GATE_CACHE is False else _GATE_CACHE

    if not GATE_CONFIG.exists():
        _GATE_CACHE = False
        return None

    try:
        with GATE_CONFIG.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            _GATE_CACHE = False
            return None
        _GATE_CACHE = data
        return data
    except Exception:
        _GATE_CACHE = False
        return None


def _gate_thresholds(target: str, snapshot_minute: int | None) -> dict:
    if target == "q3":
        defaults = {
            "min_graph_points": 18,
            "min_pbp_events": 20,
            "volatility_block_at": 0.72,
            "min_edge": 0.08,
        }
        key = "default"
    else:
        snap = snapshot_minute or 36
        if snap <= 24:
            defaults = {
                "min_graph_points": 18,
                "min_pbp_events": 20,
                "volatility_block_at": 0.72,
                "min_edge": 0.08,
            }
            key = "24"
        elif snap <= 30:
            defaults = {
                "min_graph_points": 24,
                "min_pbp_events": 26,
                "volatility_block_at": 0.72,
                "min_edge": 0.08,
            }
            key = "30"
        else:
            defaults = {
                "min_graph_points": 30,
                "min_pbp_events": 32,
                "volatility_block_at": 0.72,
                "min_edge": 0.08,
            }
            key = "36"

    cfg = _load_gate_config()
    if not cfg:
        return defaults

    block = (
        cfg.get("thresholds", {})
        .get(target, {})
        .get(key)
    )
    if not isinstance(block, dict):
        return defaults

    out = dict(defaults)
    for name in (
        "min_graph_points",
        "min_pbp_events",
        "volatility_block_at",
        "min_edge",
    ):
        if name in block:
            out[name] = block[name]
    return out


def _bet_signal(
    target: str,
    confidence: float,
    snapshot_minute: int | None,
) -> dict:
    lean_thr, bet_thr = _signal_thresholds(target, snapshot_minute)

    if confidence >= bet_thr:
        return {
            "signal": "BET",
            "suggested_units": 1.0,
            "threshold_lean": lean_thr,
            "threshold_bet": bet_thr,
        }
    if confidence >= lean_thr:
        return {
            "signal": "LEAN",
            "suggested_units": 0.5,
            "threshold_lean": lean_thr,
            "threshold_bet": bet_thr,
        }
    return {
        "signal": "NO BET",
        "suggested_units": 0.0,
        "threshold_lean": lean_thr,
        "threshold_bet": bet_thr,
    }


def _sufficiency_thresholds(target: str, snapshot_minute: int | None) -> dict:
    thr = _gate_thresholds(target, snapshot_minute)
    return {
        "min_graph_points": int(thr["min_graph_points"]),
        "min_pbp_events": int(thr["min_pbp_events"]),
    }


def _lead_changes_upto(data: dict, cutoff_minute: int) -> int:
    events = _pbp_events_upto(data, cutoff_minute)
    changes = 0
    prev_sign = 0
    for event in events:
        hs = event.get("home_score")
        as_ = event.get("away_score")
        if hs is None or as_ is None:
            continue
        diff = int(hs) - int(as_)
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            changes += 1
        prev_sign = sign
    return changes


def _volatility_index(data: dict, cutoff_minute: int) -> dict:
    gp = data.get("graph_points", [])
    vals = [
        int(p.get("value", 0))
        for p in gp
        if int(p.get("minute", 0)) <= cutoff_minute
    ]
    swings = _count_sign_swings(vals)
    lead_changes = _lead_changes_upto(data, cutoff_minute)

    # Heuristic 0..1 scale to simplify gating.
    swings_norm = min(swings / 10.0, 1.0)
    leads_norm = min(lead_changes / 12.0, 1.0)
    index = round((0.6 * swings_norm) + (0.4 * leads_norm), 6)
    return {
        "index": index,
        "swings": swings,
        "lead_changes": lead_changes,
    }


def _decision_gate(
    match_data: dict,
    target: str,
    snapshot_minute: int | None,
    confidence: float,
    model_signal: str,
) -> dict:
    cutoff = snapshot_minute if snapshot_minute is not None else (
        24 if target == "q3" else 36
    )

    gp_count = len([
        p for p in match_data.get("graph_points", [])
        if int(p.get("minute", 0)) <= cutoff
    ])
    pbp_count = len(_pbp_events_upto(match_data, cutoff))

    thr = _sufficiency_thresholds(target, snapshot_minute)
    if gp_count < thr["min_graph_points"] or pbp_count < thr["min_pbp_events"]:
        return {
            "decision_gate": "BLOCK_LOW_DATA",
            "final_recommendation": "NO BET",
            "reason": "insufficient_graph_or_pbp_coverage",
            "gp_count": gp_count,
            "pbp_count": pbp_count,
            "volatility_index": None,
            "volatility_swings": None,
            "volatility_lead_changes": None,
        }

    gate_thr = _gate_thresholds(target, snapshot_minute)
    vol = _volatility_index(match_data, cutoff)
    if vol["index"] >= float(gate_thr["volatility_block_at"]):
        return {
            "decision_gate": "BLOCK_HIGH_VOLATILITY",
            "final_recommendation": "NO BET",
            "reason": "match_too_volatile_for_current_signal",
            "gp_count": gp_count,
            "pbp_count": pbp_count,
            "volatility_index": vol["index"],
            "volatility_swings": vol["swings"],
            "volatility_lead_changes": vol["lead_changes"],
        }

    # Extra low-edge guard even if model emits LEAN/BET.
    if confidence < float(gate_thr["min_edge"]):
        return {
            "decision_gate": "BLOCK_LOW_EDGE",
            "final_recommendation": "NO BET",
            "reason": "confidence_below_minimum_edge",
            "gp_count": gp_count,
            "pbp_count": pbp_count,
            "volatility_index": vol["index"],
            "volatility_swings": vol["swings"],
            "volatility_lead_changes": vol["lead_changes"],
        }

    return {
        "decision_gate": f"ALLOW_{model_signal.replace(' ', '_')}",
        "final_recommendation": model_signal,
        "reason": "passed_all_gates",
        "gp_count": gp_count,
        "pbp_count": pbp_count,
        "volatility_index": vol["index"],
        "volatility_swings": vol["swings"],
        "volatility_lead_changes": vol["lead_changes"],
    }


def run_inference(
    match_id: str,
    metric: str,
    fetch_missing: bool,
    force_version: str,
    refresh: bool = False,
) -> dict:
    if not COMPARE_JSON.exists():
        raise FileNotFoundError(
            "Comparison file missing. "
            "Run: python training/model_cli.py compare"
        )

    with COMPARE_JSON.open("r", encoding="utf-8") as f:
        compare_blob = json.load(f)

    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)

    match_data, fetched = _load_match(conn, match_id, fetch_missing, refresh)
    if not match_data:
        conn.close()
        return {
            "ok": False,
            "reason": "match_not_found",
            "match_id": match_id,
        }

    event_snapshot = None
    try:
        event_snapshot = scraper_mod.fetch_event_snapshot(match_id)
    except Exception:
        event_snapshot = None

    match_meta = match_data.get("match", {})
    state_info = _infer_state(match_data, event_snapshot)

    out = {
        "ok": True,
        "match_id": match_id,
        "fetched_now": fetched,
        "metric_for_pick": metric,
        "force_version": force_version,
        "match": {
            "home_team": match_meta.get("home_team", ""),
            "away_team": match_meta.get("away_team", ""),
            "league": match_meta.get("league", ""),
            "date": match_meta.get("date", ""),
            "time_utc": match_meta.get("time", ""),
            "home_score": state_info["home_score"],
            "away_score": state_info["away_score"],
            "state": state_info["state"],
            "status_description": state_info["status_description"],
            "clock": state_info["clock"],
            "minute_estimate": state_info["minute_estimate"],
        },
        "predictions": {},
    }

    def forced_version_for_target(target: str) -> str | None:
        if force_version in ("v1", "v2", "v4"):
            return force_version
        if force_version == "hybrid":
            return "v2" if target == "q3" else "v4"
        return None

    for target in ("q3", "q4"):
        ok_target, reason = _required_ok(match_data, target)
        if not ok_target:
            out["predictions"][target] = {
                "available": False,
                "reason": reason,
            }
            continue

        entry = _pick_entry(compare_blob, target, metric)
        if not entry:
            out["predictions"][target] = {
                "available": False,
                "reason": "no_best_model_entry",
            }
            continue

        version = str(entry.get("version", "v1"))
        model_name = str(entry.get("model", "logreg"))
        forced = forced_version_for_target(target)
        if forced is not None:
            version = forced

        use_v3_live = (
            out["match"].get("state") == "live"
            and _v3_available()
            and force_version == "auto"
        )

        if use_v3_live:
            if target == "q3":
                if (out["match"].get("minute_estimate") or 0) < 24:
                    out["predictions"][target] = {
                        "available": False,
                        "reason": "v3_q3_available_from_minute_24",
                    }
                    continue
                snap = 24
            else:
                snap = _select_q4_snapshot(out["match"].get("minute_estimate"))
                if snap is None:
                    out["predictions"][target] = {
                        "available": False,
                        "reason": "v3_q4_available_from_minute_24",
                    }
                    continue

            features = _build_features_v3(
                conn,
                match_data,
                target=target,
                snapshot_minute=snap,
            )
            probs = _predict_prob_v3(target, snap, features)
            prob_home = probs["ensemble_avg_prob"]
            version = "v3"
            model_name = "ensemble_avg_prob"
            snapshot_used = snap
        else:
            features = _build_features(conn, match_data, version, target)
            prob_home = _predict_prob(version, target, model_name, features)
            snapshot_used = None

        out["predictions"][target] = {
            "available": True,
            "version": version,
            "model": model_name,
            "snapshot_minute": snapshot_used,
            "p_home_win": round(prob_home, 6),
            "p_away_win": round(1.0 - prob_home, 6),
            "predicted_winner": "home" if prob_home >= 0.5 else "away",
            "confidence": round(abs(prob_home - 0.5) * 2.0, 6),
        }

        sig = _bet_signal(
            target=target,
            confidence=float(out["predictions"][target]["confidence"]),
            snapshot_minute=snapshot_used,
        )
        out["predictions"][target]["bet_signal"] = sig["signal"]
        out["predictions"][target]["suggested_units"] = sig["suggested_units"]
        out["predictions"][target]["threshold_lean"] = sig["threshold_lean"]
        out["predictions"][target]["threshold_bet"] = sig["threshold_bet"]

        gate = _decision_gate(
            match_data=match_data,
            target=target,
            snapshot_minute=snapshot_used,
            confidence=float(out["predictions"][target]["confidence"]),
            model_signal=sig["signal"],
        )
        out["predictions"][target]["decision_gate"] = gate["decision_gate"]
        out["predictions"][target]["final_recommendation"] = (
            gate["final_recommendation"]
        )
        out["predictions"][target]["gate_reason"] = gate["reason"]
        out["predictions"][target]["gate_gp_count"] = gate["gp_count"]
        out["predictions"][target]["gate_pbp_count"] = gate["pbp_count"]
        out["predictions"][target]["volatility_index"] = (
            gate["volatility_index"]
        )
        out["predictions"][target]["volatility_swings"] = (
            gate["volatility_swings"]
        )
        out["predictions"][target]["volatility_lead_changes"] = (
            gate["volatility_lead_changes"]
        )

        quarter = "Q3" if target == "q3" else "Q4"
        actual = _actual_quarter_outcome(match_data, quarter)
        if actual is None:
            out["predictions"][target]["result"] = "pending"
        elif actual == "push":
            out["predictions"][target]["result"] = "push"
        elif out["predictions"][target]["predicted_winner"] == actual:
            out["predictions"][target]["result"] = "hit"
        else:
            out["predictions"][target]["result"] = "miss"
        out["predictions"][target]["actual_winner"] = actual

    conn.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer Q3/Q4 winners for one match_id",
    )
    parser.add_argument("match_id", help="SofaScore match ID")
    parser.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric used to pick best model from version comparison",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not scrape when match_id is missing in DB",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch from source and overwrite DB even if match exists",
    )
    parser.add_argument(
        "--force-version",
        choices=["auto", "v1", "v2", "v4", "hybrid"],
        default="auto",
        help="Override selected version (hybrid => q3=v2, q4=v4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of friendly betting summary",
    )
    args = parser.parse_args()

    result = run_inference(
        match_id=str(args.match_id),
        metric=args.metric,
        fetch_missing=not args.no_fetch,
        force_version=args.force_version,
        refresh=args.refresh,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not result.get("ok"):
        print("[infer] match not found and could not be fetched")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    meta = result.get("match", {})
    print("\n=== BET SUMMARY ===")
    print(
        f"Match: {meta.get('home_team', '')} vs {meta.get('away_team', '')}"
    )
    print(
        f"League: {meta.get('league', '')}  |  "
        f"Date UTC: {meta.get('date', '')} {meta.get('time_utc', '')}"
    )
    print(
        f"Match ID: {result.get('match_id')}  |  "
        f"fetched_now={result.get('fetched_now')}"
    )

    state = meta.get("state", "unknown")
    status_desc = meta.get("status_description", "")
    clock = meta.get("clock")
    minute_est = meta.get("minute_estimate")
    print(
        f"State: {state}  |  "
        f"Score: {meta.get('home_score')} - {meta.get('away_score')}"
    )
    if state != "finished":
        if clock:
            print(f"Live clock: {clock}  |  status: {status_desc}")
        elif minute_est is not None:
            print(f"Approx minute: {minute_est}  |  status: {status_desc}")
        else:
            print(f"Status: {status_desc}")

    print("\nPredictions:")
    for target in ("q3", "q4"):
        p = result.get("predictions", {}).get(target, {})
        if not p.get("available"):
            print(
                f"- {target.upper()}: unavailable "
                f"({p.get('reason', 'unknown')})"
            )
            continue

        pred = p.get("predicted_winner")
        conf = p.get("confidence")
        ph = p.get("p_home_win")
        pa = p.get("p_away_win")
        model = p.get("model")
        version = p.get("version")
        snap = p.get("snapshot_minute")
        result_txt = p.get("result", "pending")
        actual = p.get("actual_winner")
        bet_signal = p.get("bet_signal", "NO BET")
        final_rec = p.get("final_recommendation", "NO BET")
        decision_gate = p.get("decision_gate", "BLOCK_LOW_EDGE")
        gate_reason = p.get("gate_reason", "unknown")
        units = p.get("suggested_units", 0.0)
        t_lean = p.get("threshold_lean")
        t_bet = p.get("threshold_bet")
        vol_idx = p.get("volatility_index")

        snap_txt = f" m{snap}" if snap is not None else ""

        print(
            f"- {target.upper()}: pick={pred}  p_home={ph} p_away={pa} "
            f"conf={conf}  [{version}/{model}{snap_txt}]"
        )
        print(
            f"  signal={bet_signal} units={units} "
            f"(lean>={t_lean}, bet>={t_bet})"
        )
        print(
            f"  gate={decision_gate} final={final_rec} "
            f"reason={gate_reason} vol_idx={vol_idx}"
        )

        if state == "finished" or actual in ("home", "away", "push"):
            print(f"  result={result_txt}  actual={actual}")
            if final_rec == "NO BET":
                print("  note=would_skip_this_pick_by_risk_rules")
        else:
            print("  result=pending (match not FT)")


if __name__ == "__main__":
    main()
