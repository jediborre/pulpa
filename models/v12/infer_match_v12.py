"""
V12 - Conservative Inference Engine with Risk Management
========================================================
Combines:
1. Classification model (winner prediction) 
2. Regression model (points prediction for over/under)
3. League filtering (only bet on predictable leagues)
4. Conservative gates (NO_BET by default)
5. Asymmetric risk (loss penalty > win reward)
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = __import__("db")

MODEL_DIR = ROOT / "training" / "v12" / "model_outputs"
LEAGUE_STATS_FILE = MODEL_DIR / "league_stats.json"

# Risk thresholds - VERY conservative
DEFAULT_ODDS = 1.91
MIN_CONFIDENCE_TO_BET = 0.65  # Need 65%+ confidence
MIN_EDGE_FOR_OVER_UNDER = 4.0  # Need 4+ points edge
MAX_VOLATILITY = 0.70  # Block if volatility too high
MIN_GRAPH_POINTS_Q3 = 16   # aligned with bet_monitor MIN_GP_Q3
MIN_GRAPH_POINTS_Q4 = 22   # lowered from 28 (was above bet_monitor MIN_GP_Q4=26, causing false 'insuficiente')
MIN_PBP_EVENTS_Q3 = 14     # relaxed slightly
MIN_PBP_EVENTS_Q4 = 20     # relaxed slightly

# Loss penalty
LOSS_PENALTY = 2.0  # Losing costs 2x more


@dataclass
class V12Prediction:
    """Complete prediction with risk assessment."""
    match_id: str
    quarter: str
    timestamp: str
    
    # Classification
    winner_pick: str  # "home", "away", "uncertain"
    winner_confidence: float  # 0.0 - 1.0
    winner_signal: str  # "BET", "LEAN", "NO_BET"
    
    # Regression (Over/Under)
    predicted_total: float | None
    predicted_home: float | None
    predicted_away: float | None
    over_under_signal: str  # "OVER", "UNDER", "NO_BET"
    over_under_confidence: float
    
    # Risk assessment
    league_quality: str  # "strong", "moderate", "weak", "unknown"
    league_bettable: bool
    volatility_index: float
    data_quality: str  # "excellent", "good", "poor"
    
    # Final recommendation
    final_signal: str  # "BET_HOME", "BET_AWAY", "OVER", "UNDER", "NO_BET"
    final_confidence: float
    risk_level: str  # "low", "medium", "high", "extreme"
    reasoning: str


def _load_ensemble(target: str) -> dict | None:
    """Load classifier ensemble."""
    path = MODEL_DIR / f"{target}_clf_ensemble.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_regressor_ensemble(target: str, target_type: str, gender: str) -> dict | None:
    """Load regressor ensemble."""
    path = MODEL_DIR / f"{target}_{target_type}_{gender}_reg_ensemble.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_league_stats() -> dict:
    """Load historical league performance."""
    if not LEAGUE_STATS_FILE.exists():
        return {}
    with open(LEAGUE_STATS_FILE, "r") as f:
        return json.load(f)


def _predict_proba(ensemble: dict, features: dict) -> float:
    """Get probability from classifier ensemble."""
    models = ensemble.get("models", {})
    vec = ensemble["vectorizer"]
    scaler = ensemble["scaler"]
    
    x = vec.transform([features])
    x_scaled = scaler.transform(x)
    
    probs = []
    weights = []
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x_scaled if name in ["logreg"] else x)[0][1])
            probs.append(prob)
            # Weight non-linear models more
            if name in ["xgb", "lgbm", "catboost"]:
                weights.append(0.25)
            elif name == "gb":
                weights.append(0.20)
            else:
                weights.append(0.10)
    
    if not probs:
        return 0.5
    
    # Weighted average
    total_weight = sum(weights)
    weighted_prob = sum(p * w for p, w in zip(probs, weights)) / total_weight
    return weighted_prob


def _predict_regression(ensemble: dict, features: dict) -> float:
    """Get prediction from regressor ensemble."""
    models = ensemble.get("models", {})
    vec = ensemble["vectorizer"]
    scaler = ensemble["scaler"]
    
    x = vec.transform([features])
    x_scaled = scaler.transform(x)
    
    preds = []
    weights = []
    
    for name, model in models.items():
        if name == "ridge":
            pred = float(model.predict(x_scaled)[0])
        else:
            pred = float(model.predict(x)[0])
        preds.append(pred)
        
        if name in ["xgb", "catboost"]:
            weights.append(0.30)
        elif name in ["gb", "lgbm"]:
            weights.append(0.25)
        else:  # ridge
            weights.append(0.15)
    
    if not preds:
        return 0.0
    
    total_weight = sum(weights)
    weighted_pred = sum(p * w for p, w in zip(preds, weights)) / total_weight
    return weighted_pred


def _infer_gender(league: str, home_team: str, away_team: str) -> str:
    text = f"{league} {home_team} {away_team}".lower()
    markers = ["women", "woman", "female", "femen", "fem.", " ladies ", "(w)", " w ", "wnba", "girls"]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


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


def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
    points = [p for p in graph_points if int(p.get("minute", 0)) <= max_minute]
    values = [int(p.get("value", 0)) for p in points]
    if not values:
        return {
            "gp_count": 0, "gp_last": 0, "gp_peak_home": 0, "gp_peak_away": 0,
            "gp_area_home": 0, "gp_area_away": 0, "gp_area_diff": 0,
            "gp_mean_abs": 0.0, "gp_swings": 0, "gp_slope_3m": 0, "gp_slope_5m": 0,
            "gp_acceleration": 0.0,
        }

    area_home = sum(max(v, 0) for v in values)
    area_away = sum(max(-v, 0) for v in values)
    mean_abs = sum(abs(v) for v in values) / len(values)

    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else (values[-1] - values[0])

    if len(values) >= 6:
        slope_first_half = values[len(values)//2] - values[0]
        slope_second_half = values[-1] - values[len(values)//2]
        acceleration = slope_second_half - slope_first_half
    else:
        acceleration = 0

    return {
        "gp_count": len(values), "gp_last": values[-1],
        "gp_peak_home": max(values), "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home, "gp_area_away": area_away, "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs, "gp_swings": _count_sign_swings(values),
        "gp_slope_3m": slope_3m, "gp_slope_5m": slope_5m,
        "gp_acceleration": acceleration,
    }


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


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
        "pbp_home_efficiency": _safe_rate(home_pts, home_plays) if home_plays else 0,
        "pbp_away_efficiency": _safe_rate(away_pts, away_plays) if away_plays else 0,
    }


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _bucket(value: str, top_set: set[str], prefix: str) -> str:
    return value if value in top_set else f"{prefix}_OTHER"


def _build_features_from_db(data: dict, target: str, league_stats: dict, top_leagues: set, top_teams: set) -> dict:
    """Build feature dict from match data (same as training)."""
    m = data["match"]
    score = data["score"]
    pbp = data.get("play_by_play", {})
    gp = data.get("graph_points", [])
    
    q1h, q1a = _quarter_points(data, "Q1")
    q2h, q2a = _quarter_points(data, "Q2")
    q3h, q3a = _quarter_points(data, "Q3")
    
    ht, at, league = m.get("home_team", ""), m.get("away_team", ""), m.get("league", "")
    gender = _infer_gender(league, ht, at)
    
    league_info = league_stats.get(league, {})
    league_home_wr = league_info.get("home_win_rate", 0.5)
    league_avg_total = league_info.get("avg_total_points", 0)
    league_std_total = league_info.get("std_total_points", 0)
    league_samples = league_info.get("samples", 0)
    
    base = {
        "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
        "gender_bucket": gender,
        "home_team_bucket": _bucket(ht, top_teams, "TEAM"),
        "away_team_bucket": _bucket(at, top_teams, "TEAM"),
        "home_prior_wr": 0.5,  # Default without history
        "away_prior_wr": 0.5,
        "prior_wr_diff": 0.0,
        "prior_wr_sum": 1.0,
        "q1_diff": (q1h or 0) - (q1a or 0),
        "q2_diff": (q2h or 0) - (q2a or 0),
        "league_home_advantage": round(league_home_wr - 0.5, 4),
        "league_avg_total_points": round(league_avg_total, 2),
        "league_std_total_points": round(league_std_total, 2),
        "league_sample_size": min(league_samples, 500),
    }
    
    if target == "q3":
        ht_home = (q1h or 0) + (q2h or 0)
        ht_away = (q1a or 0) + (q2a or 0)
        q3_pbp = _pbp_stats_upto(pbp, ["Q1", "Q2"])
        
        f = dict(base, ht_home=ht_home, ht_away=ht_away, ht_diff=ht_home - ht_away, ht_total=ht_home + ht_away)
        f.update(_graph_stats_upto(gp, 24))
        f.update(q3_pbp)
        
    else:  # q4
        score_3q_home = (q1h or 0) + (q2h or 0) + (q3h or 0)
        score_3q_away = (q1a or 0) + (q2a or 0) + (q3a or 0)
        q4_pbp = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])
        
        f = dict(base, q3_diff=(q3h or 0) - (q3a or 0), 
                 score_3q_home=score_3q_home, score_3q_away=score_3q_away,
                 score_3q_diff=score_3q_home - score_3q_away,
                 score_3q_total=score_3q_home + score_3q_away)
        f.update(_graph_stats_upto(gp, 36))
        f.update(q4_pbp)
    
    return f


def _resolve_league_stats(league: str, league_stats: dict) -> dict:
    """Look up league stats with fallback to base name (before first comma)."""
    s = league_stats.get(league, {})
    if not s and "," in league:
        base = league.split(",")[0].strip()
        s = league_stats.get(base, {})
    if not s:
        # Partial prefix match: find longest key that is a prefix of the league name
        for key in sorted(league_stats.keys(), key=len, reverse=True):
            if league.startswith(key):
                s = league_stats[key]
                break
    return s


def _assess_league_quality(league: str, league_stats: dict) -> tuple[str, bool]:
    """Assess if league is good for betting."""
    if not league_stats:
        return "unknown", False
    
    stats = _resolve_league_stats(league, league_stats)
    samples = stats.get("samples", 0)
    home_wr = stats.get("home_win_rate", 0.5)
    
    if samples < 30:
        return "weak", False
    
    # Strong league = predictable (home advantage is clear)
    if home_wr >= 0.57 or home_wr <= 0.43:
        return "strong", True
    elif home_wr >= 0.53 or home_wr <= 0.47:
        return "moderate", True
    else:
        return "weak", False


def _compute_volatility(data: dict, cutoff: int) -> float:
    """Compute volatility index from graph points."""
    gp = data.get("graph_points", [])
    points = [p for p in gp if int(p.get("minute", 0)) <= cutoff]
    values = [int(p.get("value", 0)) for p in points]
    
    if len(values) < 2:
        return 0.0
    
    diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    mean_diff = np.mean(diffs)
    max_possible = max(abs(max(values)), abs(min(values)), 1)
    
    return min(mean_diff / max_possible, 1.0)


def _data_quality(gp_count: int, pbp_count: int, target: str) -> str:
    """Assess data quality."""
    if target == "q3":
        min_gp, min_pbp = MIN_GRAPH_POINTS_Q3, MIN_PBP_EVENTS_Q3
    else:
        min_gp, min_pbp = MIN_GRAPH_POINTS_Q4, MIN_PBP_EVENTS_Q4
    
    if gp_count >= min_gp * 1.3 and pbp_count >= min_pbp * 1.3:
        return "excellent"
    elif gp_count >= min_gp and pbp_count >= min_pbp:
        return "good"
    else:
        return "poor"


def run_inference(
    match_id: str,
    target: str = "q4",
    sportsbook_line: float | None = None,
    fetch_missing: bool = True,
) -> V12Prediction | dict:
    """
    Run V12 inference on a match.
    
    Args:
        match_id: SofaScore match ID
        target: "q3" or "q4"
        sportsbook_line: Over/under line from sportsbook (optional)
        fetch_missing: If True, scrape match if not in DB
    
    Returns:
        V12Prediction object or error dict
    """
    cutoff = 24 if target == "q3" else 36
    min_gp = MIN_GRAPH_POINTS_Q3 if target == "q3" else MIN_GRAPH_POINTS_Q4
    min_pbp = MIN_PBP_EVENTS_Q3 if target == "q3" else MIN_PBP_EVENTS_Q4
    
    # Load models
    clf_ensemble = _load_ensemble(target)
    if clf_ensemble is None:
        return {"ok": False, "reason": f"V12 classifier for {target} not found. Train first."}
    
    # Load DB
    db_path = ROOT / "matches.db"
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)
    
    data = db_mod.get_match(conn, match_id)
    conn.close()
    
    if data is None:
        if fetch_missing:
            # Try to scrape
            try:
                scraper_mod = __import__("scraper")
                data = scraper_mod.fetch_match_by_id(match_id)
                # Save to DB
                conn = db_mod.get_conn(str(db_path))
                db_mod.init_db(conn)
                db_mod.save_match(conn, match_id, data)
                conn.close()
            except Exception as e:
                return {"ok": False, "reason": f"Match not in DB and scraping failed: {e}"}
        else:
            return {"ok": False, "reason": "Match not in DB"}
    
    # Load league stats
    league_stats = _load_league_stats()
    
    # Identify top leagues/teams for bucketing
    top_leagues = set()
    top_teams = set()
    # Use default buckets
    for k in league_stats.keys():
        top_leagues.add(k)
    # Use top 25 leagues and 150 teams
    top_leagues = list(league_stats.keys())[:25]
    top_leagues_set = set(top_leagues)
    top_teams_set = set()  # Will fall to OTHER
    
    # Check data completeness
    gp = data.get("graph_points", [])
    gp_count = len([p for p in gp if int(p.get("minute", 0)) <= cutoff])
    pbp = data.get("play_by_play", {})
    pbp_quarters = ["Q1", "Q2"] if target == "q3" else ["Q1", "Q2", "Q3"]
    pbp_count = sum(len(plays) for q, plays in pbp.items() if q in pbp_quarters)
    
    quality = _data_quality(gp_count, pbp_count, target)
    volatility = _compute_volatility(data, cutoff)
    
    # Data gate
    if gp_count < min_gp:
        return V12Prediction(
            match_id=match_id, quarter=target, timestamp=datetime.now().isoformat(),
            winner_pick="uncertain", winner_confidence=0.0, winner_signal="NO_BET",
            predicted_total=None, predicted_home=None, predicted_away=None,
            over_under_signal="NO_BET", over_under_confidence=0.0,
            league_quality="unknown", league_bettable=False,
            volatility_index=volatility, data_quality="poor",
            final_signal="NO_BET", final_confidence=0.0,
            risk_level="extreme",
            reasoning=f"Insufficient graph points: {gp_count} < {min_gp}"
        )
    
    if pbp_count < min_pbp:
        return V12Prediction(
            match_id=match_id, quarter=target, timestamp=datetime.now().isoformat(),
            winner_pick="uncertain", winner_confidence=0.0, winner_signal="NO_BET",
            predicted_total=None, predicted_home=None, predicted_away=None,
            over_under_signal="NO_BET", over_under_confidence=0.0,
            league_quality="unknown", league_bettable=False,
            volatility_index=volatility, data_quality="poor",
            final_signal="NO_BET", final_confidence=0.0,
            risk_level="extreme",
            reasoning=f"Insufficient PBP events: {pbp_count} < {min_pbp}"
        )
    
    # Build features
    features = _build_features_from_db(data, target, league_stats, top_leagues_set, top_teams_set)
    
    # Classification prediction
    prob_home = _predict_proba(clf_ensemble, features)
    winner_pick = "home" if prob_home >= 0.5 else "away"
    winner_confidence = abs(prob_home - 0.5) * 2.0  # 0-1 scale
    
    # League assessment
    league = data["match"].get("league", "unknown")
    league_quality, league_bettable = _assess_league_quality(league, league_stats)
    
    # Volatility gate
    volatility_blocked = volatility >= MAX_VOLATILITY
    
    # Winner signal
    if volatility_blocked or winner_confidence < MIN_CONFIDENCE_TO_BET or not league_bettable:
        winner_signal = "NO_BET"
    elif winner_confidence >= 0.75:
        winner_signal = "BET"
    else:
        winner_signal = "LEAN"
    
    # Regression prediction (Over/Under)
    gender = _infer_gender(league, data["match"].get("home_team", ""), data["match"].get("away_team", ""))
    
    predicted_total = None
    predicted_home = None
    predicted_away = None
    ou_signal = "NO_BET"
    ou_confidence = 0.0
    
    # Load regressors
    reg_total = _load_regressor_ensemble(target, "total", gender)
    reg_home = _load_regressor_ensemble(target, "home", gender)
    reg_away = _load_regressor_ensemble(target, "away", gender)
    
    if reg_total:
        predicted_total = _predict_regression(reg_total, features)
    if reg_home:
        predicted_home = _predict_regression(reg_home, features)
    if reg_away:
        predicted_away = _predict_regression(reg_away, features)
    
    # Over/Under decision
    if sportsbook_line and predicted_total:
        edge = predicted_total - sportsbook_line
        ou_confidence = min(abs(edge) / 10.0, 1.0)
        
        if abs(edge) >= MIN_EDGE_FOR_OVER_UNDER and not volatility_blocked:
            ou_signal = "OVER" if edge > 0 else "UNDER"
        else:
            ou_signal = "NO_BET"
    
    # Final recommendation - VERY conservative
    final_signal = "NO_BET"
    final_confidence = 0.0
    reasoning_parts = []
    
    # Risk scoring
    risk_score = 0
    if volatility_blocked:
        risk_score += 3
        reasoning_parts.append("HIGH VOLATILITY")
    if not league_bettable:
        risk_score += 2
        reasoning_parts.append("LEAGUE NOT BETTABLE")
    if quality == "poor":
        risk_score += 2
        reasoning_parts.append("POOR DATA")
    elif quality == "good":
        risk_score += 1
    
    # Determine if we should bet
    can_bet_winner = (winner_signal == "BET" and league_bettable and not volatility_blocked and quality in ["excellent", "good"])
    can_bet_ou = (ou_signal in ["OVER", "UNDER"] and not volatility_blocked and quality in ["excellent", "good"])
    
    if can_bet_winner and winner_confidence >= MIN_CONFIDENCE_TO_BET:
        final_signal = f"BET_{winner_pick.upper()}"
        final_confidence = winner_confidence
        reasoning_parts.append(f"Winner: {winner_pick} (conf={winner_confidence:.2f})")
    elif can_bet_ou and sportsbook_line:
        final_signal = ou_signal
        final_confidence = ou_confidence
        reasoning_parts.append(f"{ou_signal} total={predicted_total:.1f} vs line={sportsbook_line}")
    else:
        final_signal = "NO_BET"
        if not reasoning_parts:
            reasoning_parts.append("Insufficient confidence for bet")
    
    # Risk level
    if risk_score >= 5:
        risk_level = "extreme"
    elif risk_score >= 3:
        risk_level = "high"
    elif risk_score >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No clear signal"
    
    return V12Prediction(
        match_id=match_id,
        quarter=target,
        timestamp=datetime.now().isoformat(),
        winner_pick=winner_pick,
        winner_confidence=round(winner_confidence, 4),
        winner_signal=winner_signal,
        predicted_total=round(predicted_total, 2) if predicted_total else None,
        predicted_home=round(predicted_home, 2) if predicted_home else None,
        predicted_away=round(predicted_away, 2) if predicted_away else None,
        over_under_signal=ou_signal,
        over_under_confidence=round(ou_confidence, 4),
        league_quality=league_quality,
        league_bettable=league_bettable,
        volatility_index=round(volatility, 4),
        data_quality=quality,
        final_signal=final_signal,
        final_confidence=round(final_confidence, 4),
        risk_level=risk_level,
        reasoning=reasoning,
    )


def prediction_to_dict(pred: V12Prediction) -> dict:
    """Convert prediction to JSON-serializable dict."""
    return {
        "match_id": pred.match_id,
        "quarter": pred.quarter,
        "timestamp": pred.timestamp,
        "winner": {
            "pick": pred.winner_pick,
            "confidence": pred.winner_confidence,
            "signal": pred.winner_signal,
        },
        "over_under": {
            "predicted_total": pred.predicted_total,
            "predicted_home": pred.predicted_home,
            "predicted_away": pred.predicted_away,
            "signal": pred.over_under_signal,
            "confidence": pred.over_under_confidence,
        },
        "risk": {
            "league_quality": pred.league_quality,
            "league_bettable": pred.league_bettable,
            "volatility_index": pred.volatility_index,
            "data_quality": pred.data_quality,
            "risk_level": pred.risk_level,
        },
        "final": {
            "signal": pred.final_signal,
            "confidence": pred.final_confidence,
            "reasoning": pred.reasoning,
        },
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V12 Inference Engine")
    parser.add_argument("match_id", help="SofaScore match ID")
    parser.add_argument("--target", choices=["q3", "q4"], default="q4")
    parser.add_argument("--line", type=float, default=None, help="Sportsbook over/under line")
    parser.add_argument("--no-fetch", action="store_true", help="Don't scrape if not in DB")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    result = run_inference(
        args.match_id,
        target=args.target,
        sportsbook_line=args.line,
        fetch_missing=not args.no_fetch,
    )
    
    if isinstance(result, dict) and not result.get("ok"):
        print(f"ERROR: {result.get('reason')}")
        sys.exit(1)
    
    if args.json:
        print(json.dumps(prediction_to_dict(result), indent=2))
    else:
        pred: V12Prediction = result
        print(f"\n{'='*60}")
        print(f"V12 Prediction - {pred.match_id} - {pred.quarter.upper()}")
        print(f"{'='*60}")
        print(f"Winner:        {pred.winner_pick} (confidence: {pred.winner_confidence:.2%})")
        print(f"Winner Signal: {pred.winner_signal}")
        if pred.predicted_total:
            print(f"Pred Total:    {pred.predicted_total:.1f}")
        print(f"O/U Signal:    {pred.over_under_signal}")
        print(f"League:        {pred.league_quality} ({'bettable' if pred.league_bettable else 'NO'})")
        print(f"Volatility:    {pred.volatility_index:.2f}")
        print(f"Data Quality:  {pred.data_quality}")
        print(f"Risk Level:    {pred.risk_level.upper()}")
        print(f"\nFINAL: {pred.final_signal} (confidence: {pred.final_confidence:.2%})")
        print(f"Reasoning: {pred.reasoning}")
        print(f"{'='*60}\n")
