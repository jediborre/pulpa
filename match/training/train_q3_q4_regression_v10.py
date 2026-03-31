"""
V10 - Over/Under Regression Models for Q3/Q4 Total Points

Features: Predict Q3_home, Q3_away, Q3_total, Q4_home, Q4_away, Q4_total
Ensemble: Ridge + GradientBoosting + XGBoost (3 methods: avg, weighted, stacking)
"""

from __future__ import annotations
import csv
import importlib
import sys
import time
from collections import defaultdict
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = importlib.import_module("db")

DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs_v10"


@dataclass
class OverUnderSample:
    match_id: str
    dt: datetime
    features: dict
    target_home: int | None
    target_away: int | None
    target_total: int | None


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


def build_samples(target_quarter: str) -> list[OverUnderSample]:
    """Build samples for Q3 or Q4 total/away/home prediction"""
    print(f"[V10] Building samples for {target_quarter}...")
    
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)

    league_rows = conn.execute("SELECT league, COUNT(*) AS n FROM matches GROUP BY league ORDER BY n DESC").fetchall()
    team_counter = Counter()
    for ht, at in conn.execute("SELECT home_team, away_team FROM matches").fetchall():
        if ht: team_counter[str(ht)] += 1
        if at: team_counter[str(at)] += 1

    top_leagues = {str(r[0]) if r[0] else "" for r in league_rows[:20]}
    top_teams = {team for team, _ in team_counter.most_common(120)}

    rows = conn.execute("SELECT match_id, date, time FROM matches ORDER BY date, time").fetchall()
    team_history = defaultdict(list)
    samples = []

    for row in rows:
        match_id = str(row["match_id"])
        dt = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M")
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_match(data):
            continue

        m, score, pbp, gp = data["match"], data["score"], data.get("play_by_play", {}), data.get("graph_points", [])
        
        q1h, q1a = _quarter_points(data, "Q1")
        q2h, q2a = _quarter_points(data, "Q2")
        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue

        ht, at, league = m.get("home_team", ""), m.get("away_team", ""), m.get("league", "")
        home_hist = team_history[ht][-12:]
        away_hist = team_history[at][-12:]
        home_wr = _safe_rate(sum(home_hist), len(home_hist))
        away_wr = _safe_rate(sum(away_hist), len(away_hist))

        if target_quarter == "q3":
            # Features from Q1+Q2 to predict Q3
            base = {
                "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
                "gender_bucket": _infer_gender(league, ht, at),
                "home_team_bucket": _bucket(ht, top_teams, "TEAM"),
                "away_team_bucket": _bucket(at, top_teams, "TEAM"),
                "home_prior_wr": home_wr, "away_prior_wr": away_wr,
                "prior_wr_diff": home_wr - away_wr,
                "q1_diff": q1h - q1a, "q2_diff": q2h - q2a,
            }
            
            ht_total = q1h + q1a + q2h + q2a
            ht_home = q1h + q2h
            ht_away = q1a + q2a
            
            f = dict(base)
            f["ht_home"] = ht_home
            f["ht_away"] = ht_away
            f["ht_total"] = ht_total
            f.update(_graph_stats_upto(gp, 24))
            f.update(_pbp_stats_upto(pbp, ["Q1", "Q2"]))
            
            target_home = q3h
            target_away = q3a
            target_total = q3h + q3a
            
        else:  # q4
            # Features from Q1+Q2+Q3 to predict Q4
            base = {
                "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
                "gender_bucket": _infer_gender(league, ht, at),
                "home_team_bucket": _bucket(ht, top_teams, "TEAM"),
                "away_team_bucket": _bucket(at, top_teams, "TEAM"),
                "home_prior_wr": home_wr, "away_prior_wr": away_wr,
                "prior_wr_diff": home_wr - away_wr,
                "q1_diff": q1h - q1a, "q2_diff": q2h - q2a, "q3_diff": q3h - q3a,
            }
            
            score_3q_home = q1h + q2h + q3h
            score_3q_away = q1a + q2a + q3a
            score_3q_total = score_3q_home + score_3q_away
            
            f = dict(base)
            f["score_3q_home"] = score_3q_home
            f["score_3q_away"] = score_3q_away
            f["score_3q_total"] = score_3q_total
            f.update(_graph_stats_upto(gp, 36))
            f.update(_pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"]))
            
            target_home = q4h
            target_away = q4a
            target_total = q4h + q4a

        samples.append(OverUnderSample(match_id, dt, f, target_home, target_away, target_total))
        
        team_history[ht].append(1 if score["home"] > score["away"] else 0)
        team_history[at].append(1 if score["away"] > score["home"] else 0)

    conn.close()
    print(f"[V10] Built {len(samples)} samples")
    return samples


def train_regression(samples: list[OverUnderSample], target_type: str, target_name: str) -> dict:
    """Train regression model for home/away/total"""
    print(f"[V10] Training {target_name} {target_type}...")
    
    # Get target
    if target_type == "home":
        targets = [s.target_home for s in samples]
    elif target_type == "away":
        targets = [s.target_away for s in samples]
    else:  # total
        targets = [s.target_total for s in samples]
    
    # Filter valid
    valid = [(s, t) for s, t in zip(samples, targets) if t is not None]
    if not valid:
        return {"error": "No valid samples"}
    
    samples_valid = [s for s, t in valid]
    y = np.array([t for s, t in valid])
    
    # Split 80/20 temporal
    n = len(samples_valid)
    n_train = int(n * 0.8)
    
    x_dict = [s.features for s in samples_valid]
    
    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_dict)
    
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    models = {}
    predictions = {}
    
    # 1. Ridge Regression
    print(f"  [Ridge]...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_scaled, y_train)
    pred_ridge = ridge.predict(x_test_scaled)
    models["ridge"] = ridge
    predictions["ridge"] = pred_ridge
    
    # 2. GradientBoosting
    print(f"  [GB]...")
    gb = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    gb.fit(x_train, y_train)
    pred_gb = gb.predict(x_test)
    models["gb"] = gb
    predictions["gb"] = pred_gb
    
    # 3. XGBoost if available
    if HAS_XGB:
        print(f"  [XGB]...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
            random_state=42
        )
        xgb_model.fit(x_train, y_train)
        pred_xgb = xgb_model.predict(x_test)
        models["xgb"] = xgb_model
        predictions["xgb"] = pred_xgb
    
    # Ensemble Methods
    # 1. Simple Average
    if HAS_XGB:
        avg_pred = (pred_ridge + pred_gb + pred_xgb) / 3
    else:
        avg_pred = (pred_ridge + pred_gb) / 2
    
    # 2. Weighted (GB/XGB get more weight)
    if HAS_XGB:
        weighted_pred = 0.20 * pred_ridge + 0.35 * pred_gb + 0.45 * pred_xgb
    else:
        weighted_pred = 0.30 * pred_ridge + 0.70 * pred_gb
    
    # 3. Stacking (use predictions as features)
    # Need predictions on training set to train stacking
    if HAS_XGB:
        pred_ridge_train = ridge.predict(x_train_scaled)
        pred_gb_train = gb.predict(x_train)
        pred_xgb_train = xgb_model.predict(x_train)
        stack_X_train = np.column_stack([pred_ridge_train, pred_gb_train, pred_xgb_train])
        stack_X_test = np.column_stack([pred_ridge, pred_gb, pred_xgb])
    else:
        pred_ridge_train = ridge.predict(x_train_scaled)
        pred_gb_train = gb.predict(x_train)
        stack_X_train = np.column_stack([pred_ridge_train, pred_gb_train])
        stack_X_test = np.column_stack([pred_ridge, pred_gb])
    
    stack_lr = Ridge(alpha=0.5)
    stack_lr.fit(stack_X_train, y_train)
    stack_pred = stack_lr.predict(stack_X_test)
    models["stacking"] = stack_lr
    
    # Metrics
    metrics = []
    
    def metrics_row(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"model": name, "n_train": n_train, "n_test": n_test, 
                "mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3)}
    
    n_test = n - n_train
    metrics.append(metrics_row("ridge", y_test, pred_ridge))
    metrics.append(metrics_row("gb", y_test, pred_gb))
    if HAS_XGB:
        metrics.append(metrics_row("xgb", y_test, pred_xgb))
    metrics.append(metrics_row("ensemble_avg", y_test, avg_pred))
    metrics.append(metrics_row("ensemble_weighted", y_test, weighted_pred))
    metrics.append(metrics_row("stacking", y_test, stack_pred))
    
    # Save models
    for name, model in models.items():
        artifact = {"version": "v10", "target": target_name, "type": target_type,
                    "model_name": name, "vectorizer": vec, "scaler": scaler, "model": model}
        joblib.dump(artifact, OUT_DIR / f"{target_name}_{target_type}_{name}.joblib")
    
    # Save ensemble
    joblib.dump({
        "version": "v10", "target": target_name, "type": target_type,
        "models": models, "has_xgb": HAS_XGB
    }, OUT_DIR / f"{target_name}_{target_type}_ensemble.joblib")
    
    print(f"[V10] {target_name}_{target_type} done - MAE: {metrics[-3]['mae']}")
    
    return {"metrics": metrics, "n_samples": n}


def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    # Train Q3 models
    samples_q3 = build_samples("q3")
    for target_type in ["home", "away", "total"]:
        result = train_regression(samples_q3, target_type, "q3")
        if "metrics" in result:
            all_metrics.extend(result["metrics"])
    
    # Train Q4 models
    samples_q4 = build_samples("q4")
    for target_type in ["home", "away", "total"]:
        result = train_regression(samples_q4, target_type, "q4")
        if "metrics" in result:
            all_metrics.extend(result["metrics"])
    
    # Save metrics
    with (OUT_DIR / "metrics.csv").open("w", newline="") as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    
    elapsed = time.time() - start
    print(f"\n[V10] Done in {elapsed:.1f}s")
    print(f"[V10] Output: {OUT_DIR}")


if __name__ == "__main__":
    main()