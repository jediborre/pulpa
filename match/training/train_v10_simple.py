"""V10 - Over/Under Regression Models - Read directly from DB"""

import csv
import sqlite3
import time
from pathlib import Path
from collections import Counter

import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs_v10"


def load_data(target_quarter: str) -> tuple:
    """Load features and actual Q3/Q4 scores from DB"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get matches with all 4 quarters and play_by_play
    matches = conn.execute("""
        SELECT m.match_id, m.league, m.home_team, m.away_team,
               q1.home as q1h, q1.away as q1a,
               q2.home as q2h, q2.away as q2a,
               q3.home as q3h, q3.away as q3a,
               q4.home as q4h, q4.away as q4a
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        WHERE q1.home IS NOT NULL AND q3.home IS NOT NULL
        AND (SELECT COUNT(*) FROM play_by_play WHERE match_id = m.match_id) > 20
    """).fetchall()
    
    # Get team counts
    team_counter = Counter()
    for m in conn.execute("SELECT home_team, away_team FROM matches").fetchall():
        if m[0]: team_counter[str(m[0])] += 1
        if m[1]: team_counter[str(m[1])] += 1
    
    top_teams = {t for t, _ in team_counter.most_common(120)}
    
    # Build features and targets
    data = []
    for m in matches:
        league = m["league"] or ""
        ht, at = m["home_team"] or "", m["away_team"] or ""
        
        # Determine gender
        gender = "women" if any(x in f"{league} {ht} {at}".lower() 
                                for x in ["women", "woman", "female", "(w)"]) else "men_or_open"
        
        # League bucket
        league_bucket = league if league else "OTHER"
        
        # Team buckets
        home_bucket = ht if ht in top_teams else "TEAM_OTHER"
        away_bucket = at if at in top_teams else "TEAM_OTHER"
        
        # First half scores
        ht_home = m["q1h"] + m["q2h"]
        ht_away = m["q1a"] + m["q2a"]
        ht_total = ht_home + ht_away
        
        if target_quarter == "q3":
            target_home = m["q3h"]
            target_away = m["q3a"]
            target_total = m["q3h"] + m["q3a"]
        else:  # q4
            target_home = m["q4h"]
            target_away = m["q4a"]
            target_total = m["q4h"] + m["q4a"]
        
        # Features (simplified - no graph points to save time)
        features = {
            "league_bucket": league_bucket,
            "gender_bucket": gender,
            "home_team": home_bucket,
            "away_team": away_bucket,
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_total": ht_total,
            "q1_diff": m["q1h"] - m["q1a"],
            "q2_diff": m["q2h"] - m["q2a"],
        }
        
        # Add Q3 info for Q4
        if target_quarter == "q4":
            features["q3_diff"] = m["q3h"] - m["q3a"]
            features["score_3q_home"] = ht_home + m["q3h"]
            features["score_3q_away"] = ht_away + m["q3a"]
            features["score_3q_total"] = ht_total + m["q3h"] + m["q3a"]
        
        data.append({
            "match_id": m["match_id"],
            "features": features,
            "target_home": target_home,
            "target_away": target_away,
            "target_total": target_total
        })
    
    conn.close()
    print(f"[V10] Loaded {len(data)} matches for {target_quarter}")
    return data


def train_model(data, target_type: str) -> dict:
    """Train regression for home/away/total"""
    name = f"{target_type}"
    
    # Extract target
    y = np.array([d[f"target_{target_type}"] for d in data])
    
    n = len(data)
    n_train = int(n * 0.8)
    n_test = n - n_train
    
    X = [d["features"] for d in data]
    
    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(X)
    
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    
    models = {}
    preds = {}
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_s, y_train)
    preds["ridge"] = ridge.predict(x_test_s)
    models["ridge"] = ridge
    
    # GB
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(x_train, y_train)
    preds["gb"] = gb.predict(x_test)
    models["gb"] = gb
    
    # XGB
    if HAS_XGB:
        xgb_m = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        xgb_m.fit(x_train, y_train)
        preds["xgb"] = xgb_m.predict(x_test)
        models["xgb"] = xgb_m
    
    # Ensembles
    if HAS_XGB:
        avg = (preds["ridge"] + preds["gb"] + preds["xgb"]) / 3
        weighted = 0.2 * preds["ridge"] + 0.35 * preds["gb"] + 0.45 * preds["xgb"]
        stack_X = np.column_stack([preds["ridge"], preds["gb"], preds["xgb"]])
    else:
        avg = (preds["ridge"] + preds["gb"]) / 2
        weighted = 0.3 * preds["ridge"] + 0.7 * preds["gb"]
        stack_X = np.column_stack([preds["ridge"], preds["gb"]])
    
    stack = Ridge(alpha=0.5)
    stack.fit(stack_X[:n_train], y_train)
    stack_pred = stack.predict(stack_X[n_train:])
    preds["stacking"] = stack_pred
    models["stacking"] = stack
    
    # Metrics
    metrics = []
    for m_name, pred in preds.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        metrics.append({"target": target_type, "model": m_name, "n_train": n_train, 
                        "n_test": n_test, "mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3)})
    
    # Save models
    for m_name, m in models.items():
        joblib.dump({"model": m, "vectorizer": vec, "scaler": scaler}, 
                    OUT_DIR / f"{name}_{m_name}.joblib")
    
    print(f"  {name}: MAE={metrics[-3]['mae']}, R2={metrics[-3]['r2']}")
    return metrics


def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    # Q3 models
    data_q3 = load_data("q3")
    for target_type in ["home", "away", "total"]:
        metrics = train_model(data_q3, f"q3_{target_type}")
        all_metrics.extend(metrics)
    
    # Q4 models
    data_q4 = load_data("q4")
    for target_type in ["home", "away", "total"]:
        metrics = train_model(data_q4, f"q4_{target_type}")
        all_metrics.extend(metrics)
    
    # Save metrics
    with (OUT_DIR / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    
    elapsed = time.time() - start
    print(f"\n[V10] Done in {elapsed:.1f}s - {OUT_DIR}")


if __name__ == "__main__":
    main()