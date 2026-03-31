"""Populate ONLY V9 predictions - simplified."""

import sqlite3
import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db as db_mod
from training import train_q3_q4_models_v6 as v6

OUT_DIR = ROOT / "training" / "model_outputs_v9"


def load_v9_ensemble(target):
    path = OUT_DIR / f"{target}_ensemble.joblib"
    if not path.exists():
        print(f"[ERROR] {path} not found")
        return None
    
    data = joblib.load(path)
    print(f"[V9] Loaded {target} ensemble with models: {list(data.get('models', {}).keys())}")
    print(f"[V9] Weights: {data.get('weights')}")
    return data


def predict_v9(features, ensemble_data):
    models = ensemble_data.get("models", {})
    weights = ensemble_data.get("weights", [0.5, 0.5])
    
    feature_names = [
        "away_prior_wr", "clutch_away_max_run_pts", "clutch_away_points",
        "clutch_home_event_share", "clutch_home_max_run_pts", "clutch_home_points",
        "clutch_last_scoring_away", "clutch_last_scoring_home", "clutch_points_diff",
        "clutch_run_diff", "clutch_scoring_events", "clutch_window_minutes",
        "global_abs_diff", "global_diff", "gp_area_away", "gp_area_diff",
        "gp_area_home", "gp_count", "gp_last", "gp_mean_abs", "gp_peak_away",
        "gp_peak_home", "gp_slope_3m", "gp_slope_5m", "gp_swings",
        "home_prior_wr", "ht_away", "ht_diff", "ht_home", "ht_total",
        "is_tied", "leading_points_per_min", "mc_home_win_prob", "pbp_3pt_diff",
        "pbp_away_3pt", "pbp_away_plays", "pbp_away_pts_per_play", "pbp_home_3pt",
        "pbp_home_3pt_share", "pbp_home_plays", "pbp_home_plays_share",
        "pbp_home_pts_per_play", "pbp_plays_diff", "pbp_pts_per_play_diff",
        "pressure_ratio_lead", "pressure_ratio_tie", "prior_wr_diff",
        "prior_wr_sum", "q1_diff", "q2_diff", "remaining_minutes_target",
        "req_pts_per_trailing_event", "required_ppm_lead", "required_ppm_tie",
        "scoring_gap_per_min", "trailing_3pt_rate", "trailing_is_away",
        "trailing_is_home", "trailing_play_share", "trailing_plays_per_min",
        "trailing_points_per_min", "trailing_points_per_play",
        "trailing_points_to_lead", "trailing_points_to_tie", "urgency_index"
    ]
    
    x = [[features.get(name, 0.0) for name in feature_names]]
    
    probs = []
    if "logreg" in models:
        p = models["logreg"].predict_proba(x)[0][1] * weights[0]
        probs.append(p)
    if "gb" in models:
        p = models["gb"].predict_proba(x)[0][1] * weights[1]
        probs.append(p)
    
    if not probs:
        print("[ERROR] No models available")
        return 0.5
    
    return sum(probs) / sum(weights)


def main():
    conn = db_mod.get_conn(str(ROOT / "matches.db"))
    db_mod.init_db(conn)
    
    # Load V9 ensembles
    ensemble_q3 = load_v9_ensemble("q3")
    ensemble_q4 = load_v9_ensemble("q4")
    
    if not ensemble_q3 or not ensemble_q4:
        print("[ERROR] Failed to load V9 ensembles")
        return
    
    # Build samples using V6 module (same features)
    print("[V9] Building samples...")
    samples = v6._build_samples(ROOT / "matches.db")
    print(f"[V9] Built {len(samples)} samples")
    
    # Populate predictions
    count = 0
    for sample in samples:
        try:
            p_q3 = predict_v9(sample.features_q3, ensemble_q3)
            p_q4 = predict_v9(sample.features_q4, ensemble_q4)
            
            q3_pick = "home" if p_q3 >= 0.5 else "away"
            q4_pick = "home" if p_q4 >= 0.5 else "away"
            
            conf_q3 = max(p_q3, 1-p_q3)
            conf_q4 = max(p_q4, 1-p_q4)
            
            signal_q3 = "BET" if conf_q3 > 0.65 else "LEAN"
            signal_q4 = "BET" if conf_q4 > 0.65 else "LEAN"
            
            # Get actual results
            m_id = sample.match_id
            q3_row = conn.execute("SELECT home, away FROM quarter_scores WHERE match_id=? AND quarter='Q3'", (m_id,)).fetchone()
            q4_row = conn.execute("SELECT home, away FROM quarter_scores WHERE match_id=? AND quarter='Q4'", (m_id,)).fetchone()
            
            if q3_row and q4_row:
                q3_actual = "home" if q3_row[0] > q3_row[1] else "away"
                q4_actual = "home" if q4_row[0] > q4_row[1] else "away"
            else:
                q3_actual, q4_actual = None, None
            
            q3_outcome = None
            if q3_actual and sample.target_q3 is not None:
                predicted = 1 if q3_pick == "home" else 0
                actual = sample.target_q3
                q3_outcome = "hit" if predicted == actual else "miss"
            
            q4_outcome = None
            if q4_actual and sample.target_q4 is not None:
                predicted = 1 if q4_pick == "home" else 0
                actual = sample.target_q4
                q4_outcome = "hit" if predicted == actual else "miss"
            
            # Insert/Update
            conn.execute("""
                INSERT OR REPLACE INTO eval_match_results 
                (event_date, match_id, home_team, away_team,
                 q3_home_score, q3_away_score, q3_winner,
                 q4_home_score, q4_away_score, q4_winner,
                 q3_pick__v9, q3_signal__v9, q3_outcome__v9, q3_conf__v9,
                 q4_pick__v9, q4_signal__v9, q4_outcome__v9, q4_conf__v9)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.dt.strftime("%Y-%m-%d"), m_id,
                sample.features_q3.get("home_team_bucket", ""),
                sample.features_q3.get("away_team_bucket", ""),
                q3_row[0] if q3_row else None,
                q3_row[1] if q3_row else None,
                q3_actual,
                q4_row[0] if q4_row else None,
                q4_row[1] if q4_row else None,
                q4_actual,
                q3_pick, signal_q3, q3_outcome, round(conf_q3, 3),
                q4_pick, signal_q4, q4_outcome, round(conf_q4, 3)
            ))
            
            count += 1
            if count % 500 == 0:
                print(f"[V9] Progress: {count}/{len(samples)}")
                
        except Exception as e:
            print(f"[ERROR] {sample.match_id}: {e}")
    
    conn.commit()
    print(f"[V9] Done - {count} predictions saved")
    conn.close()


if __name__ == "__main__":
    main()