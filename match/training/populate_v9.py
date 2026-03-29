"""Populate only V9 predictions - fast version."""

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


def load_model(version, target):
    path = OUT_DIR / f"{target}_ensemble.joblib"
    if not path.exists():
        print(f"[v9] {path} not found")
        return None, None, None
    
    data = joblib.load(path)
    return data.get("models", {}), data.get("vectorizer"), data.get("scaler")


def predict_ensemble(vec, scaler, models, features):
    if vec is not None:
        x = vec.transform([features])
        x = scaler.transform(x)
    else:
        import pandas as pd
        df = pd.DataFrame([features])
        CATEGORICAL = ["league", "gender_bucket", "home_team", "away_team"]
        for col in CATEGORICAL:
            if col in df.columns:
                df[col] = df[col].astype('category')
        x = df
    
    probs = []
    for name, m in models.items():
        p = m.predict_proba(x)[:, 1]
        probs.append(p[0])
    
    return sum(probs) / len(probs)


def main():
    conn = db_mod.get_conn(str(ROOT / "matches.db"))
    db_mod.init_db(conn)
    
    # Clear V9 columns only
    conn.execute("""
        DELETE FROM eval_match_results 
        WHERE q3_signal__v9 IS NOT NULL OR q4_signal__v9 IS NOT NULL
    """)
    conn.commit()
    print("[v9] Cleared existing V9 predictions")
    
    print("[v9] Building samples...")
    samples = v6._build_samples(ROOT / "matches.db")
    print(f"[v9] Built {len(samples)} samples")
    
    # Load V9 models
    vec_q3, models_q3, scaler_q3 = load_model("v9", "q3")
    vec_q4, models_q4, scaler_q4 = load_model("v9", "q4")
    
    if not models_q3 or not models_q4:
        print("[v9] Models not found")
        return
    
    print(f"[v9] Predicting {len(samples)} samples...")
    
    for sample in samples:
        m_id = sample.match_id
        
        # Get actual results
        q1h, q1a = v6._quarter_points({"score": {"quarters": {"Q1": sample.features_q3.get("_q1")}},"Q1")
        if q1h is None:
            continue
        
        p_q3 = predict_ensemble(vec_q3, scaler_q3, models_q3, sample.features_q3)
        p_q4 = predict_ensemble(vec_q4, scaler_q4, models_q4, sample.features_q4)
        
        q3_pick = "home" if p_q3 >= 0.5 else "away"
        q4_pick = "home" if p_q4 >= 0.5 else "away"
        
        conf_q3 = p_q3 if p_q3 >= 0.5 else 1-p_q3
        conf_q4 = p_q4 if p_q4 >= 0.5 else 1-p_q4
        
        signal_q3 = "BET" if conf_q3 > 0.65 else "LEAN"
        signal_q4 = "BET" if conf_q4 > 0.65 else "LEAN"
        
        # Update or insert
        conn.execute("""
            INSERT OR REPLACE INTO eval_match_results 
            (event_date, match_id, home_team, away_team,
             q3_signal__v9, q3_conf__v9, q4_signal__v9, q4_conf__v9,
             q3_pred__v9, q4_pred__v9)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sample.dt.strftime("%Y-%m-%d"), m_id, 
             sample.features_q3.get("home_team_bucket", ""),
             sample.features_q3.get("away_team_bucket", ""),
             signal_q3, conf_q3, signal_q4, conf_q4,
             q3_pick, q4_pick))
    
    conn.commit()
    print("[v9] Done")
    conn.close()


if __name__ == "__main__":
    main()