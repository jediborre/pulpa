import sqlite3
import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db
from training import train_q3_q4_models_v4 as v4
from training import train_q3_q4_models_v5 as v5
from training import train_q3_q4_models_v6 as v6
from training import train_q3_q4_models_v7 as v7
import pandas as pd

MODELS = {
    "v4": ROOT / "training" / "model_outputs_v4",
    "v5": ROOT / "training" / "model_outputs_v5",
    "v6": ROOT / "training" / "model_outputs_v6",
    "v7": ROOT / "training" / "model_outputs_v7",
    "v9": ROOT / "training" / "model_outputs_v9",
}

def load_ensemble(version, target):
    models = {}
    if version == "v4":
        names = ["logreg", "rf", "gb"]
    elif version in ["v5", "v6"]:
        names = ["xgb", "hist_gb", "mlp"]
    elif version == "v7":
        names = ["catboost", "xgb_cat"]
    elif version == "v9":
        names = ["logreg", "gb"]
        
    vec = None
    for name in names:
        path = MODELS[version] / f"{target}_{name}.joblib"
        artifact = joblib.load(path)
        vec = artifact.get("vectorizer")
        models[name] = artifact["model"]
    
    return vec, models

def predict_ensemble(vec, models, features):
    if vec is not None:
        x = vec.transform([features])
    else:
        df = pd.DataFrame([features])
        CATEGORICAL = ["league", "gender_bucket", "home_team", "away_team"]
        for col in CATEGORICAL:
            if col in df.columns:
                df[col] = df[col].astype('category')
        x = df
        
    probs = []
    for m in models.values():
        probs.append(float(m.predict_proba(x)[0][1]))
    return sum(probs) / len(probs)

def main():
    conn = db.get_conn(str(ROOT / "matches.db"))
    
    # 1. Clear eval_match_results
    conn.execute("DELETE FROM eval_match_results")
    conn.commit()
    print("Cleared eval_match_results table.")
    
    # 2. Iterate versions
    versions = [("v4", v4), ("v5", v5), ("v6", v6), ("v7", v7), ("v9", v6)]
    
    print("Caching global sparse match metadata...")
    match_scores = {}
    for row in conn.execute("SELECT * FROM matches").fetchall():
        m_id = row["match_id"]
        qs = {r["quarter"]: (r["home"], r["away"]) for r in conn.execute("SELECT * FROM quarter_scores WHERE match_id=?", (m_id,))}
        match_scores[m_id] = {
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "q3_home_score": qs.get("Q3", (None, None))[0],
            "q3_away_score": qs.get("Q3", (None, None))[1],
            "q4_home_score": qs.get("Q4", (None, None))[0],
            "q4_away_score": qs.get("Q4", (None, None))[1],
        }
    
    for v_name, v_mod in versions:
        print(f"Building samples and features for {v_name}...")
        samples = v_mod._build_samples(ROOT / "matches.db")
        print(f"Loaded {len(samples)} samples. Loading models...")
        
        vec_q3, models_q3 = load_ensemble(v_name, "q3")
        vec_q4, models_q4 = load_ensemble(v_name, "q4")
        
        print(f"Populating predictions and backfilling db for {v_name}...")
        
        for sample in samples:
            m_id = sample.match_id
            m_data = match_scores.get(m_id)
            if not m_data:
                continue
            
            p_q3 = predict_ensemble(vec_q3, models_q3, sample.features_q3)
            p_q4 = predict_ensemble(vec_q4, models_q4, sample.features_q4)
            
            q3_pick = "home" if p_q3 >= 0.5 else "away"
            q4_pick = "home" if p_q4 >= 0.5 else "away"
            
            conf_q3 = p_q3 if p_q3 >= 0.5 else 1-p_q3
            conf_q4 = p_q4 if p_q4 >= 0.5 else 1-p_q4
            
            signal_q3 = "BET" if conf_q3 > 0.65 else "LEAN"
            signal_q4 = "BET" if conf_q4 > 0.65 else "LEAN"
            
            q3_hit = None
            if sample.target_q3 is not None:
                q3_hit = "hit" if ((p_q3>=0.5) == sample.target_q3) else "miss"
                
            q4_hit = None
            if sample.target_q4 is not None:
                q4_hit = "hit" if ((p_q4>=0.5) == sample.target_q4) else "miss"
            
            predictions = {
                "q3": {
                    "available": True,
                    "predicted_winner": q3_pick,
                    "confidence": conf_q3,
                    "final_recommendation": signal_q3,
                    "result": q3_hit,
                    "threshold_lean": 0.55,
                    "threshold_bet": 0.65,
                },
                "q4": {
                    "available": True,
                    "predicted_winner": q4_pick,
                    "confidence": conf_q4,
                    "final_recommendation": signal_q4,
                    "result": q4_hit,
                    "threshold_lean": 0.55,
                    "threshold_bet": 0.65,
                }
            }
            
            db.save_eval_match_result(
                conn,
                event_date=m_data["date"],
                match_id=m_id,
                home_team=m_data["home_team"],
                away_team=m_data["away_team"],
                q3_home_score=m_data["q3_home_score"],
                q3_away_score=m_data["q3_away_score"],
                q4_home_score=m_data["q4_home_score"],
                q4_away_score=m_data["q4_away_score"],
                result_tag=v_name,
                predictions=predictions
            )
            
    conn.close()
    print("All backfills inserted successfully!")

if __name__ == "__main__":
    main()
