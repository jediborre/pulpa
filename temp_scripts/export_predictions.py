import sys
import json
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"c:\Users\borre\OneDrive\OLD\Escritorio\pulpa").resolve()
sys.path.insert(0, str(ROOT / 'match'))

from training.v13 import dataset
from training.v13.eval_v13 import load_trained_models

def export():
    print('Loading models...')
    models = load_trained_models()
    
    print('Loading samples cache (this can take a few seconds)...')
    samples, meta = dataset.build_samples(use_cache=True)
    train_s, val_s, cal_s = dataset.split_temporal(samples)
    
    unseen = val_s + cal_s
    print(f'Unseen samples: {len(unseen)}')

    print('Fetching metadata for matches...')
    match_ids = list(set(s.match_id for s in unseen))
    conn = dataset.get_db_connection()
    c = conn.cursor()
    
    # We load league and volatility from DB to have them in JSON if you want to explore it
    c.execute(f"SELECT match_id, date, league FROM matches WHERE match_id IN ({','.join(['?']*len(match_ids))})", match_ids)
    match_meta = {}
    for r in c.fetchall():
        match_meta[r['match_id']] = {
            'date': r['date'],
            'league': r['league'],
            'volatility': 0  # Since it's not in DB, use default or ignore
        }
    conn.close()

    results = []

    print('Generating predictions...')
    for s in tqdm(unseen, desc="Exporting predictions"):
        key = f"{s.target}_{s.pace_bucket}_{s.gender}"
        if key not in models:
            continue
            
        model_info = models[key]
        clf = model_info['clf']
        vec = model_info['vec']
        scaler = model_info['scaler']
        
        # We process singly here for simplicity, although batched is faster
        x = vec.transform([s.features])
        x = scaler.transform(x)
        
        # Weighted prediction
        proba_home = 0.0
        for m_info in clf['models']:
            model = m_info['model']
            weight = m_info['weight']
            pred_proba = model.predict_proba(x)[0]
            proba_home += pred_proba[1] * weight
            
        confidence = abs(proba_home - 0.5) * 2
        winner_pick = "home" if proba_home > 0.5 else "away"
        
        meta_m = match_meta.get(s.match_id, {})
        
        # Quarter scores to check NBA direction gate
        scores = {
            'q1_home': s.features.get('q1_home', 0),
            'q1_away': s.features.get('q1_away', 0),
            'q2_home': s.features.get('q2_home', 0),
            'q2_away': s.features.get('q2_away', 0),
            'q3_home': s.features.get('q3_home', 0),
            'q3_away': s.features.get('q3_away', 0),
        }
        
        # Minimal data for frontend portal
        res = {
            'match_id': s.match_id,
            'target': s.target,
            'date': meta_m.get('date', 'unknown'),
            'league': meta_m.get('league', 'unknown'),
            'volatility': meta_m.get('volatility', 0),
            'gender': s.gender,
            'pace_bucket': s.pace_bucket,
            'confidence': round(float(confidence), 4),
            'winner_pick': winner_pick,
            'target_winner': 'home' if s.target_winner == 1 else 'away',
            'hit': (winner_pick == ('home' if s.target_winner == 1 else 'away'))
        }
        res.update(scores)
        results.append(res)
        
    out_path = ROOT / "match" / "training" / "v13" / "model_outputs" / "dashboard_data.json"
    print(f'Saving {len(results)} items to {out_path}')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    
if __name__ == "__main__":
    export()
