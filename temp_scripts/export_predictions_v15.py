import sys
import json
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"c:\Users\borre\OneDrive\OLD\Escritorio\pulpa").resolve()
sys.path.insert(0, str(ROOT / 'match'))

from training.v15 import dataset as ds
from training.v15.inference import V15Engine, SUMMARY_PATH

def export():
    print('Loading v15 engine...')
    engine = V15Engine.load(SUMMARY_PATH)
    
    print('Loading samples cache (v15)...')
    samples, _ = ds.build_samples(use_cache=True, verbose=False)
    splits = ds.split_temporal(samples)
    
    # Validation and holdout (unseen)
    unseen = splits.get("holdout", [])
    if not unseen:
        unseen = splits.get("validation", []) + splits.get("calibration", [])
    if not unseen:
        unseen = samples
        
    print(f'Unseen samples for V15: {len(unseen)}')

    print('Fetching DB metadata for matches...')
    match_ids = list(set(s.match_id for s in unseen))
    conn = ds.get_db_connection()
    c = conn.cursor()
    
    c.execute(f"SELECT match_id, date, league FROM matches WHERE match_id IN ({','.join(['?']*len(match_ids))})", match_ids)
    match_meta = {}
    for r in c.fetchall():
        match_meta[r['match_id']] = {
            'date': r['date'],
            'league': r['league'],
            'volatility': 0  # not universally in matches table anymore or requires compute
        }
    conn.close()

    results = []

    print('Generating predictions V15...')
    for s in tqdm(unseen, desc="Exporting V15"):
        bundle = engine.models.get((s.league, s.target))
        if bundle is None:
            continue
            
        clf = bundle.clf
        vec = bundle.vec
        
        f = s.features
        x = vec.transform([f])
        probas = clf.predict_proba(x)[0]
        proba_home = float(probas[1])
        
        confidence = abs(proba_home - 0.5) * 2
        winner_pick = "home" if proba_home >= 0.5 else "away"
        
        meta_m = match_meta.get(s.match_id, {})
        
        scores = {
            'q1_home': f.get('q1_home', 0),
            'q1_away': f.get('q1_away', 0),
            'q2_home': f.get('q2_home', 0),
            'q2_away': f.get('q2_away', 0),
            'q3_home': f.get('q3_home', 0),
            'q3_away': f.get('q3_away', 0),
        }
        
        gender = "women" if "women" in s.league.lower() or "femenina" in s.league.lower() else "men"
        
        res = {
            'match_id': s.match_id,
            'target': s.target,
            'date': meta_m.get('date', 'unknown'),
            'league': meta_m.get('league', 'unknown'),
            'volatility': 0, # Placeholder
            'gender': gender,
            'pace_bucket': "medium", # V15 uses pure leagues, so pace is mixed
            'confidence': round(float(confidence), 4),
            'winner_pick': winner_pick,
            'target_winner': 'home' if s.target_winner == 1 else 'away',
            'hit': (winner_pick == ('home' if s.target_winner == 1 else 'away'))
        }
        res.update(scores)
        results.append(res)
        
    out_path = ROOT / "v13_dashboard" / "public" / "dashboard_data_v15.json"
    print(f'Saving {len(results)} items to {out_path}')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    
if __name__ == "__main__":
    export()
