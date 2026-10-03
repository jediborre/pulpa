import sys
from pathlib import Path
import time
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "match" / "training"))

import train_q3_q4_models_v6 as v6

def main():
    t0 = time.time()
    print("Loading splits cache...")
    splits_path = ROOT / "match" / "training" / "model_outputs_m_v1" / "q4_roi_splits_cache.joblib"
    splits = joblib.load(splits_path)
    test_rows = splits["test_rows"]
    print(f"Loaded {len(test_rows)} test rows in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    print("Loading v6_2 model...")
    v6_2_path = ROOT / "match" / "training" / "model_outputs_v6_2" / "q4_champion.joblib"
    art = joblib.load(v6_2_path)
    vec = art["vectorizer"]
    models = art["models"]
    keep_leagues = set(art["league_filter"]["kept_leagues"])
    other_token = art["league_filter"]["other_token"]
    print(f"Loaded v6_2 model in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    print("Predicting v6_2...")
    # Rules
    import json
    cfg = json.loads((ROOT / "match" / "training" / "v6_2_league_name_exclusions.json").read_text(encoding="utf-8"))
    rules = []
    for cat in cfg.get("categories", []):
        for p in cat.get("patterns", []):
            p = str(p).strip()
            if p:
                rules.append((cat.get("name", "uncategorized"), p, p.lower()))
                
    probs_v6_2 = [None] * len(test_rows)
    excluded_flags_v6_2 = [False] * len(test_rows)
    excluded_reasons_v6_2 = [None] * len(test_rows)
    transformed = []
    transformed_idx = []
    
    for i, s in enumerate(test_rows):
        rec = dict(s.features_q4)
        lg = str(rec.get("league", ""))
        lg_lc = lg.lower()
        hit = None
        for cname, raw, low in rules:
            if low in lg_lc:
                hit = (cname, raw)
                break
        if hit is not None:
            excluded_flags_v6_2[i] = True
            excluded_reasons_v6_2[i] = f"excluded_league_name:{hit[0]}:{hit[1]}"
            continue
            
        if lg not in keep_leagues:
            rec["league"] = other_token
            rec["league_bucket"] = other_token
        transformed.append(rec)
        transformed_idx.append(i)
        
    if transformed:
        x_valid = vec.transform(transformed)
        p_xgb = models["xgb"].predict_proba(x_valid)[:, 1]
        p_hgb = models["hist_gb"].predict_proba(x_valid)[:, 1]
        p_blend = (0.6 * p_xgb + 0.4 * p_hgb)
        for j, orig_idx in enumerate(transformed_idx):
            probs_v6_2[orig_idx] = float(p_blend[j])
            
    print(f"Predicted v6_2 on {len(test_rows)} rows in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    print("Loading m27_v3 predictions from cache...")
    pred_path = ROOT / "match" / "training" / "model_outputs_m_v1" / "q4_roi_pred_cache.joblib"
    pred_cache = joblib.load(pred_path)
    probs_m27_v3 = pred_cache["predictions"]["p_m27_v3"]
    excl_m27_v3_flags = pred_cache["predictions"]["excl_m27_v3_flags"]
    excl_m27_v3_reasons = pred_cache["predictions"]["excl_m27_v3_reasons"]
    print(f"Loaded m27_v3 predictions in {time.time()-t0:.2f}s")
    
    # Verify sizes match
    print(f"v6_2 predictions={len(probs_v6_2)}, m27_v3 predictions={len(probs_m27_v3)}")
    
if __name__ == "__main__":
    main()
