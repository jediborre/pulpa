"""
V10 - Evaluate Over/Under predictions on test data
Calculate ROI and effectiveness for betting simulation
"""

import csv
import sqlite3
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs_v10"

# Thresholds for O/U (based on EDA: Q3~39, Q4~39 mean)
THRESHOLDS = {
    "q3_home": 19,
    "q3_away": 19,
    "q3_total": 39,
    "q4_home": 19,
    "q4_away": 19,
    "q4_total": 39,
}

# Odds (typical -110 / 1.91)
ODDS = 1.91


def load_model(target: str):
    """Load ensemble model for target (not stacking - uses different input)"""
    path = OUT_DIR / f"{target}_gb.joblib"  # Use GB - works directly
    if not path.exists():
        print(f"[V10] {path} not found")
        return None
    return joblib.load(path)


def get_test_data(target: str):
    """Get test data from DB (same as training but test set only)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    if target.startswith("q3"):
        quarter_filter = "q1.q2"
    else:
        quarter_filter = "q1.q2.q3"
    
    matches = conn.execute(f"""
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
    
    # Get team counts for buckets
    team_counter = {}
    for m in conn.execute("SELECT home_team, away_team FROM matches").fetchall():
        if m[0]: team_counter[str(m[0])] = team_counter.get(str(m[0]), 0) + 1
        if m[1]: team_counter[str(m[1])] = team_counter.get(str(m[1]), 0) + 1
    
    top_teams = {t for t, _ in sorted(team_counter.items(), key=lambda x: x[1], reverse=True)[:120]}
    
    data = []
    for m in matches:
        league = m["league"] or ""
        ht, at = m["home_team"] or "", m["away_team"] or ""
        
        gender = "women" if any(x in f"{league} {ht} {at}".lower() 
                                for x in ["women", "woman", "female", "(w)"]) else "men_or_open"
        
        league_bucket = league if league else "OTHER"
        home_bucket = ht if ht in top_teams else "TEAM_OTHER"
        away_bucket = at if at in top_teams else "TEAM_OTHER"
        
        ht_home = m["q1h"] + m["q2h"]
        ht_away = m["q1a"] + m["q2a"]
        ht_total = ht_home + ht_away
        
        if target == "q3_home":
            actual = m["q3h"]
        elif target == "q3_away":
            actual = m["q3a"]
        elif target == "q3_total":
            actual = m["q3h"] + m["q3a"]
        elif target == "q4_home":
            actual = m["q4h"]
        elif target == "q4_away":
            actual = m["q4a"]
        else:  # q4_total
            actual = m["q4h"] + m["q4a"]
        
        features = {
            "league_bucket": league_bucket,
            "gender_bucket": gender,
            "home_team": home_bucket,
            "away_team": away_bucket,
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_total": ht_total,
            "q1_diff": m["q1h"] - m["q1a"],
            "q2_diff": m["q1h"] - m["q1a"],
        }
        
        if target.startswith("q4"):
            features["q3_diff"] = m["q3h"] - m["q3a"]
            features["score_3q_home"] = ht_home + m["q3h"]
            features["score_3q_away"] = ht_away + m["q3a"]
            features["score_3q_total"] = ht_total + m["q3h"] + m["q3a"]
        
        data.append({
            "match_id": m["match_id"],
            "home_team": ht,
            "away_team": at,
            "features": features,
            "actual": actual
        })
    
    conn.close()
    
    # Take last 20% as test (temporal split)
    n = len(data)
    test_start = int(n * 0.8)
    return data[test_start:]


def evaluate_target(target: str):
    """Evaluate single target"""
    print(f"\n[V10] Evaluating {target}...")
    
    model = load_model(target)
    if not model:
        return []
    
    data = get_test_data(target)
    print(f"  Test samples: {len(data)}")
    
    if not data:
        return []
    
    # Extract features
    from sklearn.feature_extraction import DictVectorizer
    
    X_dict = [d["features"] for d in data]
    y_actual = [d["actual"] for d in data]
    
    vec = model["vectorizer"]
    reg = model["model"]
    
    X = vec.transform(X_dict)
    y_pred = reg.predict(X)  # GB doesn't use scaled data
    
    # Calculate betting results
    threshold = THRESHOLDS.get(target, 39)
    
    results = []
    for i, d in enumerate(data):
        pred = y_pred[i]
        actual = y_actual[i]
        
        # Bet Over
        bet_over = pred > threshold
        over_hit = actual > threshold
        over_profit = ODDS - 1 if over_hit else -1
        over_roi = over_profit * 100
        
        # Bet Under
        bet_under = pred < threshold
        under_hit = actual < threshold
        under_profit = ODDS - 1 if under_hit else -1
        under_roi = under_profit * 100
        
        # Only count if prediction is clear (edge)
        edge = abs(pred - threshold)
        
        results.append({
            "target": target,
            "match_id": d["match_id"],
            "home_team": d["home_team"],
            "away_team": d["away_team"],
            "threshold": threshold,
            "prediction": round(pred, 1),
            "actual": actual,
            "error": round(pred - actual, 1),
            "bet_over": bet_over,
            "over_hit": over_hit,
            "over_roi": round(over_roi, 1),
            "bet_under": bet_under,
            "under_hit": under_hit,
            "under_roi": round(under_roi, 1),
            "edge": round(edge, 1)
        })
    
    # Summary stats
    over_bets = [r for r in results if r["bet_over"]]
    under_bets = [r for r in results if r["bet_under"]]
    
    if over_bets:
        over_hits = sum(1 for r in over_bets if r["over_hit"])
        over_total_profit = sum(r["over_roi"] for r in over_bets) / 100
        over_roi_pct = (over_total_profit / len(over_bets)) * 100 if over_bets else 0
        print(f"  OVER: {len(over_bets)} bets, {over_hits} hits, {over_roi_pct:.1f}% ROI")
    
    if under_bets:
        under_hits = sum(1 for r in under_bets if r["under_hit"])
        under_total_profit = sum(r["under_roi"] for r in under_bets) / 100
        under_roi_pct = (under_total_profit / len(under_bets)) * 100 if under_bets else 0
        print(f"  UNDER: {len(under_bets)} bets, {under_hits} hits, {under_roi_pct:.1f}% ROI")
    
    return results


def main():
    print("=" * 60)
    print("V10 - Over/Under Betting Evaluation")
    print("=" * 60)
    
    all_results = []
    
    for target in ["q3_home", "q3_away", "q3_total", "q4_home", "q4_away", "q4_total"]:
        results = evaluate_target(target)
        all_results.extend(results)
    
    # Save results (ASCII only)
    if all_results:
        # Use ASCII-safe team names
        for r in all_results:
            r["home_team"] = r["home_team"].encode('ascii', 'replace').decode('ascii')
            r["away_team"] = r["away_team"].encode('ascii', 'replace').decode('ascii')
        
        with (OUT_DIR / "betting_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n[V10] Saved {len(all_results)} predictions to betting_results.csv")
        
        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        
        # Group by target
        for target in ["q3_home", "q3_away", "q3_total", "q4_home", "q4_away", "q4_total"]:
            target_results = [r for r in all_results if r["target"] == target]
            if not target_results:
                continue
            
            over_bets = [r for r in target_results if r["bet_over"]]
            under_bets = [r for r in target_results if r["bet_under"]]
            
            print(f"\n{target.upper()}:")
            if over_bets:
                hits = sum(1 for r in over_bets if r["over_hit"])
                profit = sum(r["over_roi"] for r in over_bets)
                roi = (profit / len(over_bets))
                print(f"  OVER: {len(over_bets)} bets, {hits} hits ({hits/len(over_bets)*100:.1f}%), ROI: {roi:.1f}%")
            if under_bets:
                hits = sum(1 for r in under_bets if r["under_hit"])
                profit = sum(r["under_roi"] for r in under_bets)
                roi = (profit / len(under_bets))
                print(f"  UNDER: {len(under_bets)} bets, {hits} hits ({hits/len(under_bets)*100:.1f}%), ROI: {roi:.1f}%")


if __name__ == "__main__":
    main()