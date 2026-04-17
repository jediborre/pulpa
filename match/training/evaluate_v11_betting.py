"""
V11 - Evaluate Over/Under predictions with sportsbook line
Calculate edge and betting decisions based on prediction vs line

Betting Logic V2:
- Model predicts actual total (e.g., 24 pts in test data)
- Suggested line = prediction + MARGIN (sportsbook margin, typically +5 pts)
- Sportsbook line is provided as input (e.g., 45, 55, 65)
- Edge = suggested_line - sportsbook_line
- MIN_EDGE filters out low-value bets

Usage:
    python evaluate_v11_betting.py --line 55.5
    python evaluate_v11_betting.py --line 45 --margin 5 --min-edge 3
"""

import csv
import sqlite3
import sys
import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_outputs_v11"

ODDS = 1.91
DEFAULT_MARGIN = 5.0  # Sportsbook typically adds ~5 pts to predicted line
DEFAULT_MIN_EDGE = 3.0  # Only bet if |edge| >= MIN_EDGE, else NO_BET
DEFAULT_LINE = 45.0

# Global variables that can be modified by CLI args
MARGIN = DEFAULT_MARGIN
MIN_EDGE = DEFAULT_MIN_EDGE


def load_model(target: str, gender: str):
    """Load ensemble model for target and gender"""
    path = OUT_DIR / f"{target}_{gender}_gb.joblib"
    if not path.exists():
        print(f"[V11] {path} not found")
        return None
    return joblib.load(path)


def get_test_data(target: str):
    """Get test data from DB"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
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
        
        if target == "q3_total":
            actual = m["q3h"] + m["q3a"]
        elif target == "q4_total":
            actual = m["q4h"] + m["q4a"]
        else:
            continue
        
        features = {
            "league_bucket": league_bucket,
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
            "gender": gender,
            "features": features,
            "actual": actual
        })
    
    conn.close()
    
    n = len(data)
    test_start = int(n * 0.8)
    return data[test_start:]


def evaluate_target(target: str, sportsbook_line: float):
    """Evaluate single target with sportsbook line
    
    V11 betting logic (V2 - with margin):
    - Model predicts actual total (e.g., 45 pts)
    - Suggested line = prediction + MARGIN (e.g., 50 pts)
    - Sportsbook offers line (e.g., 55.5)
    - Edge = suggested_line - sportsbook_line (e.g., -5.5)
    - If edge > MIN_EDGE: bet OVER (value - prediction underestimates)
    - If edge < -MIN_EDGE: bet UNDER (value - prediction overestimates)
    - Hit: actual > line (OVER wins) or actual < line (UNDER wins)
    """
    print(f"\n[V11] Evaluating {target} with sportsbook line {sportsbook_line}...")
    
    data = get_test_data(target)
    print(f"  Test samples: {len(data)}")
    
    if not data:
        return []
    
    results = []
    for d in data:
        gender = d["gender"]
        model = load_model(target, gender)
        if not model:
            continue
        
        X_dict = [d["features"]]
        vec = model["vectorizer"]
        reg = model["model"]
        
        X = vec.transform(X_dict)
        prediction = reg.predict(X)[0]
        
        actual = d["actual"]
        
        suggested_line = prediction + MARGIN
        edge = suggested_line - sportsbook_line
        
        if edge > MIN_EDGE:
            bet = "OVER"
            bet_hit = actual > sportsbook_line
        elif edge < -MIN_EDGE:
            bet = "UNDER"
            bet_hit = actual < sportsbook_line
        else:
            bet = "NO_BET"
            bet_hit = None
        
        if bet in ["OVER", "UNDER"]:
            profit = ODDS - 1 if bet_hit else -1
            roi_pct = profit * 100
        else:
            profit = 0
            roi_pct = 0
        
        results.append({
            "target": target,
            "gender": gender,
            "match_id": d["match_id"],
            "home_team": d["home_team"],
            "away_team": d["away_team"],
            "sportsbook_line": sportsbook_line,
            "prediction": round(prediction, 1),
            "suggested_line": round(suggested_line, 1),
            "actual": actual,
            "edge": round(edge, 1),
            "bet": bet,
            "bet_hit": bet_hit,
            "profit": round(profit, 2),
            "roi_pct": round(roi_pct, 1)
        })
    
    if results:
        over_bets = [r for r in results if r["bet"] == "OVER"]
        under_bets = [r for r in results if r["bet"] == "UNDER"]
        
        if over_bets:
            hits = sum(1 for r in over_bets if r["bet_hit"])
            profit = sum(r["profit"] for r in over_bets)
            roi = (profit / len(over_bets)) * 100 if over_bets else 0
            print(f"  OVER: {len(over_bets)} bets, {hits} hits ({hits/len(over_bets)*100:.1f}%), ROI: {roi:.1f}%")
        
        if under_bets:
            hits = sum(1 for r in under_bets if r["bet_hit"])
            profit = sum(r["profit"] for r in under_bets)
            roi = (profit / len(under_bets)) * 100 if under_bets else 0
            print(f"  UNDER: {len(under_bets)} bets, {hits} hits ({hits/len(under_bets)*100:.1f}%), ROI: {roi:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="V11 Betting Evaluation")
    parser.add_argument("--line", type=float, default=DEFAULT_LINE, help="Sportsbook line (default: 45.0)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Sportsbook margin (default: 5.0)")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE, help="Min edge to bet (default: 3.0)")
    args = parser.parse_args()
    
    global MARGIN, MIN_EDGE
    MARGIN = args.margin
    MIN_EDGE = args.min_edge
    sportsbook_line = args.line
    
    print("=" * 60)
    print(f"V11 - Over/Under Betting Evaluation")
    print(f"Line: {sportsbook_line}, Margin: {MARGIN}, Min Edge: {MIN_EDGE}")
    print("=" * 60)
    
    all_results = []
    
    for target in ["q3_total", "q4_total"]:
        results = evaluate_target(target, sportsbook_line)
        all_results.extend(results)
    
    if all_results:
        for r in all_results:
            r["home_team"] = r["home_team"].encode('ascii', 'replace').decode('ascii')
            r["away_team"] = r["away_team"].encode('ascii', 'replace').decode('ascii')
        
        with (OUT_DIR / f"betting_results_{int(sportsbook_line*10)}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n[V11] Saved {len(all_results)} predictions")
        
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        
        over_bets = [r for r in all_results if r["bet"] == "OVER"]
        under_bets = [r for r in all_results if r["bet"] == "UNDER"]
        
        if over_bets:
            hits = sum(1 for r in over_bets if r["bet_hit"])
            profit = sum(r["profit"] for r in over_bets)
            roi = (profit / len(over_bets)) * 100
            print(f"OVER: {len(over_bets)} bets, {hits} hits ({hits/len(over_bets)*100:.1f}%), ROI: {roi:.1f}%")
        
        if under_bets:
            hits = sum(1 for r in under_bets if r["bet_hit"])
            profit = sum(r["profit"] for r in under_bets)
            roi = (profit / len(under_bets)) * 100
            print(f"UNDER: {len(under_bets)} bets, {hits} hits ({hits/len(under_bets)*100:.1f}%), ROI: {roi:.1f}%")


if __name__ == "__main__":
    main()