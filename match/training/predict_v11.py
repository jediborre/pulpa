"""
V11 - Predictor en tiempo real para Over/Under Q3/Q4

Usage:
    python predict_v11.py --match-id 15556170 --line 55.5
    python predict_v11.py --league "NBA" --home "Lakers" --away "Celtics" --q1h 28 --q2h 27 --q1a 25 --q2a 20 --line 55.5
"""

import argparse
import importlib
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = importlib.import_module("db")

DB_PATH = ROOT / "matches.db"
MODEL_DIR = ROOT / "training" / "model_outputs_v11"

ODDS = 1.91
DEFAULT_MARGIN = 5.0
DEFAULT_MIN_EDGE = 3.0


def load_model(target: str, gender: str):
    """Load model for target and gender"""
    path = MODEL_DIR / f"{target}_{gender}_gb.joblib"
    if not path.exists():
        print(f"[ERROR] Model not found: {path}")
        return None
    return joblib.load(path)


def infer_gender(league: str, home_team: str, away_team: str) -> str:
    """Infer gender from league/team names"""
    text = f"{league} {home_team} {away_team}".lower()
    markers = ["women", "woman", "female", "femen", "fem.", " ladies ", "(w)", " w ", "wnba", "girls"]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


def predict_from_match(match_id: str, line: float, margin: float = DEFAULT_MARGIN, min_edge: float = DEFAULT_MIN_EDGE):
    """Predict from existing match in DB"""
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    
    data = db_mod.get_match(conn, match_id)
    conn.close()
    
    if not data:
        print(f"[ERROR] Match {match_id} not found in DB")
        return
    
    m = data["match"]
    score = data.get("score", {})
    q1 = score.get("quarters", {}).get("Q1", {})
    q2 = score.get("quarters", {}).get("Q2", {})
    q3 = score.get("quarters", {}).get("Q3", {})
    
    league = m.get("league", "")
    home_team = m.get("home_team", "")
    away_team = m.get("away_team", "")
    gender = infer_gender(league, home_team, away_team)
    
    q1h = int(q1.get("home", 0))
    q1a = int(q1.get("away", 0))
    q2h = int(q2.get("home", 0))
    q2a = int(q2.get("away", 0))
    
    ht_home = q1h + q2h
    ht_away = q1a + q2a
    ht_total = ht_home + ht_away
    
    features = {
        "league_bucket": league if league else "OTHER",
        "home_team": home_team if home_team else "TEAM_OTHER",
        "away_team": away_team if away_team else "TEAM_OTHER",
        "ht_home": ht_home,
        "ht_away": ht_away,
        "ht_total": ht_total,
        "q1_diff": q1h - q1a,
        "q2_diff": q2h - q2a,
    }
    
    for target in ["q3_total", "q4_total"]:
        model = load_model(target, gender)
        if not model:
            continue
        
        X = model["vectorizer"].transform([features])
        prediction = model["model"].predict(X)[0]
        
        suggested_line = prediction + margin
        edge = suggested_line - line
        
        if edge > min_edge:
            bet = "OVER"
            recommendation = "APOSTAR OVER"
        elif edge < -min_edge:
            bet = "UNDER"
            recommendation = "APOSTAR UNDER"
        else:
            bet = "NO_BET"
            recommendation = "SIN VALOR"
        
        print(f"\n{'='*60}")
        print(f"MATCH: {home_team} vs {away_team}")
        print(f"Target: {target.upper()} | Gender: {gender}")
        print(f"{'='*60}")
        print(f"Q1+Q2 Score: {ht_home}-{ht_away} (total: {ht_total})")
        print(f"Model Prediction: {prediction:.1f} pts")
        print(f"Suggested Line: {suggested_line:.1f} pts (prediction + {margin})")
        print(f"Sportsbook Line: {line}")
        print(f"Edge: {edge:.1f}")
        print(f"Bet: {bet}")
        print(f"Recommendation: {recommendation}")
        
        if bet != "NO_BET":
            print(f"Odds: {ODDS}")
            print(f"If bet $100: Win ${(ODDS-1)*100:.0f} if hit, Lose $100 if miss")


def predict_from_params(league: str, home: str, away: str, q1h: int, q2h: int, q1a: int, q2a: int, line: float, margin: float = DEFAULT_MARGIN, min_edge: float = DEFAULT_MIN_EDGE):
    """Predict from manual parameters (live betting)"""
    gender = infer_gender(league, home, away)
    
    ht_home = q1h + q2h
    ht_away = q1a + q2a
    ht_total = ht_home + ht_away
    
    features = {
        "league_bucket": league if league else "OTHER",
        "home_team": home if home else "TEAM_OTHER",
        "away_team": away if away else "TEAM_OTHER",
        "ht_home": ht_home,
        "ht_away": ht_away,
        "ht_total": ht_total,
        "q1_diff": q1h - q1a,
        "q2_diff": q2h - q2a,
    }
    
    print(f"\n{'='*60}")
    print(f"LIVE PREDICTION")
    print(f"{'='*60}")
    print(f"League: {league}")
    print(f"Match: {home} vs {away}")
    print(f"Q1+Q2: {q1h}-{q1a} + {q2h}-{q2a} = {ht_home}-{ht_away} (total: {ht_total})")
    print(f"Gender: {gender}")
    print(f"Sportsbook Line: {line}")
    print(f"Margin: {margin}, Min Edge: {min_edge}")
    
    for target in ["q3_total", "q4_total"]:
        model = load_model(target, gender)
        if not model:
            continue
        
        X = model["vectorizer"].transform([features])
        prediction = model["model"].predict(X)[0]
        
        suggested_line = prediction + margin
        edge = suggested_line - line
        
        if edge > min_edge:
            bet = "OVER"
            rec = "OVER"
        elif edge < -min_edge:
            bet = "UNDER"
            rec = "UNDER"
        else:
            bet = "NO_BET"
            rec = "NO BET"
        
        print(f"\n{target.upper()}:")
        print(f"  Prediction: {prediction:.1f} | Suggested: {suggested_line:.1f} | Edge: {edge:.1f}")
        print(f"  -> {rec}")


def main():
    parser = argparse.ArgumentParser(description="V11 Live Predictor")
    parser.add_argument("--match-id", type=str, help="Match ID from DB")
    parser.add_argument("--line", type=float, required=True, help="Sportsbook line")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Margin (default: 5)")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE, help="Min edge (default: 3)")
    
    parser.add_argument("--league", type=str, help="League name")
    parser.add_argument("--home", type=str, help="Home team")
    parser.add_argument("--away", type=str, help="Away team")
    parser.add_argument("--q1h", type=int, help="Q1 home points")
    parser.add_argument("--q2h", type=int, help="Q2 home points")
    parser.add_argument("--q1a", type=int, help="Q1 away points")
    parser.add_argument("--q2a", type=int, help="Q2 away points")
    
    args = parser.parse_args()
    
    if args.match_id:
        predict_from_match(args.match_id, args.line, args.margin, args.min_edge)
    elif args.q1h is not None and args.q2h is not None and args.q1a is not None and args.q2a is not None:
        predict_from_params(
            args.league or "", args.home or "", args.away or "",
            args.q1h, args.q2h, args.q1a, args.q2a,
            args.line, args.margin, args.min_edge
        )
    else:
        print("Usage:")
        print("  python training/predict_v11.py --match-id 15556170 --line 55.5")
        print("  python training/predict_v11.py --league NBA --home Lakers --away Celtics --q1h 28 --q2h 27 --q1a 25 --q2a 20 --line 55.5")


if __name__ == "__main__":
    main()
