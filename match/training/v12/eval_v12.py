"""
V12 - Evaluation Script
========================
Evaluates V12 models on historical data with full risk management.
Reports: accuracy, hit rate, ROI, profit, coverage, risk metrics.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = __import__("db")
infer_mod = __import__("training.v12.infer_match_v12", fromlist=["run_inference", "prediction_to_dict"])

DB_PATH = ROOT / "matches.db"
MODEL_DIR = ROOT / "training" / "v12" / "model_outputs"
EVAL_DIR = ROOT / "training" / "v12" / "eval_outputs"

DEFAULT_ODDS = 1.91
LOSS_PENALTY = 2.0


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _is_complete_match(data: dict) -> bool:
    quarters = data.get("score", {}).get("quarters", {})
    required = {"Q1", "Q2", "Q3", "Q4"}
    if not required.issubset(quarters.keys()):
        return False
    gp = data.get("graph_points", [])
    if not gp or len(gp) < 20:
        return False
    pbp = data.get("play_by_play", {})
    if not pbp or not required.issubset(pbp.keys()):
        return False
    for q in required:
        if len(pbp[q]) == 0:
            return False
    return True


def evaluate_v12(
    limit_matches: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    odds: float = DEFAULT_ODDS,
) -> dict:
    """Evaluate V12 on historical matches."""
    print("[v12-eval] Loading models...")
    
    # Check models exist
    for target in ["q3", "q4"]:
        clf_path = MODEL_DIR / f"{target}_clf_ensemble.joblib"
        if not clf_path.exists():
            return {"error": f"Model {clf_path} not found. Train first."}
    
    print("[v12-eval] Loading DB...")
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    
    # Build query
    sql = "SELECT match_id, date, time FROM matches WHERE status_type = 'finished' ORDER BY date, time"
    params = []
    if start_date:
        sql = sql.replace("WHERE", "WHERE date >= ? AND")
        params.insert(0, start_date)
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)
    if limit_matches:
        sql += " LIMIT ?"
        params.append(limit_matches)
    
    rows = conn.execute(sql, params).fetchall()
    print(f"[v12-eval] Evaluating {len(rows)} matches...")
    
    # Load league stats
    league_stats_file = MODEL_DIR / "league_stats.json"
    league_stats = {}
    if league_stats_file.exists():
        with open(league_stats_file, "r") as f:
            league_stats = json.load(f)
    
    # Results tracking
    results = {
        "q3": {
            "total": 0, "valid": 0, "bets": 0,
            "winner_correct": 0, "winner_wrong": 0, "winner_push": 0,
            "ou_correct": 0, "ou_wrong": 0, "ou_push": 0,
            "profit_winner": 0.0, "profit_ou": 0.0,
            "leagues": defaultdict(lambda: {"bets": 0, "hits": 0, "profit": 0.0}),
        },
        "q4": {
            "total": 0, "valid": 0, "bets": 0,
            "winner_correct": 0, "winner_wrong": 0, "winner_push": 0,
            "ou_correct": 0, "ou_wrong": 0, "ou_push": 0,
            "profit_winner": 0.0, "profit_ou": 0.0,
            "leagues": defaultdict(lambda: {"bets": 0, "hits": 0, "profit": 0.0}),
        },
    }
    
    skipped_data = 0
    skipped_model = 0
    
    for idx, row in enumerate(rows):
        if idx % 500 == 0:
            print(f"[v12-eval] Progress: {idx}/{len(rows)}")
        
        match_id = str(row["match_id"])
        data = db_mod.get_match(conn, match_id)
        
        if data is None or not _is_complete_match(data):
            skipped_data += 1
            continue
        
        league = data["match"].get("league", "unknown")
        
        for target in ["q3", "q4"]:
            results[target]["total"] += 1
            
            q_label = "Q3" if target == "q3" else "Q4"
            q_home, q_away = _quarter_points(data, q_label)
            
            if q_home is None or q_away is None:
                continue
            
            results[target]["valid"] += 1
            
            # Actual outcome
            if q_home > q_away:
                actual_winner = "home"
            elif q_away > q_home:
                actual_winner = "away"
            else:
                actual_winner = "push"
                results[target]["winner_push"] += 1
                continue
            
            # Get prediction from V12 (simplified - use model directly for speed)
            # For full evaluation, we'd run the full inference
            # Here we do a quick evaluation using the models directly
            
            # Build features (simplified version for eval)
            from training.v12.infer_match_v12 import _build_features_from_db, _load_ensemble, _predict_proba
            
            # Get top leagues/teams for bucketing
            top_leagues_set = set(list(league_stats.keys())[:25])
            top_teams_set = set()
            
            features = _build_features_from_db(data, target, league_stats, top_leagues_set, top_teams_set)
            
            clf_ensemble = _load_ensemble(target)
            if clf_ensemble is None:
                skipped_model += 1
                continue
            
            prob_home = _predict_proba(clf_ensemble, features)
            predicted_winner = "home" if prob_home >= 0.5 else "away"
            confidence = abs(prob_home - 0.5) * 2.0
            
            # Conservative gate
            if confidence < 0.65:
                continue
            
            results[target]["bets"] += 1
            results[target]["leagues"][league]["bets"] += 1
            
            # Track winner
            if predicted_winner == actual_winner:
                results[target]["winner_correct"] += 1
                results[target]["profit_winner"] += (odds - 1.0)
                results[target]["leagues"][league]["hits"] += 1
                results[target]["leagues"][league]["profit"] += (odds - 1.0)
            else:
                results[target]["winner_wrong"] += 1
                results[target]["profit_winner"] -= LOSS_PENALTY
                results[target]["leagues"][league]["profit"] -= LOSS_PENALTY
    
    conn.close()
    
    # Build report
    report = {
        "version": "v12",
        "evaluated_at": datetime.now().isoformat(),
        "odds": odds,
        "loss_penalty": LOSS_PENALTY,
        "skipped_data": skipped_data,
        "skipped_model": skipped_model,
        "quarters": {},
    }
    
    for target in ["q3", "q4"]:
        r = results[target]
        total_bets = r["bets"]
        total_correct = r["winner_correct"]
        total_wrong = r["winner_wrong"]
        
        accuracy = total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0
        hit_rate = total_correct / total_bets if total_bets > 0 else 0
        roi = r["profit_winner"] / total_bets if total_bets > 0 else 0
        
        # League breakdown
        league_report = {}
        for league, lr in sorted(r["leagues"].items(), key=lambda x: x[1]["bets"], reverse=True)[:20]:
            if lr["bets"] >= 10:  # Only leagues with enough bets
                league_report[league] = {
                    "bets": lr["bets"],
                    "hits": lr["hits"],
                    "hit_rate": round(lr["hits"] / lr["bets"], 4) if lr["bets"] > 0 else 0,
                    "profit": round(lr["profit"], 2),
                }
        
        report["quarters"][target] = {
            "total_matches": r["total"],
            "valid_matches": r["valid"],
            "bets_placed": total_bets,
            "coverage": round(total_bets / r["valid"], 4) if r["valid"] > 0 else 0,
            "correct": total_correct,
            "wrong": total_wrong,
            "pushes": r["winner_push"],
            "accuracy": round(accuracy, 4),
            "hit_rate": round(hit_rate, 4),
            "profit": round(r["profit_winner"], 2),
            "roi": round(roi, 4),
            "top_leagues": league_report,
        }
    
    # Save report
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_file = EVAL_DIR / "eval_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    return report


def print_report(report: dict):
    """Pretty print evaluation report."""
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    
    print(f"\n{'='*70}")
    print(f"V12 EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"Evaluated at: {report.get('evaluated_at', 'N/A')}")
    print(f"Odds: {report.get('odds', 1.91)} | Loss Penalty: {report.get('loss_penalty', 2.0)}x")
    print(f"Skipped (data): {report.get('skipped_data', 0)}")
    print(f"Skipped (model): {report.get('skipped_model', 0)}")
    
    for target, qr in report.get("quarters", {}).items():
        print(f"\n{'─'*70}")
        print(f"  {target.upper()}")
        print(f"{'─'*70}")
        print(f"  Total matches:   {qr.get('total_matches', 0)}")
        print(f"  Valid matches:   {qr.get('valid_matches', 0)}")
        print(f"  Bets placed:     {qr.get('bets_placed', 0)}")
        print(f"  Coverage:        {qr.get('coverage', 0):.2%}")
        print(f"  Correct:         {qr.get('correct', 0)}")
        print(f"  Wrong:           {qr.get('wrong', 0)}")
        print(f"  Accuracy:        {qr.get('accuracy', 0):.2%}")
        print(f"  Hit Rate:        {qr.get('hit_rate', 0):.2%}")
        print(f"  Profit:          {qr.get('profit', 0):.2f} units")
        print(f"  ROI:             {qr.get('roi', 0):.4f} per bet")
        
        # Top leagues
        top_leagues = qr.get("top_leagues", {})
        if top_leagues:
            print(f"\n  Top Leagues (by bets):")
            for league, lr in list(top_leagues.items())[:10]:
                print(f"    {league}: bets={lr['bets']}, hit_rate={lr['hit_rate']:.2%}, profit={lr['profit']:.2f}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V12 Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit matches")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--odds", type=float, default=DEFAULT_ODDS, help="Decimal odds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    start = time.time()
    report = evaluate_v12(
        limit_matches=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        odds=args.odds,
    )
    elapsed = time.time() - start
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
        print(f"Evaluation completed in {elapsed:.1f}s")
