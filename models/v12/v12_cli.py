"""
V12 CLI - Command Line Interface for V12 Models
================================================
Usage:
  python v12_cli.py train                      # Train all V12 models
  python v12_cli.py predict <match_id>         # Predict for a match
  python v12_cli.py predict <match_id> --line 185.5  # With sportsbook line
  python v12_cli.py eval                       # Evaluate on historical data
  python v12_cli.py validate                   # Check data leakage + simulation
  python v12_cli.py validate --leakage         # Only leakage check
  python v12_cli.py validate --simulation      # Only bankroll simulation
  python v12_cli.py validate --full            # Full validation with plots
  python v12_cli.py live --simulate            # Simulate comeback betting
  python v12_cli.py live --scenarios           # Show example scenarios
  python v12_cli.py leagues                    # Show league stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_train(args):
    """Train V12 models."""
    from training.v12 import train_v12
    train_v12.main()


def cmd_predict(args):
    """Run prediction for a match."""
    from training.v12.infer_match_v12 import run_inference, prediction_to_dict
    
    result = run_inference(
        args.match_id,
        target=args.target,
        sportsbook_line=args.line,
        fetch_missing=not args.no_fetch,
    )
    
    if isinstance(result, dict) and not result.get("ok", True):
        print(f"ERROR: {result.get('reason', 'Unknown error')}")
        sys.exit(1)
    
    if args.json:
        print(json.dumps(prediction_to_dict(result), indent=2))
    else:
        pred = result
        print(f"\n{'='*60}")
        print(f"V12 PREDICTION - {pred.match_id} - {pred.quarter.upper()}")
        print(f"{'='*60}")
        print(f"")
        print(f"WINNER PREDICTION:")
        print(f"  Pick:          {pred.winner_pick}")
        print(f"  Confidence:    {pred.winner_confidence:.2%}")
        print(f"  Signal:        {pred.winner_signal}")
        print(f"")
        if pred.predicted_total:
            print(f"POINTS PREDICTION:")
            print(f"  Predicted Total: {pred.predicted_total:.1f}")
            if pred.predicted_home:
                print(f"  Predicted Home:  {pred.predicted_home:.1f}")
            if pred.predicted_away:
                print(f"  Predicted Away:  {pred.predicted_away:.1f}")
            print(f"  O/U Signal:      {pred.over_under_signal}")
            print(f"")
        print(f"RISK ASSESSMENT:")
        print(f"  League Quality:  {pred.league_quality}")
        print(f"  League Bettable: {'YES' if pred.league_bettable else 'NO'}")
        print(f"  Volatility:      {pred.volatility_index:.2f}")
        print(f"  Data Quality:    {pred.data_quality}")
        print(f"")
        print(f"{'─'*60}")
        print(f"FINAL RECOMMENDATION:")
        print(f"  Signal:        {pred.final_signal}")
        print(f"  Confidence:    {pred.final_confidence:.2%}")
        print(f"  Risk Level:    {pred.risk_level.upper()}")
        print(f"  Reasoning:     {pred.reasoning}")
        print(f"{'='*60}\n")


def cmd_eval(args):
    """Evaluate V12 models."""
    from training.v12 import eval_v12
    
    report = eval_v12.evaluate_v12(
        limit_matches=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        odds=args.odds,
    )
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        eval_v12.print_report(report)


def cmd_leagues(args):
    """Show league statistics."""
    league_stats_file = ROOT / "training" / "v12" / "model_outputs" / "league_stats.json"
    
    if not league_stats_file.exists():
        print("League stats not found. Train V12 models first.")
        return
    
    with open(league_stats_file, "r") as f:
        league_stats = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"V12 LEAGUE STATISTICS")
    print(f"{'='*70}")
    print(f"Total leagues: {len(league_stats)}")
    print(f"")
    
    # Sort by sample size
    sorted_leagues = sorted(league_stats.items(), key=lambda x: x[1].get("samples", 0), reverse=True)
    
    print(f"{'League':<50} {'Samples':>8} {'Home WR':>10} {'Avg Total':>12}")
    print(f"{'─'*70}")
    
    for league, stats in sorted_leagues[:30]:
        samples = stats.get("samples", 0)
        home_wr = stats.get("home_win_rate", 0)
        avg_total = stats.get("avg_total_points", 0)
        
        print(f"{league:<50} {samples:>8} {home_wr:>10.2%} {avg_total:>12.1f}")
    
    print(f"\n{'='*70}\n")


def cmd_validate(args):
    """Run validation and bankroll simulation."""
    from training.v12 import validate_v12
    
    if args.leakage:
        result = validate_v12.check_data_leakage()
        print(json.dumps(result, indent=2))
    elif args.simulation:
        result = validate_v12.simulate_bankroll(
            odds=args.odds,
            initial_bank=args.bank,
            limit_matches=args.limit,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"V12 BANKROLL SIMULATION")
            print(f"{'='*60}")
            print(f"Odds:            {result['odds']}")
            print(f"Initial Bank:    ${result['initial_bankroll']:,.2f}")
            print(f"Final Bank:      ${result['final_bankroll']:,.2f}")
            print(f"Profit:          ${result['total_profit']:,.2f} ({result['roi']:.1%})")
            print(f"Hit Rate:        {result['hit_rate']:.1%}")
            print(f"Total Bets:      {result['total_bets']}")
            print(f"Max Drawdown:    {result['max_drawdown']:.1%}")
            print(f"Consec Losses:   {result['max_consecutive_losses']}")
            print(f"{'='*60}\n")
            
            if result['roi'] > 1.0:  # >100% ROI
                print(f"⚠️  WARNING: ROI of {result['roi']:.0%} is UNREALISTIC")
                print(f"   This indicates DATA LEAKAGE in the model.")
                print(f"   Realistic ROI should be -10% to +10%")
                print(f"   See: HONEST_ASSESSMENT.md")
                print(f"{'='*60}\n")
    elif args.full:
        validate_v12.run_full_validation()
    else:
        # Default: run leakage + simulation
        print("Running data leakage check...")
        leakage = validate_v12.check_data_leakage()
        print(f"\nCritical issues: {leakage['critical_count']}")
        print(f"Moderate issues: {leakage['moderate_count']}")
        
        print(f"\nRunning bankroll simulation (odds={args.odds})...")
        result = validate_v12.simulate_bankroll(
            odds=args.odds,
            initial_bank=args.bank,
            limit_matches=args.limit,
        )
        print(f"\nFinal Bank: ${result['final_bankroll']:,.2f}")
        print(f"ROI: {result['roi']:.1%}")
        print(f"Hit Rate: {result['hit_rate']:.1%}")


def cmd_live(args):
    """Run live betting engine."""
    from training.v12.live_engine import live_betting
    
    if args.scenarios:
        live_betting.print_live_scenario()
    elif args.simulate:
        start = time.time()
        result = live_betting.simulate_historical_comebacks(limit_matches=args.limit)
        elapsed = time.time() - start
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*70}")
            print(f"V12 LIVE - COMEBACK BETTING SIMULATION")
            print(f"{'='*70}")
            print(f"Matches analyzed: {result['sample_size']}")
            print(f"Total opportunities: {result['total_opportunities']}")
            print(f"Comeback wins: {result['comeback_wins']}")
            print(f"Comeback losses: {result['comeback_losses']}")
            print(f"Comeback rate: {result['comeback_rate']:.1%}")
            print(f"Avg odds on comebacks: {result['avg_comeback_odds']:.2f}")
            print(f"Avg profit per bet: ${result['avg_profit_per_bet']:.2f}")
            print(f"Avg trailing by: {result['avg_trailing_by']:.1f} pts")
            print(f"{'='*70}")
            print(f"Completed in {elapsed:.1f}s")
            print(f"Results saved to: {live_betting.LIVE_DIR / 'comeback_simulation.json'}")
            print(f"{'='*70}\n")
    else:
        # Default: show scenarios
        live_betting.print_live_scenario()


def main():
    parser = argparse.ArgumentParser(
        description="V12 Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train V12 models")
    
    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict for a match")
    predict_parser.add_argument("match_id", help="SofaScore match ID")
    predict_parser.add_argument("--target", choices=["q3", "q4"], default="q4")
    predict_parser.add_argument("--line", type=float, default=None, help="Sportsbook over/under line")
    predict_parser.add_argument("--no-fetch", action="store_true", help="Don't scrape if not in DB")
    predict_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate V12 models")
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit matches")
    eval_parser.add_argument("--start-date", type=str, default=None)
    eval_parser.add_argument("--end-date", type=str, default=None)
    eval_parser.add_argument("--odds", type=float, default=1.91)
    eval_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # Leagues
    leagues_parser = subparsers.add_parser("leagues", help="Show league stats")
    
    # Validate
    validate_parser = subparsers.add_parser("validate", help="Run validation & simulation")
    validate_parser.add_argument("--leakage", action="store_true", help="Only check data leakage")
    validate_parser.add_argument("--simulation", action="store_true", help="Only run bankroll sim")
    validate_parser.add_argument("--full", action="store_true", help="Full validation")
    validate_parser.add_argument("--odds", type=float, default=1.41, help="Odds for simulation")
    validate_parser.add_argument("--bank", type=float, default=1000, help="Initial bankroll")
    validate_parser.add_argument("--limit", type=int, default=1000, help="Limit matches")
    validate_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # Live
    live_parser = subparsers.add_parser("live", help="Live betting engine")
    live_parser.add_argument("--simulate", action="store_true", help="Simulate comeback betting")
    live_parser.add_argument("--scenarios", action="store_true", help="Show example scenarios")
    live_parser.add_argument("--limit", type=int, default=500, help="Limit matches for simulation")
    live_parser.add_argument("--json", action="store_true", help="JSON output")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "leagues":
        cmd_leagues(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "live":
        cmd_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
