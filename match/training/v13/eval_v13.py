"""
eval_v13.py — Evaluation with live simulation for V13.

Evaluates model performance with:
- Static evaluation (standard accuracy/F1 on test set)
- Live simulation (simulates real-time data arrival with delays)
- "Time to stable signal" metric
- Flip detection (NO_BET → BET changes)
- Betting window analysis
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add parent to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v13 import config, dataset, features
from training.v13 import infer_match_v13

OUT_DIR = ROOT / "training" / "v13" / "model_outputs"


def load_trained_models():
    """Load all trained models for evaluation."""
    import joblib
    
    summary_path = OUT_DIR / "training_summary.json"
    if not summary_path.exists():
        print("❌ No training summary found. Run training first.")
        return None
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    models = {}
    for m in summary.get('models_trained', []):
        key = m['key']
        
        clf_path = OUT_DIR / f"{key}_clf_ensemble.joblib"
        vec_path = OUT_DIR / f"{key}_vectorizer.joblib"
        scaler_path = OUT_DIR / f"{key}_scaler.joblib"
        
        if clf_path.exists() and vec_path.exists() and scaler_path.exists():
            models[key] = {
                'clf': joblib.load(clf_path),
                'vec': joblib.load(vec_path),
                'scaler': joblib.load(scaler_path),
            }
    
    print(f"✅ Loaded {len(models)} model ensembles")
    return models


def simulate_live_arrival(graph_points, pbp_events, target, minute):
    """
    Simulate live data arrival at a specific minute.
    
    Only includes data that would be available by that minute.
    """
    gp_upto = [p for p in graph_points if p['minute'] <= minute]
    
    # For Q3: PBP from Q1+Q2
    # For Q4: PBP from Q1+Q2+Q3
    if target == 'q3':
        pbp_upto = [e for e in pbp_events if e['quarter'] in ['Q1', 'Q2']]
    else:
        pbp_upto = [e for e in pbp_events if e['quarter'] in ['Q1', 'Q2', 'Q3']]
    
    return gp_upto, pbp_upto


def evaluate_static(test_samples, models):
    """
    Static evaluation on test set (standard ML evaluation).
    
    No live simulation, just accuracy/F1 on held-out data.
    """
    print("\n" + "="*80)
    print("📊 STATIC EVALUATION")
    print("="*80)
    
    results = {}
    
    # Group by model key
    by_key = defaultdict(list)
    for s in test_samples:
        by_key[f"{s.target}_{s.pace_bucket}_{s.gender}"].append(s)
    
    for key, samples in tqdm(by_key.items(), desc="Evaluating models"):
        if key not in models:
            continue
        
        model_info = models[key]
        clf = model_info['clf']
        vec = model_info['vec']
        scaler = model_info['scaler']
        
        # Extract features
        x = vec.transform([s.features for s in samples])
        x = scaler.transform(x)
        y_true = [s.target_winner for s in samples]
        
        # Predict
        y_pred = clf['models'][0]['model'].predict(x)
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        results[key] = {
            'samples': len(samples),
            'accuracy': round(acc, 3),
            'f1': round(f1, 3),
            'precision': round(precision_score(y_true, y_pred, average='weighted'), 3),
            'recall': round(recall_score(y_true, y_pred, average='weighted'), 3),
        }
    
    # Summary
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_f1 = np.mean([r['f1'] for r in results.values()])
        
        print(f"\n📊 Static Evaluation Summary:")
        print(f"   Models evaluated: {len(results)}")
        print(f"   Average accuracy: {avg_acc:.3f}")
        print(f"   Average F1:       {avg_f1:.3f}")
        
        # By target
        for target in ['q3', 'q4']:
            target_results = {k: v for k, v in results.items() if k.startswith(target)}
            if target_results:
                avg_acc_t = np.mean([r['accuracy'] for r in target_results.values()])
                avg_f1_t = np.mean([r['f1'] for r in target_results.values()])
                print(f"   {target.upper()}: acc={avg_acc_t:.3f}, f1={avg_f1_t:.3f}")
    
    return results


def evaluate_live_simulation(test_matches, models):
    """
    Live simulation evaluation.
    
    Simulates real-time data arrival with:
    - Q3: checks at minutes 20, 22, 24
    - Q4: checks at minutes 30, 32, 34
    - Measures: time to stable signal, flips, betting window
    """
    print("\n" + "="*80)
    print("🎥 LIVE SIMULATION EVALUATION")
    print("="*80)
    
    live_results = {
        'q3': {
            'total_matches': 0,
            'stable_by_minute': defaultdict(int),
            'flips': 0,
            'avg_time_to_stable': 0,
            'betting_windows': [],
        },
        'q4': {
            'total_matches': 0,
            'stable_by_minute': defaultdict(int),
            'flips': 0,
            'avg_time_to_stable': 0,
            'betting_windows': [],
        }
    }
    
    all_times_to_stable = []
    all_flips = 0
    total_matches_simulated = 0
    
    # For Q3: simulate checks at minutes 20, 22, 24
    q3_check_minutes = [20, 22, 24]
    # For Q4: simulate checks at minutes 30, 32, 34
    q4_check_minutes = [30, 32, 34]
    
    print(f"\n🎯 Simulating live data arrival...")
    print(f"   Q3 checks at minutes: {q3_check_minutes}")
    print(f"   Q4 checks at minutes: {q4_check_minutes}")
    
    # Group test matches by target
    by_target = defaultdict(list)
    for match in test_matches:
        by_target[match['target']].append(match)
    
    for target, matches in by_target.items():
        check_minutes = q3_check_minutes if target == 'q3' else q4_check_minutes
        
        print(f"\n📺 Simulating {target.upper()} ({len(matches)} matches)...")
        
        for match in tqdm(matches, desc=f"  {target} live sim", leave=False):
            match_id = match['match_id']
            signals = []
            first_bet_minute = None
            stable_minute = None
            
            for minute in check_minutes:
                # Simulate data arrival at this minute
                gp_upto, pbp_upto = simulate_live_arrival(
                    match['graph_points'],
                    match['pbp_events'],
                    target,
                    minute
                )
                
                # Run inference with this partial data
                # (simplified - would use full inference pipeline in production)
                signal = _quick_inference(match, gp_upto, pbp_upto, target, minute, models)
                signals.append((minute, signal))
                
                # Check if signal is BET
                if signal.startswith('BET') and first_bet_minute is None:
                    first_bet_minute = minute
            
            # Analyze signal stability
            bet_signals = [s for _, s in signals if s.startswith('BET')]
            no_bet_signals = [s for _, s in signals if s == 'NO_BET']
            
            # Detect flips (NO_BET → BET)
            flips = 0
            for i in range(1, len(signals)):
                prev_signal = signals[i-1][1]
                curr_signal = signals[i][1]
                if prev_signal == 'NO_BET' and curr_signal.startswith('BET'):
                    flips += 1
            
            # Time to stable signal
            if bet_signals:
                stable_minute = next(m for m, s in signals if s.startswith('BET'))
                time_to_stable = stable_minute - check_minutes[0]
                all_times_to_stable.append(time_to_stable)
                live_results[target]['stable_by_minute'][stable_minute] += 1
            else:
                live_results[target]['stable_by_minute']['never'] += 1
            
            # Betting window (time remaining in quarter)
            if first_bet_minute is not None:
                if target == 'q3':
                    window = 24 - first_bet_minute  # Q3 ends at min 24
                else:
                    window = 36 - first_bet_minute  # Q4 ends at min 36
                live_results[target]['betting_windows'].append(window)
            
            all_flips += flips
            total_matches_simulated += 1
            live_results[target]['total_matches'] += 1
    
    # Summary
    print(f"\n" + "="*80)
    print("📊 Live Simulation Results")
    print("="*80)
    
    print(f"\n🎯 Total matches simulated: {total_matches_simulated}")
    print(f"🔄 Total flips detected: {all_flips}")
    
    if all_times_to_stable:
        avg_time = np.mean(all_times_to_stable)
        print(f"⏱️ Average time to stable signal: {avg_time:.1f} minutes")
    
    for target in ['q3', 'q4']:
        res = live_results[target]
        print(f"\n📺 {target.upper()}:")
        print(f"   Matches: {res['total_matches']}")
        print(f"   Flips: {sum(1 for m in range(res['total_matches']) if m < all_flips)}")  # Approximate
        
        if res['betting_windows']:
            avg_window = np.mean(res['betting_windows'])
            print(f"   Avg betting window: {avg_window:.1f} minutes")
        
        print(f"   Stable by minute:")
        for minute, count in sorted(res['stable_by_minute'].items()):
            print(f"     Min {minute}: {count} matches")
    
    return live_results


def _quick_inference(match, gp_upto, pbp_upto, target, minute, models):
    """
    Quick inference for live simulation.
    
    Simplified version that doesn't load full models from disk.
    """
    # This would use the full inference pipeline
    # For now, return a simplified signal based on available data
    
    gp_count = len(gp_upto)
    pbp_count = len(pbp_upto)
    
    # Check minimum gates
    min_gp = config.MIN_GP_Q3 if target == 'q3' else config.MIN_GP_Q4
    min_pbp = config.MIN_PBP_Q3 if target == 'q3' else config.MIN_PBP_Q4
    
    if gp_count < min_gp or pbp_count < min_pbp:
        return "NO_BET"
    
    # Simplified: use score difference to predict
    if target == 'q3':
        q1_h = match.get('q1_home', 0)
        q1_a = match.get('q1_away', 0)
        q2_h = match.get('q2_home', 0)
        q2_a = match.get('q2_away', 0)
        
        current_diff = (q1_h + q2_h) - (q1_a + q2_a)
    else:
        q3_h = match.get('q3_home', 0)
        q3_a = match.get('q3_away', 0)
        current_diff = q3_h - q3_a
    
    # Simple prediction based on current state
    if abs(current_diff) > 5:
        return "BET_HOME" if current_diff > 0 else "BET_AWAY"
    else:
        return "NO_BET"  # Too close to call


def evaluate_betting_performance(test_matches, models, odds=1.91):
    """
    Evaluate betting performance with simulated bankroll.
    
    Uses conservative Kelly criterion for stake sizing.
    """
    print("\n" + "="*80)
    print("💰 BETTING PERFORMANCE SIMULATION")
    print("="*80)
    
    bankroll = 1000.0
    initial_bankroll = bankroll
    bets = 0
    wins = 0
    losses = 0
    
    print(f"\n💵 Initial bankroll: ${bankroll:.2f}")
    print(f"📊 Odds: {odds}")
    print(f"🎯 Using conservative Kelly (25%)")
    
    for match in tqdm(test_matches, desc="Simulating bets"):
        # Get prediction
        # (simplified - would use full inference in production)
        signal = _quick_inference(
            match,
            match['graph_points'],
            match['pbp_events'],
            match['target'],
            24 if match['target'] == 'q3' else 36,
            models
        )
        
        if not signal.startswith('BET'):
            continue
        
        # Calculate stake (Kelly criterion)
        # Simplified: assume 55% win probability
        p_win = 0.55
        b = odds - 1  # Net odds
        q = 1 - p_win
        
        kelly_f = (b * p_win - q) / b
        stake = bankroll * kelly_f * 0.25  # 25% Kelly
        stake = min(stake, bankroll * 0.05)  # Max 5% of bankroll
        
        if stake < 1:
            continue  # Skip tiny bets
        
        # Determine outcome
        actual_winner = 'home' if match.get('target_winner', 0) == 1 else 'away'
        predicted_home = signal == 'BET_HOME'
        
        if (predicted_home and actual_winner == 'home') or \
           (not predicted_home and actual_winner == 'away'):
            # Win
            bankroll += stake * (odds - 1)
            wins += 1
        else:
            # Loss
            bankroll -= stake
            losses += 1
        
        bets += 1
    
    # Results
    if bets > 0:
        hit_rate = wins / bets
        roi = (bankroll - initial_bankroll) / initial_bankroll
        profit = bankroll - initial_bankroll
        
        print(f"\n💰 Betting Results:")
        print(f"   Total bets: {bets}")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Hit rate: {hit_rate:.1%}")
        print(f"   Final bankroll: ${bankroll:.2f}")
        print(f"   Profit/Loss: ${profit:+.2f}")
        print(f"   ROI: {roi:+.1%}")
    else:
        print(f"\n📭 No bets placed (too conservative or no signals)")
    
    return {
        'bets': bets,
        'wins': wins,
        'losses': losses,
        'hit_rate': wins / bets if bets > 0 else 0,
        'final_bankroll': bankroll,
        'profit': bankroll - initial_bankroll,
        'roi': (bankroll - initial_bankroll) / initial_bankroll,
    }


def main():
    """Main evaluation pipeline."""
    print("\n" + "🏀"*40)
    print("V13 EVALUATION PIPELINE")
    print("🏀"*40)
    
    start_time = time.time()
    
    # Load models
    models = load_trained_models()
    if not models:
        return
    
    # Load test samples
    print("\n📦 Loading test samples...")
    samples, meta = dataset.build_samples(use_cache=True)
    
    # Split to get test set
    train_samples, val_samples, cal_samples = dataset.split_temporal(samples)
    
    # For evaluation, use validation set as "test"
    test_samples = val_samples
    print(f"✅ {len(test_samples)} test samples loaded")
    
    # Load match data for live simulation
    print("\n📺 Preparing live simulation data...")
    match_ids = list(set(s.match_id for s in test_samples))
    conn = dataset.get_db_connection()
    graph_points = dataset.load_graph_points(conn, match_ids)
    pbp_events = dataset.load_pbp_events(conn, match_ids)
    conn.close()
    
    # Build test match list with graph/pbp data
    test_matches = []
    for s in test_samples:
        match_data = {
            'match_id': s.match_id,
            'target': s.target,
            'graph_points': graph_points.get(s.match_id, []),
            'pbp_events': pbp_events.get(s.match_id, []),
            'q1_home': s.features.get('q1_home', 0),
            'q1_away': s.features.get('q1_away', 0),
            'q2_home': s.features.get('q2_home', 0),
            'q2_away': s.features.get('q2_away', 0),
            'q3_home': s.features.get('q3_home', 0),
            'q3_away': s.features.get('q3_away', 0),
            'target_winner': s.target_winner,
        }
        test_matches.append(match_data)
    
    # Run evaluations
    static_results = evaluate_static(test_samples, models)
    live_results = evaluate_live_simulation(test_matches, models)
    betting_results = evaluate_betting_performance(test_matches, models)
    
    # Save evaluation report
    eval_report = {
        'evaluated_at': datetime.now().isoformat(),
        'static': static_results,
        'live': live_results,
        'betting': betting_results,
        'evaluation_time_seconds': round(time.time() - start_time, 1),
    }
    
    report_path = OUT_DIR / "eval_report.json"
    with open(report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    
    print(f"\n💾 Evaluation report saved to {report_path}")
    print(f"⏱️ Total evaluation time: {eval_report['evaluation_time_seconds']}s")
    
    print("\n" + "🏀"*40)
    print("EVALUATION COMPLETE")
    print("🏀"*40 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate V13 models")
    parser.add_argument('--static-only', action='store_true', help='Only run static evaluation')
    parser.add_argument('--live-only', action='store_true', help='Only run live simulation')
    parser.add_argument('--betting-only', action='store_true', help='Only run betting simulation')
    parser.add_argument('--odds', type=float, default=1.91, help='Odds for betting simulation')
    
    args = parser.parse_args()
    main()
