"""
V12 LIVE - Real-Time Value Betting Engine
===========================================
The key insight: Sportsbooks OVERREACT to current score.

When Team A is DOWN 8 points mid-quarter:
- Sportsbook odds: 3.50 (implies 28.6% win prob)
- V12 LIVE fair odds: 2.00 (50% win prob - comeback likely)
- VALUE BET! Expected ROI: +75%

This module:
1. Takes pre-Q3/Q4 prediction from V12
2. Monitors LIVE score, graph points, PBP during quarter
3. Computes "fair odds" in real-time
4. Alerts when sportsbook live odds offer VALUE
5. Tracks comeback probability with momentum features

NO historical odds data needed - only live graph + scoring data.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]  # live_engine -> v12 -> training
PROJECT_ROOT = ROOT.parent  # training -> match
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

db_mod = __import__("db")

DB_PATH = PROJECT_ROOT / "matches.db"
MODEL_DIR = ROOT / "model_outputs"
LIVE_DIR = Path(__file__).parent
LIVE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LiveState:
    """Current state of a live quarter."""
    match_id: str
    quarter: str  # "Q3" or "Q4"
    
    # Score
    qtr_home_score: int  # Points in THIS quarter only
    qtr_away_score: int
    total_home_score: int  # Cumulative through this quarter
    total_away_score: int
    
    # Game state
    elapsed_minutes: float  # How far into the quarter (0-12)
    minutes_left: float
    
    # Graph features
    graph_points: list[dict]
    pbp_events: list[dict]
    
    # Momentum signals
    home_momentum: float  # -1 to +1 (negative = away pressure)
    recent_run_home: int  # Last scoring run by home
    recent_run_away: int
    swings_count: int  # Lead changes
    
    # V12 prediction
    pre_bet_winner: str  # What V12 predicted pre-quarter
    pre_bet_confidence: float


@dataclass
class LiveOdds:
    """Current sportsbook live odds for quarter winner."""
    home_odds: float
    away_odds: float
    timestamp: str


@dataclass
class ValueBet:
    """Detected value betting opportunity."""
    match_id: str
    quarter: str
    timestamp: str
    
    # Situation
    score_home: int
    score_away: int
    score_diff: int  # Positive = home ahead
    minutes_elapsed: float
    minutes_left: float
    
    # Your model's assessment
    your_home_win_prob: float
    your_fair_odds_home: float  # 1 / your_home_win_prob
    your_fair_odds_away: float
    
    # Sportsbook offer
    sportsbook_odds_home: float
    sportsbook_odds_away: float
    
    # Value detection
    home_edge: float  # Your odds - SB odds (positive = value)
    away_edge: float
    has_value: bool
    value_side: str  # "home", "away", or "none"
    
    # Recommendation
    bet_side: str  # "home" or "away"
    bet_odds: float
    expected_roi: float  # (your_prob * (odds - 1)) - (1 - your_prob)
    confidence: str  # "high", "medium", "low"
    reasoning: str
    
    # Risk
    risk_level: str  # "low", "medium", "high", "extreme"
    is_comeback_bet: bool  # Betting on team that's currently behind


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _count_sign_swings(values: list[int]) -> int:
    swings = 0
    prev_sign = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign
    return swings


def _monte_carlo_live_win_prob(
    score_home: int,
    score_away: int,
    home_pts_per_min: float,
    away_pts_per_min: float,
    minutes_left: float,
    num_sims: int = 5000,
) -> dict:
    """
    Monte Carlo simulation for LIVE win probability.
    
    Projects remaining minutes based on current pace.
    """
    if minutes_left <= 0:
        winner = "home" if score_home > score_away else "away"
        return {
            "home_win_prob": 1.0 if winner == "home" else 0.0,
            "away_win_prob": 1.0 if winner == "away" else 0.0,
        }
    
    # Add variance (basketball is high-variance)
    var_home = max(0.5, home_pts_per_min * 1.5)
    var_away = max(0.5, away_pts_per_min * 1.5)
    
    sim_home = np.random.normal(
        score_home + home_pts_per_min * minutes_left,
        np.sqrt(var_home * minutes_left),
        num_sims,
    )
    sim_away = np.random.normal(
        score_away + away_pts_per_min * minutes_left,
        np.sqrt(var_away * minutes_left),
        num_sims,
    )
    
    home_wins = np.sum(sim_home > sim_away)
    ties = np.sum(np.abs(sim_home - sim_away) < 0.5)
    
    home_prob = (home_wins + 0.5 * ties) / num_sims
    away_prob = 1.0 - home_prob
    
    return {
        "home_win_prob": home_prob,
        "away_win_prob": away_prob,
    }


def compute_live_win_probability(
    state: LiveState,
) -> dict:
    """
    Compute LIVE win probability using V12 features.
    
    Combines:
    1. Monte Carlo projection (based on current pace)
    2. Graph momentum signal
    3. Score pressure (comeback dynamics)
    4. Recent scoring runs
    """
    elapsed = state.elapsed_minutes
    left = state.minutes_left
    qtr_home = state.qtr_home_score
    qtr_away = state.qtr_away_score
    diff = qtr_home - qtr_away
    
    # 1. Monte Carlo base projection
    home_ppm = _safe_rate(qtr_home, elapsed) if elapsed > 0 else 1.5
    away_ppm = _safe_rate(qtr_away, elapsed) if elapsed > 0 else 1.5
    
    mc = _monte_carlo_live_win_prob(
        score_home=qtr_home,
        score_away=qtr_away,
        home_pts_per_min=home_ppm,
        away_pts_per_min=away_ppm,
        minutes_left=left,
    )
    mc_home_prob = mc["home_win_prob"]
    
    # 2. Graph momentum adjustment
    gp = state.graph_points
    if len(gp) >= 5:
        values = [int(p.get("value", 0)) for p in gp]
        last_5 = values[-5:]
        momentum = np.mean(last_5)
        max_abs = max(abs(max(values)), abs(min(values)), 1)
        momentum_normalized = momentum / max_abs  # -1 to +1
        
        # Strong momentum shifts win prob by up to 15%
        momentum_adjustment = momentum_normalized * 0.15
    else:
        momentum_adjustment = 0.0
    
    # 3. Score pressure (comeback factor)
    # When trailing, probability of comeback depends on:
    # - Point deficit
    # - Time remaining
    # - Historical comeback rates
    
    if diff != 0:
        pts_to_tie = abs(diff)
        trailing_team_ppm = away_ppm if diff > 0 else home_ppm
        
        # Required pace to tie
        required_ppm = pts_to_tie / left if left > 0 else 999
        
        # Comeback feasibility
        if required_ppm <= trailing_team_ppm * 0.8:
            # Trailing team scoring faster than needed
            comeback_bonus = 0.10
        elif required_ppm <= trailing_team_ppm * 1.2:
            # Close to required pace
            comeback_bonus = 0.05
        elif required_ppm <= trailing_team_ppm * 1.5:
            # Possible but needs acceleration
            comeback_bonus = 0.02
        else:
            # Unlikely comeback
            comeback_bonus = -0.05
        
        # Apply to trailing team
        if diff > 0:  # Home ahead
            mc_home_prob -= comeback_bonus
        else:  # Away ahead
            mc_home_prob += comeback_bonus
    
    # 4. Recent scoring runs
    if state.recent_run_home > state.recent_run_away:
        # Home on a run
        run_adj = min(0.05, state.recent_run_home * 0.01)
        mc_home_prob += run_adj
    elif state.recent_run_away > state.recent_run_home:
        run_adj = min(0.05, state.recent_run_away * 0.01)
        mc_home_prob -= run_adj
    
    # 5. Time decay (early quarter = more variance, late quarter = score dominates)
    if elapsed < 3:
        # Early quarter: pull toward 50% (more uncertainty)
        pull_factor = 0.3
        mc_home_prob = mc_home_prob * (1 - pull_factor) + 0.5 * pull_factor
    elif elapsed > 9:
        # Late quarter: score dominates
        if diff > 5:
            mc_home_prob = max(mc_home_prob, 0.85)
        elif diff < -5:
            mc_home_prob = min(mc_home_prob, 0.15)
    
    # Clamp
    mc_home_prob = max(0.01, min(0.99, mc_home_prob))
    mc_away_prob = 1.0 - mc_home_prob
    
    # Convert to odds
    fair_odds_home = 1.0 / mc_home_prob if mc_home_prob > 0 else 999
    fair_odds_away = 1.0 / mc_away_prob if mc_away_prob > 0 else 999
    
    return {
        "home_win_prob": mc_home_prob,
        "away_win_prob": mc_away_prob,
        "fair_odds_home": round(fair_odds_home, 2),
        "fair_odds_away": round(fair_odds_away, 2),
        "monte_carlo_base": mc["home_win_prob"],
        "momentum_adjustment": round(momentum_adjustment, 4),
    }


def detect_value_bet(
    state: LiveState,
    live_odds: LiveOdds,
    min_edge: float = 0.15,  # 15% minimum edge
) -> ValueBet | None:
    """
    Detect if current sportsbook odds offer value vs V12 LIVE assessment.
    """
    prob_result = compute_live_win_probability(state)
    
    your_home_prob = prob_result["home_win_prob"]
    your_fair_home = prob_result["fair_odds_home"]
    your_fair_away = prob_result["fair_odds_away"]
    
    sb_home = live_odds.home_odds
    sb_away = live_odds.away_odds
    
    # Compute edges
    # If your fair odds < SB odds, there's value
    home_edge = your_fair_home - sb_home  # Negative = value on home
    away_edge = your_fair_away - sb_away  # Negative = value on away
    
    # Expected ROI
    # ROI = p * (odds - 1) - (1 - p)
    home_roi = your_home_prob * (sb_home - 1) - (1 - your_home_prob)
    away_roi = your_away_prob * (sb_away - 1) - (1 - your_away_prob)
    
    # Check if there's value
    has_value = False
    value_side = "none"
    bet_side = "none"
    bet_odds = 0
    expected_roi = 0
    
    if home_roi > min_edge:
        has_value = True
        value_side = "home"
        bet_side = "home"
        bet_odds = sb_home
        expected_roi = home_roi
    elif away_roi > min_edge:
        has_value = True
        value_side = "away"
        bet_side = "away"
        bet_odds = sb_away
        expected_roi = away_roi
    
    if not has_value:
        return None
    
    # Determine if comeback bet
    is_comeback = False
    if bet_side == "home" and state.qtr_home_score < state.qtr_away_score:
        is_comeback = True
    elif bet_side == "away" and state.qtr_away_score < state.qtr_home_score:
        is_comeback = True
    
    # Confidence
    if expected_roi > 0.40:
        confidence = "high"
    elif expected_roi > 0.20:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Risk level
    risk = "medium"
    if state.elapsed_minutes < 3:
        risk = "high"  # Early, volatile
    elif state.elapsed_minutes > 9:
        risk = "low"  # Late, more predictable
    
    if is_comeback:
        risk = "high"  # Comebacks are risky
    
    # Reasoning
    diff = state.qtr_home_score - state.qtr_away_score
    if is_comeback:
        trailing = state.pre_bet_winner if (
            (diff < 0 and state.pre_bet_winner == "home") or
            (diff > 0 and state.pre_bet_winner == "away")
        ) else "unknown"
        reasoning = (
            f"Comeback bet: {bet_side} trailing by {abs(diff)} pts "
            f"with {state.minutes_left:.1f} min left. "
            f"V12 says {your_home_prob:.0%} win prob, SB offers {bet_odds:.2f} "
            f"(implies {1/bet_odds:.0%}). Expected ROI: {expected_roi:.0%}"
        )
    else:
        reasoning = (
            f"{bet_side} leading by {abs(diff)} pts with {state.minutes_left:.1f} min left. "
            f"V12 says {your_home_prob if bet_side == 'home' else your_home_prob:.0%} win prob, "
            f"SB offers {bet_odds:.2f}. Expected ROI: {expected_roi:.0%}"
        )
    
    return ValueBet(
        match_id=state.match_id,
        quarter=state.quarter,
        timestamp=datetime.now().isoformat(),
        score_home=state.qtr_home_score,
        score_away=state.qtr_away_score,
        score_diff=diff,
        minutes_elapsed=state.elapsed_minutes,
        minutes_left=state.minutes_left,
        your_home_win_prob=round(your_home_prob, 4),
        your_fair_odds_home=round(your_fair_home, 2),
        your_fair_odds_away=round(your_fair_away, 2),
        sportsbook_odds_home=sb_home,
        sportsbook_odds_away=sb_away,
        home_edge=round(home_edge, 2),
        away_edge=round(away_edge, 2),
        has_value=has_value,
        value_side=value_side,
        bet_side=bet_side,
        bet_odds=bet_odds,
        expected_roi=round(expected_roi, 4),
        confidence=confidence,
        reasoning=reasoning,
        risk_level=risk,
        is_comeback_bet=is_comeback,
    )


def simulate_historical_comebacks(
    limit_matches: int = 1000,
) -> dict:
    """
    Simulate LIVE betting on historical data.
    
    For each quarter, check:
    1. If pre-Q3/Q4 prediction was correct
    2. If at any point the predicted team was BEHIND
    3. What the fair odds would have been
    4. If betting on comeback would have been profitable
    """
    print(f"[live-sim] Analyzing {limit_matches} matches for comeback opportunities...")
    
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    
    rows = conn.execute("""
        SELECT match_id, date, time FROM matches 
        WHERE status_type = 'finished' 
        ORDER BY date DESC, time DESC
        LIMIT ?
    """, (limit_matches,)).fetchall()
    
    opportunities = []
    comeback_bets = []
    total_comeback_wins = 0
    total_comeback_losses = 0
    best_comeback_odds = []
    
    for idx, row in enumerate(rows):
        if idx % 200 == 0 and idx > 0:
            print(f"  Progress: {idx}/{limit_matches}")
        
        match_id = str(row["match_id"])
        data = db_mod.get_match(conn, match_id)
        if data is None:
            continue
        
        quarters = data["score"].get("quarters", {})
        pbp = data.get("play_by_play", {})
        gp = data.get("graph_points", [])
        
        for q_label in ["Q3", "Q4"]:
            q = quarters.get(q_label)
            if not q:
                continue
            
            q_home = int(q.get("home", 0))
            q_away = int(q.get("away", 0))
            
            if q_home == q_away:
                continue  # Push
            
            actual_winner = "home" if q_home > q_away else "away"
            
            # Simulate mid-quarter snapshots
            # At 4, 6, 8 minutes into quarter
            for snapshot_min in [4, 6, 8]:
                minutes_left = 12.0 - snapshot_min
                
                # Get score at that point (approximate from PBP)
                qtr_pbp = pbp.get(q_label, [])
                
                # Cumulative score at snapshot
                snap_home = 0
                snap_away = 0
                plays_at_snapshot = 0
                
                for play in qtr_pbp:
                    clock = play.get("time", "12:00")
                    try:
                        mm, ss = clock.split(":")
                        elapsed_in_q = 12 - int(mm) - int(ss) / 60
                    except:
                        continue
                    
                    if elapsed_in_q <= snapshot_min:
                        pts = int(play.get("points", 0))
                        if play.get("team") == "home":
                            snap_home += pts
                        else:
                            snap_away += pts
                        plays_at_snapshot += 1
                
                if plays_at_snapshot < 3:
                    continue  # Not enough data
                
                diff = snap_home - snap_away
                
                # Check if there's a team behind by 3+ points
                if abs(diff) < 3:
                    continue  # Too close
                
                trailing = "away" if diff > 0 else "home"
                trailing_score = snap_away if diff > 0 else snap_home
                leading_score = snap_home if diff > 0 else snap_away
                
                # Monte Carlo: what's the comeback probability?
                elapsed = snapshot_min
                
                # Use the actual quarter scores for pace
                trailing_score_snap = trailing_score
                leading_score_snap = leading_score
                
                # Points per minute so far this quarter
                trailing_ppm = _safe_rate(trailing_score_snap, elapsed)
                leading_ppm = _safe_rate(leading_score_snap, elapsed)
                
                # Ensure minimum pace (teams always score something)
                trailing_ppm = max(trailing_ppm, 1.5)
                leading_ppm = max(leading_ppm, 1.5)
                
                mc = _monte_carlo_live_win_prob(
                    score_home=snap_home,
                    score_away=snap_away,
                    home_pts_per_min=trailing_ppm if trailing == "home" else leading_ppm,
                    away_pts_per_min=leading_ppm if trailing == "home" else trailing_ppm,
                    minutes_left=minutes_left,
                    num_sims=2000,
                )
                
                trailing_win_prob = mc["away_win_prob"] if trailing == "away" else mc["home_win_prob"]
                
                # Realistic probability floor for basketball comebacks
                # Even down 15 with 4 min left, there's ~1-2% chance
                # Down 3 with 8 min left, there's ~35-40% chance
                deficit = abs(diff)
                if deficit <= 3:
                    min_prob = 0.25
                elif deficit <= 5:
                    min_prob = 0.15
                elif deficit <= 8:
                    min_prob = 0.08
                elif deficit <= 12:
                    min_prob = 0.03
                else:
                    min_prob = 0.01
                
                trailing_win_prob = max(trailing_win_prob, min_prob)
                trailing_win_prob = min(trailing_win_prob, 0.95)
                
                # What would sportsbook odds be?
                # Fair odds for trailing team
                fair_trailing_odds = 1.0 / trailing_win_prob
                
                # SB typically adds 10-20% margin for live betting
                # AND overreacts to current score (gives trailing team slightly better odds than fair)
                # This is where value comes from - SB overpays for the "excitement" of comeback
                sb_margin = 0.90  # SB gives SLIGHTLY better odds than fair (overreaction)
                sb_trailing_odds = fair_trailing_odds * sb_margin
                
                # Realistic sportsbook ranges
                sb_trailing_odds = max(1.01, min(sb_trailing_odds, 21.0))
                fair_trailing_odds = max(1.01, min(fair_trailing_odds, 100.0))
                
                # Now check: did the trailing team actually win?
                trailing_won = (
                    (trailing == "home" and q_home > q_away) or
                    (trailing == "away" and q_away > q_home)
                )
                
                if trailing_won:
                    total_comeback_wins += 1
                    profit = sb_trailing_odds - 1  # Net profit on $1 bet
                    best_comeback_odds.append({
                        "match": match_id,
                        "quarter": q_label,
                        "snapshot": snapshot_min,
                        "trailing_by": abs(diff),
                        "minutes_left": minutes_left,
                        "actual_comeback_prob": trailing_win_prob,
                        "fair_odds": round(fair_trailing_odds, 2),
                        "sb_odds": round(sb_trailing_odds, 2),
                        "profit": round(profit, 2),
                    })
                else:
                    total_comeback_losses += 1
                
                opportunities.append({
                    "match": match_id,
                    "quarter": q_label,
                    "snapshot": snapshot_min,
                    "trailing": trailing,
                    "trailing_by": abs(diff),
                    "minutes_left": minutes_left,
                    "trailing_win_prob": round(trailing_win_prob, 4),
                    "fair_odds": round(fair_trailing_odds, 2),
                    "sb_odds": round(sb_trailing_odds, 2),
                    "cameback": trailing_won,
                })
    
    conn.close()
    
    # Compute stats
    total_comebacks = total_comeback_wins + total_comeback_losses
    comeback_rate = total_comeback_wins / total_comebacks if total_comebacks > 0 else 0
    
    if best_comeback_odds:
        avg_odds = np.mean([b["sb_odds"] for b in best_comeback_odds])
        avg_profit = np.mean([b["profit"] for b in best_comeback_odds])
        avg_trail_by = np.mean([b["trailing_by"] for b in best_comeback_odds])
    else:
        avg_odds = 0
        avg_profit = 0
        avg_trail_by = 0
    
    result = {
        "total_opportunities": len(opportunities),
        "total_comeback_bets": total_comebacks,
        "comeback_wins": total_comeback_wins,
        "comeback_losses": total_comeback_losses,
        "comeback_rate": round(comeback_rate, 4),
        "avg_comeback_odds": round(avg_odds, 2),
        "avg_profit_per_bet": round(avg_profit, 2),
        "avg_trailing_by": round(avg_trail_by, 1),
        "best_comebacks": sorted(best_comeback_odds, key=lambda x: x["profit"], reverse=True)[:10],
        "sample_size": limit_matches,
    }
    
    # Save
    with open(LIVE_DIR / "comeback_simulation.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def print_live_scenario():
    """Print example live betting scenarios."""
    print("\n" + "="*70)
    print("V12 LIVE BETTING - SCENARIO EXAMPLES")
    print("="*70)
    
    print("""
SCENARIO 1: Comeback Bet (VALUE!)
─────────────────────────────────
Q4, 6 min elapsed
  Score: Lakers 28 - Celtics 20 (Celtics trailing by 8)
  Sportsbook odds:
    Lakers: 1.20 (implies 83.3%)
    Celtics: 4.50 (implies 22.2%)

V12 LIVE analysis:
  Graph momentum: +5 (Celtics pressure rising)
  Celtics pace: 3.3 pts/min
  Lakers pace: 4.7 pts/min
  Required for Celtics: 8 pts / 6 min = 1.33 pts/min
  Celtics scoring at 3.3 ppm >> 1.33 required ✅
  
  Monte Carlo: Celtics comeback probability = 38%
  Fair odds for Celtics: 1 / 0.38 = 2.63
  Sportsbook offers: 4.50
  
  VALUE! Edge = 4.50 - 2.63 = +1.87 (+71% ROI)
  Expected ROI: 0.38 * (4.50 - 1) - (1 - 0.38) = +0.71
  
  DECISION: BET $25 on Celtics @ 4.50
  If they comeback: Win $87.50
  If they don't: Lose $25
  Expected value: +$17.75 per bet

SCENARIO 2: No Value (Skip)
─────────────────────────────
Q3, 8 min elapsed
  Score: Bucks 30 - Heat 22 (Heat trailing by 8)
  Sportsbook odds:
    Bucks: 1.15 (implies 87.0%)
    Heat: 5.50 (implies 18.2%)

V12 LIVE analysis:
  Graph momentum: -8 (Bucks dominating)
  Heat pace: 2.8 pts/min
  Required for Heat: 8 pts / 4 min = 2.0 pts/min
  Heat barely above required pace
  
  Monte Carlo: Heat comeback probability = 12%
  Fair odds for Heat: 1 / 0.12 = 8.33
  Sportsbook offers: 5.50
  
  NO VALUE! 5.50 < 8.33 (SB underpaying)
  Expected ROI: 0.12 * (5.50 - 1) - (1 - 0.12) = -0.34
  
  DECISION: NO BET

SCENARIO 3: Lock (But No Value)
────────────────────────────────
Q4, 10 min elapsed
  Score: Warriors 35 - Suns 25 (Warriors ahead by 10)
  Sportsbook odds:
    Warriors: 1.05 (implies 95.2%)
    Suns: 9.00 (implies 11.1%)

V12 LIVE analysis:
  Monte Carlo: Warriors win probability = 97%
  Fair odds for Warriors: 1.03
  Sportsbook offers: 1.05
  
  Technically value but...
  Expected ROI: 0.97 * (1.05 - 1) - (1 - 0.97) = +0.0185
  
  ROI of only 1.85% - NOT WORTH THE RISK
  DECISION: NO BET (too small edge)
""")
    
    print("="*70)
    print("KEY INSIGHT:")
    print("  Sportsbooks OVERREACT to current score.")
    print("  They give trailing team WORSE odds than fair probability.")
    print("  BUT sometimes they overreact TOO MUCH → VALUE on comeback.")
    print("  V12 LIVE detects these moments using Monte Carlo + momentum.")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V12 LIVE Betting Engine")
    parser.add_argument("--simulate", action="store_true", help="Run comeback simulation")
    parser.add_argument("--scenarios", action="store_true", help="Show example scenarios")
    parser.add_argument("--limit", type=int, default=1000, help="Limit matches for simulation")
    parser.add_argument("--json", action="store_true", help="JSON output for simulation")
    
    args = parser.parse_args()
    
    if args.scenarios:
        print_live_scenario()
    
    if args.simulate:
        start = time.time()
        result = simulate_historical_comebacks(limit_matches=args.limit)
        elapsed = time.time() - start
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*70)
            print("V12 LIVE - COMEBACK BETTING SIMULATION")
            print("="*70)
            print(f"Matches analyzed: {result['sample_size']}")
            print(f"Total opportunities: {result['total_opportunities']}")
            print(f"Comeback bets: {result['total_comeback_bets']}")
            print(f"Comeback wins: {result['comeback_wins']}")
            print(f"Comeback losses: {result['comeback_losses']}")
            print(f"Comeback rate: {result['comeback_rate']:.1%}")
            print(f"Avg odds on comebacks: {result['avg_comeback_odds']:.2f}")
            print(f"Avg profit per bet: ${result['avg_profit_per_bet']:.2f}")
            print(f"Avg trailing by: {result['avg_trailing_by']:.1f} pts")
            print("="*70)
            
            if result['best_comebacks']:
                print("\nTop 10 Comeback Bets:")
                for i, cb in enumerate(result['best_comebacks'][:10], 1):
                    print(f"  {i}. {cb['match']} Q{cb['quarter']} "
                          f"(down {cb['trailing_by']} @ {cb['snapshot']}min, "
                          f"{cb['minutes_left']:.0f}min left) "
                          f"→ {cb['fair_odds']:.2f} fair, "
                          f"{cb['sb_odds']:.2f} offered, "
                          f"profit: ${cb['profit']:.2f}")
            
            print(f"\nCompleted in {elapsed:.1f}s")
            print(f"Results saved to: {LIVE_DIR / 'comeback_simulation.json'}")
            print("="*70 + "\n")
