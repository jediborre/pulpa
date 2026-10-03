"""
infer_match_v13.py — Live inference engine for V13.

Loads trained models and runs inference on a live match,
using dynamic cutoffs and pace-aware model selection.
"""

import sys
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v13 import config, features as feat_module
from training.v13.dataset import get_db_connection, classify_pace_bucket

MODEL_DIR = ROOT / "training" / "v13" / "model_outputs"

@dataclass
class V13Prediction:
    """Prediction result."""
    match_id: str
    quarter: str
    winner_pick: str
    winner_confidence: float
    winner_signal: str
    predicted_total: Optional[float]
    predicted_home: Optional[float]
    predicted_away: Optional[float]
    league_quality: str
    league_bettable: bool
    volatility_index: float
    data_quality: str
    final_signal: str
    reasoning: str
    model_quality: Optional[str] = None  # ✅ Good / ⚠️ Moderate / 🚨 Low samples
    model_samples: Optional[int] = None
    model_gap: Optional[float] = None
    model_f1: Optional[float] = None
    mae: Optional[float] = None
    mae_home: Optional[float] = None
    mae_away: Optional[float] = None
    fallback_used: bool = False


def run_inference(match_id: str, target: str, fetch_missing: bool = False) -> Dict:
    """
    Run V13 inference for a match.

    Args:
        match_id: Match ID
        target: 'q3' or 'q4'
        fetch_missing: If True, try to fetch missing data

    Returns:
        Prediction dict
    """
    try:
        # Step 1: Load match data
        match_data = _load_match_data(match_id)
        if not match_data:
            return {"ok": False, "reason": "Match data not found"}

        # Step 2: Determine pace bucket
        pace_bucket = _determine_pace_bucket(match_data, target)

        # Step 3: Get gender
        gender = match_data.get('gender', 'men')

        # Step 4: Load models
        model_key = f"{target}_{pace_bucket}_{gender}"
        clf_ensemble = _load_model(model_key, "clf")
        fallback_used = False
        if not clf_ensemble:
            # Fallback to medium bucket
            model_key = f"{target}_medium_{gender}"
            clf_ensemble = _load_model(model_key, "clf")
            fallback_used = True
            if not clf_ensemble:
                return {"ok": False, "reason": f"No models found for {model_key}"}

        # Step 4.5: Get model quality from training summary
        model_quality = _get_model_quality(model_key)

        # Step 5: Build features
        # build_features_for_sample expects a TrainingSample-like object and
        # graph_points/pbp_events as dicts keyed by match_id.
        # Here we adapt the live match_data dict to that interface.
        cutoff = config.Q3_GRAPH_CUTOFF if target == 'q3' else config.Q4_GRAPH_CUTOFF
        q1_home = int(match_data.get('q1_home') or 0)
        q1_away = int(match_data.get('q1_away') or 0)
        q2_home = int(match_data.get('q2_home') or 0)
        q2_away = int(match_data.get('q2_away') or 0)

        class _LiveSample:
            pass

        sample_adapter = _LiveSample()
        sample_adapter.match_id = match_id
        sample_adapter.target = target
        sample_adapter.snapshot_minute = cutoff
        sample_adapter.features = {
            'halftime_diff': (q1_home + q2_home) - (q1_away + q2_away),
            'halftime_total': q1_home + q1_away + q2_home + q2_away,
        }

        gp_list = match_data.get('graph_points', [])
        pbp_list = match_data.get('pbp_events', [])
        gp_dict = {match_id: gp_list if isinstance(gp_list, list) else []}
        pbp_dict = {match_id: pbp_list if isinstance(pbp_list, list) else []}

        features = feat_module.build_features_for_sample(sample_adapter, gp_dict, pbp_dict)

        # Step 6: Run prediction
        winner_pick, winner_confidence = _predict_winner(clf_ensemble, features)

        # Step 7: Run regression
        pred_home, pred_away, pred_total, mae, mae_home, mae_away = _predict_points(
            model_key, features
        )

        # Step 8: Apply gates with model quality context
        signal, reasoning = _apply_gates(
            winner_pick, winner_confidence, pred_total, match_data, target,
            model_quality=model_quality,
            pred_home=pred_home, pred_away=pred_away,
            mae_home=mae_home, mae_away=mae_away,
        )
        
        result = V13Prediction(
            match_id=match_id,
            quarter=target,
            winner_pick=winner_pick,
            winner_confidence=winner_confidence,
            winner_signal=signal,
            predicted_total=pred_total,
            predicted_home=pred_home,
            predicted_away=pred_away,
            league_quality=_assess_league_quality(match_data),
            league_bettable=True,
            volatility_index=0.0,
            data_quality="good",
            final_signal=signal,
            reasoning=reasoning,
            model_quality=model_quality.get('quality', 'unknown'),
            model_samples=model_quality.get('samples', 0),
            model_gap=model_quality.get('gap', 0),
            model_f1=model_quality.get('f1', 0),
            mae=mae,
            mae_home=mae_home,
            mae_away=mae_away,
            fallback_used=fallback_used,
        )
        
        return {"ok": True, "prediction": result}
        
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def _load_match_data(match_id: str) -> Optional[Dict]:
    """Load match data from DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get metadata
    cursor.execute("""
        SELECT match_id, date, league,
               CASE 
                   WHEN LOWER(league) LIKE '%women%' OR LOWER(league) LIKE '%femenino%' THEN 'women'
                   ELSE 'men'
               END as gender
        FROM matches
        WHERE match_id = ?
    """, (match_id,))
    
    meta = cursor.fetchone()
    if not meta:
        conn.close()
        return None
    
    match_data = {
        'match_id': meta['match_id'],
        'date': meta['date'],
        'league': meta['league'],
        'gender': meta['gender'],
    }
    
    # Get quarter scores
    cursor.execute("""
        SELECT quarter, home, away
        FROM quarter_scores
        WHERE match_id = ?
    """, (match_id,))
    
    for row in cursor.fetchall():
        q = row['quarter'].lower()
        match_data[f'{q}_home'] = row['home']
        match_data[f'{q}_away'] = row['away']
    
    # Get graph points
    cursor.execute("""
        SELECT minute, value
        FROM graph_points
        WHERE match_id = ?
        ORDER BY minute
    """, (match_id,))
    
    match_data['graph_points'] = [
        {'minute': r['minute'], 'value': r['value']}
        for r in cursor.fetchall()
    ]
    
    # Get PBP events
    cursor.execute("""
        SELECT quarter, seq, time, player, points, team, home_score, away_score
        FROM play_by_play
        WHERE match_id = ?
        ORDER BY quarter, seq
    """, (match_id,))
    
    match_data['pbp_events'] = [
        {
            'quarter': r['quarter'],
            'seq': r['seq'],
            'time': r['time'],
            'player': r['player'],
            'points': r['points'],
            'team': r['team'],
            'home_score': r['home_score'],
            'away_score': r['away_score']
        }
        for r in cursor.fetchall()
    ]
    
    conn.close()
    
    return match_data


def _determine_pace_bucket(match_data: Dict, target: str) -> str:
    """Determine pace bucket from match data."""
    # Load pace thresholds from training summary
    summary_path = MODEL_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        pace_thresholds = summary.get('dataset', {}).get('pace_thresholds', {})
    else:
        pace_thresholds = {}
    
    # Calculate pace total
    if target == 'q3':
        q1_h = match_data.get('q1_home', 0)
        q1_a = match_data.get('q1_away', 0)
        q2_h = match_data.get('q2_home', 0)
        q2_a = match_data.get('q2_away', 0)
        pace_total = q1_h + q1_a + q2_h + q2_a
    else:  # q4
        pace_total = sum(match_data.get(f'q{i}_{side}', 0) 
                        for i in [1,2,3] for side in ['home', 'away'])
    
    return classify_pace_bucket(target, pace_total, pace_thresholds)


def _load_model(model_key: str, model_type: str):
    """Load model from disk."""
    if model_type == "clf":
        path = MODEL_DIR / f"{model_key}_clf_ensemble.joblib"
    else:
        path = MODEL_DIR / f"{model_key}_reg_ensemble.joblib"
    
    if not path.exists():
        return None
    
    return joblib.load(path)


def _predict_winner(ensemble: Dict, features: Dict) -> tuple:
    """Predict winner using ensemble."""
    # Load vectorizer and scaler
    model_key = ensemble['model_key']
    vec_path = MODEL_DIR / f"{model_key}_vectorizer.joblib"
    scaler_path = MODEL_DIR / f"{model_key}_scaler.joblib"
    
    if not vec_path.exists() or not scaler_path.exists():
        return "uncertain", 0.0
    
    vec = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    
    # Transform features
    x = vec.transform([features])
    x = scaler.transform(x)
    
    # Weighted prediction
    proba_home = 0.0
    
    for model_info in ensemble['models']:
        model = model_info['model']
        weight = model_info['weight']
        
        pred_proba = model.predict_proba(x)[0]
        proba_home += pred_proba[1] * weight  # Class 1 = home
    
    winner_pick = "home" if proba_home > 0.5 else "away"
    confidence = abs(proba_home - 0.5) * 2  # Normalize to 0-1
    
    return winner_pick, round(confidence, 3)


def _predict_points(model_key: str, features: Dict) -> tuple:
    """Predict points using regressor ensembles."""
    pred_home, pred_away, pred_total = None, None, None
    mae = None
    
    # Load vectorizer and scaler
    vec_path = MODEL_DIR / f"{model_key}_vectorizer.joblib"
    scaler_path = MODEL_DIR / f"{model_key}_scaler.joblib"
    
    if not vec_path.exists() or not scaler_path.exists():
        return pred_home, pred_away, pred_total, mae
    
    vec = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    
    x = vec.transform([features])
    x = scaler.transform(x)
    
    # Predict home
    home_reg_path = MODEL_DIR / f"{model_key}_home_reg_ensemble.joblib"
    mae_home = None
    if home_reg_path.exists():
        home_reg = joblib.load(home_reg_path)
        pred_home = _predict_from_ensemble(home_reg, x)
        mae_home = sum(m['mae'] * w for m, w in zip(home_reg['models'], home_reg['weights']))

    # Predict away
    away_reg_path = MODEL_DIR / f"{model_key}_away_reg_ensemble.joblib"
    mae_away = None
    if away_reg_path.exists():
        away_reg = joblib.load(away_reg_path)
        pred_away = _predict_from_ensemble(away_reg, x)
        mae_away = sum(m['mae'] * w for m, w in zip(away_reg['models'], away_reg['weights']))

    # Predict total
    total_reg_path = MODEL_DIR / f"{model_key}_total_reg_ensemble.joblib"
    if total_reg_path.exists():
        total_reg = joblib.load(total_reg_path)
        pred_total = _predict_from_ensemble(total_reg, x)
        mae = sum(m['mae'] * w for m, w in zip(total_reg['models'], total_reg['weights']))

    return pred_home, pred_away, pred_total, mae, mae_home, mae_away


def _predict_from_ensemble(ensemble: Dict, x) -> float:
    """Predict from regressor ensemble."""
    prediction = 0.0
    
    for model_info, weight in zip(ensemble['models'], ensemble['weights']):
        model = model_info['model']
        prediction += model.predict(x)[0] * weight
    
    return round(prediction, 1)


def _get_model_quality(model_key: str) -> Dict:
    """Get model quality info from training summary."""
    summary_path = MODEL_DIR / "training_summary.json"
    if not summary_path.exists():
        return {'quality': 'unknown', 'samples': 0, 'gap': 0.0, 'f1': 0.0}

    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception:
        return {'quality': 'unknown', 'samples': 0, 'gap': 0.0, 'f1': 0.0}

    for m in summary.get('models_trained', []):
        if m.get('key') == model_key:
            samples = m.get('samples_train', 0)
            gap = m.get('train_val_gap', 0.0)
            f1 = m.get('val_f1', 0.0)
            if samples < 200:
                quality = 'low'
            elif samples < 500:
                quality = 'moderate'
            else:
                quality = 'good'
            return {'quality': quality, 'samples': samples, 'gap': gap, 'f1': f1}

    return {'quality': 'unknown', 'samples': 0, 'gap': 0.0, 'f1': 0.0}


_WOMEN_KEYWORDS = (
    "women", "femenin", "féminin", "feminino", "damen", "kobiet", "donna",
    "lbf", "wnba", "wcba", "liga femenina", "division femenina",
    "superettan dam", "1 liga kobiet", "kadetkinje",
)

# Leagues with confirmed negative ROI in April out-of-sample.
# Substrings are matched case-insensitively against the full league name.
# Tuple = (substring, confidence_boost)
# ROI < -40% → +0.20 boost;  ROI -10% to -40% → +0.12 boost
_PENALTY_LEAGUES: tuple[tuple[str, float], ...] = (
    # Very bad ROI (< -40%)
    ("pba commissioner",   0.20),   # PBA Commissioner's Cup  -76.7%
    ("rapid league",       0.20),   # Rapid League            -47.5%
    ("liga nationala",     0.20),   # Liga Nationala          -53.3%
    ("euroleague",         0.20),   # Euroleague              -44.0%
    ("eybl u17",           0.20),   # EYBL U17 Chal. Cup      -40.0%
    # NBA handled separately via volume gate (already ≥20 bets → Gate 3)
    # Moderate negative ROI (-10% to -40%)
    ("bnxt league",        0.12),   # BNXT League              -6.7% (updated)
    ("serie b, group",     0.12),   # Serie B Group A          -6.7%
    ("chile lnb",          0.12),   # Chile LNB                -6.7%
    ("eybl u20",           0.12),   # EYBL U20 Superfinal      -2.0%
    ("egbl u14",           0.12),   # EGBL U14 Superfinal      -2.0%
    ("the basketball league", 0.12),# The Basketball League   -16.0%
    ("1st division",       0.12),   # 1st Division            -16.0%
    ("hungary nb",         0.12),   # Hungary NB 1.A          -30.0%
)

# Leagues requiring ALL ultra-conservative checks to pass (Gate 4):
# confidence threshold already raised by Gate 3, plus:
#   - MAE must be low enough that predictions are non-overlapping
#   - model quality must be 'good' (≥500 training samples)
#   - model F1 must be above minimum
#   - train/val gap must be below overfitting threshold
_ULTRA_CONSERVATIVE_LEAGUES: tuple[str, ...] = (
    # NBA removed — handled by Direction Confirmation Gate (Gate 5) instead
    "ncaa men",
    "bnxt league",
    "hungary nb",
    "egbl u14",
    "iceland basketball premier league",
)

# Gate 4 thresholds (for other ultra-conservative leagues)
_UC_MAX_MAE         = 6.5   # max allowed MAE on home or away prediction
_UC_MIN_SPREAD      = 4.0   # predicted |home-away| must exceed this + MAE to be non-overlapping
_UC_MIN_F1          = 0.58  # minimum val_f1 for the model bucket
_UC_MAX_GAP         = 0.12  # maximum train/val gap (overfitting indicator)
_UC_MIN_SAMPLES     = 500   # model must have been trained on ≥500 samples

# Gate 5 — NBA Direction Confirmation
# Only back a team that the current game score also supports.
# Q3 bets: predicted winner can be AT MOST this many pts behind at halftime.
# Q4 bets: predicted winner must be LEADING (or tied) at end of Q3.
_NBA_LEAGUES = ("nba",)   # extend if needed
_NBA_MAX_DEFICIT_Q3 = -3  # halftime: pick can be at most 3 pts down (close game)
_NBA_MAX_DEFICIT_Q4 = 0   # Q3 end:   pick must not be trailing at all


def _is_nba_league(league: str) -> bool:
    lg = league.lower()
    return any(kw in lg for kw in _NBA_LEAGUES)


def _is_ultra_conservative_league(league: str) -> bool:
    lg = league.lower()
    for kw in _ULTRA_CONSERVATIVE_LEAGUES:
        if kw in lg:
            return True
    return False


def _penalty_boost(league: str) -> float:
    """Return the confidence boost for a known-problematic league, or 0.0."""
    lg = league.lower()
    for keyword, boost in _PENALTY_LEAGUES:
        if keyword in lg:
            return boost
    return 0.0


def _is_women_league(league: str) -> bool:
    """Return True if the league name indicates a women's competition."""
    lg = league.lower()
    for kw in _WOMEN_KEYWORDS:
        if kw in lg:
            return True
    return False


def _count_league_bets(league: str) -> int:
    """Count how many BET signals were stored in eval_match_results for this league
    (last 60 days, both q3 and q4 combined, tag bot_hybrid_f1)."""
    try:
        import sqlite3
        db_path = ROOT / "matches.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # Count rows where this league had a BET signal under the active tag
        result = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM eval_match_results e
            JOIN matches m ON m.match_id = e.match_id
            WHERE m.league = ?
              AND e.event_date >= DATE('now', '-60 days')
              AND (
                e.q3_signal__bot_hybrid_f1 IN ('BET','BET_HOME','BET_AWAY')
                OR e.q4_signal__bot_hybrid_f1 IN ('BET','BET_HOME','BET_AWAY')
              )
            """,
            (league,),
        ).fetchone()
        conn.close()
        return int(result["cnt"]) if result else 0
    except Exception:
        return 0


def _apply_gates(winner_pick, confidence, pred_total, match_data, target,
                 model_quality=None, pred_home=None, pred_away=None,
                 mae_home=None, mae_away=None) -> tuple:
    """Apply betting gates."""
    league = match_data.get("league", "")

    # ── Gate 1: Block women's leagues (overfitting, insufficient data) ────────
    if _is_women_league(league):
        return "NO_BET", "Liga femenina bloqueada (v13)"

    # ── Gate 2: Base confidence threshold ────────────────────────────────────
    min_conf = config.MIN_CONFIDENCE_Q3 if target == "q3" else config.MIN_CONFIDENCE_Q4

    # ── Gate 2b: NBA fixed threshold — 70% required (data shows only 70%+ has
    #    meaningful signal; lower bands sit at 25-27% hit rate vs 71.4% needed)
    if _is_nba_league(league):
        min_conf = max(min_conf, 0.70)

    # ── Gate 3a: Named problematic leagues — raise bar by known ROI penalty ──
    boost = _penalty_boost(league)
    if boost > 0:
        min_conf = min(min_conf + boost, 0.90)

    # ── Gate 3b: Extreme-volume leagues (≥20 prior bets) ─────────────────────
    # NBA is handled by Gate 2b above — skip volume boost to avoid double-raising
    league_bets = 0 if _is_nba_league(league) else _count_league_bets(league)
    if league_bets >= 20:
        # +0.05 for 20-49 bets, +0.08 for 50+ (accumulates with Gate 3a)
        extra = 0.08 if league_bets >= 50 else 0.05
        min_conf = min(min_conf + extra, 0.90)

    if confidence < min_conf:
        reasons = []
        if boost > 0:
            reasons.append(f"liga penalizada +{boost:.2f}")
        if league_bets >= 20:
            reasons.append(f"{league_bets} apuestas previas")
        suffix = f" ({', '.join(reasons)})" if reasons else ""
        return "NO_BET", f"Confidence {confidence:.3f} < {min_conf:.2f}{suffix}"

    if match_data.get("volatility", 0) > config.MAX_VOLATILITY:
        return "NO_BET", "Volatility too high"

    # ── Gate 5: NBA Direction Confirmation — game state must agree with pick ─
    if _is_nba_league(league):
        q1_h = int(match_data.get('q1_home') or 0)
        q1_a = int(match_data.get('q1_away') or 0)
        q2_h = int(match_data.get('q2_home') or 0)
        q2_a = int(match_data.get('q2_away') or 0)
        q3_h = int(match_data.get('q3_home') or 0)
        q3_a = int(match_data.get('q3_away') or 0)

        pick_lower = (winner_pick or "").lower()
        if target == "q4":
            # Cumulative lead through Q3
            cum_home = q1_h + q2_h + q3_h
            cum_away = q1_a + q2_a + q3_a
            # Lead from the PICK's perspective
            if pick_lower == "home":
                pick_lead = cum_home - cum_away
            else:
                pick_lead = cum_away - cum_home
            if pick_lead < _NBA_MAX_DEFICIT_Q4:
                return (
                    "NO_BET",
                    f"NBA Gate5: {winner_pick} trailing {-pick_lead} pts al inicio Q4"
                    f" (lim {_NBA_MAX_DEFICIT_Q4})",
                )
        else:  # q3
            # Halftime lead (Q1+Q2)
            ht_home = q1_h + q2_h
            ht_away = q1_a + q2_a
            if pick_lower == "home":
                pick_lead = ht_home - ht_away
            else:
                pick_lead = ht_away - ht_home
            if pick_lead < _NBA_MAX_DEFICIT_Q3:
                return (
                    "NO_BET",
                    f"NBA Gate5: {winner_pick} trailing {-pick_lead} pts al medio tiempo"
                    f" (lim {_NBA_MAX_DEFICIT_Q3})",
                )

    # ── Gate 4: Ultra-conservative multi-check for historically bad leagues ──
    if _is_ultra_conservative_league(league):
        mq = model_quality or {}
        uc_reasons = []

        # 4a: Model must have ≥500 training samples
        samples = mq.get("samples", 0)
        if samples < _UC_MIN_SAMPLES:
            uc_reasons.append(f"muestras insuficientes ({samples}<{_UC_MIN_SAMPLES})")

        # 4b: Model val_f1 must be above minimum
        f1 = mq.get("f1", 0.0)
        if f1 < _UC_MIN_F1:
            uc_reasons.append(f"F1 bajo ({f1:.3f}<{_UC_MIN_F1})")

        # 4c: Train/val gap must not indicate overfitting
        gap = mq.get("gap", 0.0)
        if gap > _UC_MAX_GAP:
            uc_reasons.append(f"overfitting gap ({gap:.3f}>{_UC_MAX_GAP})")

        # 4d: MAE must be small enough to trust the prediction direction
        if mae_home is not None and mae_home > _UC_MAX_MAE:
            uc_reasons.append(f"MAE_home alto ({mae_home:.1f}>{_UC_MAX_MAE})")
        if mae_away is not None and mae_away > _UC_MAX_MAE:
            uc_reasons.append(f"MAE_away alto ({mae_away:.1f}>{_UC_MAX_MAE})")

        # 4e: Predicted spread must clearly exceed MAE (non-overlapping intervals)
        if pred_home is not None and pred_away is not None:
            spread = abs(pred_home - pred_away)
            combined_mae = (mae_home or 0) + (mae_away or 0)
            if spread < _UC_MIN_SPREAD + combined_mae * 0.5:
                uc_reasons.append(
                    f"spread ambiguo ({spread:.1f} pts, MAE combinado {combined_mae:.1f})"
                )

        if uc_reasons:
            return "NO_BET", "Gate UC: " + " | ".join(uc_reasons)

    signal = f"BET_{winner_pick.upper()}"
    reasoning = f"Confidence: {confidence:.3f}"

    if boost > 0 or league_bets >= 20:
        reasoning += f" | umbral elevado ({min_conf:.2f}"
        if boost > 0:
            reasoning += f", penalización +{boost:.2f}"
        if league_bets >= 20:
            reasoning += f", {league_bets} apuestas históricas"
        reasoning += ")"

    # Add model quality note when samples are low (warning, but not blocking)
    if model_quality and model_quality.get("quality") == "low":
        n = model_quality.get("samples", 0)
        reasoning += f" | Low training samples: {n}"

    return signal, reasoning


def _assess_league_quality(match_data: Dict) -> str:
    """Assess league quality for betting."""
    # Simplified: would use league stats from walk-forward
    return "moderate"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python infer_match_v13.py <match_id>")
        sys.exit(1)
    
    match_id = sys.argv[1]
    result = run_inference(match_id, 'q3')
    
    if result['ok']:
        pred = result['prediction']
        print(f"\n✅ Prediction for {match_id}:")
        print(f"   Winner: {pred.winner_pick} ({pred.winner_confidence:.3f})")
        print(f"   Signal: {pred.winner_signal}")
        print(f"   Reasoning: {pred.reasoning}")
    else:
        print(f"❌ {result['reason']}")
