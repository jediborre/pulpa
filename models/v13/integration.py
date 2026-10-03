"""
integration.py — Integration helpers for V13 with telegram_bot and bet_monitor.

Provides functions to:
- Register V13 as available model
- Run inference from bet_monitor
- Format predictions for telegram
- Handle model loading and caching
"""

import sys
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "training" / "v13" / "model_outputs"


@dataclass
class V13InferenceResult:
    """Standardized inference result for V13."""
    ok: bool
    match_id: str
    target: str
    winner_pick: Optional[str] = None
    winner_confidence: Optional[float] = None
    winner_signal: Optional[str] = None
    predicted_total: Optional[float] = None
    predicted_home: Optional[float] = None
    predicted_away: Optional[float] = None
    mae: Optional[float] = None
    mae_home: Optional[float] = None
    mae_away: Optional[float] = None
    league_quality: Optional[str] = None
    league_bettable: Optional[bool] = None
    volatility_index: Optional[float] = None
    data_quality: Optional[str] = None
    reasoning: Optional[str] = None
    final_signal: Optional[str] = None
    error: Optional[str] = None


def check_v13_available() -> bool:
    """Check if V13 models are trained and available."""
    summary_path = OUT_DIR / "training_summary.json"
    if not summary_path.exists():
        return False
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    models = summary.get('models_trained', [])
    return len(models) > 0


def get_v13_model_info() -> Dict:
    """Get V13 model information."""
    summary_path = OUT_DIR / "training_summary.json"
    if not summary_path.exists():
        return {"available": False, "reason": "No training summary found"}
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    models = summary.get('models_trained', [])
    leakage = summary.get('leakage_detection', {})
    
    return {
        "available": True,
        "version": summary.get('version', 'v13'),
        "trained_at": summary.get('trained_at', 'unknown'),
        "models_trained": len(models),
        "leakage_assessment": leakage.get('assessment', 'unknown'),
        "dataset": {
            "total_matches": summary.get('dataset', {}).get('total_matches', 0),
            "total_samples": summary.get('dataset', {}).get('total_samples', 0),
        },
        "models": [
            {
                "key": m['key'],
                "f1": m.get('val_f1', 0),
                "samples": m.get('samples_train', 0),
                "gap": m.get('train_val_gap', 0),
            }
            for m in models
        ]
    }


def run_v13_inference(match_id: str, target: str) -> V13InferenceResult:
    """
    Run V13 inference for a match.
    
    This is the main function called by bet_monitor.
    Returns standardized result for telegram formatting.
    """
    try:
        # Import V13 inference module
        v13_mod = importlib.import_module("training.v13.infer_match_v13")
        
        result = v13_mod.run_inference(match_id, target)
        
        if not result.get('ok', False):
            return V13InferenceResult(
                ok=False,
                match_id=match_id,
                target=target,
                error=result.get('reason', 'Unknown error'),
            )
        
        pred = result['prediction']
        
        return V13InferenceResult(
            ok=True,
            match_id=match_id,
            target=target,
            winner_pick=getattr(pred, 'winner_pick', None),
            winner_confidence=getattr(pred, 'winner_confidence', None),
            winner_signal=getattr(pred, 'winner_signal', None),
            predicted_total=getattr(pred, 'predicted_total', None),
            predicted_home=getattr(pred, 'predicted_home', None),
            predicted_away=getattr(pred, 'predicted_away', None),
            mae=getattr(pred, 'mae', None),
            mae_home=getattr(pred, 'mae_home', None),
            mae_away=getattr(pred, 'mae_away', None),
            league_quality=getattr(pred, 'league_quality', None),
            league_bettable=getattr(pred, 'league_bettable', None),
            volatility_index=getattr(pred, 'volatility_index', None),
            data_quality=getattr(pred, 'data_quality', None),
            reasoning=getattr(pred, 'reasoning', None),
            final_signal=getattr(pred, 'final_signal', None),
        )
        
    except Exception as e:
        return V13InferenceResult(
            ok=False,
            match_id=match_id,
            target=target,
            error=str(e),
        )


def format_for_telegram(result: V13InferenceResult) -> Dict[str, Any]:
    """
    Format V13 inference result for telegram bot display.
    
    Returns dict compatible with telegram_bot._render_inference_debug.
    """
    if not result.ok:
        return {
            "ok": False,
            "reason": result.error,
        }
    
    return {
        "ok": True,
        "predictions": {
            result.target: {
                "available": True,
                "predicted_winner": result.winner_pick,
                "confidence": result.winner_confidence,
                "bet_signal": result.winner_signal,
                "final_recommendation": result.final_signal,
                "predicted_total": result.predicted_total,
                "predicted_home": result.predicted_home,
                "predicted_away": result.predicted_away,
                "reasoning": result.reasoning,
                "mae": result.mae,
                "mae_home": result.mae_home,
                "mae_away": result.mae_away,
                "league_quality": result.league_quality,
                "league_bettable": result.league_bettable,
                "volatility_index": result.volatility_index,
                "data_quality": result.data_quality,
            }
        },
    }


def get_v13_config() -> Dict[str, Any]:
    """Get V13 configuration for bet_monitor."""
    from training.v13 import config
    
    return {
        "q3_graph_cutoff": config.Q3_GRAPH_CUTOFF,
        "q4_graph_cutoff": config.Q4_GRAPH_CUTOFF,
        "min_gp_q3": config.MIN_GP_Q3,
        "min_gp_q4": config.MIN_GP_Q4,
        "min_pbp_q3": config.MIN_PBP_Q3,
        "min_pbp_q4": config.MIN_PBP_Q4,
        "min_confidence_q3": config.MIN_CONFIDENCE_Q3,
        "min_confidence_q4": config.MIN_CONFIDENCE_Q4,
        "max_volatility": config.MAX_VOLATILITY,
        "monitor_q3_minute": config.MONITOR_Q3_MINUTE,
        "monitor_q4_minute": config.MONITOR_Q4_MINUTE,
        "monitor_wake_before": config.MONITOR_WAKE_BEFORE,
        "monitor_confirm_ticks_q3": config.MONITOR_CONFIRM_TICKS_Q3,
        "monitor_confirm_ticks_q4": config.MONITOR_CONFIRM_TICKS_Q4,
    }


def register_v13_models(bot_config: Dict) -> Dict:
    """
    Register V13 as available model in bot config.
    
    Call this during bot initialization.
    """
    if check_v13_available():
        if "v13" not in bot_config.get('available_models', []):
            bot_config.setdefault('available_models', []).append("v13")
            bot_config.setdefault('model_config', {})
            bot_config['model_config'].setdefault('q3', 'v13')
            bot_config['model_config'].setdefault('q4', 'v13')
    
    return bot_config


def integrate_with_bet_monitor(monitor_config: Dict) -> Dict:
    """
    Integrate V13 with bet_monitor.
    
    Updates monitor config to use V13 thresholds.
    """
    v13_config = get_v13_config()
    
    # Update monitor thresholds to match V13
    monitor_config.update({
        'q3_minute': v13_config['monitor_q3_minute'],
        'q4_minute': v13_config['monitor_q4_minute'],
        'wake_before': v13_config['monitor_wake_before'],
        'confirm_ticks_q3': v13_config['monitor_confirm_ticks_q3'],
        'confirm_ticks_q4': v13_config['monitor_confirm_ticks_q4'],
        'min_gp_q3': v13_config['min_gp_q3'],
        'min_gp_q4': v13_config['min_gp_q4'],
        'min_pbp_q3': v13_config['min_pbp_q3'],
        'min_pbp_q4': v13_config['min_pbp_q4'],
    })
    
    # Register V13 as available
    if check_v13_available():
        monitor_config.setdefault('available_models', []).append("v13")
        monitor_config.setdefault('default_model', 'v13')
    
    return monitor_config
