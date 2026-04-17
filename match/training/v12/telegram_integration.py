
def _run_v12_inference(match_id: str, target: str = "q4") -> dict:
    """Run V12 inference via subprocess to avoid import issues."""
    import subprocess
    import json
    
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(V12_INFERENCE_SCRIPT), match_id, "--target", target],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # Parse JSON output
        # The script outputs to stdout, we need to capture it differently
        # Let's use the direct import method instead
        
        import importlib
        v12_infer = importlib.import_module("training.v12.infer_match_v12")
        v12_result = v12_infer.run_inference(
            match_id=match_id,
            target=target,
            fetch_missing=False,
        )
        
        if isinstance(v12_result, dict) and not v12_result.get("ok", True):
            return {
                "ok": False,
                "reason": v12_result.get("reason", "Unknown error"),
            }
        
        # Convert V12Prediction to dict
        return {
            "ok": True,
            "predictions": {
                target: {
                    "available": True,
                    "predicted_winner": v12_result.winner_pick,
                    "confidence": v12_result.winner_confidence,
                    "bet_signal": v12_result.winner_signal,
                    "final_recommendation": v12_result.final_signal,
                    "predicted_total": v12_result.predicted_total,
                    "predicted_home": v12_result.predicted_home,
                    "predicted_away": v12_result.predicted_away,
                    "over_under_signal": v12_result.over_under_signal,
                    "league_quality": v12_result.league_quality,
                    "league_bettable": v12_result.league_bettable,
                    "volatility_index": v12_result.volatility_index,
                    "data_quality": v12_result.data_quality,
                    "risk_level": v12_result.risk_level,
                    "reasoning": v12_result.reasoning,
                }
            }
        }
        
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"V12 inference failed: {exc}",
        }


def _run_v12_live_analysis(
    match_id: str,
    quarter: str,
    qtr_home_score: int,
    qtr_away_score: int,
    total_home_score: int,
    total_away_score: int,
    elapsed_minutes: float,
    graph_points: list,
    pbp_events: list,
) -> dict:
    """Run V12 LIVE virtual bookmaker analysis."""
    try:
        import importlib
        v12_live = importlib.import_module("training.v12.live_engine.virtual_bookmaker")
        
        analysis = v12_live.analizar_quarter_en_vivo(
            match_id=match_id,
            quarter=quarter,
            qtr_home_score=qtr_home_score,
            qtr_away_score=qtr_away_score,
            total_home_score=total_home_score,
            total_away_score=total_away_score,
            elapsed_minutes=elapsed_minutes,
            graph_points=graph_points,
            pbp_events=pbp_events,
        )
        
        if analysis is None:
            return {"ok": False, "reason": "No se pudo analizar"}
        
        # Convert to dict-like structure
        markets_text = []
        for m in analysis.markets:
            markets_text.append(
                f"{m.description}\n"
                f"  Prob: {m.our_probability:.1%} | Fair: {m.fair_odds:.2f}\n"
                f"  → Si tu casa ofrece > {m.fair_odds * 1.15:.2f} → VALUE"
            )
        
        return {
            "ok": True,
            "analysis": analysis,
            "markets_text": "\n\n".join(markets_text),
            "projections": {
                "home": analysis.projected_home_pts,
                "away": analysis.projected_away_pts,
                "total": analysis.projected_total_pts,
                "diff": analysis.projected_diff,
            },
            "momentum": analysis.graph_momentum,
            "recommendation": analysis.overall_recommendation,
            "best_market": analysis.best_market,
        }
        
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"V12 LIVE analysis failed: {exc}",
        }
