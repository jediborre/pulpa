"""
Ubicación original: scratch/test_borrego.py
Propósito / Qué hacía:
Script de prueba personalizada para validación de pipelines de datos.
"""

import sqlite3
from tools.stats_cli import Prediction, FusionBorregoEngine, FusionConsensusEngine, get_actual_winner

def main():
    conn = sqlite3.connect('match/matches.db')
    conn.row_factory = sqlite3.Row
    
    matches = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league, s.event_date
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date = '2026-05-25'
    """).fetchall()
    
    engine = FusionBorregoEngine()
    consensus_engine = FusionConsensusEngine()
    
    print(f"{'PARTIDO':<60} | {'v6_2':<12} | {'m27_v3':<12} | {'Borrego':<12} | {'Ganador':<8}")
    print("-" * 115)
    
    for m in matches:
        mid = m["match_id"]
        
        # Get scores
        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()
        
        # Get logs
        logs = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,)).fetchall()
        
        deduped = {}
        for l in logs:
            deduped[l["model_version"]] = l
            
        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        
        v6_log = deduped.get("v6_2")
        if v6_log:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
                
        m27_log = deduped.get("m27_v3")
        if m27_log:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)
                
        # Determine actual winner
        actual_winner = get_actual_winner(qs, deduped)
                    
        borrego_pick = engine.evaluate(v6_pred, m27_pred)
        
        v6_str = f"{v6_pred.pick}({int(v6_pred.confidence)})"
        m27_str = f"{m27_pred.pick}({int(m27_pred.confidence)})"
        
        print(f"{m['home_team'] + ' vs ' + m['away_team']:<60} | {v6_str:<12} | {m27_str:<12} | {borrego_pick:<12} | {str(actual_winner):<8}")

if __name__ == '__main__':
    main()
