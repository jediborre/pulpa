"""
walk_forward.py — Walk-forward validation for league statistics.

Computes league stats using ONLY historical data up to each match date,
preventing data leakage from future matches.
"""

import sqlite3
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime


def compute_league_stats_walkforward(conn, matches: List[Dict]) -> Dict:
    """
    Compute league statistics using walk-forward approach.
    
    For each match, league stats are computed using ONLY matches
    that occurred BEFORE this match's date.
    
    Returns dict of league stats that can be used as features.
    """
    cursor = conn.cursor()
    
    # Sort matches by date
    sorted_matches = sorted(matches, key=lambda m: m.get('date', ''))
    
    # Rolling league stats
    league_history = defaultdict(list)  # league -> list of past match results
    
    print("\n📊 Computing league stats with walk-forward validation...")
    
    for match in tqdm(sorted_matches, desc="Processing matches"):
        league = match.get('league', 'Unknown')
        date = match.get('date', '')
        
        # Get league stats from historical data only
        stats = _get_league_stats_from_history(league, league_history[league])
        
        # Store stats for this match
        match['league_stats'] = stats
        
        # Add this match to history (for future matches)
        # Only add if match is completed
        if 'q1_home' in match and match.get('q1_home') is not None:
            league_history[league].append({
                'date': date,
                'q1_home': match.get('q1_home', 0),
                'q1_away': match.get('q1_away', 0),
                'q2_home': match.get('q2_home', 0),
                'q2_away': match.get('q2_away', 0),
                'q3_home': match.get('q3_home', 0),
                'q3_away': match.get('q3_away', 0),
                'q4_home': match.get('q4_home', 0),
                'q4_away': match.get('q4_away', 0),
            })
    
    return dict(league_history)


def _get_league_stats_from_history(league: str, history: List[Dict]) -> Dict:
    """
    Compute league stats from historical matches.
    
    Returns dict with:
    - league_home_advantage: avg home - away point diff
    - league_avg_total_points: avg total points per match
    - league_std_total_points: std of total points
    - league_samples: number of historical matches
    """
    if not history:
        return {
            'league_home_advantage': 0.0,
            'league_avg_total_points': 80.0,  # Default
            'league_std_total_points': 10.0,
            'league_samples': 0,
        }
    
    total_points_list = []
    home_advantage_list = []
    
    for m in history:
        # Total points for Q1+Q2 (halftime)
        q1_h = m.get('q1_home', 0)
        q1_a = m.get('q1_away', 0)
        q2_h = m.get('q2_home', 0)
        q2_a = m.get('q2_away', 0)
        
        ht_total = q1_h + q1_a + q2_h + q2_a
        total_points_list.append(ht_total)
        
        # Home advantage
        home_pts = q1_h + q2_h
        away_pts = q1_a + q2_a
        home_advantage_list.append(home_pts - away_pts)
    
    import numpy as np
    
    return {
        'league_home_advantage': float(np.mean(home_advantage_list)),
        'league_avg_total_points': float(np.mean(total_points_list)),
        'league_std_total_points': float(np.std(total_points_list)),
        'league_samples': len(history),
    }


def compute_league_stats_static(conn) -> Dict:
    """
    Compute league stats using ALL data (FOR COMPARISON ONLY).
    
    This has data leakage and should NOT be used in training.
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            m.league,
            AVG(qs1.home + qs2.home - qs1.away - qs2.away) as home_advantage,
            AVG(qs1.home + qs1.away + qs2.home + qs2.away) as avg_total,
            COUNT(*) as samples
        FROM matches m
        JOIN quarter_scores qs1 ON m.match_id = qs1.match_id AND qs1.quarter = 'Q1'
        JOIN quarter_scores qs2 ON m.match_id = qs2.match_id AND qs2.quarter = 'Q2'
        GROUP BY m.league
    """)
    
    stats = {}
    for row in cursor.fetchall():
        stats[row['league']] = {
            'league_home_advantage': row['home_advantage'],
            'league_avg_total_points': row['avg_total'],
            'league_samples': row['samples'],
        }
    
    return stats
