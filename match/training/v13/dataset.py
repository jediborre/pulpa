"""
dataset.py — Dataset construction with pace buckets and dynamic cutoffs for V13.

Handles:
- Loading matches from DB with quarter scores
- Pivot quarter scores to wide format
- Calculate pace buckets (low/medium/high)
- Split into train/val/cal sets temporally
- Generate samples for each (target, gender, pace) combination
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import json

from training.v13 import config

DB_PATH = Path(__file__).parents[2] / "matches.db"


@dataclass
class MatchData:
    """Complete match data with quarter scores."""
    match_id: str
    date: str
    league: str
    gender: str
    q1_home: int
    q1_away: int
    q2_home: int
    q2_away: int
    q3_home: int
    q3_away: int
    q4_home: int
    q4_away: int
    graph_points: List[Dict] = field(default_factory=list)
    pbp_events: List[Dict] = field(default_factory=list)


@dataclass
class TrainingSample:
    """A single training sample with features and targets."""
    match_id: str
    target: str  # 'q3' or 'q4'
    gender: str
    pace_bucket: str  # 'low', 'medium', 'high'
    snapshot_minute: int
    features: Dict[str, Any]
    target_winner: int | None  # 1=home, 0=away
    target_home_pts: int | None
    target_away_pts: int | None
    target_total_pts: int | None
    date: str
    league: str


def get_db_connection():
    """Create DB connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_quarter_scores(conn) -> Dict[str, Dict[str, int]]:
    """Load and pivot quarter scores to wide format."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT match_id, quarter, home, away
        FROM quarter_scores
        WHERE quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    """)
    
    pivot = {}
    for row in cursor.fetchall():
        mid = row['match_id']
        q = row['quarter'].lower()
        
        if mid not in pivot:
            pivot[mid] = {}
        
        pivot[mid][f'{q}_home'] = row['home']
        pivot[mid][f'{q}_away'] = row['away']
    
    return pivot


def load_matches_metadata(conn) -> Dict[str, Dict]:
    """Load match metadata (date, league, gender)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT match_id, date, league,
               CASE 
                   WHEN LOWER(league) LIKE '%women%' OR LOWER(league) LIKE '%femenino%' THEN 'women'
                   ELSE 'men'
               END as gender
        FROM matches
    """)
    
    return {
        row['match_id']: {
            'date': row['date'],
            'league': row['league'],
            'gender': row['gender']
        }
        for row in cursor.fetchall()
    }


def load_graph_points(conn, match_ids: List[str]) -> Dict[str, List[Dict]]:
    """Load graph points for given matches."""
    cursor = conn.cursor()
    
    # Load in batches to avoid memory issues
    graph_points = {}
    batch_size = 500
    
    for i in tqdm(range(0, len(match_ids), batch_size), desc="Loading graph points"):
        batch = match_ids[i:i+batch_size]
        placeholders = ','.join('?' for _ in batch)
        
        cursor.execute(f"""
            SELECT match_id, minute, value
            FROM graph_points
            WHERE match_id IN ({placeholders})
            ORDER BY match_id, minute
        """, batch)
        
        for row in cursor.fetchall():
            mid = row['match_id']
            if mid not in graph_points:
                graph_points[mid] = []
            graph_points[mid].append({
                'minute': row['minute'],
                'value': row['value'],  # This is the diff (home - away)
            })
    
    return graph_points


def load_pbp_events(conn, match_ids: List[str]) -> Dict[str, List[Dict]]:
    """Load play-by-play events for given matches."""
    cursor = conn.cursor()
    
    pbp_events = {}
    batch_size = 500
    
    for i in tqdm(range(0, len(match_ids), batch_size), desc="Loading PBP events"):
        batch = match_ids[i:i+batch_size]
        placeholders = ','.join('?' for _ in batch)
        
        cursor.execute(f"""
            SELECT match_id, quarter, seq, time, player, points, team, home_score, away_score
            FROM play_by_play
            WHERE match_id IN ({placeholders})
            ORDER BY match_id, quarter, seq
        """, batch)
        
        for row in cursor.fetchall():
            mid = row['match_id']
            if mid not in pbp_events:
                pbp_events[mid] = []
            pbp_events[mid].append({
                'quarter': row['quarter'],
                'seq': row['seq'],
                'time': row['time'],
                'player': row['player'],
                'points': row['points'],
                'team': row['team'],
                'home_score': row['home_score'],
                'away_score': row['away_score']
            })
    
    return pbp_events


def classify_pace_bucket(target: str, total_pts: int, pace_thresholds: Dict) -> str:
    """Classify match into pace bucket based on total points."""
    if target == 'q3':
        low_upper = pace_thresholds.get('q3_low_upper', config.PACE_Q3_LOW_UPPER)
        high_lower = pace_thresholds.get('q3_high_lower', config.PACE_Q3_HIGH_LOWER)
    else:  # q4
        low_upper = pace_thresholds.get('q4_low_upper', config.PACE_Q4_LOW_UPPER)
        high_lower = pace_thresholds.get('q4_high_lower', config.PACE_Q4_HIGH_LOWER)
    
    if total_pts <= low_upper:
        return 'low'
    elif total_pts >= high_lower:
        return 'high'
    else:
        return 'medium'


def calculate_pace_thresholds(samples: List[Dict]) -> Dict:
    """Calculate pace thresholds from actual data (percentiles 33/66)."""
    q3_totals = []
    q4_totals = []
    
    for s in samples:
        # Q3 pace = Q1+Q2 total
        q3_total = s['q1_home'] + s['q1_away'] + s['q2_home'] + s['q2_away']
        q3_totals.append(q3_total)
        
        # Q4 pace = Q1+Q2+Q3 total
        q4_total = q3_total + s['q3_home'] + s['q3_away']
        q4_totals.append(q4_total)
    
    thresholds = {
        'q3_low_upper': float(np.percentile(q3_totals, 33)),
        'q3_high_lower': float(np.percentile(q3_totals, 66)),
        'q4_low_upper': float(np.percentile(q4_totals, 33)),
        'q4_high_lower': float(np.percentile(q4_totals, 66)),
    }
    
    return thresholds


def build_samples(
    use_cache: bool = True,
    pace_thresholds: Optional[Dict] = None,
) -> Tuple[List[TrainingSample], Dict]:
    """
    Build training samples from DB.
    
    Args:
        use_cache: If True, try to load from cache file
        pace_thresholds: Pre-calculated thresholds (if None, calculate from data)
    
    Returns:
        (samples, metadata) tuple
    """
    cache_path = DB_PATH.parent / "training" / "v13" / "model_outputs" / "samples_cache.json"
    
    if use_cache and cache_path.exists():
        print(f"📦 Loading cached samples from {cache_path}")
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        samples = [TrainingSample(**s) for s in cache['samples']]
        metadata = cache['metadata']
        return samples, metadata
    
    print("🔨 Building samples from database...")
    conn = get_db_connection()
    
    try:
        # Step 1: Load quarter scores
        print("\n📊 Loading quarter scores...")
        quarter_scores = load_quarter_scores(conn)
        print(f"✅ Loaded {len(quarter_scores)} matches with quarter scores")
        
        # Step 2: Load metadata
        print("\n📋 Loading match metadata...")
        metadata_map = load_matches_metadata(conn)
        print(f"✅ Loaded {len(metadata_map)} match metadata entries")
        
        # Step 3: Filter matches with complete Q1+Q2+Q3
        print("\n🔍 Filtering complete matches...")
        complete_matches = []
        for mid, scores in quarter_scores.items():
            if all(f'q{i}_{side}' in scores for i in [1,2,3] for side in ['home', 'away']):
                meta = metadata_map.get(mid, {})
                complete_matches.append({
                    'match_id': mid,
                    'date': meta.get('date', ''),
                    'league': meta.get('league', ''),
                    'gender': meta.get('gender', 'men'),
                    **scores
                })
        
        print(f"✅ {len(complete_matches)} matches with Q1+Q2+Q3 complete")
        
        # Step 4: Calculate pace thresholds if not provided
        if pace_thresholds is None:
            print("\n📏 Calculating pace thresholds...")
            pace_thresholds = calculate_pace_thresholds(complete_matches)
            print(f"✅ Q3: low ≤{pace_thresholds['q3_low_upper']:.0f}, medium, high ≥{pace_thresholds['q3_high_lower']:.0f}")
            print(f"✅ Q4: low ≤{pace_thresholds['q4_low_upper']:.0f}, medium, high ≥{pace_thresholds['q4_high_lower']:.0f}")
        
        # Step 5: Build samples for Q3 and Q4
        samples = []
        
        for match in tqdm(complete_matches, desc="Building samples"):
            mid = match['match_id']
            date = match['date']
            league = match['league']
            gender = match['gender']
            
            # Calculate pace totals
            q1_h, q1_a = match['q1_home'], match['q1_away']
            q2_h, q2_a = match['q2_home'], match['q2_away']
            q3_h, q3_a = match['q3_home'], match['q3_away']
            
            q3_pace_total = q1_h + q1_a + q2_h + q2_a  # For Q3 bucket
            q4_pace_total = q3_pace_total + q3_h + q3_a  # For Q4 bucket
            
            # Q3 sample
            q3_pace = classify_pace_bucket('q3', q3_pace_total, pace_thresholds)
            q3_winner = 1 if q3_h > q3_a else 0
            q3_total = q3_h + q3_a
            
            # Halftime total (Q1+Q2) for Q4 prediction
            q1_h, q1_a, q2_h, q2_a = match['q1_home'], match['q1_away'], match['q2_home'], match['q2_away']
            halftime_total = q1_h + q1_a + q2_h + q2_a
            halftime_diff = (q1_h + q2_h) - (q1_a + q2_a)
            
            # Generate multiple snapshots for Q3 (dynamic cutoff)
            for snapshot_min in [18, 20, 21, 22, 23]:
                samples.append(TrainingSample(
                    match_id=mid,
                    target='q3',
                    gender=gender,
                    pace_bucket=q3_pace,
                    snapshot_minute=snapshot_min,
                    features={},  # Will be filled by features.py
                    target_winner=q3_winner,
                    target_home_pts=q3_h,
                    target_away_pts=q3_a,
                    target_total_pts=q3_total,
                    date=date,
                    league=league,
                ))
            
            # Q4 sample (only if Q4 is complete)
            if all(f'q{i}_{side}' in match for i in [4] for side in ['home', 'away']):
                q4_h, q4_a = match['q4_home'], match['q4_away']
                q4_pace = classify_pace_bucket('q4', q4_pace_total, pace_thresholds)
                q4_winner = 1 if q4_h > q4_a else 0
                q4_total = q4_h + q4_a
                
                # Generate multiple snapshots for Q4 (dynamic cutoff)
                for snapshot_min in [28, 29, 30, 31, 32]:
                    samples.append(TrainingSample(
                        match_id=mid,
                        target='q4',
                        gender=gender,
                        pace_bucket=q4_pace,
                        snapshot_minute=snapshot_min,
                        features={
                            # Store halftime info for Q4 features (NOT Q3 final!)
                            'halftime_total': halftime_total,
                            'halftime_diff': halftime_diff,
                        },
                        target_winner=q4_winner,
                        target_home_pts=q4_h,
                        target_away_pts=q4_a,
                        target_total_pts=q4_total,
                        date=date,
                        league=league,
                    ))
        
        print(f"\n✅ Total samples built: {len(samples)}")
        print(f"   Q3 samples: {sum(1 for s in samples if s.target == 'q3')}")
        print(f"   Q4 samples: {sum(1 for s in samples if s.target == 'q4')}")
        
        # Save cache
        metadata = {
            'pace_thresholds': pace_thresholds,
            'n_samples': len(samples),
            'n_matches': len(complete_matches),
        }
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({
                'samples': [vars(s) for s in samples],
                'metadata': metadata
            }, f)
        
        print(f"💾 Samples cached to {cache_path}")
        
        return samples, metadata
        
    finally:
        conn.close()


def split_temporal(
    samples: List[TrainingSample],
    train_end: str = '2026-01-31',
    val_end: str = '2026-03-31',
) -> Tuple[List[TrainingSample], List[TrainingSample], List[TrainingSample]]:
    """
    Split samples temporally into train/val/cal sets.
    
    Train: up to train_end
    Val: train_end to val_end
    Cal (calibration): val_end onwards
    """
    train = [s for s in samples if s.date <= train_end]
    val = [s for s in samples if train_end < s.date <= val_end]
    cal = [s for s in samples if s.date > val_end]
    
    return train, val, cal


def get_subset(
    samples: List[TrainingSample],
    target: str,
    gender: str,
    pace_bucket: str,
) -> List[TrainingSample]:
    """Get subset of samples for specific (target, gender, pace) combination."""
    return [
        s for s in samples
        if s.target == target and s.gender == gender and s.pace_bucket == pace_bucket
    ]


def iter_model_keys() -> List[Tuple[str, str, str]]:
    """Generate all (target, gender, pace) combinations."""
    keys = []
    for target in ['q3', 'q4']:
        for gender in ['men', 'women']:
            for pace in ['low', 'medium', 'high']:
                keys.append((target, gender, pace))
    return keys


if __name__ == "__main__":
    # Test sample building
    samples, metadata = build_samples(use_cache=False)
    print(f"\nBuilt {len(samples)} samples")
    print(f"Pace thresholds: {metadata['pace_thresholds']}")
