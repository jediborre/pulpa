"""
features.py — Feature engineering for V13.

Builds features from graph points and PBP events up to a dynamic cutoff minute.
"""

import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm


def build_features_for_sample(sample, graph_points: Dict, pbp_events: Dict) -> Dict[str, Any]:
    """
    Build feature dict for a single sample.
    
    Uses graph points and PBP up to snapshot_minute.
    """
    mid = sample.match_id
    target = sample.target
    cutoff = sample.snapshot_minute
    
    features = {}
    
    # Basic metadata features
    features['target'] = target
    features['snapshot_minute'] = cutoff
    
    # Graph features up to cutoff
    gp = graph_points.get(mid, [])
    gp_upto_cutoff = [p for p in gp if p['minute'] <= cutoff]
    
    features.update(_graph_features(gp_upto_cutoff, cutoff))
    
    # PBP features
    pbp = pbp_events.get(mid, [])
    pbp_upto = _pbp_upto_quarter(pbp, target)
    
    features.update(_pbp_features(pbp_upto))
    
    # Quarter score features
    features.update(_score_features(sample))
    
    return features


def _graph_features(gp: List[Dict], cutoff: int) -> Dict[str, Any]:
    """Extract features from graph points up to cutoff."""
    features = {}
    
    features['gp_count'] = len(gp)
    
    if not gp:
        features['gp_diff'] = 0
        features['gp_slope_3m'] = 0
        features['gp_slope_5m'] = 0
        features['gp_acceleration'] = 0
        features['gp_peak'] = 0
        features['gp_valley'] = 0
        features['gp_amplitude'] = 0
        features['gp_swings'] = 0
        return features
    
    # Current diff (value is already home - away)
    latest = gp[-1]
    features['gp_diff'] = latest['value']
    
    # Slope 3m
    if len(gp) >= 3:
        recent_3 = gp[-3:]
        features['gp_slope_3m'] = (recent_3[-1]['value'] - recent_3[0]['value']) / 3.0
    else:
        features['gp_slope_3m'] = 0
    
    # Slope 5m
    if len(gp) >= 5:
        recent_5 = gp[-5:]
        features['gp_slope_5m'] = (recent_5[-1]['value'] - recent_5[0]['value']) / 5.0
    else:
        features['gp_slope_5m'] = features['gp_slope_3m']
    
    # Acceleration
    if len(gp) >= 6:
        slope_1 = (gp[-3]['value'] - gp[-6]['value']) / 3.0 if gp[-6]['value'] != 0 else 0
        slope_2 = (gp[-1]['value'] - gp[-3]['value']) / 3.0 if gp[-3]['value'] != 0 else 0
        features['gp_acceleration'] = slope_2 - slope_1
    else:
        features['gp_acceleration'] = 0
    
    # Peak and valley (values are already diffs)
    values = [p['value'] for p in gp]
    features['gp_peak'] = max(values)
    features['gp_valley'] = min(values)
    features['gp_amplitude'] = features['gp_peak'] - features['gp_valley']
    
    # Swings (direction changes)
    swings = 0
    for i in range(2, len(values)):
        if (values[i] - values[i-1]) * (values[i-1] - values[i-2]) < 0:
            swings += 1
    features['gp_swings'] = swings
    
    return features


def _pbp_upto_quarter(pbp: List[Dict], target: str) -> List[Dict]:
    """Filter PBP events up to the target quarter."""
    if target == 'q3':
        return [e for e in pbp if e['quarter'] in ['Q1', 'Q2']]
    else:  # q4
        return [e for e in pbp if e['quarter'] in ['Q1', 'Q2', 'Q3']]


def _pbp_features(pbp: List[Dict]) -> Dict[str, Any]:
    """Extract features from PBP events."""
    features = {}
    
    features['pbp_count'] = len(pbp)
    
    if not pbp:
        features['pbp_pts_per_event'] = 0
        features['pbp_home_pts'] = 0
        features['pbp_away_pts'] = 0
        features['pbp_home_3pt'] = 0
        features['pbp_away_3pt'] = 0
        return features
    
    # Points per event
    total_pts = sum(e['points'] for e in pbp if e['points'] > 0)
    features['pbp_pts_per_event'] = total_pts / len(pbp) if pbp else 0
    
    # Home/away points
    home_pts = sum(e['points'] for e in pbp if e['team'] == 'home' and e['points'] > 0)
    away_pts = sum(e['points'] for e in pbp if e['team'] == 'away' and e['points'] > 0)
    
    features['pbp_home_pts'] = home_pts
    features['pbp_away_pts'] = away_pts
    
    # 3PT rate (approximation: points > 2)
    home_3pt = sum(1 for e in pbp if e['team'] == 'home' and e['points'] == 3)
    away_3pt = sum(1 for e in pbp if e['team'] == 'away' and e['points'] == 3)
    
    features['pbp_home_3pt'] = home_3pt
    features['pbp_away_3pt'] = away_3pt
    
    return features


def _score_features(sample) -> Dict[str, Any]:
    """Extract quarter score features.
    
    IMPORTANT: Only use data available UP TO the snapshot minute.
    
    For Q3 prediction (snapshot at min 18-23):
      - Q1 is complete → can use Q1 diff
      - Q2 is in progress → CANNOT use Q2 final diff
    
    For Q4 prediction (snapshot at min 28-32):
      - Q1, Q2 are complete → can use Q1+Q2 diff (halftime)
      - Q3 is in progress → CANNOT use Q3 final diff
      - Using Q3 final diff would leak the target (who won Q3 ≈ who wins Q4)
    """
    features = {}
    
    if sample.target == 'q3':
        # Q1 diff is available (Q1 complete before min 12)
        # But we don't have Q1-only scores in the sample currently
        # This would need to be loaded from DB separately
        features['q1_diff'] = 0  # Placeholder
        
    else:  # q4
        # Halftime diff (Q1+Q2) is available (halftime at min 24)
        # This is passed in sample.features during dataset building
        features['halftime_diff'] = sample.features.get('halftime_diff', 0)
        features['halftime_total'] = sample.features.get('halftime_total', 0)
        # DO NOT use q3_diff - it leaks the Q4 target!
    
    return features


def enrich_samples_with_features(samples, graph_points, pbp_events):
    """Add features to all samples."""
    print("\n🔧 Building features for samples...")
    
    for sample in tqdm(samples, desc="Enriching samples"):
        sample.features = build_features_for_sample(sample, graph_points, pbp_events)
    
    print(f"✅ Features built for {len(samples)} samples")
    return samples
