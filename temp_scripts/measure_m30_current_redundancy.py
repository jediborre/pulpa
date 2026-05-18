"""Measure redundancy among current-state features in M30_V1.

Checks how much signal is unique vs duplicated among:
- current_leader=*
- current_trailing_side=*
- current_leader_won_q1, current_leader_won_q3_partial
- current_trailer_won_q3_partial, current_trailer_was_halftime_leader
- current_leader_was_halftime_trailer, trailing_now_deficit_abs
- score_est_diff (reference)

Also measures how many halftime_transition flags are redundant with halftime_diff.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import train_q4_m30_v1 as m30


def _corr(values, targets):
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(values) < 2 or np.std(values) == 0 or np.std(targets) == 0:
        return None
    corr = np.corrcoef(values, targets)[0, 1]
    if math.isnan(corr):
        return None
    return float(corr)


print("[redundancy] Loading data...")
samples, preloaded = m30._build_base_samples_and_data(m30.DB_PATH)
dynamic_rows = m30._build_dynamic_samples(samples, preloaded=preloaded)
_, _, test_rows, _ = m30._split_rows_temporal_by_match(dynamic_rows)

df = pd.DataFrame([row["features"] | {"target": int(row["target"])} for row in test_rows])
y = df["target"].astype(int)

# --- Current-state features to check ---
current_state_vars = [
    "current_leader", "current_trailing_side",
    "current_leader_won_q1", "current_leader_won_q3_partial",
    "current_trailer_won_q3_partial", "current_trailer_was_halftime_leader",
    "current_leader_was_halftime_trailer", "trailing_now_deficit_abs",
]

halftime_transition_vars = [
    "halftime_trailer_cutting_in_q3", "halftime_trailer_won_q3_partial",
    "halftime_trailer_now_leads", "halftime_trailer_now_tied",
    "halftime_trailer_neutralized_deficit", "halftime_leader_lost_lead",
    "halftime_trailer_gain", "halftime_to_current_margin_delta",
    "abs_margin_delta_from_halftime", "halftime_trailer_strong_recovery",
    "big_halftime_lead_now_close", "lead_flip_from_halftime",
]

reference_vars = ["score_est_diff", "halftime_diff", "prior_wr_diff", "q2_diff"]

print("\n=== CURRENT-STATE FEATURES: signal vs reference ===")
# For categorical vars, one-hot encode; for numeric, use directly
all_expanded = {}

for col in df.columns:
    if col == "target":
        continue
    series = df[col]
    if series.dtype == object or series.dtype.name == "category":
        dummies = pd.get_dummies(series, prefix=col)
        for dc in dummies.columns:
            all_expanded[dc] = dummies[dc].astype(float)
    else:
        all_expanded[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

expanded_df = pd.DataFrame(all_expanded)

# Reference signals
ref_signal = {}
for ref in reference_vars:
    ref_val = _corr(expanded_df.get(ref, pd.Series([0.0])), y)
    ref_signal[ref] = ref_val
    print(f"  {ref}: abs_corr = {abs(ref_val) if ref_val else 0:.4f} vs target")

print("\n--- Current-state one-hot features vs target ---")
current_state_onehots = [c for c in expanded_df.columns if any(c.startswith(p) for p in [
    "current_leader=", "current_trailing_side=",
    "current_leader_won_q", "current_trailer_won_q",
    "current_trailer_was", "current_leader_was",
    "trailing_now_deficit_abs", "trailing_now_is_",
])]
for col in sorted(current_state_onehots):
    s = expanded_df[col]
    c = _corr(s, y)
    if c is not None:
        print(f"  {col:50s} abs_corr={abs(c):.4f}  rate={s.mean():.3f}")

print("\n--- Current-state vs score_est_diff (redundancy) ---")
score_diff = expanded_df.get("score_est_diff", pd.Series([0.0]))
for col in sorted(current_state_onehots):
    s = expanded_df[col]
    c = _corr(s, score_diff)
    if c is not None:
        print(f"  {col:50s} corr_with_score_diff={c:+.4f}")

print("\n--- Halftime-transition features vs halftime_diff (redundancy) ---")
halftime_diff = expanded_df.get("halftime_diff", pd.Series([0.0]))

for col in sorted(expanded_df.columns):
    is_ht = any(col.startswith(p) for p in [
        "halftime_trailer_", "halftime_leader_", "halftime_to_",
        "abs_margin_delta", "lead_flip_", "big_halftime_",
    ])
    if not is_ht:
        continue
    s = expanded_df[col]
    c = _corr(s, y)
    r = _corr(s, halftime_diff)
    if c is not None:
        print(f"  {col:50s} target_corr={abs(c):+.4f}  halftime_diff_corr={r:+.4f}  ")

print("\n=== REDUNDANCY SUMMARY ===")
# Check: current_leader=home + current_leader=away 1-hot vs score_est_diff
for leader_val in ["home", "away", "tied"]:
    col = f"current_leader={leader_val}"
    if col in expanded_df.columns:
        s = expanded_df[col]
        r = _corr(s, score_diff)
        print(f"  {col}: corr with score_est_diff = {r:+.4f}")

# How many of the current-state features have abs_corr < 0.02?
weak_state = []
for col in current_state_onehots:
    s = expanded_df[col]
    c = _corr(s, y)
    if c is not None and abs(c) < 0.02:
        weak_state.append((col, abs(c)))
print(f"\n  Current-state features with abs_corr < 0.02: {len(weak_state)}")
for name, val in sorted(weak_state, key=lambda x: x[1]):
    print(f"    {name}: {val:.4f}")
