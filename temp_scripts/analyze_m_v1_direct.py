import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import report_m_v1_roi as rep  # noqa: E402
import train_q4_m27_v1 as m27_v1_train  # noqa: E402
import train_q4_m30_v1 as m30_v1_train  # noqa: E402


_, _, test_rows = rep._prepare_splits(force_rebuild=False, only_date=None)
teams_map = rep._load_match_teams_map()
match_ids = [sample.match_id for sample in test_rows]
q4_scores_map = rep._load_match_q4_scores_map(match_ids)
q3_min27_scores_map = rep._load_match_score_at_minute_map(match_ids, 27)
q3_min30_scores_map = rep._load_match_score_at_minute_map(
    match_ids,
    30,
    score_upto_fn=m30_v1_train._score_upto_m30,
)
q3_score_maps = {27: q3_min27_scores_map, 30: q3_min30_scores_map}

p_m27, excl27, reasons27, snap27 = rep._predict_m_v1_probs_for_snapshot_mode(
    test_rows,
    "m27_v1",
    rep.M27_V1_CHAMPION_PATH,
    m27_v1_train._build_m27_v1_features,
    int(m27_v1_train.SNAPSHOT_MINUTE),
    apply_filters=(not rep.LEAGUE_NAME_FILTERS_DISABLED),
)
p_m30, excl30, reasons30, snap30 = rep._predict_m_v1_probs_for_snapshot_mode(
    test_rows,
    "m30_v1",
    rep.M30_V1_CHAMPION_PATH,
    m30_v1_train._build_m30_v1_features,
    int(m30_v1_train.SNAPSHOT_MINUTE),
    apply_filters=(not rep.LEAGUE_NAME_FILTERS_DISABLED),
)

p_m27_raw = rep._predict_m_v1_probs_raw(
    test_rows,
    "m27_v1",
    rep.M27_V1_CHAMPION_PATH,
    m27_v1_train._build_m27_v1_features,
    int(m27_v1_train.SNAPSHOT_MINUTE),
)
p_m30_raw = rep._predict_m_v1_probs_raw(
    test_rows,
    "m30_v1",
    rep.M30_V1_CHAMPION_PATH,
    m30_v1_train._build_m30_v1_features,
    int(m30_v1_train.SNAPSHOT_MINUTE),
)

_, summary27 = rep._simulate(
    "m27_v1",
    test_rows,
    p_m27,
    int(m27_v1_train.SNAPSHOT_MINUTE),
    excluded_flags=excl27,
    excluded_reasons=reasons27,
    mode=rep.MODE,
    kelly_mult=rep.KELLY_MULT,
    kelly_cap=rep.KELLY_CAP,
    min_conf_prob=rep.MIN_CONF_PROB,
    stake_step=rep.STAKE_STEP,
    min_stake=rep.MIN_STAKE,
    max_stake=rep.MAX_STAKE,
    teams_map=teams_map,
    q4_scores_map=q4_scores_map,
    q3_score_maps=q3_score_maps,
    selected_snapshot_minutes=snap27,
)
_, summary30 = rep._simulate(
    "m30_v1",
    test_rows,
    p_m30,
    int(m30_v1_train.SNAPSHOT_MINUTE),
    excluded_flags=excl30,
    excluded_reasons=reasons30,
    mode=rep.MODE,
    kelly_mult=rep.KELLY_MULT,
    kelly_cap=rep.KELLY_CAP,
    min_conf_prob=rep.MIN_CONF_PROB,
    stake_step=rep.STAKE_STEP,
    min_stake=rep.MIN_STAKE,
    max_stake=rep.MAX_STAKE,
    teams_map=teams_map,
    q4_scores_map=q4_scores_map,
    q3_score_maps=q3_score_maps,
    selected_snapshot_minutes=snap30,
)

raw_summary = [
    rep._build_raw_comparison_summary("m27_v1_raw", test_rows, p_m27_raw),
    rep._build_raw_comparison_summary("m30_v1_raw", test_rows, p_m30_raw),
]
raw_compare = rep._build_raw_model_comparison(
    test_rows,
    p_m27_raw,
    p_m30_raw,
    teams_map,
    q4_scores_map,
    q3_min27_scores_map,
    q3_min30_scores_map,
)

raw_compare["hit_m27_raw"] = pd.to_numeric(raw_compare["hit_m27_raw"], errors="coerce")
raw_compare["hit_m30_raw"] = pd.to_numeric(raw_compare["hit_m30_raw"], errors="coerce")
raw_compare["delta_pick_strength_m30_m27"] = pd.to_numeric(
    raw_compare["delta_pick_strength_m30_m27"],
    errors="coerce",
)
only27 = raw_compare[
    (raw_compare["hit_m27_raw"] == 1) & (raw_compare["hit_m30_raw"] != 1)
]
only30 = raw_compare[
    (raw_compare["hit_m30_raw"] == 1) & (raw_compare["hit_m27_raw"] != 1)
]
both = raw_compare[
    (raw_compare["hit_m27_raw"] == 1) & (raw_compare["hit_m30_raw"] == 1)
]
both_wrong = raw_compare[
    (raw_compare["hit_m27_raw"] == 0) & (raw_compare["hit_m30_raw"] == 0)
]

result = {
    "raw_summary": raw_summary,
    "raw_compare_stats": {
        "rows": int(len(raw_compare)),
        "only_m27_correct": int(len(only27)),
        "only_m30_correct": int(len(only30)),
        "both_correct": int(len(both)),
        "both_wrong": int(len(both_wrong)),
        "agreement_rate": float(raw_compare["ambos_mismo_pick"].fillna(0).mean()),
        "avg_delta_pick_strength_m30_m27": float(
            raw_compare["delta_pick_strength_m30_m27"].dropna().mean()
        ),
        "avg_delta_when_only_m30_correct": float(
            only30["delta_pick_strength_m30_m27"].dropna().mean()
        ),
        "avg_delta_when_only_m27_correct": float(
            only27["delta_pick_strength_m30_m27"].dropna().mean()
        ),
        "top10_leagues_only_m27_correct": only27.groupby("liga")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
    },
    "filtered_summary": [summary27, summary30],
    "filtered_exclusion_counts": {
        "m27_v1": pd.Series(reasons27).fillna("passed").value_counts().head(20).to_dict(),
        "m30_v1": pd.Series(reasons30).fillna("passed").value_counts().head(20).to_dict(),
    },
}

print(json.dumps(result, ensure_ascii=False, indent=2))
