import json
from pathlib import Path

import pandas as pd

path = Path(
    r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\training\model_outputs_m_v1\Q4_ROI_match_by_match_m_v1_20260516_223437.xlsx"
)

xl = pd.ExcelFile(path)
out = {"sheets": xl.sheet_names}

raw_summary = pd.read_excel(path, sheet_name="raw_summary")
out["raw_summary"] = raw_summary.to_dict(orient="records")

raw = pd.read_excel(path, sheet_name="raw_compare")
raw["hit_m27_raw"] = pd.to_numeric(raw["hit_m27_raw"], errors="coerce")
raw["hit_m30_raw"] = pd.to_numeric(raw["hit_m30_raw"], errors="coerce")
raw["delta_pick_strength_m30_m27"] = pd.to_numeric(
    raw["delta_pick_strength_m30_m27"],
    errors="coerce",
)

only27 = raw[(raw["hit_m27_raw"] == 1) & (raw["hit_m30_raw"] != 1)]
only30 = raw[(raw["hit_m30_raw"] == 1) & (raw["hit_m27_raw"] != 1)]
both = raw[(raw["hit_m27_raw"] == 1) & (raw["hit_m30_raw"] == 1)]
both_wrong = raw[(raw["hit_m27_raw"] == 0) & (raw["hit_m30_raw"] == 0)]

out["raw_compare_stats"] = {
    "rows": int(len(raw)),
    "only_m27_correct": int(len(only27)),
    "only_m30_correct": int(len(only30)),
    "both_correct": int(len(both)),
    "both_wrong": int(len(both_wrong)),
    "agreement_rate": float(raw["ambos_mismo_pick"].fillna(0).mean()),
    "avg_delta_pick_strength_m30_m27": float(
        raw["delta_pick_strength_m30_m27"].dropna().mean()
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
}

summary = pd.read_excel(path, sheet_name="summary")
out["summary"] = summary.to_dict(orient="records")

print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
