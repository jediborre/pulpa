"""Quick overfitting check for v2/v4 using temporal holdout IDs.

For each version/target:
- take holdout IDs from the tail (last n_test rows) of q*_dataset.csv
- run inference forced to that version
- compute accuracy_all and BET-only metrics
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

infer_mod = importlib.import_module("training.infer_match")

VERSIONS = {
    "v2": ROOT / "training" / "model_outputs_v2",
    "v4": ROOT / "training" / "model_outputs_v4",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _to_int(s: str | None, default: int = 0) -> int:
    try:
        return int(float(s or default))
    except ValueError:
        return default


def _n_test(version_dir: Path, target: str) -> int:
    rows = _read_csv(version_dir / f"{target}_metrics.csv")
    if not rows:
        return 0
    n = _to_int(rows[0].get("samples_test"), 0)
    return n


def _holdout_ids(version_dir: Path, target: str) -> list[str]:
    rows = _read_csv(version_dir / f"{target}_dataset.csv")
    n_test = _n_test(version_dir, target)
    if not rows or n_test <= 0:
        return []
    tail = rows[-n_test:]
    return [
        str(r.get("match_id", "")).strip()
        for r in tail
        if r.get("match_id")
    ]


def _eval(
    version: str,
    target: str,
    ids: list[str],
    odds: float = 1.91,
) -> dict:
    hits_all = 0
    misses_all = 0
    bets = 0
    hits_bet = 0
    losses_bet = 0

    for match_id in ids:
        res = infer_mod.run_inference(
            match_id=match_id,
            metric="f1",
            fetch_missing=False,
            force_version=version,
        )
        if not res.get("ok"):
            continue

        p = res.get("predictions", {}).get(target, {})
        if not p.get("available"):
            continue

        result = str(p.get("result", ""))
        if result in ("", "pending", "push"):
            continue

        if result == "hit":
            hits_all += 1
        elif result == "miss":
            misses_all += 1

        if p.get("final_recommendation") == "BET":
            bets += 1
            if result == "hit":
                hits_bet += 1
            elif result == "miss":
                losses_bet += 1

    samples = hits_all + misses_all
    hit_rate_bet = _safe_rate(hits_bet, bets)
    coverage = _safe_rate(bets, samples)
    acc_all = _safe_rate(hits_all, samples)
    if bets:
        profit_units = (hits_bet * (odds - 1.0)) - losses_bet
        roi_units_per_bet = profit_units / bets
    else:
        profit_units = 0.0
        roi_units_per_bet = 0.0

    return {
        "samples": samples,
        "accuracy_all": round(acc_all, 6),
        "bets": bets,
        "coverage": round(coverage, 6),
        "hit_rate_bet": round(hit_rate_bet, 6),
        "roi_units_per_bet": round(roi_units_per_bet, 6),
        "profit_units": round(profit_units, 6),
    }


def main() -> None:
    infer_mod.scraper_mod.fetch_event_snapshot = lambda _mid: None

    report = {"f1_holdout": {}}
    for version, vdir in VERSIONS.items():
        report["f1_holdout"][version] = {}
        for target in ("q3", "q4"):
            ids = _holdout_ids(vdir, target)
            report["f1_holdout"][version][target] = {
                "holdout_ids": len(ids),
                "metrics": _eval(version, target, ids),
            }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
