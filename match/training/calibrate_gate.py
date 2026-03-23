"""Calibrate decision-gate thresholds from historical finished matches.

This script tunes gate parameters used in training/infer_match.py:
- min_graph_points
- min_pbp_events
- volatility_block_at
- min_edge

It optimizes thresholds independently for q3 and q4 (snapshot=36 behavior)
using a temporal split (train/validation) to reduce overfitting.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

infer_mod = importlib.import_module("training.infer_match")
db_mod = importlib.import_module("db")

DB_PATH = ROOT / "matches.db"
COMPARE_JSON = (
    ROOT / "training" / "model_comparison" / "version_comparison.json"
)
OUT_DIR = ROOT / "training" / "model_outputs_v2"
OUT_CONFIG = OUT_DIR / "gate_config.json"
OUT_REPORT = OUT_DIR / "gate_calibration_report.json"


@dataclass
class Record:
    target: str
    match_id: str
    dt: datetime
    signal: str
    confidence: float
    gp_count: int
    pbp_count: int
    volatility_index: float
    hit: int


def _parse_dt(date_str: str, time_str: str) -> datetime:
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")


def _snapshot_for_target(target: str) -> int | None:
    if target == "q3":
        return 24
    return 36


def _cutoff_for_target(target: str) -> int:
    return 24 if target == "q3" else 36


def _model_file(version: str, target: str, model_name: str) -> Path:
    model_dir = (
        infer_mod.MODEL_DIR_V2
        if version == "v2"
        else infer_mod.MODEL_DIR_V1
    )
    return model_dir / f"{target}_{model_name}.joblib"


def _load_model(version: str, target: str, model_name: str):
    path = _model_file(version, target, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _predict_prob(
    artifacts: dict[str, dict],
    model_name: str,
    features: dict,
) -> float:
    def single(name: str) -> float:
        artifact = artifacts[name]
        vec = artifact["vectorizer"]
        model = artifact["model"]
        x = vec.transform([features])
        return float(model.predict_proba(x)[0][1])

    if model_name == "ensemble_avg_prob":
        probs = [single("logreg"), single("rf"), single("gb")]
        return float(sum(probs) / len(probs))

    return single(model_name)


def _iter_candidate_rows(conn, limit: int | None):
    sql = (
        "SELECT match_id, date, time FROM matches "
        "WHERE date IS NOT NULL AND time IS NOT NULL "
        "ORDER BY date DESC, time DESC"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)

    for row in conn.execute(sql, params).fetchall():
        yield str(row["match_id"]), str(row["date"]), str(row["time"])


def _collect_records(
    metric: str,
    limit: int | None,
) -> dict[str, list[Record]]:
    if not COMPARE_JSON.exists():
        raise FileNotFoundError(
            "Comparison summary missing. "
            "Run: python training/model_cli.py compare"
        )

    with COMPARE_JSON.open("r", encoding="utf-8") as f:
        compare_blob = json.load(f)

    selections: dict[str, dict] = {}
    artifacts: dict[str, dict[str, dict]] = {}

    for target in ("q3", "q4"):
        entry = infer_mod._pick_entry(compare_blob, target, metric)
        if not entry:
            raise RuntimeError(
                f"No best model entry for target={target} metric={metric}"
            )
        version = str(entry.get("version", "v2"))
        model_name = str(entry.get("model", "logreg"))
        selections[target] = {
            "version": version,
            "model_name": model_name,
        }

        if model_name == "ensemble_avg_prob":
            artifacts[target] = {
                "logreg": _load_model(version, target, "logreg"),
                "rf": _load_model(version, target, "rf"),
                "gb": _load_model(version, target, "gb"),
            }
        else:
            artifacts[target] = {
                model_name: _load_model(version, target, model_name),
            }

    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)

    by_target: dict[str, list[Record]] = {"q3": [], "q4": []}

    for match_id, date_str, time_str in _iter_candidate_rows(conn, limit):
        data = db_mod.get_match(conn, match_id)
        if data is None:
            continue

        dt = _parse_dt(date_str, time_str)
        for target in ("q3", "q4"):
            ok, _reason = infer_mod._required_ok(data, target)
            if not ok:
                continue

            quarter = "Q3" if target == "q3" else "Q4"
            actual = infer_mod._actual_quarter_outcome(data, quarter)
            if actual in (None, "push"):
                continue

            version = selections[target]["version"]
            model_name = selections[target]["model_name"]
            features = infer_mod._build_features(conn, data, version, target)
            prob_home = _predict_prob(artifacts[target], model_name, features)
            predicted = "home" if prob_home >= 0.5 else "away"
            confidence = abs(prob_home - 0.5) * 2.0

            snapshot = _snapshot_for_target(target)
            signal = infer_mod._bet_signal(
                target,
                confidence,
                snapshot,
            )["signal"]

            cutoff = _cutoff_for_target(target)
            gp_count = len([
                p for p in data.get("graph_points", [])
                if int(p.get("minute", 0)) <= cutoff
            ])
            pbp_count = len(infer_mod._pbp_events_upto(data, cutoff))
            volatility = float(
                infer_mod._volatility_index(data, cutoff)["index"]
            )
            hit = int(predicted == actual)

            by_target[target].append(
                Record(
                    target=target,
                    match_id=match_id,
                    dt=dt,
                    signal=signal,
                    confidence=confidence,
                    gp_count=gp_count,
                    pbp_count=pbp_count,
                    volatility_index=volatility,
                    hit=hit,
                )
            )

    conn.close()

    for target in by_target:
        by_target[target].sort(key=lambda r: r.dt)

    return by_target


def _grid_for_target(target: str) -> dict[str, list[float | int]]:
    if target == "q3":
        return {
            "min_graph_points": [12, 14, 16, 18, 20, 22, 24],
            "min_pbp_events": [14, 16, 18, 20, 22, 24, 26],
            "volatility_block_at": [0.62, 0.66, 0.70, 0.72, 0.76, 0.80],
            "min_edge": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12],
        }

    return {
        "min_graph_points": [22, 24, 26, 28, 30, 32, 34],
        "min_pbp_events": [24, 26, 28, 30, 32, 34, 36],
        "volatility_block_at": [0.62, 0.66, 0.70, 0.72, 0.76, 0.80],
        "min_edge": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12],
    }


def _evaluate(records: list[Record], params: dict, odds: float) -> dict:
    total = len(records)
    if total == 0:
        return {
            "samples": 0,
            "bets": 0,
            "coverage": 0.0,
            "wins": 0,
            "losses": 0,
            "hit_rate": 0.0,
            "roi_units_per_bet": 0.0,
            "profit_units": 0.0,
        }

    bets = 0
    wins = 0
    losses = 0

    for r in records:
        if r.signal != "BET":
            continue
        if r.gp_count < params["min_graph_points"]:
            continue
        if r.pbp_count < params["min_pbp_events"]:
            continue
        if r.volatility_index >= params["volatility_block_at"]:
            continue
        if r.confidence < params["min_edge"]:
            continue

        bets += 1
        if r.hit:
            wins += 1
        else:
            losses += 1

    coverage = bets / total if total else 0.0
    hit_rate = wins / bets if bets else 0.0
    if bets:
        win_profit = max(odds - 1.0, 0.01)
        profit_units = (wins * win_profit) - losses
        roi = profit_units / bets
    else:
        profit_units = 0.0
        roi = 0.0

    return {
        "samples": total,
        "bets": bets,
        "coverage": round(coverage, 6),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(hit_rate, 6),
        "roi_units_per_bet": round(roi, 6),
        "profit_units": round(profit_units, 6),
    }


def _score(eval_metrics: dict, min_coverage: float) -> float:
    bets = int(eval_metrics["bets"])
    if bets <= 0:
        return -1e9

    coverage = float(eval_metrics["coverage"])
    roi = float(eval_metrics["roi_units_per_bet"])
    hit_rate = float(eval_metrics["hit_rate"])

    # Soft penalty if below required action volume.
    penalty = 0.0
    if coverage < min_coverage:
        penalty = (min_coverage - coverage) * 2.5

    return roi + (0.20 * hit_rate) - penalty


def _find_best_params(
    target: str,
    train_records: list[Record],
    odds: float,
    min_coverage: float,
) -> tuple[dict, dict]:
    grid = _grid_for_target(target)

    best_params = None
    best_metrics = None
    best_score = -math.inf

    for gp in grid["min_graph_points"]:
        for pbp in grid["min_pbp_events"]:
            for vol in grid["volatility_block_at"]:
                for edge in grid["min_edge"]:
                    params = {
                        "min_graph_points": int(gp),
                        "min_pbp_events": int(pbp),
                        "volatility_block_at": float(vol),
                        "min_edge": float(edge),
                    }
                    metrics = _evaluate(train_records, params, odds)
                    score = _score(metrics, min_coverage=min_coverage)
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError(f"Failed to calibrate target={target}")

    return best_params, best_metrics


def _split_temporal(records: list[Record], train_ratio: float = 0.8):
    if not records:
        return [], []
    cut = max(1, int(len(records) * train_ratio))
    if cut >= len(records):
        cut = len(records) - 1
    if cut < 1:
        cut = 1
    return records[:cut], records[cut:]


def _default_params_for_target(target: str) -> dict:
    if target == "q3":
        return {
            "min_graph_points": 18,
            "min_pbp_events": 20,
            "volatility_block_at": 0.72,
            "min_edge": 0.08,
        }

    return {
        "min_graph_points": 30,
        "min_pbp_events": 32,
        "volatility_block_at": 0.72,
        "min_edge": 0.08,
    }


def calibrate(
    metric: str,
    limit: int | None,
    odds: float,
    min_coverage: float,
) -> dict:
    by_target = _collect_records(metric=metric, limit=limit)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "odds": odds,
        "min_coverage": min_coverage,
        "targets": {},
    }

    thresholds = {
        "q3": {},
        "q4": {},
    }

    for target in ("q3", "q4"):
        records = by_target[target]
        train_records, val_records = _split_temporal(records, train_ratio=0.8)

        if len(records) < 50 or not val_records:
            raise RuntimeError(
                "Insufficient samples for calibration "
                f"target={target}: {len(records)}"
            )

        best_params, train_best = _find_best_params(
            target=target,
            train_records=train_records,
            odds=odds,
            min_coverage=min_coverage,
        )

        default_params = _default_params_for_target(target)

        val_best = _evaluate(val_records, best_params, odds)
        val_default = _evaluate(val_records, default_params, odds)
        full_best = _evaluate(records, best_params, odds)
        full_default = _evaluate(records, default_params, odds)

        key = "default" if target == "q3" else "36"
        thresholds[target][key] = best_params

        report["targets"][target] = {
            "samples_total": len(records),
            "samples_train": len(train_records),
            "samples_val": len(val_records),
            "default_params": default_params,
            "best_params": best_params,
            "train_best": train_best,
            "val_default": val_default,
            "val_best": val_best,
            "full_default": full_default,
            "full_best": full_best,
        }

    config = {
        "version": 1,
        "generated_at_utc": report["generated_at_utc"],
        "source": {
            "metric": metric,
            "odds": odds,
            "min_coverage": min_coverage,
            "limit": limit,
        },
        "thresholds": thresholds,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "config_path": str(OUT_CONFIG),
        "report_path": str(OUT_REPORT),
        "config": config,
        "report": report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate decision gate thresholds from historical data",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric used to select best model from version comparison",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Max recent matches to evaluate (default: 3000)",
    )
    parser.add_argument(
        "--odds",
        type=float,
        default=1.91,
        help="Assumed decimal odds for ROI proxy (default: 1.91)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.08,
        help="Minimum desired BET coverage for optimization (default: 0.08)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = calibrate(
        metric=args.metric,
        limit=args.limit,
        odds=args.odds,
        min_coverage=args.min_coverage,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("[calibrate-gate] done")
    print(f"config: {result['config_path']}")
    print(f"report: {result['report_path']}")
    for target in ("q3", "q4"):
        t = result["report"]["targets"][target]
        print(
            f"- {target}: samples={t['samples_total']} "
            f"val_default_roi={t['val_default']['roi_units_per_bet']} "
            f"val_best_roi={t['val_best']['roi_units_per_bet']} "
            f"best={t['best_params']}"
        )


if __name__ == "__main__":
    main()
