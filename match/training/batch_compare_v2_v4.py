"""Batch compare v2, v4 and hybrid inference performance on stored matches.

Computes per-target metrics for both versions and a delta table:
- samples
- bets
- coverage
- hit_rate (on BET recommendations)
- roi_units_per_bet (proxy from fixed decimal odds)
- accuracy_all (on all available non-push samples)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = importlib.import_module("db")
infer_mod = importlib.import_module("training.infer_match")

DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "model_comparison"
OUT_JSON = OUT_DIR / "batch_v2_v4_hybrid.json"

POLICIES = ("v2", "v4", "hybrid")


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _iter_match_ids(conn, limit: int | None) -> list[str]:
    sql = (
        "SELECT match_id FROM matches "
        "WHERE home_score IS NOT NULL AND away_score IS NOT NULL "
        "ORDER BY date DESC, time DESC"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()
    return [str(r[0]) for r in rows]


def _empty_stats() -> dict:
    return {
        "samples": 0,
        "bets": 0,
        "hits": 0,
        "losses": 0,
        "all_hits": 0,
        "all_misses": 0,
    }


def _finalize(stats: dict, odds: float) -> dict:
    samples = int(stats["samples"])
    bets = int(stats["bets"])
    hits = int(stats["hits"])
    losses = int(stats["losses"])
    all_hits = int(stats["all_hits"])
    all_misses = int(stats["all_misses"])

    coverage = _safe_rate(bets, samples)
    hit_rate = _safe_rate(hits, bets)
    accuracy_all = _safe_rate(all_hits, all_hits + all_misses)

    if bets:
        profit_units = (hits * (odds - 1.0)) - losses
        roi_units_per_bet = profit_units / bets
    else:
        profit_units = 0.0
        roi_units_per_bet = 0.0

    return {
        "samples": samples,
        "bets": bets,
        "coverage": round(coverage, 6),
        "hit_rate": round(hit_rate, 6),
        "roi_units_per_bet": round(roi_units_per_bet, 6),
        "profit_units": round(profit_units, 6),
        "accuracy_all": round(accuracy_all, 6),
    }


def _delta(a: dict, b: dict) -> dict:
    keys = [
        "coverage",
        "hit_rate",
        "roi_units_per_bet",
        "profit_units",
        "accuracy_all",
    ]
    out = {}
    for key in keys:
        out[key] = round(float(a.get(key, 0.0)) - float(b.get(key, 0.0)), 6)
    out["bets"] = int(a.get("bets", 0)) - int(b.get("bets", 0))
    out["samples"] = int(a.get("samples", 0)) - int(b.get("samples", 0))
    return out


def run_batch(metric: str, limit: int | None, odds: float) -> dict:
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    match_ids = _iter_match_ids(conn, limit)
    conn.close()

    # Avoid network calls during batch evaluation.
    infer_mod.scraper_mod.fetch_event_snapshot = lambda _mid: None

    raw_stats: dict[str, dict[str, dict]] = {
        p: {"q3": _empty_stats(), "q4": _empty_stats()}
        for p in POLICIES
    }

    for match_id in match_ids:
        for version in POLICIES:
            res = infer_mod.run_inference(
                match_id=match_id,
                metric=metric,
                fetch_missing=False,
                force_version=version,
            )
            if not res.get("ok"):
                continue

            for target in ("q3", "q4"):
                p = res.get("predictions", {}).get(target, {})
                if not p.get("available"):
                    continue

                result = str(p.get("result", ""))
                if result in ("", "pending", "push"):
                    continue

                s = raw_stats[version][target]
                s["samples"] += 1
                if result == "hit":
                    s["all_hits"] += 1
                elif result == "miss":
                    s["all_misses"] += 1

                if p.get("final_recommendation") == "BET":
                    s["bets"] += 1
                    if result == "hit":
                        s["hits"] += 1
                    elif result == "miss":
                        s["losses"] += 1

    report = {
        "metric": metric,
        "limit": limit,
        "odds": odds,
        "matches_seen": len(match_ids),
        "targets": {},
    }

    for target in ("q3", "q4"):
        v2f = _finalize(raw_stats["v2"][target], odds)
        v4f = _finalize(raw_stats["v4"][target], odds)
        hybf = _finalize(raw_stats["hybrid"][target], odds)
        report["targets"][target] = {
            "v2": v2f,
            "v4": v4f,
            "hybrid": hybf,
            "delta_v4_minus_v2": _delta(v4f, v2f),
            "delta_hybrid_minus_v2": _delta(hybf, v2f),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch compare v2 vs v4")
    p.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric for model selection context",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=800,
        help="Max recent matches to evaluate",
    )
    p.add_argument(
        "--odds",
        type=float,
        default=1.91,
        help="Fixed decimal odds for ROI proxy",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    report = run_batch(metric=args.metric, limit=args.limit, odds=args.odds)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("[batch-v2-v4-hybrid] done")
    print(f"matches_seen={report['matches_seen']}")
    for target in ("q3", "q4"):
        blob = report["targets"][target]
        print(f"[{target}] v2={blob['v2']}")
        print(f"[{target}] v4={blob['v4']}")
        print(f"[{target}] hybrid={blob['hybrid']}")
        print(f"[{target}] delta={blob['delta_v4_minus_v2']}")
        print(f"[{target}] delta_hybrid={blob['delta_hybrid_minus_v2']}")
    print(f"report={OUT_JSON}")


if __name__ == "__main__":
    main()
