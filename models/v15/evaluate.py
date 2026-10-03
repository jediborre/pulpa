"""
evaluate.py - Backtest y reportes de V15.

Genera:
- eval_report_v15.json   : metricas por (liga, target), ROI, curvas de threshold
- eval_predictions_v15.csv: todas las predicciones del holdout con contexto
- threshold_curves/<liga>_<target>.csv: para graficar tuning por liga

Importante:
- Se evalua SOLO sobre el holdout (el split mas reciente, nunca visto).
- Se usa inferencia real (con calibracion + gates), NO simplificaciones.
- Soporta snapshot unico (22 / 31) o barrido de multiples snapshots.

CLI: python -m training.v15.evaluate --odds 1.40 --out training/v15/reports/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from training.v15 import config, dataset as ds
from training.v15.inference import V15Engine, MODEL_DIR, SUMMARY_PATH


REPORTS_DIR = Path(__file__).parent / "reports"


# ============================================================================
# Backtest
# ============================================================================

def run_backtest(
    use_cache: bool = True,
    odds: float = config.DEFAULT_ODDS,
    use_holdout: bool = True,
    snapshot_only_prod: bool = True,
) -> dict[str, Any]:
    """
    Corre inferencia sobre el split de holdout (o todo si use_holdout=False).

    snapshot_only_prod: si True usa solo los cutoffs de produccion
      (Q3=22, Q4=31) — el escenario live real.
    """
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    print("[v15/eval] cargando engine...")
    engine = V15Engine.load(SUMMARY_PATH)

    print("[v15/eval] cargando muestras...")
    samples, _ = ds.build_samples(use_cache=use_cache, verbose=False)
    splits = ds.split_temporal(samples)
    target_samples = splits["holdout"] if use_holdout else samples

    if snapshot_only_prod:
        target_samples = [
            s for s in target_samples
            if (s.target == "q3" and s.snapshot_minute == config.Q3_GRAPH_CUTOFF)
            or (s.target == "q4" and s.snapshot_minute == config.Q4_GRAPH_CUTOFF)
        ]
    print(f"[v15/eval] {len(target_samples)} muestras de evaluacion")

    # Preload de graph/pbp para todas
    match_ids = sorted({s.match_id for s in target_samples})
    conn = ds.get_db_connection()
    try:
        gp = ds.load_graph_points(conn, match_ids)
        pbp = ds.load_pbp_events(conn, match_ids)
    finally:
        conn.close()

    # Recolectamos predicciones
    preds: list[dict] = []
    for s in target_samples:
        quarter_scores = _reconstruct_quarter_scores(s)
        pred = engine.predict(
            match_id=s.match_id,
            target=s.target,
            league=s.league,
            quarter_scores=quarter_scores,
            graph_points=gp.get(s.match_id, []),
            pbp_events=pbp.get(s.match_id, []),
        )
        preds.append({
            "match_id": s.match_id,
            "target": s.target,
            "league": s.league,
            "date": s.date,
            "snapshot_minute": s.snapshot_minute,
            "true_winner": s.target_winner,
            "true_home_pts": s.target_home_pts,
            "true_away_pts": s.target_away_pts,
            "signal": pred.signal,
            "reason": pred.reason,
            "probability_home": pred.probability,
            "confidence": pred.confidence,
            "threshold": pred.threshold,
            "pred_home": pred.debug.pred_home,
            "pred_away": pred.debug.pred_away,
            "pred_total": pred.debug.pred_total,
            "pace_bucket": pred.debug.pace_bucket,
            "gates_failed": [
                g.name for g in pred.debug.gates if not g.passed and "not_run" not in g.reason
            ],
        })

    # Agregaciones
    report = _build_report(preds, odds)
    _write_report(preds, report)
    return report


# ============================================================================
# Reconstruccion de quarter_scores desde sample
# ============================================================================

def _reconstruct_quarter_scores(s: ds.Sample) -> dict[str, int]:
    """Reconstruye marcadores por cuarto a partir de s.features (halftime_diff,
    halftime_total, q1_diff, q2_diff, q1_total, q2_total, q3_diff, q3_total)."""
    f = s.features
    q1_total = int(f.get("q1_total", 0))
    q1_diff = int(f.get("q1_diff", 0))
    q1_home = (q1_total + q1_diff) // 2
    q1_away = q1_total - q1_home
    q2_total = int(f.get("q2_total", 0))
    q2_diff = int(f.get("q2_diff", 0))
    q2_home = (q2_total + q2_diff) // 2
    q2_away = q2_total - q2_home
    out = {
        "q1_home": q1_home, "q1_away": q1_away,
        "q2_home": q2_home, "q2_away": q2_away,
    }
    if s.target == "q4":
        q3_total = int(f.get("q3_total", 0))
        q3_diff = int(f.get("q3_diff", 0))
        q3_home = (q3_total + q3_diff) // 2
        q3_away = q3_total - q3_home
        out["q3_home"] = q3_home
        out["q3_away"] = q3_away
    return out


# ============================================================================
# Reporte
# ============================================================================

def _build_report(preds: list[dict], odds: float) -> dict[str, Any]:
    global_stats = _summarize_bets(preds, odds)
    by_league_target: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in preds:
        by_league_target[(p["league"], p["target"])].append(p)

    per_group: list[dict] = []
    for (league, target), items in by_league_target.items():
        per_group.append({
            "league": league,
            "target": target,
            "n": len(items),
            **_summarize_bets(items, odds),
            "threshold_curve": _threshold_curve(items, odds),
        })
    per_group.sort(key=lambda r: (-(r.get("roi") or -99), -(r.get("n_bets") or 0)))

    # Razones de NO_BET
    reason_counts: dict[str, int] = defaultdict(int)
    for p in preds:
        if p["signal"] == "NO_BET":
            reason_counts[p["reason"]] += 1

    gate_failure_counts: dict[str, int] = defaultdict(int)
    for p in preds:
        for g in p.get("gates_failed", []):
            gate_failure_counts[g] += 1

    return {
        "odds": odds,
        "global": global_stats,
        "n_predictions": len(preds),
        "reason_counts": dict(reason_counts),
        "gate_failure_counts": dict(gate_failure_counts),
        "per_league_target": per_group,
    }


def _summarize_bets(items: list[dict], odds: float) -> dict:
    bets = [p for p in items if p["signal"] in ("BET_HOME", "BET_AWAY")]
    if not bets:
        return {"n_bets": 0, "coverage": 0.0, "hit_rate": None, "roi": None}
    preds = [1 if p["signal"] == "BET_HOME" else 0 for p in bets]
    truths = [p["true_winner"] for p in bets]
    correct = sum(1 for a, b in zip(preds, truths) if a == b)
    n = len(bets)
    hit = correct / n
    roi = (correct * (odds - 1) - (n - correct)) / n
    return {
        "n_bets": n,
        "wins": correct,
        "coverage": n / len(items) if items else 0.0,
        "hit_rate": hit,
        "roi": roi,
        "pnl": correct * (odds - 1) - (n - correct),
    }


def _threshold_curve(items: list[dict], odds: float) -> list[dict]:
    with_proba = [
        p for p in items
        if p.get("probability_home") is not None and p.get("true_winner") is not None
    ]
    if not with_proba:
        return []
    curve = []
    for t in np.round(np.arange(0.55, 0.93, 0.01), 2):
        mask = []
        for p in with_proba:
            conf = max(p["probability_home"], 1 - p["probability_home"])
            if conf >= t:
                pred_winner = 1 if p["probability_home"] >= 0.5 else 0
                mask.append((pred_winner, p["true_winner"]))
        n = len(mask)
        if n < 10:
            curve.append({"threshold": float(t), "n_bets": n, "hit_rate": None, "roi": None})
            continue
        wins = sum(1 for a, b in mask if a == b)
        hit = wins / n
        roi = (wins * (odds - 1) - (n - wins)) / n
        curve.append({
            "threshold": float(t), "n_bets": n,
            "hit_rate": float(hit), "roi": float(roi),
        })
    return curve


# ============================================================================
# Escritura
# ============================================================================

def _write_report(preds: list[dict], report: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    # JSON
    json_path = REPORTS_DIR / "eval_report_v15.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_jsonable)
    print(f"[v15/eval] -> {json_path}")

    # CSV predicciones
    csv_path = REPORTS_DIR / "eval_predictions_v15.csv"
    if preds:
        fieldnames = list(preds[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in preds:
                row = dict(p)
                row["gates_failed"] = "|".join(row.get("gates_failed") or [])
                writer.writerow(row)
        print(f"[v15/eval] -> {csv_path}")

    # Curvas de threshold por liga
    curves_dir = REPORTS_DIR / "threshold_curves"
    curves_dir.mkdir(exist_ok=True)
    for entry in report["per_league_target"]:
        slug = ds.slugify_league(entry["league"])
        name = f"{slug}_{entry['target']}.csv"
        path = curves_dir / name
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "n_bets", "hit_rate", "roi"])
            for row in entry.get("threshold_curve", []):
                writer.writerow([
                    row.get("threshold"),
                    row.get("n_bets"),
                    row.get("hit_rate"),
                    row.get("roi"),
                ])
    print(f"[v15/eval] curvas de threshold -> {curves_dir}")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="V15 evaluator")
    ap.add_argument("--odds", type=float, default=config.DEFAULT_ODDS)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="Evalua TODO el dataset, no solo holdout")
    ap.add_argument("--all-snapshots", action="store_true",
                    help="Evalua tambien cutoffs alternativos (no solo los de produccion)")
    args = ap.parse_args()
    run_backtest(
        use_cache=not args.no_cache,
        odds=args.odds,
        use_holdout=not args.full,
        snapshot_only_prod=not args.all_snapshots,
    )


if __name__ == "__main__":
    main()
