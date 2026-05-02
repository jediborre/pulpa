"""Train V6.2 predictors for quarter winners (Q3, Q4).

V6.2 keeps V6 feature engineering, but adds automatic league pruning:
- leagues with weak/no signal in train are collapsed to LEAGUE_OTHER_SIGNAL_WEAK
- keeps temporal 80/20 split like V6 for holdout evaluation
- removes MLP from the production stack
- exports A/B comparison against V6 baseline metrics
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import train_q3_q4_models_v6 as v6

ROOT = v6.ROOT
DB_PATH = v6.DB_PATH
OUT_DIR = ROOT / "training" / "model_outputs_v6_2"
BASELINE_DIR = ROOT / "training" / "model_outputs_v6"
LEAGUE_EXCLUSION_CONFIG_PATH = ROOT / "training" / "v6_2_league_name_exclusions.json"

# League pruning rules (train split only).
LEAGUE_MIN_TRAIN_ROWS = 30
LEAGUE_MIN_EFFECT_ABS_DIFF = 0.015
LEAGUE_OTHER_TOKEN = "LEAGUE_OTHER_SIGNAL_WEAK"


def _load_league_exclusion_rules(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    categories = cfg.get("categories", [])
    rules: list[dict[str, str]] = []
    for cat in categories:
        cat_name = str(cat.get("name", "uncategorized")).strip() or "uncategorized"
        for p in cat.get("patterns", []):
            pat = str(p).strip()
            if not pat:
                continue
            rules.append({
                "category": cat_name,
                "pattern": pat,
                "pattern_lc": pat.lower(),
            })
    return rules


def _exclude_rows_by_league_name(
    rows_sorted: list[v6.MatchSample],
    target_name: str,
    rules: list[dict[str, str]],
) -> tuple[list[v6.MatchSample], list[dict], dict]:
    kept: list[v6.MatchSample] = []
    excluded_rows: list[dict] = []

    for s in rows_sorted:
        league = str((s.features_q3 if target_name == "q3" else s.features_q4).get("league", ""))
        league_lc = league.lower()

        hit_rule: dict[str, str] | None = None
        for rule in rules:
            if rule["pattern_lc"] in league_lc:
                hit_rule = rule
                break

        if hit_rule is None:
            kept.append(s)
            continue

        excluded_rows.append(
            {
                "target": target_name,
                "match_id": s.match_id,
                "datetime": s.dt.isoformat(),
                "league": league,
                "exclude_category": hit_rule["category"],
                "exclude_pattern": hit_rule["pattern"],
            }
        )

    total = len(rows_sorted)
    excluded = len(excluded_rows)
    summary = {
        "target": target_name,
        "rows_before_exclusion": total,
        "rows_excluded_by_name": excluded,
        "rows_after_exclusion": len(kept),
        "exclude_ratio": round(float(excluded) / float(total), 6) if total else 0.0,
        "active_patterns": len(rules),
    }
    return kept, excluded_rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_auc(y_true: list[int], probs: list[float]) -> float | None:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return None


def _metric_row(
    target: str,
    model_name: str,
    n_total: int,
    n_train: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    row = {
        "target": target,
        "model": model_name,
        "samples_total": n_total,
        "samples_train": n_train,
        "samples_test": n_test,
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds)), 6),
        "precision": round(float(precision_score(y_true, preds)), 6),
        "recall": round(float(recall_score(y_true, preds)), 6),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }
    auc = _safe_auc(y_true, probs)
    row["roc_auc"] = None if auc is None else round(auc, 6)
    return row


def _compute_league_signal(
    rows_sorted: list[v6.MatchSample],
    target_name: str,
    n_train: int,
) -> tuple[set[str], list[dict]]:
    if target_name == "q3":
        y_train = [int(s.target_q3) for s in rows_sorted[:n_train]]
        leagues = [str(s.features_q3.get("league", "")) for s in rows_sorted[:n_train]]
    else:
        y_train = [int(s.target_q4) for s in rows_sorted[:n_train]]
        leagues = [str(s.features_q4.get("league", "")) for s in rows_sorted[:n_train]]

    global_pos_rate = float(np.mean(y_train)) if y_train else 0.0
    stats: dict[str, dict] = {}

    for league, y in zip(leagues, y_train):
        rec = stats.setdefault(league, {"league": league, "train_rows": 0, "positives": 0})
        rec["train_rows"] += 1
        rec["positives"] += int(y)

    keep: set[str] = set()
    out_rows: list[dict] = []
    for league, rec in sorted(stats.items(), key=lambda kv: (-kv[1]["train_rows"], kv[0])):
        train_rows = int(rec["train_rows"])
        pos_rate = float(rec["positives"]) / float(train_rows) if train_rows else 0.0
        abs_effect = abs(pos_rate - global_pos_rate)

        if train_rows < LEAGUE_MIN_TRAIN_ROWS:
            cls = "weak_support"
        elif abs_effect < LEAGUE_MIN_EFFECT_ABS_DIFF:
            cls = "weak_effect"
        else:
            cls = "keep"
            keep.add(league)

        out_rows.append(
            {
                "target": target_name,
                "league": league,
                "train_rows": train_rows,
                "train_pos_rate": round(pos_rate, 6),
                "global_train_pos_rate": round(global_pos_rate, 6),
                "abs_effect_vs_global": round(abs_effect, 6),
                "signal_class": cls,
            }
        )

    return keep, out_rows


def _apply_league_filter(
    x_dict: list[dict],
    keep_leagues: set[str],
) -> tuple[list[dict], dict]:
    filtered: list[dict] = []
    replaced = 0
    for row in x_dict:
        rec = dict(row)
        lg = str(rec.get("league", ""))
        if lg not in keep_leagues:
            rec["league"] = LEAGUE_OTHER_TOKEN
            rec["league_bucket"] = LEAGUE_OTHER_TOKEN
            replaced += 1
        filtered.append(rec)

    summary = {
        "rows_total": len(x_dict),
        "rows_replaced": replaced,
        "rows_kept": len(x_dict) - replaced,
        "replace_ratio": round(float(replaced) / float(len(x_dict)), 6) if x_dict else 0.0,
        "kept_leagues_count": len(keep_leagues),
    }
    return filtered, summary


def _make_models() -> dict:
    return {
        "xgb": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        ),
    }


def _train_target(samples: list[v6.MatchSample], target_name: str, league_name_rules: list[dict[str, str]]) -> dict:
    if target_name == "q3":
        rows = [s for s in samples if s.target_q3 is not None]
        rows = sorted(rows, key=lambda item: item.dt)
    else:
        rows = [s for s in samples if s.target_q4 is not None]
        rows = sorted(rows, key=lambda item: item.dt)

    rows, league_name_excluded_rows, league_name_exclusion_summary = _exclude_rows_by_league_name(
        rows,
        target_name,
        league_name_rules,
    )

    if target_name == "q3":
        x_raw = [s.features_q3 for s in rows]
        y = [int(s.target_q3) for s in rows]
    else:
        x_raw = [s.features_q4 for s in rows]
        y = [int(s.target_q4) for s in rows]

    n_total = len(rows)
    if n_total < 200:
        raise RuntimeError(f"Not enough rows for {target_name}. Got {n_total}.")

    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    keep_leagues, league_signal_rows = _compute_league_signal(rows, target_name, n_train)
    x_filtered, filter_summary = _apply_league_filter(x_raw, keep_leagues)

    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_filtered)
    x_train = x_all[:n_train]
    x_test = x_all[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    models = _make_models()
    proba_map: dict[str, list[float]] = {}
    metrics_rows: list[dict] = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        probs = list(model.predict_proba(x_test)[:, 1])
        proba_map[model_name] = probs

        metrics_rows.append(
            _metric_row(
                target=target_name,
                model_name=model_name,
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                y_true=y_test,
                probs=probs,
            )
        )

        artifact = {
            "version": "v6.2",
            "target": target_name,
            "model_name": model_name,
            "vectorizer": vec,
            "model": model,
            "league_filter": {
                "min_train_rows": LEAGUE_MIN_TRAIN_ROWS,
                "min_abs_effect_diff": LEAGUE_MIN_EFFECT_ABS_DIFF,
                "other_token": LEAGUE_OTHER_TOKEN,
                "kept_leagues": sorted(keep_leagues),
            },
            "league_name_exclusion": {
                "config_path": str(LEAGUE_EXCLUSION_CONFIG_PATH),
                "active_patterns": len(league_name_rules),
                "rows_excluded": league_name_exclusion_summary["rows_excluded_by_name"],
            },
            "trained_rows": n_total,
            "feature_count": len(vec.feature_names_),
        }
        joblib.dump(artifact, OUT_DIR / f"{target_name}_{model_name}.joblib")

    # Champion strategy by target.
    if target_name == "q3":
        champion_probs = proba_map["xgb"]
        champion_name = "champion_q3_xgb"
    else:
        champion_probs = [
            0.6 * proba_map["xgb"][i] + 0.4 * proba_map["hist_gb"][i]
            for i in range(len(y_test))
        ]
        champion_name = "champion_q4_blend_xgb_0.6_hist_0.4"

    metrics_rows.append(
        _metric_row(
            target=target_name,
            model_name=champion_name,
            n_total=n_total,
            n_train=n_train,
            n_test=n_test,
            y_true=y_test,
            probs=champion_probs,
        )
    )

    champion_artifact = {
        "version": "v6.2",
        "target": target_name,
        "model_name": champion_name,
        "vectorizer": vec,
        "models": models,
        "champion_strategy": {
            "q3": "xgb_only",
            "q4": "0.6*xgb + 0.4*hist_gb",
        },
        "league_filter": {
            "min_train_rows": LEAGUE_MIN_TRAIN_ROWS,
            "min_abs_effect_diff": LEAGUE_MIN_EFFECT_ABS_DIFF,
            "other_token": LEAGUE_OTHER_TOKEN,
            "kept_leagues": sorted(keep_leagues),
        },
        "league_name_exclusion": {
            "config_path": str(LEAGUE_EXCLUSION_CONFIG_PATH),
            "active_patterns": len(league_name_rules),
            "rows_excluded": league_name_exclusion_summary["rows_excluded_by_name"],
        },
    }
    joblib.dump(champion_artifact, OUT_DIR / f"{target_name}_champion.joblib")

    return {
        "target": target_name,
        "metrics": metrics_rows,
        "league_signal": league_signal_rows,
        "league_name_exclusions": league_name_excluded_rows,
        "league_name_exclusion_summary": league_name_exclusion_summary,
        "filter_summary": {
            "target": target_name,
            **filter_summary,
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "features_after_vectorizer": len(vec.feature_names_),
        },
    }


def _ab_comparison_rows(v6_rows: list[dict], v62_rows: list[dict]) -> list[dict]:
    by_target_v6: dict[str, dict[str, dict]] = {}
    for row in v6_rows:
        by_target_v6.setdefault(row["target"], {})[row["model"]] = row

    out: list[dict] = []
    for row in v62_rows:
        target = row["target"]
        model = row["model"]

        if model.startswith("champion_q3"):
            base_model = "xgb"
        elif model.startswith("champion_q4"):
            base_model = "ensemble_avg_prob"
        else:
            base_model = model

        base = by_target_v6.get(target, {}).get(base_model)
        if not base:
            continue

        comp = {
            "target": target,
            "v6_model_reference": base_model,
            "v6_2_model": model,
            "v6_accuracy": base["accuracy"],
            "v6_2_accuracy": row["accuracy"],
            "delta_accuracy": round(float(row["accuracy"]) - float(base["accuracy"]), 6),
            "v6_f1": base["f1"],
            "v6_2_f1": row["f1"],
            "delta_f1": round(float(row["f1"]) - float(base["f1"]), 6),
            "v6_log_loss": base["log_loss"],
            "v6_2_log_loss": row["log_loss"],
            "delta_log_loss": round(float(row["log_loss"]) - float(base["log_loss"]), 6),
            "v6_brier": base["brier"],
            "v6_2_brier": row["brier"],
            "delta_brier": round(float(row["brier"]) - float(base["brier"]), 6),
            "v6_roc_auc": base["roc_auc"],
            "v6_2_roc_auc": row["roc_auc"],
            "delta_roc_auc": round(float(row["roc_auc"]) - float(base["roc_auc"]), 6),
        }
        out.append(comp)

    out.sort(key=lambda r: (r["target"], r["v6_2_model"]))
    return out


def _write_markdown_report(
    q3_result: dict,
    q4_result: dict,
    ab_rows: list[dict],
    league_name_rules: list[dict[str, str]],
) -> None:
    lines: list[str] = []
    lines.append("# V6.2 Report")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- league_min_train_rows: {LEAGUE_MIN_TRAIN_ROWS}")
    lines.append(f"- league_min_effect_abs_diff: {LEAGUE_MIN_EFFECT_ABS_DIFF}")
    lines.append(f"- league_other_token: {LEAGUE_OTHER_TOKEN}")
    lines.append(f"- league_name_exclusion_config: {LEAGUE_EXCLUSION_CONFIG_PATH}")
    lines.append(f"- league_name_exclusion_patterns: {len(league_name_rules)}")
    lines.append("- league_name_exclusion_match_mode: contains(case-insensitive)")
    lines.append("- split: temporal 80/20 (same as v6)")
    lines.append("- models: xgb, hist_gb")
    lines.append("- champion_q3: xgb")
    lines.append("- champion_q4: 0.6*xgb + 0.4*hist_gb")
    lines.append("")

    lines.append("## League Name Exclusion Summary")
    lines.append("| target | rows_before | rows_excluded | rows_after | exclude_ratio |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in [q3_result["league_name_exclusion_summary"], q4_result["league_name_exclusion_summary"]]:
        lines.append(
            f"| {r['target']} | {r['rows_before_exclusion']} | {r['rows_excluded_by_name']} | {r['rows_after_exclusion']} | {r['exclude_ratio']:.4f} |"
        )
    lines.append("")

    lines.append("## League Filter Summary")
    lines.append("| target | kept_leagues | rows_replaced | replace_ratio | features_after_vectorizer |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in [q3_result["filter_summary"], q4_result["filter_summary"]]:
        lines.append(
            f"| {r['target']} | {r['kept_leagues_count']} | {r['rows_replaced']} | {r['replace_ratio']:.4f} | {r['features_after_vectorizer']} |"
        )
    lines.append("")

    lines.append("## Holdout Metrics (V6.2)")
    lines.append("| target | model | accuracy | f1 | log_loss | brier | roc_auc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in q3_result["metrics"] + q4_result["metrics"]:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['accuracy']:.6f} | {row['f1']:.6f} | {row['log_loss']:.6f} | {row['brier']:.6f} | {row['roc_auc']:.6f} |"
        )
    lines.append("")

    lines.append("## A/B vs V6")
    lines.append("| target | v6_ref | v6.2_model | d_acc | d_f1 | d_log_loss | d_brier | d_auc |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in ab_rows:
        lines.append(
            f"| {row['target']} | {row['v6_model_reference']} | {row['v6_2_model']} | {row['delta_accuracy']:+.6f} | {row['delta_f1']:+.6f} | {row['delta_log_loss']:+.6f} | {row['delta_brier']:+.6f} | {row['delta_roc_auc']:+.6f} |"
        )

    (OUT_DIR / "V6_2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    league_name_rules = _load_league_exclusion_rules(LEAGUE_EXCLUSION_CONFIG_PATH)

    print("[v6.2] building samples using v6 feature pipeline...")
    samples = v6._build_samples(DB_PATH)

    print("[v6.2] training q3...")
    q3_result = _train_target(samples, "q3", league_name_rules)

    print("[v6.2] training q4...")
    q4_result = _train_target(samples, "q4", league_name_rules)

    v62_metrics = q3_result["metrics"] + q4_result["metrics"]
    _write_csv(OUT_DIR / "q3_metrics.csv", [r for r in q3_result["metrics"] if r["target"] == "q3"])
    _write_csv(OUT_DIR / "q4_metrics.csv", [r for r in q4_result["metrics"] if r["target"] == "q4"])
    _write_csv(OUT_DIR / "league_signal_q3.csv", q3_result["league_signal"])
    _write_csv(OUT_DIR / "league_signal_q4.csv", q4_result["league_signal"])
    _write_csv(OUT_DIR / "league_name_exclusions_q3.csv", q3_result["league_name_exclusions"])
    _write_csv(OUT_DIR / "league_name_exclusions_q4.csv", q4_result["league_name_exclusions"])
    _write_csv(
        OUT_DIR / "league_name_exclusion_summary.csv",
        [q3_result["league_name_exclusion_summary"], q4_result["league_name_exclusion_summary"]],
    )
    _write_csv(OUT_DIR / "league_filter_summary.csv", [q3_result["filter_summary"], q4_result["filter_summary"]])

    baseline_q3 = []
    baseline_q4 = []
    with (BASELINE_DIR / "q3_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        baseline_q3 = list(csv.DictReader(f))
    with (BASELINE_DIR / "q4_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        baseline_q4 = list(csv.DictReader(f))

    v6_rows = baseline_q3 + baseline_q4
    ab_rows = _ab_comparison_rows(v6_rows, v62_metrics)
    _write_csv(OUT_DIR / "ab_comparison_v6_vs_v6_2.csv", ab_rows)
    _write_markdown_report(q3_result, q4_result, ab_rows, league_name_rules)

    summary = {
        "version": "v6.2",
        "baseline_version": "v6",
        "split": "temporal_80_20",
        "league_filter": {
            "min_train_rows": LEAGUE_MIN_TRAIN_ROWS,
            "min_abs_effect_diff": LEAGUE_MIN_EFFECT_ABS_DIFF,
            "other_token": LEAGUE_OTHER_TOKEN,
        },
        "league_name_exclusion": {
            "config_path": str(LEAGUE_EXCLUSION_CONFIG_PATH),
            "match_mode": "contains_case_insensitive",
            "active_patterns": len(league_name_rules),
            "q3": q3_result["league_name_exclusion_summary"],
            "q4": q4_result["league_name_exclusion_summary"],
        },
        "q3": q3_result["filter_summary"],
        "q4": q4_result["filter_summary"],
    }
    with (OUT_DIR / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[v6.2] done")
    print(f"[v6.2] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
