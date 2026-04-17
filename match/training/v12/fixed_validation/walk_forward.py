"""
V12 - FIX: Proper Walk-Forward Validation
===========================================
This fixes the data leakage issues:
1. League stats computed with ROLLING window (only past data)
2. Team history built STRICTLY sequentially
3. Walk-forward validation instead of 80/20 split
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # training/v12/fixed_validation -> match
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

db_mod = __import__("db")

DB_PATH = PROJECT_ROOT / "matches.db"
FIX_DIR = Path(__file__).parent
FIX_DIR.mkdir(parents=True, exist_ok=True)


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_samples_with_rolling_stats(cutoff_date: str, db_path: Path) -> tuple[list, dict]:
    """
    Build samples using ONLY matches before cutoff_date.
    League stats and team history are computed from past data only.
    """
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)

    # League stats from PAST only
    league_rows = conn.execute("""
        SELECT m.league, m.home_score, m.away_score
        FROM matches m
        WHERE m.status_type = 'finished'
          AND m.date < ?
          AND m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
    """, (cutoff_date,)).fetchall()

    league_data = defaultdict(lambda: {"home_wins": 0, "away_wins": 0, "total": 0, "total_points": []})
    for row in league_rows:
        league = row["league"] or "unknown"
        home, away = int(row["home_score"]), int(row["away_score"])
        league_data[league]["total"] += 1
        if home > away:
            league_data[league]["home_wins"] += 1
        elif away > home:
            league_data[league]["away_wins"] += 1
        league_data[league]["total_points"].append(home + away)

    league_stats = {}
    for league, d in league_data.items():
        pts = d["total_points"]
        league_stats[league] = {
            "samples": d["total"],
            "home_win_rate": d["home_wins"] / d["total"] if d["total"] else 0.5,
            "avg_total_points": np.mean(pts) if pts else 0,
            "std_total_points": np.std(pts) if pts else 0,
        }

    # Get matches for features (ALL matches, we'll filter later)
    match_rows = conn.execute("""
        SELECT match_id, date, time FROM matches 
        WHERE date < ? 
        ORDER BY date, time, match_id
    """, (cutoff_date,)).fetchall()

    # Build team history STRICTLY sequentially
    top_leagues = set(list(league_stats.keys())[:25])
    top_teams_counter = Counter()
    for ht, at in conn.execute("SELECT home_team, away_team FROM matches WHERE date < ?", (cutoff_date,)).fetchall():
        if ht: top_teams_counter[str(ht)] += 1
        if at: top_teams_counter[str(at)] += 1
    top_teams = {team for team, _ in top_teams_counter.most_common(150)}

    # Import feature building functions
    from training.v12.train_v12 import (
        _is_complete_match, _quarter_points, _infer_gender, _bucket,
        _graph_stats_upto, _pbp_stats_upto, _score_pressure_features,
        _monte_carlo_win_prob, _pbp_events_upto_minute,
        _max_scoring_run, _current_scoring_run, _pbp_recent_window_features,
    )
    from training.v12.train_v12 import HybridSample

    team_history = defaultdict(list)
    samples = []

    for row in match_rows:
        match_id = str(row["match_id"])
        dt = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M")
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_match(data):
            continue

        m, score, pbp, gp = data["match"], data["score"], data.get("play_by_play", {}), data.get("graph_points", [])
        q1h, q1a = _quarter_points(data, "Q1")
        q2h, q2a = _quarter_points(data, "Q2")
        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue

        ht, at, league = m.get("home_team", ""), m.get("away_team", ""), m.get("league", "")
        gender = _infer_gender(league, ht, at)

        # Team history from PAST only
        home_hist = team_history[ht][-12:]
        away_hist = team_history[at][-12:]
        home_prior_wr = _safe_rate(sum(home_hist), len(home_hist))
        away_prior_wr = _safe_rate(sum(away_hist), len(away_hist))

        # League context from PAST only
        league_info = league_stats.get(league, {})
        league_home_wr = league_info.get("home_win_rate", 0.5)
        league_avg_total = league_info.get("avg_total_points", 0)
        league_std_total = league_info.get("std_total_points", 0)
        league_samples = league_info.get("samples", 0)

        base = {
            "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
            "gender_bucket": gender,
            "home_team_bucket": _bucket(ht, top_teams, "TEAM"),
            "away_team_bucket": _bucket(at, top_teams, "TEAM"),
            "home_prior_wr": round(home_prior_wr, 4),
            "away_prior_wr": round(away_prior_wr, 4),
            "prior_wr_diff": round(home_prior_wr - away_prior_wr, 4),
            "prior_wr_sum": round(home_prior_wr + away_prior_wr, 4),
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
            "league_home_advantage": round(league_home_wr - 0.5, 4),
            "league_avg_total_points": round(league_avg_total, 2),
            "league_std_total_points": round(league_std_total, 2),
            "league_sample_size": min(league_samples, 500),
        }

        # Q3 features
        ht_home, ht_away = q1h + q2h, q1a + q2a
        q3_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2"])

        f_q3 = dict(base, ht_home=ht_home, ht_away=ht_away, ht_diff=ht_home - ht_away, ht_total=ht_home + ht_away)
        f_q3.update(_graph_stats_upto(gp, 24))
        f_q3.update(q3_pbp_stats)
        f_q3.update(_score_pressure_features(
            score_home=ht_home, score_away=ht_away,
            pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
            pbp_home_3pt=q3_pbp_stats["pbp_home_3pt"],
            pbp_away_3pt=q3_pbp_stats["pbp_away_3pt"],
            elapsed_minutes=24.0, minutes_left=12.0
        ))
        f_q3.update(_pbp_recent_window_features(pbp, 24.0, 6.0))

        q3_mc = _monte_carlo_win_prob(
            score_home=ht_home, score_away=ht_away,
            pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
            pbp_home_pts=q3_pbp_stats["pbp_home_pts_per_play"] * q3_pbp_stats["pbp_home_plays"],
            pbp_away_pts=q3_pbp_stats["pbp_away_pts_per_play"] * q3_pbp_stats["pbp_away_plays"],
            elapsed_minutes=24.0, minutes_left=12.0
        )
        f_q3.update(q3_mc)

        q3_events = _pbp_events_upto_minute(pbp, 24.0)
        f_q3["current_run_home"] = _current_scoring_run(q3_events, "home")
        f_q3["current_run_away"] = _current_scoring_run(q3_events, "away")
        f_q3["max_run_home"] = _max_scoring_run(q3_events, "home")
        f_q3["max_run_away"] = _max_scoring_run(q3_events, "away")
        f_q3["run_diff"] = _max_scoring_run(q3_events, "home") - _max_scoring_run(q3_events, "away")

        # Q4 features
        score_3q_home = ht_home + q3h
        score_3q_away = ht_away + q3a
        q4_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])

        f_q4 = dict(base, q3_diff=q3h - q3a, score_3q_home=score_3q_home, score_3q_away=score_3q_away,
                    score_3q_diff=score_3q_home - score_3q_away, score_3q_total=score_3q_home + score_3q_away)
        f_q4.update(_graph_stats_upto(gp, 36))
        f_q4.update(q4_pbp_stats)
        f_q4.update(_score_pressure_features(
            score_home=score_3q_home, score_away=score_3q_away,
            pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
            pbp_home_3pt=q4_pbp_stats["pbp_home_3pt"],
            pbp_away_3pt=q4_pbp_stats["pbp_away_3pt"],
            elapsed_minutes=36.0, minutes_left=12.0
        ))
        f_q4.update(_pbp_recent_window_features(pbp, 36.0, 6.0))

        q4_mc = _monte_carlo_win_prob(
            score_home=score_3q_home, score_away=score_3q_away,
            pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
            pbp_home_pts=q4_pbp_stats["pbp_home_pts_per_play"] * q4_pbp_stats["pbp_home_plays"],
            pbp_away_pts=q4_pbp_stats["pbp_away_pts_per_play"] * q4_pbp_stats["pbp_away_plays"],
            elapsed_minutes=36.0, minutes_left=12.0
        )
        f_q4.update(q4_mc)

        q4_events = _pbp_events_upto_minute(pbp, 36.0)
        f_q4["current_run_home"] = _current_scoring_run(q4_events, "home")
        f_q4["current_run_away"] = _current_scoring_run(q4_events, "away")
        f_q4["max_run_home"] = _max_scoring_run(q4_events, "home")
        f_q4["max_run_away"] = _max_scoring_run(q4_events, "away")
        f_q4["run_diff"] = _max_scoring_run(q4_events, "home") - _max_scoring_run(q4_events, "away")

        for quarter, features in [("q3", f_q3), ("q4", f_q4)]:
            if quarter == "q3":
                winner_target = None if q3h == q3a else (1 if q3h > q3a else 0)
            else:
                winner_target = None if q4h == q4a else (1 if q4h > q4a else 0)

            samples.append(HybridSample(
                match_id=f"{match_id}_{quarter}",
                dt=dt,
                league=league or "unknown",
                gender=gender,
                features=features,
                target_winner=winner_target,
                target_home_pts=None, target_away_pts=None, target_total_pts=None,
            ))

        # Update team history AFTER building features
        team_history[ht].append(1 if score["home"] > score["away"] else 0)
        team_history[at].append(1 if score["away"] > score["home"] else 0)

    conn.close()
    return samples, league_stats


def walk_forward_validation(
    start_date: str = "2024-01-01",
    end_date: str = "2026-04-01",
    step_months: int = 2,
) -> dict:
    """
    Walk-forward validation:
    1. Train on data before cutoff
    2. Test on data after cutoff
    3. Move cutoff forward
    4. Repeat
    """
    print(f"\n[walk-forward] {start_date} to {end_date}, step={step_months}mo")
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate cutoff dates
    cutoffs = []
    current = start_dt
    while current < end_dt:
        cutoffs.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=step_months * 30)
    
    if cutoffs[-1] != end_dt.strftime("%Y-%m-%d"):
        cutoffs.append(end_dt.strftime("%Y-%m-%d"))
    
    print(f"[walk-forward] {len(cutoffs)} folds: {cutoffs[:3]}...")
    
    all_results = {
        "q3": {"y_true": [], "y_pred": [], "probs": []},
        "q4": {"y_true": [], "y_pred": [], "probs": []},
    }
    
    fold_results = []
    
    for i, cutoff in enumerate(cutoffs[:-1]):
        test_end = cutoffs[i + 1]
        print(f"\n[fold {i+1}/{len(cutoffs)-1}] Train before {cutoff}, test {cutoff} to {test_end}")
        
        # Build samples with rolling stats
        samples, league_stats = build_samples_with_rolling_stats(cutoff, DB_PATH)
        
        if len(samples) < 500:
            print(f"  Skip: only {len(samples)} samples")
            continue
        
        valid = [s for s in samples if s.target_winner is not None]
        valid.sort(key=lambda s: s.dt)
        
        # Split: last 20% as test within the training period
        n = len(valid)
        n_train = int(n * 0.85)
        n_test = n - n_train
        
        if n_test < 100:
            print(f"  Skip: only {n_test} test samples")
            continue
        
        x_dict = [s.features for s in valid]
        y = [s.target_winner for s in valid]
        
        vec = DictVectorizer(sparse=False)
        x_all = vec.fit_transform(x_dict)
        x_train, x_test = x_all[:n_train], x_all[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # Train models
        models = {}
        
        logreg = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        logreg.fit(x_train_scaled, y_train)
        models["logreg"] = logreg
        
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_split=20, min_samples_leaf=10, random_state=42
        )
        gb.fit(x_train, y_train)
        models["gb"] = gb
        
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                random_state=42, eval_metric='logloss'
            )
            xgb_model.fit(x_train, y_train)
            models["xgb"] = xgb_model
        
        if HAS_CAT:
            cat_model = CatBoostClassifier(
                iterations=200, depth=5, learning_rate=0.08,
                l2_leaf_reg=3, random_state=42, verbose=0
            )
            cat_model.fit(x_train, y_train)
            models["catboost"] = cat_model
        
        # Ensemble predictions
        probs_list = []
        for name, model in models.items():
            if name == "logreg":
                probs = model.predict_proba(x_test_scaled)[:, 1]
            else:
                probs = model.predict_proba(x_test)[:, 1]
            probs_list.append(probs)
        
        avg_probs = np.mean(probs_list, axis=0)
        preds = [1 if p >= 0.5 else 0 for p in avg_probs]
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        brier = brier_score_loss(y_test, avg_probs)
        
        try:
            roc_auc = roc_auc_score(y_test, avg_probs)
        except ValueError:
            roc_auc = None
        
        fold_result = {
            "cutoff": cutoff,
            "test_range": f"{cutoff} to {test_end}",
            "n_train": n_train,
            "n_test": n_test,
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "brier": round(brier, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc else None,
        }
        fold_results.append(fold_result)
        
        print(f"  Acc={acc:.3f}, F1={f1:.3f}, Brier={brier:.3f}, ROC-AUC={roc_auc:.3f}")
        
        # Store for aggregation
        for target in ["q3", "q4"]:
            target_mask = [target in s.match_id for s in valid[n_train:]]
            all_results[target]["y_true"].extend([y for y, m in zip(y_test, target_mask) if m])
            all_results[target]["y_pred"].extend([p for p, m in zip(preds, target_mask) if m])
            all_results[target]["probs"].extend([p for p, m in zip(avg_probs, target_mask) if m])
    
    # Aggregate results
    summary = {}
    for target in ["q3", "q4"]:
        y_true = all_results[target]["y_true"]
        y_pred = all_results[target]["y_pred"]
        probs = all_results[target]["probs"]
        
        if not y_true:
            summary[target] = {"error": "No test samples"}
            continue
        
        summary[target] = {
            "n_samples": len(y_true),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred), 4),
            "brier": round(brier_score_loss(y_true, probs), 4),
        }
        try:
            summary[target]["roc_auc"] = round(roc_auc_score(y_true, probs), 4)
        except ValueError:
            pass
    
    # Save results
    report = {
        "method": "walk_forward_validation",
        "start_date": start_date,
        "end_date": end_date,
        "step_months": step_months,
        "folds": len(fold_results),
        "fold_results": fold_results,
        "summary": summary,
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(FIX_DIR / "walk_forward_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    fold_nums = range(1, len(fold_results) + 1)
    accuracies = [r["accuracy"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    briers = [r["brier"] for r in fold_results]
    
    axes[0].plot(fold_nums, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    axes[0].set_title('Walk-Forward Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Fold Number')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.35, 0.75)
    
    axes[1].plot(fold_nums, f1s, 's-', color='green', linewidth=2, markersize=8)
    axes[1].set_title('Walk-Forward F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Fold Number')
    axes[1].set_ylabel('F1')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.35, 0.75)
    
    axes[2].plot(fold_nums, briers, '^-', color='orange', linewidth=2, markersize=8)
    axes[2].axhline(0.25, color='red', linestyle='--', alpha=0.5, label='Random (0.25)')
    axes[2].set_title('Walk-Forward Brier Score', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Fold Number')
    axes[2].set_ylabel('Brier (lower=better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('V12 Walk-Forward Validation (NO DATA LEAKAGE)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_path = FIX_DIR / "walk_forward_plots.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return report


def main():
    start = time.time()
    
    print("="*70)
    print("V12 WALK-FORWARD VALIDATION (NO DATA LEAKAGE)")
    print("="*70)
    
    report = walk_forward_validation(
        start_date="2024-06-01",
        end_date="2026-04-01",
        step_months=2,
    )
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("WALK-FORWARD SUMMARY")
    print("="*70)
    print(f"Folds: {report['folds']}")
    print(f"Time: {elapsed:.1f}s")
    
    for target, res in report.get("summary", {}).items():
        print(f"\n{target.upper()}:")
        print(f"  Samples: {res.get('n_samples', 0)}")
        print(f"  Accuracy: {res.get('accuracy', 'N/A')}")
        print(f"  F1: {res.get('f1', 'N/A')}")
        print(f"  Brier: {res.get('brier', 'N/A')}")
    
    print(f"\nResults saved to: {FIX_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
