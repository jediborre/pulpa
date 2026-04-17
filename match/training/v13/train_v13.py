"""
train_v13.py — Main training orchestrator for V13.

Trains all (target, gender, pace_bucket) models with:
- Walk-forward validation for league stats
- Dynamic cutoff snapshots
- Probability calibration
- Comprehensive dataset metadata tracking
- Diagnostic plots for leakage detection
"""

import sys
import json
import time
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Add parent to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v13 import config, dataset, features, walk_forward
from training.v13 import train_clf, train_reg, plots

OUT_DIR = ROOT / "training" / "v13" / "model_outputs"


def compute_dataset_metadata(samples, train_samples, val_samples, cal_samples):
    """Compute comprehensive dataset metadata."""
    print("\n📊 Computing dataset metadata...")
    
    # Date ranges
    all_dates = [s.date for s in samples if s.date]
    train_dates = [s.date for s in train_samples if s.date]
    val_dates = [s.date for s in val_samples if s.date]
    cal_dates = [s.date for s in cal_samples if s.date]
    
    metadata = {
        "total_samples": len(samples),
        "total_matches": len(set(s.match_id for s in samples)),
        "date_range": {
            "oldest": min(all_dates) if all_dates else None,
            "newest": max(all_dates) if all_dates else None,
        },
        "splits": {
            "train": {
                "samples": len(train_samples),
                "matches": len(set(s.match_id for s in train_samples)),
                "date_range": {
                    "oldest": min(train_dates) if train_dates else None,
                    "newest": max(train_dates) if train_dates else None,
                }
            },
            "validation": {
                "samples": len(val_samples),
                "matches": len(set(s.match_id for s in val_samples)),
                "date_range": {
                    "oldest": min(val_dates) if val_dates else None,
                    "newest": max(val_dates) if val_dates else None,
                }
            },
            "calibration": {
                "samples": len(cal_samples),
                "matches": len(set(s.match_id for s in cal_samples)),
                "date_range": {
                    "oldest": min(cal_dates) if cal_dates else None,
                    "newest": max(cal_dates) if cal_dates else None,
                }
            }
        },
    }
    
    # Distribution by target
    target_counts = Counter(s.target for s in samples)
    metadata["by_target"] = dict(target_counts)
    
    # Distribution by gender
    gender_counts = Counter(s.gender for s in samples)
    metadata["by_gender"] = dict(gender_counts)
    
    # Distribution by pace bucket
    pace_counts = Counter(s.pace_bucket for s in samples)
    metadata["by_pace"] = dict(pace_counts)
    
    # Distribution by (target, gender, pace)
    combo_counts = Counter(f"{s.target}_{s.pace_bucket}_{s.gender}" for s in samples)
    metadata["by_model_key"] = dict(combo_counts)
    
    # League distribution
    league_counts = Counter(s.league for s in samples if s.league)
    metadata["top_leagues"] = dict(league_counts.most_common(20))
    
    # Pace bucket details
    metadata["pace_bucket_details"] = {}
    for target in ['q3', 'q4']:
        for gender in ['men', 'women']:
            for pace in ['low', 'medium', 'high']:
                key = f"{gender}_{pace}"
                subset = [s for s in samples if s.target == target and s.gender == gender and s.pace_bucket == pace]
                if subset:
                    metadata["pace_bucket_details"][key] = {
                        "target": target,
                        "samples": len(subset),
                        "positive_rate": round(np.mean([s.target_winner for s in subset]), 3),
                    }
    
    print(f"✅ Metadata computed for {metadata['total_samples']} samples")
    return metadata


def build_learning_curve_data(clf_ensemble, x_train, y_train, x_val, y_val):
    """Build learning curve data at different training sizes."""
    sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    val_scores = []
    train_sizes = []
    
    n_total = x_train.shape[0]  # Fix for sparse matrices
    
    for frac in tqdm(sizes, desc="    Learning curve", leave=False):
        n = int(n_total * frac)
        if n < 10:
            continue
        
        x_sub = x_train[:n]
        y_sub = y_train[:n]
        
        # Quick eval with first model in ensemble
        if clf_ensemble and clf_ensemble.get('models'):
            model_info = clf_ensemble['models'][0]
            model = model_info['model']
            model.fit(x_sub, y_sub)
            
            train_pred = model.predict(x_sub)
            val_pred = model.predict(x_val)
            
            train_scores.append(f1_score(y_sub, train_pred, average='weighted'))
            val_scores.append(f1_score(y_val, val_pred, average='weighted'))
            train_sizes.append(n)
    
    return train_sizes, train_scores, val_scores


def extract_feature_importance(clf_ensemble, feature_names):
    """Extract feature importance from ensemble."""
    importance = defaultdict(float)
    
    for model_info in clf_ensemble['models']:
        model = model_info['model']
        weight = model_info['weight']
        
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if coefs.ndim == 2:
                coefs = coefs[0]
            for i, coef in enumerate(coefs):
                if i < len(feature_names):
                    importance[feature_names[i]] += float(coef) * weight
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, imp in enumerate(importances):
                if i < len(feature_names):
                    importance[feature_names[i]] += float(imp) * weight
    
    return dict(importance)


def main(skip_tuning: bool = False, subset: str = None):
    """Main training pipeline."""
    print("\n" + "🏀"*40)
    print("V13 TRAINING PIPELINE")
    print("🏀"*40)
    
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # STEP 1: BUILD SAMPLES
    # ================================================================
    print("\n" + "="*80)
    print("📦 STEP 1: BUILDING DATASET")
    print("="*80)
    
    samples, meta = dataset.build_samples(use_cache=True)
    pace_thresholds = meta['pace_thresholds']
    
    print(f"✅ {len(samples)} samples from {meta['n_matches']} matches")
    print(f"   Pace thresholds: {pace_thresholds}")
    
    # ================================================================
    # STEP 2: LOAD GRAPH AND PBP DATA
    # ================================================================
    print("\n" + "="*80)
    print("📊 STEP 2: LOADING GRAPH POINTS AND PBP")
    print("="*80)
    
    match_ids = list(set(s.match_id for s in samples))
    conn = dataset.get_db_connection()
    graph_points = dataset.load_graph_points(conn, match_ids)
    pbp_events = dataset.load_pbp_events(conn, match_ids)
    conn.close()
    
    print(f"✅ Graph points: {len(graph_points)} matches")
    print(f"✅ PBP events: {len(pbp_events)} matches")
    
    # ================================================================
    # STEP 3: BUILD FEATURES
    # ================================================================
    print("\n" + "="*80)
    print("🔧 STEP 3: BUILDING FEATURES")
    print("="*80)
    
    features.enrich_samples_with_features(samples, graph_points, pbp_events)
    
    # ================================================================
    # STEP 4: TEMPORAL SPLIT
    # ================================================================
    print("\n" + "="*80)
    print("✂️ STEP 4: TEMPORAL SPLIT")
    print("="*80)
    
    train_samples, val_samples, cal_samples = dataset.split_temporal(samples)
    
    print(f"✅ Train: {len(train_samples)} samples ({min(s.date for s in train_samples) if train_samples else 'N/A'} to {max(s.date for s in train_samples) if train_samples else 'N/A'})")
    print(f"✅ Val:   {len(val_samples)} samples ({min(s.date for s in val_samples) if val_samples else 'N/A'} to {max(s.date for s in val_samples) if val_samples else 'N/A'})")
    print(f"✅ Cal:   {len(cal_samples)} samples ({min(s.date for s in cal_samples) if cal_samples else 'N/A'} to {max(s.date for s in cal_samples) if cal_samples else 'N/A'})")
    
    # ================================================================
    # STEP 5: COMPUTE DATASET METADATA
    # ================================================================
    print("\n" + "="*80)
    print("📊 STEP 5: COMPUTING DATASET METADATA")
    print("="*80)
    
    dataset_metadata = compute_dataset_metadata(samples, train_samples, val_samples, cal_samples)
    
    # ================================================================
    # STEP 6: TRAIN MODELS
    # ================================================================
    print("\n" + "="*80)
    print("🎯 STEP 6: TRAINING MODELS")
    print("="*80)
    
    training_summary = {
        "version": "v13",
        "trained_at": datetime.now().isoformat(),
        "dataset": dataset_metadata,
        "pace_thresholds": pace_thresholds,
        "models_trained": [],
        "leakage_detection": {},
        "gates": {
            "q3_graph_cutoff_min": config.Q3_GRAPH_CUTOFF,
            "q4_graph_cutoff_min": config.Q4_GRAPH_CUTOFF,
            "min_gp_q3": config.MIN_GP_Q3,
            "min_gp_q4": config.MIN_GP_Q4,
            "min_pbp_q3": config.MIN_PBP_Q3,
            "min_pbp_q4": config.MIN_PBP_Q4,
            "min_confidence_q3": config.MIN_CONFIDENCE_Q3,
            "min_confidence_q4": config.MIN_CONFIDENCE_Q4,
            "max_volatility": config.MAX_VOLATILITY,
            "league_samples_block": config.LEAGUE_MIN_SAMPLES_BLOCK,
            "league_samples_penalize": config.LEAGUE_MIN_SAMPLES_PENALIZE,
        },
    }
    
    # For leakage detection across all models
    all_model_gaps = {}
    
    for target, gender, pace in dataset.iter_model_keys():
        if subset:
            # Match if subset starts with target (e.g., --subset q4 matches all q4_*)
            # OR if subset matches the full key pattern
            if not (target.startswith(subset) or f"{target}_{pace}" in subset or subset in f"{target}_{gender}_{pace}"):
                continue
        
        model_key = f"{target}_{pace}_{gender}"
        
        print(f"\n{'─'*80}")
        print(f"🎯 {model_key.upper()}")
        print(f"{'─'*80}")
        
        # Get subsets
        train_sub = dataset.get_subset(train_samples, target, gender, pace)
        val_sub = dataset.get_subset(val_samples, target, gender, pace)
        cal_sub = dataset.get_subset(cal_samples, target, gender, pace)
        
        if len(train_sub) < 50:
            print(f"  ⚠️ Skipping {model_key}: only {len(train_sub)} training samples")
            continue
        
        # Extract features and targets
        vec = DictVectorizer(sparse=True)
        
        x_train = vec.fit_transform([s.features for s in train_sub])
        y_train = np.array([s.target_winner for s in train_sub])
        
        x_val = vec.transform([s.features for s in val_sub])
        y_val = np.array([s.target_winner for s in val_sub])
        
        # Scale features
        scaler = StandardScaler(with_mean=False)
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        
        print(f"  📊 Train: {len(train_sub)}, Val: {len(val_sub)}, Cal: {len(cal_sub)}")
        print(f"  📊 Positive class: {y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")
        
        # Train classifier ensemble
        clf_ensemble = train_clf.train_ensemble(
            x_train_scaled, y_train, x_val_scaled, y_val,
            model_key=model_key,
            best_params=None,
        )
        
        model_summary = {
            "key": model_key,
            "samples_train": len(train_sub),
            "samples_val": len(val_sub),
            "samples_cal": len(cal_sub),
        }
        
        if clf_ensemble:
            # Save ensemble
            clf_path = OUT_DIR / f"{model_key}_clf_ensemble.joblib"
            train_clf.save_ensemble(clf_ensemble, clf_path)
            
            # Save vectorizer and scaler
            joblib.dump(vec, OUT_DIR / f"{model_key}_vectorizer.joblib")
            joblib.dump(scaler, OUT_DIR / f"{model_key}_scaler.joblib")
            
            # Get best metrics
            best_f1 = max(r['val_f1'] for r in clf_ensemble['models'])
            best_acc = max(r['val_acc'] for r in clf_ensemble['models'])
            
            # Calculate train scores for leakage detection
            train_preds = clf_ensemble['models'][0]['model'].predict(x_train_scaled)
            train_f1 = f1_score(y_train, train_preds, average='weighted')
            
            gap = train_f1 - best_f1
            all_model_gaps[model_key] = {
                'train': round(train_f1, 3),
                'val': round(best_f1, 3),
                'gap': round(gap, 3),
            }
            
            model_summary.update({
                "val_accuracy": round(best_acc, 3),
                "val_f1": round(best_f1, 3),
                "train_f1": round(train_f1, 3),
                "train_val_gap": round(gap, 3),
                "algorithms": [r['algorithm'] for r in clf_ensemble['models']],
                "weights": {r['algorithm']: round(r['weight'], 3) for r in clf_ensemble['models']},
            })
            
            # Extract feature names
            try:
                feature_names = vec.get_feature_names_out().tolist()
            except:
                feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]
            
            # Feature importance
            feat_importance = extract_feature_importance(clf_ensemble, feature_names)
            
            # Learning curve data
            lc_sizes, lc_train, lc_val = build_learning_curve_data(
                clf_ensemble, x_train_scaled, y_train, x_val_scaled, y_val
            ) if clf_ensemble else ([], [], [])
            
            # Save plots
            plots.save_diagnostic_plots(
                model_key=model_key,
                train_scores=lc_train,
                val_scores=lc_val,
                train_sizes=lc_sizes,
                feature_importance=feat_importance,
                output_dir=OUT_DIR / "plots",
            )
            
            print(f"  ✅ {model_key}: acc={best_acc:.3f}, f1={best_f1:.3f}, gap={gap:.3f}")
        else:
            print(f"  ❌ Failed to train {model_key}")
            continue
        
        # Train regressor ensembles
        reg_results = {}
        for target_type in ['home', 'away', 'total']:
            if target_type == 'home':
                y_train_reg = np.array([s.target_home_pts for s in train_sub])
                y_val_reg = np.array([s.target_home_pts for s in val_sub])
            elif target_type == 'away':
                y_train_reg = np.array([s.target_away_pts for s in train_sub])
                y_val_reg = np.array([s.target_away_pts for s in val_sub])
            else:
                y_train_reg = np.array([s.target_total_pts for s in train_sub])
                y_val_reg = np.array([s.target_total_pts for s in val_sub])
            
            reg_ensemble = train_reg.train_regressor_ensemble(
                x_train_scaled, y_train_reg, x_val_scaled, y_val_reg,
                model_key=model_key,
                target_type=target_type,
                best_params=None,
            )
            
            if reg_ensemble:
                reg_path = OUT_DIR / f"{model_key}_{target_type}_reg_ensemble.joblib"
                train_reg.save_regressor_ensemble(reg_ensemble, reg_path)
                
                avg_mae = np.mean([m['mae'] for m in reg_ensemble['models']])
                reg_results[target_type] = round(float(avg_mae), 2)
        
        model_summary['regression_mae'] = reg_results
        training_summary['models_trained'].append(model_summary)
    
    # ================================================================
    # STEP 7: LEAKAGE DETECTION
    # ================================================================
    print("\n" + "="*80)
    print("🔍 STEP 7: LEAKAGE DETECTION")
    print("="*80)
    
    if all_model_gaps:
        avg_gap = np.mean([g['gap'] for g in all_model_gaps.values()])
        max_gap = max([g['gap'] for g in all_model_gaps.values()])
        
        training_summary['leakage_detection'] = {
            "average_train_val_gap": round(float(avg_gap), 3),
            "max_gap": round(float(max_gap), 3),
            "model_gaps": all_model_gaps,
            "assessment": "PASS" if avg_gap < 0.10 else ("WARNING" if avg_gap < 0.15 else "FAIL"),
            "note": "Gap < 0.10 is healthy, 0.10-0.15 needs review, > 0.15 indicates possible leakage"
        }
        
        print(f"\n📊 Average train-val gap: {avg_gap:.3f}")
        print(f"📊 Max gap: {max_gap:.3f}")
        print(f"📊 Assessment: {training_summary['leakage_detection']['assessment']}")
        
        # Plot leakage detection across all models
        plots.plot_leakage_detection(all_model_gaps, OUT_DIR / "plots")
    
    # ================================================================
    # STEP 8: DATASET SUMMARY PLOT
    # ================================================================
    print("\n" + "="*80)
    print("📊 STEP 8: GENERATING DATASET SUMMARY PLOT")
    print("="*80)
    
    plots.plot_dataset_summary(
        samples_metadata={
            'q3_total': sum(1 for s in samples if s.target == 'q3'),
            'q4_total': sum(1 for s in samples if s.target == 'q4'),
            'men_q3': {
                'low': sum(1 for s in samples if s.target == 'q3' and s.gender == 'men' and s.pace_bucket == 'low'),
                'medium': sum(1 for s in samples if s.target == 'q3' and s.gender == 'men' and s.pace_bucket == 'medium'),
                'high': sum(1 for s in samples if s.target == 'q3' and s.gender == 'men' and s.pace_bucket == 'high'),
            },
            'women_q3': {
                'low': sum(1 for s in samples if s.target == 'q3' and s.gender == 'women' and s.pace_bucket == 'low'),
                'medium': sum(1 for s in samples if s.target == 'q3' and s.gender == 'women' and s.pace_bucket == 'medium'),
                'high': sum(1 for s in samples if s.target == 'q3' and s.gender == 'women' and s.pace_bucket == 'high'),
            },
            'top_leagues': dataset_metadata.get('top_leagues', {}),
        },
        train_range=dataset_metadata['splits']['train'],
        val_range=dataset_metadata['splits']['validation'],
        cal_range=dataset_metadata['splits']['calibration'],
        output_dir=OUT_DIR / "plots",
    )
    
    # ================================================================
    # STEP 9: SAVE TRAINING SUMMARY
    # ================================================================
    print("\n" + "="*80)
    print("💾 STEP 9: SAVING TRAINING SUMMARY")
    print("="*80)
    
    training_summary['training_time_seconds'] = round(time.time() - start_time, 1)
    training_summary['training_completed_at'] = datetime.now().isoformat()
    
    summary_path = OUT_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    print(f"\n✅ Summary saved to {summary_path}")
    print(f"✅ Models trained: {len(training_summary['models_trained'])}")
    print(f"✅ Total time: {training_summary['training_time_seconds']}s")
    print(f"✅ Plots saved to: {OUT_DIR / 'plots'}")
    
    print("\n" + "🏀"*40)
    print("TRAINING COMPLETE")
    print("🏀"*40 + "\n")
    
    # Final assessment
    if training_summary.get('leakage_detection', {}).get('assessment') == 'FAIL':
        print("🚨 WARNING: Possible data leakage detected!")
        print("   Review training_summary.json for details")
        print("   Check leakage_detection plot")
    elif training_summary.get('leakage_detection', {}).get('assessment') == 'WARNING':
        print("⚡ WARNING: Moderate train-val gaps detected")
        print("   Review models with largest gaps")
    else:
        print("✅ No systematic leakage detected")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train V13 models")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--subset", type=str, default=None, help="Train only specific subset (e.g., 'q3_high')")
    
    args = parser.parse_args()
    
    main(skip_tuning=args.skip_tuning, subset=args.subset)
