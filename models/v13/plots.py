"""
plots.py — Gráficas de diagnóstico para V13.

Genera visualizaciones para:
- Detectar data leakage (train vs val performance)
- Learning curves
- Calibration curves
- Feature importance
- Walk-forward performance over time
- Dataset distribution analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


def save_diagnostic_plots(
    model_key: str,
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: List[int],
    feature_importance: Optional[Dict[str, float]] = None,
    calibration_data: Optional[Dict] = None,
    walkforward_data: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
):
    """
    Save all diagnostic plots for a model.
    
    Args:
        model_key: Model identifier (e.g., 'q3_high_men')
        train_scores: Training scores at different sizes
        val_scores: Validation scores at different sizes
        train_sizes: Number of training samples for each score
        feature_importance: Dict of feature -> importance
        calibration_data: Data for calibration plot
        walkforward_data: Data for walk-forward over time
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "model_outputs" / "plots" / model_key
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📊 Generating plots for {model_key}...")
    
    # 1. Learning curves
    _plot_learning_curve(
        train_sizes, train_scores, val_scores,
        output_dir / "learning_curve.png",
        model_key
    )
    
    # 2. Feature importance
    if feature_importance:
        _plot_feature_importance(
            feature_importance,
            output_dir / "feature_importance.png",
            model_key
        )
    
    # 3. Calibration curve
    if calibration_data:
        _plot_calibration_curve(
            calibration_data,
            output_dir / "calibration_curve.png",
            model_key
        )
    
    # 4. Walk-forward over time
    if walkforward_data:
        _plot_walkforward_time(
            walkforward_data,
            output_dir / "walkforward_over_time.png",
            model_key
        )
    
    print(f"  ✅ Plots saved to {output_dir}")


def _plot_learning_curve(train_sizes, train_scores, val_scores, output_path, model_key):
    """Plot learning curves to detect overfitting/underfitting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_scores, 'o-', label='Train Score', color='blue')
    ax.plot(train_sizes, val_scores, 'o-', label='Validation Score', color='red')
    
    # Add baseline (random = 0.5 for binary classification)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Baseline (random)')
    
    # Detect leakage
    gap = np.mean(train_scores) - np.mean(val_scores)
    if gap > 0.15:
        color = 'red'
        text = f'⚠️ POSSIBLE LEAKAGE\nGap: {gap:.3f}'
    elif gap > 0.10:
        color = 'orange'
        text = f'⚡ MODERATE GAP\nGap: {gap:.3f}'
    else:
        color = 'green'
        text = f'✅ HEALTHY GAP\nGap: {gap:.3f}'
    
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
    )
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score (F1 / Accuracy)')
    ax.set_title(f'Learning Curve - {model_key}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    📈 Learning curve: {output_path.name}")


def _plot_feature_importance(feature_importance: Dict[str, float], output_path, model_key):
    """Plot top 20 feature importances."""
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    top_20 = sorted_features[:20]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = [f[0] for f in top_20]
    importances = [f[1] for f in top_20]
    
    colors = ['red' if v < 0 else 'blue' for v in importances]
    
    ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    
    ax.set_xlabel('Feature Importance (coefficient weight)')
    ax.set_title(f'Top 20 Features - {model_key}')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Check for leakage indicators
    leakage_features = [f for f in features if any(kw in f.lower() for kw in ['date', 'match_id', 'event'])]
    if leakage_features:
        ax.text(
            0.02, 0.02,
            f'🚨 LEAKAGE DETECTED: {", ".join(leakage_features)}',
            transform=ax.transAxes,
            fontsize=10,
            color='red',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.2)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    📊 Feature importance: {output_path.name}")


def _plot_calibration_curve(calibration_data: Dict, output_path, model_key):
    """Plot calibration curve (reliability diagram)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Calibration curve
    predicted_probs = calibration_data['predicted_probs']
    true_probs = calibration_data['true_probs']
    counts = calibration_data.get('counts', np.ones_like(true_probs))
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(predicted_probs, true_probs, 'o-', label='Model', linewidth=2)
    
    # Calculate calibration error
    cal_error = np.mean(np.abs(predicted_probs - true_probs))
    
    if cal_error < 0.05:
        color = 'green'
        text = f'✅ Well calibrated\nError: {cal_error:.3f}'
    elif cal_error < 0.10:
        color = 'orange'
        text = f'⚡ Moderate calibration\nError: {cal_error:.3f}'
    else:
        color = 'red'
        text = f'❌ Poor calibration\nError: {cal_error:.3f}'
    
    ax1.text(
        0.98, 0.02, text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
    )
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Observed Frequency')
    ax1.set_title(f'Calibration Curve - {model_key}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(predicted_probs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Predicted Probabilities')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    📉 Calibration curve: {output_path.name}")


def _plot_walkforward_time(walkforward_data: Dict, output_path, model_key):
    """Plot walk-forward performance over time."""
    dates = walkforward_data['dates']
    scores = walkforward_data['scores']
    baseline = walkforward_data.get('baseline', 0.5)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, scores, 'o-', label='Model F1', linewidth=2, color='blue')
    ax.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.2f})')
    
    # Rolling average
    if len(scores) >= 5:
        rolling_avg = np.convolve(scores, np.ones(5)/5, mode='valid')
        ax.plot(dates[4:], rolling_avg, label='5-period rolling avg', linewidth=2, color='green')
    
    # Check for performance degradation over time
    if len(scores) >= 10:
        first_half = np.mean(scores[:len(scores)//2])
        second_half = np.mean(scores[len(scores)//2:])
        
        if second_half < first_half - 0.05:
            text = f'⚠️ Performance degrading\n{first_half:.3f} → {second_half:.3f}'
            color = 'orange'
        else:
            text = f'✅ Stable performance\n{first_half:.3f} → {second_half:.3f}'
            color = 'green'
        
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Walk-Forward Performance Over Time - {model_key}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    📅 Walk-forward over time: {output_path.name}")


def plot_dataset_summary(
    samples_metadata: Dict,
    train_range: Dict,
    val_range: Dict,
    cal_range: Dict,
    output_dir: Path,
):
    """Plot dataset distribution and summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Temporal distribution
    ax1 = axes[0, 0]
    all_dates = []
    for split, dates in [('Train', train_range), ('Val', val_range), ('Cal', cal_range)]:
        if dates.get('matches'):
            ax1.bar(split, dates['matches'], alpha=0.7, label=split)
    
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Dataset Split Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pace bucket distribution
    ax2 = axes[0, 1]
    buckets = ['Low', 'Medium', 'High']
    men_counts = [
        samples_metadata.get('men_q3', {}).get('low', 0),
        samples_metadata.get('men_q3', {}).get('medium', 0),
        samples_metadata.get('men_q3', {}).get('high', 0),
    ]
    women_counts = [
        samples_metadata.get('women_q3', {}).get('low', 0),
        samples_metadata.get('women_q3', {}).get('medium', 0),
        samples_metadata.get('women_q3', {}).get('high', 0),
    ]
    
    x = np.arange(len(buckets))
    width = 0.35
    
    ax2.bar(x - width/2, men_counts, width, label='Men', alpha=0.7)
    ax2.bar(x + width/2, women_counts, width, label='Women', alpha=0.7)
    
    ax2.set_xlabel('Pace Bucket')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Q3 Samples by Pace and Gender')
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Target distribution
    ax3 = axes[1, 0]
    q3_count = samples_metadata.get('q3_total', 0)
    q4_count = samples_metadata.get('q4_total', 0)
    
    ax3.pie([q3_count, q4_count], labels=['Q3', 'Q4'], autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Target Distribution\nTotal: {q3_count + q4_count}')
    
    # 4. League distribution
    ax4 = axes[1, 1]
    leagues = samples_metadata.get('top_leagues', {})
    if leagues:
        top_10 = dict(sorted(leagues.items(), key=lambda x: x[1], reverse=True)[:10])
        ax4.barh(range(len(top_10)), list(top_10.values()), alpha=0.7)
        ax4.set_yticks(range(len(top_10)))
        ax4.set_yticklabels(list(top_10.keys()), fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Number of Matches')
        ax4.set_title('Top 10 Leagues by Sample Count')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "dataset_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Dataset summary: {output_path.name}")


def plot_leakage_detection(
    model_performances: Dict[str, Dict],
    output_dir: Path,
):
    """
    Plot leakage detection across all models.
    
    If train_score >> val_score consistently, indicates leakage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(model_performances.keys())
    train_scores = [model_performances[m]['train'] for m in models]
    val_scores = [model_performances[m]['val'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, train_scores, width, label='Train Score', alpha=0.7, color='blue')
    ax.bar(x + width/2, val_scores, width, label='Validation Score', alpha=0.7, color='red')
    
    # Add baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Baseline (0.5)')
    
    # Highlight models with large gaps
    for i, (train_s, val_s) in enumerate(zip(train_scores, val_scores)):
        gap = train_s - val_s
        if gap > 0.15:
            ax.annotate(
                '⚠️',
                xy=(i, val_s),
                fontsize=12,
                ha='center'
            )
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Leakage Detection: Train vs Validation Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary text
    avg_gap = np.mean([t - v for t, v in zip(train_scores, val_scores)])
    if avg_gap > 0.10:
        text = f'⚠️ Average gap: {avg_gap:.3f}\nPossible systematic leakage'
        color = 'red'
    else:
        text = f'✅ Average gap: {avg_gap:.3f}\nNo systematic leakage detected'
        color = 'green'
    
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
    )
    
    plt.tight_layout()
    output_path = output_dir / "leakage_detection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  🔍 Leakage detection: {output_path.name}")
