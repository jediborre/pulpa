"""
cli.py — Command-line interface for V13 model operations.

Provides an interactive menu and CLI commands for:
- Training models
- Running inference
- Evaluating models
- Viewing diagnostics
- Managing datasets
- Checking model status
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add parent to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v13 import config

OUT_DIR = ROOT / "training" / "v13" / "model_outputs"


def cmd_train(args):
    """Execute training command."""
    from training.v13 import train_v13
    
    print("\n" + "="*80)
    print("🎯 V13 TRAINING")
    print("="*80)
    
    train_v13.main(
        skip_tuning=args.skip_tuning,
        subset=args.subset,
    )


def cmd_infer(args):
    """Execute inference command."""
    from training.v13 import infer_match_v13
    
    print("\n" + "="*80)
    print("🔮 V13 INFERENCE")
    print("="*80)
    
    match_id = args.match_id
    target = args.target
    
    result = infer_match_v13.run_inference(match_id, target)
    
    if result['ok']:
        pred = result['prediction']
        print(f"\n✅ Prediction for {match_id} ({target.upper()})")
        print(f"{'─'*60}")
        print(f"   Winner:      {pred.winner_pick}")
        print(f"   Confidence:  {pred.winner_confidence:.3f}")
        print(f"   Signal:      {pred.winner_signal}")
        print(f"   Pred Total:  {pred.predicted_total}")
        print(f"   Pred Home:   {pred.predicted_home}")
        print(f"   Pred Away:   {pred.predicted_away}")
        print(f"   League:      {pred.league_quality}")
        print(f"   Volatility:  {pred.volatility_index:.3f}")
        print(f"   Data Qual:   {pred.data_quality}")
        print(f"   Reasoning:   {pred.reasoning}")
    else:
        print(f"\n❌ Failed: {result['reason']}")


def cmd_status(args):
    """Show model status."""
    print("\n" + "="*80)
    print("📊 V13 MODEL STATUS")
    print("="*80)
    
    # Check training summary
    summary_path = OUT_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        
        print(f"\n✅ Training summary found")
        print(f"   Version: {summary.get('version', 'unknown')}")
        print(f"   Trained at: {summary.get('trained_at', 'unknown')}")
        print(f"   Total models: {len(summary.get('models_trained', []))}")
        
        # Dataset info
        dataset = summary.get('dataset', {})
        print(f"\n📦 Dataset:")
        print(f"   Total samples: {dataset.get('total_samples', 'N/A')}")
        print(f"   Total matches: {dataset.get('total_matches', 'N/A')}")
        print(f"   Date range: {dataset.get('date_range', {}).get('oldest', 'N/A')} to {dataset.get('date_range', {}).get('newest', 'N/A')}")
        
        # Splits
        splits = dataset.get('splits', {})
        for split_name in ['train', 'validation', 'calibration']:
            split = splits.get(split_name, {})
            print(f"   {split_name.capitalize()}: {split.get('samples', 0)} samples, {split.get('matches', 0)} matches")
        
        # Leakage detection
        leakage = summary.get('leakage_detection', {})
        if leakage:
            print(f"\n🔍 Leakage Detection:")
            print(f"   Assessment: {leakage.get('assessment', 'N/A')}")
            print(f"   Average gap: {leakage.get('average_train_val_gap', 'N/A')}")
            print(f"   Max gap: {leakage.get('max_gap', 'N/A')}")
        
        # Models trained
        models = summary.get('models_trained', [])
        if models:
            print(f"\n🎯 Models Trained:")
            for m in models:
                gap = m.get('train_val_gap', 0)
                gap_indicator = "✅" if gap < 0.10 else ("⚠️" if gap < 0.15 else "🚨")
                print(f"   {gap_indicator} {m['key']:25s} F1={m.get('val_f1', 0):.3f}  Gap={gap:.3f}  N={m.get('samples_train', 0)}")
    else:
        print(f"\n❌ No training summary found at {summary_path}")
        print(f"   Run training first: python training/v13/cli.py train")
    
    # Check plots
    plots_dir = OUT_DIR / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.rglob("*.png"))
        print(f"\n📈 Diagnostic Plots: {len(plot_files)} files")
        for pf in plot_files[:10]:
            print(f"   - {pf.relative_to(plots_dir.parent)}")
        if len(plot_files) > 10:
            print(f"   ... and {len(plot_files) - 10} more")
    else:
        print(f"\n📈 No diagnostic plots found")
    
    # Check model files
    model_files = list(OUT_DIR.glob("*.joblib"))
    print(f"\n📦 Model Files: {len(model_files)} files")
    
    # Config
    print(f"\n⚙️ Configuration:")
    print(f"   Q3 Graph Cutoff: {config.Q3_GRAPH_CUTOFF}")
    print(f"   Q4 Graph Cutoff: {config.Q4_GRAPH_CUTOFF}")
    print(f"   Min GP Q3: {config.MIN_GP_Q3}")
    print(f"   Min GP Q4: {config.MIN_GP_Q4}")
    print(f"   Min Confidence Q3: {config.MIN_CONFIDENCE_Q3}")
    print(f"   Min Confidence Q4: {config.MIN_CONFIDENCE_Q4}")
    print(f"   Pace Q3: low ≤{config.PACE_Q3_LOW_UPPER}, high ≥{config.PACE_Q3_HIGH_LOWER}")
    print(f"   Pace Q4: low ≤{config.PACE_Q4_LOW_UPPER}, high ≥{config.PACE_Q4_HIGH_LOWER}")


def cmd_show_config(args):
    """Show current configuration."""
    print("\n" + "="*80)
    print("⚙️ V13 CONFIGURATION")
    print("="*80)
    
    config_items = [
        ("Graph Cutoffs", [
            ("Q3_GRAPH_CUTOFF", config.Q3_GRAPH_CUTOFF),
            ("Q4_GRAPH_CUTOFF", config.Q4_GRAPH_CUTOFF),
        ]),
        ("Minimum Gates", [
            ("MIN_GP_Q3", config.MIN_GP_Q3),
            ("MIN_GP_Q4", config.MIN_GP_Q4),
            ("MIN_PBP_Q3", config.MIN_PBP_Q3),
            ("MIN_PBP_Q4", config.MIN_PBP_Q4),
        ]),
        ("Confidence Thresholds", [
            ("MIN_CONFIDENCE_Q3", config.MIN_CONFIDENCE_Q3),
            ("MIN_CONFIDENCE_Q4", config.MIN_CONFIDENCE_Q4),
            ("MAX_VOLATILITY", config.MAX_VOLATILITY),
        ]),
        ("League Gates", [
            ("LEAGUE_MIN_SAMPLES_BLOCK", config.LEAGUE_MIN_SAMPLES_BLOCK),
            ("LEAGUE_MIN_SAMPLES_PENALIZE", config.LEAGUE_MIN_SAMPLES_PENALIZE),
            ("LEAGUE_MIN_SAMPLES_FULL", config.LEAGUE_MIN_SAMPLES_FULL),
        ]),
        ("Pace Buckets (Q3)", [
            ("PACE_Q3_LOW_UPPER", config.PACE_Q3_LOW_UPPER),
            ("PACE_Q3_HIGH_LOWER", config.PACE_Q3_HIGH_LOWER),
        ]),
        ("Pace Buckets (Q4)", [
            ("PACE_Q4_LOW_UPPER", config.PACE_Q4_LOW_UPPER),
            ("PACE_Q4_HIGH_LOWER", config.PACE_Q4_HIGH_LOWER),
        ]),
        ("Monitor Timing", [
            ("MONITOR_Q3_MINUTE", config.MONITOR_Q3_MINUTE),
            ("MONITOR_Q4_MINUTE", config.MONITOR_Q4_MINUTE),
            ("MONITOR_WAKE_BEFORE", config.MONITOR_WAKE_BEFORE),
            ("MONITOR_CONFIRM_TICKS_Q3", config.MONITOR_CONFIRM_TICKS_Q3),
            ("MONITOR_CONFIRM_TICKS_Q4", config.MONITOR_CONFIRM_TICKS_Q4),
        ]),
    ]
    
    for section, items in config_items:
        print(f"\n{section}:")
        for name, value in items:
            print(f"   {name:35s} = {value}")


def cmd_show_dataset(args):
    """Show dataset information."""
    print("\n" + "="*80)
    print("📦 V13 DATASET INFO")
    print("="*80)
    
    summary_path = OUT_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        
        dataset = summary.get('dataset', {})
        
        print(f"\n📊 General:")
        print(f"   Total samples: {dataset.get('total_samples', 'N/A')}")
        print(f"   Total matches: {dataset.get('total_matches', 'N/A')}")
        print(f"   Date range: {dataset.get('date_range', {}).get('oldest', 'N/A')} to {dataset.get('date_range', {}).get('newest', 'N/A')}")
        
        # By target
        by_target = dataset.get('by_target', {})
        print(f"\n🎯 By Target:")
        for target, count in by_target.items():
            print(f"   {target:10s}: {count:>8,} samples")
        
        # By gender
        by_gender = dataset.get('by_gender', {})
        print(f"\n👥 By Gender:")
        for gender, count in by_gender.items():
            print(f"   {gender:10s}: {count:>8,} samples")
        
        # By pace
        by_pace = dataset.get('by_pace', {})
        print(f"\n🏃 By Pace Bucket:")
        for pace, count in by_pace.items():
            print(f"   {pace:10s}: {count:>8,} samples")
        
        # By model key
        by_model = dataset.get('by_model_key', {})
        print(f"\n🔧 By Model Key (top 12):")
        sorted_models = sorted(by_model.items(), key=lambda x: x[1], reverse=True)
        for model, count in sorted_models[:12]:
            print(f"   {model:25s}: {count:>8,} samples")
        
        # Splits
        splits = dataset.get('splits', {})
        print(f"\n✂️ Splits:")
        for split_name in ['train', 'validation', 'calibration']:
            split = splits.get(split_name, {})
            print(f"   {split_name.capitalize():15s}: {split.get('samples', 0):>8,} samples, {split.get('matches', 0):>6,} matches")
            dr = split.get('date_range', {})
            if dr.get('oldest'):
                print(f"                   {dr['oldest']} to {dr['newest']}")
        
        # Top leagues
        top_leagues = dataset.get('top_leagues', {})
        if top_leagues:
            print(f"\n🏆 Top 10 Leagues:")
            for i, (league, count) in enumerate(list(top_leagues.items())[:10], 1):
                print(f"   {i:2d}. {league[:50]:50s}: {count:>6,}")
    else:
        print(f"\n❌ No dataset info available. Run training first.")


def cmd_show_plots(args):
    """List or show diagnostic plots."""
    plots_dir = OUT_DIR / "plots"
    
    if not plots_dir.exists():
        print(f"\n❌ No plots directory found at {plots_dir}")
        return
    
    print("\n" + "="*80)
    print("📈 V13 DIAGNOSTIC PLOTS")
    print("="*80)
    
    plot_files = sorted(plots_dir.rglob("*.png"))
    
    if not plot_files:
        print(f"\n📭 No plots generated yet. Run training first.")
        return
    
    print(f"\n📊 Total plots: {len(plot_files)}\n")
    
    # Group by type
    categories = {
        'dataset_summary': [],
        'leakage_detection': [],
        'learning_curve': [],
        'feature_importance': [],
        'calibration_curve': [],
        'walkforward': [],
    }
    
    for pf in plot_files:
        name = pf.name.lower()
        if 'dataset_summary' in name:
            categories['dataset_summary'].append(pf)
        elif 'leakage' in name:
            categories['leakage_detection'].append(pf)
        elif 'learning' in name:
            categories['learning_curve'].append(pf)
        elif 'feature' in name:
            categories['feature_importance'].append(pf)
        elif 'calibration' in name:
            categories['calibration_curve'].append(pf)
        elif 'walkforward' in name or 'walkforward' in name:
            categories['walkforward'].append(pf)
    
    for cat, files in categories.items():
        if files:
            print(f"\n{cat.upper().replace('_', ' ')} ({len(files)} files):")
            for f in files:
                rel = f.relative_to(plots_dir)
                print(f"   📄 {rel}")


def cmd_check_leakage(args):
    """Check for data leakage."""
    print("\n" + "="*80)
    print("🔍 V13 LEAKAGE CHECK")
    print("="*80)
    
    summary_path = OUT_DIR / "training_summary.json"
    if not summary_path.exists():
        print(f"\n❌ No training summary found. Run training first.")
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    leakage = summary.get('leakage_detection', {})
    
    if not leakage:
        print(f"\n⚠️ No leakage detection data available.")
        return
    
    assessment = leakage.get('assessment', 'UNKNOWN')
    avg_gap = leakage.get('average_train_val_gap', 0)
    max_gap = leakage.get('max_gap', 0)
    
    print(f"\n📊 Overall Assessment: {assessment}")
    print(f"   Average gap: {avg_gap:.3f}")
    print(f"   Max gap:     {max_gap:.3f}")
    
    if assessment == "PASS":
        print(f"\n✅ No systematic leakage detected")
        print(f"   All models have healthy train-validation gaps")
    elif assessment == "WARNING":
        print(f"\n⚡ Moderate gaps detected in some models")
        print(f"   Review models with gaps > 0.10")
    else:  # FAIL
        print(f"\n🚨 POSSIBLE LEAKAGE DETECTED")
        print(f"   DO NOT USE THESE MODELS FOR PRODUCTION")
        print(f"   Review training data and feature engineering")
    
    # Show models with largest gaps
    model_gaps = leakage.get('model_gaps', {})
    if model_gaps:
        sorted_gaps = sorted(model_gaps.items(), key=lambda x: x[1]['gap'], reverse=True)
        
        print(f"\n📋 Models by gap (largest first):")
        for model, gap_info in sorted_gaps[:10]:
            gap = gap_info['gap']
            indicator = "✅" if gap < 0.10 else ("⚠️" if gap < 0.15 else "🚨")
            print(f"   {indicator} {model:25s} Train={gap_info['train']:.3f}  Val={gap_info['val']:.3f}  Gap={gap:.3f}")


def cmd_list_models(args):
    """List all trained models."""
    print("\n" + "="*80)
    print("🎯 V13 TRAINED MODELS")
    print("="*80)
    
    summary_path = OUT_DIR / "training_summary.json"
    if not summary_path.exists():
        print(f"\n❌ No training summary found. Run training first.")
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    models = summary.get('models_trained', [])
    
    if not models:
        print(f"\n📭 No models trained yet.")
        return
    
    print(f"\n📊 Total models: {len(models)}\n")
    
    # Table header
    print(f"{'Model':25s} {'Train':>7s} {'Val':>7s} {'F1':>7s} {'Gap':>7s} {'Home':>6s} {'Away':>6s} {'Total':>6s}")
    print("─" * 80)
    
    for m in models:
        key = m['key']
        n_train = m.get('samples_train', 0)
        n_val = m.get('samples_val', 0)
        f1 = m.get('val_f1', 0)
        gap = m.get('train_val_gap', 0)
        
        reg_mae = m.get('regression_mae', {})
        mae_home = reg_mae.get('home', '-')
        mae_away = reg_mae.get('away', '-')
        mae_total = reg_mae.get('total', '-')
        
        print(f"{key:25s} {n_train:>7,} {n_val:>7,} {f1:>7.3f} {gap:>7.3f} {str(mae_home):>6s} {str(mae_away):>6s} {str(mae_total):>6s}")


def cmd_evaluate(args):
    """Run evaluation."""
    from training.v13 import eval_v13
    
    print("\n" + "="*80)
    print("📈 V13 EVALUATION")
    print("="*80)
    
    eval_v13.main()


def interactive_menu():
    """Show interactive menu."""
    while True:
        print("\n" + "="*80)
        print("🏀 V13 BASKETBALL PREDICTION SYSTEM")
        print("="*80)
        print("\n📋 Available Commands:")
        print("  1. Train models")
        print("  2. Run inference on a match")
        print("  3. Check model status")
        print("  4. Show configuration")
        print("  5. Show dataset info")
        print("  6. List trained models")
        print("  7. Check for data leakage")
        print("  8. Show diagnostic plots")
        print("  9. Run evaluation")
        print("  10. Exit")
        print("\n" + "─"*80)
        
        choice = input("\nSelect option (1-10): ").strip()
        
        if choice == '1':
            print("\n🎯 TRAINING OPTIONS:")
            print("  a. Full training (with tuning)")
            print("  b. Fast training (skip tuning)")
            print("  c. Train specific subset")
            print("  d. Back to menu")
            
            subchoice = input("\nSelect option (a-d): ").strip().lower()
            
            if subchoice == 'a':
                cmd_train(argparse.Namespace(skip_tuning=False, subset=None))
            elif subchoice == 'b':
                cmd_train(argparse.Namespace(skip_tuning=True, subset=None))
            elif subchoice == 'c':
                subset = input("Enter subset (e.g., q3_high): ").strip()
                cmd_train(argparse.Namespace(skip_tuning=True, subset=subset))
            # else: back to menu
            
        elif choice == '2':
            match_id = input("Enter match ID: ").strip()
            target = input("Target quarter (q3/q4) [q3]: ").strip() or 'q3'
            cmd_infer(argparse.Namespace(match_id=match_id, target=target))
            
        elif choice == '3':
            cmd_status(argparse.Namespace())
            
        elif choice == '4':
            cmd_show_config(argparse.Namespace())
            
        elif choice == '5':
            cmd_show_dataset(argparse.Namespace())
            
        elif choice == '6':
            cmd_list_models(argparse.Namespace())
            
        elif choice == '7':
            cmd_check_leakage(argparse.Namespace())
            
        elif choice == '8':
            cmd_show_plots(argparse.Namespace())
            
        elif choice == '9':
            cmd_evaluate(argparse.Namespace())
            
        elif choice == '10':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid option. Please select 1-9.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="V13 Basketball Prediction System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/v13/cli.py train                  # Full training
  python training/v13/cli.py train --skip-tuning    # Fast training
  python training/v13/cli.py infer 15736636 --target q3  # Inference
  python training/v13/cli.py status                 # Check status
  python training/v13/cli.py config                 # Show config
  python training/v13/cli.py dataset                # Dataset info
  python training/v13/cli.py models                 # List models
  python training/v13/cli.py leakage                # Check leakage
  python training/v13/cli.py plots                  # Show plots
  python training/v13/cli.py                        # Interactive menu
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train V13 models')
    train_parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning')
    train_parser.add_argument('--subset', type=str, default=None, help='Train specific subset (e.g., q3_high)')
    train_parser.set_defaults(func=cmd_train)
    
    # Infer
    infer_parser = subparsers.add_parser('infer', help='Run inference on a match')
    infer_parser.add_argument('match_id', type=str, help='Match ID to predict')
    infer_parser.add_argument('--target', type=str, default='q3', choices=['q3', 'q4'], help='Target quarter')
    infer_parser.set_defaults(func=cmd_infer)
    
    # Status
    status_parser = subparsers.add_parser('status', help='Check model status')
    status_parser.set_defaults(func=cmd_status)
    
    # Config
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.set_defaults(func=cmd_show_config)
    
    # Dataset
    dataset_parser = subparsers.add_parser('dataset', help='Show dataset info')
    dataset_parser.set_defaults(func=cmd_show_dataset)
    
    # Models
    models_parser = subparsers.add_parser('models', help='List trained models')
    models_parser.set_defaults(func=cmd_list_models)
    
    # Leakage
    leakage_parser = subparsers.add_parser('leakage', help='Check for data leakage')
    leakage_parser.set_defaults(func=cmd_check_leakage)
    
    # Plots
    plots_parser = subparsers.add_parser('plots', help='Show diagnostic plots')
    plots_parser.set_defaults(func=cmd_show_plots)
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    args = parser.parse_args()
    
    if args.command is None:
        # No command specified, show interactive menu
        interactive_menu()
    else:
        # Execute command
        args.func(args)


if __name__ == "__main__":
    main()
