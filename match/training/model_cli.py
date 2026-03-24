"""Unified CLI for model training workflows (EDA, train, compare)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"

SCRIPT_EDA = TRAINING_DIR / "eda.py"
SCRIPT_TRAIN_V1 = TRAINING_DIR / "train_q3_q4_models.py"
SCRIPT_TRAIN_V2 = TRAINING_DIR / "train_q3_q4_models_v2.py"
SCRIPT_TRAIN_V3 = TRAINING_DIR / "train_q3_q4_models_v3.py"
SCRIPT_TRAIN_V4 = TRAINING_DIR / "train_q3_q4_models_v4.py"
SCRIPT_COMPARE = TRAINING_DIR / "compare_model_versions.py"
SCRIPT_INFER = TRAINING_DIR / "infer_match.py"
SCRIPT_CALIBRATE = TRAINING_DIR / "calibrate_gate.py"
SCRIPT_MAIN_CLI = ROOT / "cli.py"

COMPARE_JSON = TRAINING_DIR / "model_comparison" / "version_comparison.json"


def _run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}"
        )


def cmd_eda(_: argparse.Namespace) -> None:
    _run_script(SCRIPT_EDA)


def cmd_train_v1(_: argparse.Namespace) -> None:
    _run_script(SCRIPT_TRAIN_V1)


def cmd_train_v2(_: argparse.Namespace) -> None:
    _run_script(SCRIPT_TRAIN_V2)


def cmd_train_v3(_: argparse.Namespace) -> None:
    _run_script(SCRIPT_TRAIN_V3)


def cmd_train_v4(_: argparse.Namespace) -> None:
    _run_script(SCRIPT_TRAIN_V4)


def cmd_compare(_: argparse.Namespace) -> None:
    _run_script(SCRIPT_COMPARE)


def cmd_all(args: argparse.Namespace) -> None:
    if not args.skip_eda:
        print("[model-cli] running: eda")
        _run_script(SCRIPT_EDA)

    if not args.skip_v1:
        print("[model-cli] running: train-v1")
        _run_script(SCRIPT_TRAIN_V1)

    if not args.skip_v2:
        print("[model-cli] running: train-v2")
        _run_script(SCRIPT_TRAIN_V2)

    if not args.skip_v4:
        print("[model-cli] running: train-v4")
        _run_script(SCRIPT_TRAIN_V4)

    print("[model-cli] running: compare")
    _run_script(SCRIPT_COMPARE)


def _pick_entry(target_blob: dict, metric: str) -> dict | None:
    metric_map = {
        "accuracy": "best_accuracy",
        "f1": "best_f1",
        "log_loss": "best_log_loss",
    }
    key = metric_map[metric]
    value = target_blob.get(key)
    return value if isinstance(value, dict) else None


def cmd_summary(args: argparse.Namespace) -> None:
    if not COMPARE_JSON.exists():
        raise FileNotFoundError(
            "Comparison summary not found. Run 'compare' first."
        )

    with COMPARE_JSON.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    print("[model-cli] summary")
    print(f"metric_for_pick={args.metric}")

    for target in ("q3", "q4"):
        target_blob = blob.get(target, {})
        entry = _pick_entry(target_blob, args.metric)
        if not entry:
            print(f"- {target}: no data")
            continue

        model = entry.get("model")
        version = entry.get("version")
        acc = entry.get("accuracy")
        f1 = entry.get("f1", "n/a")
        ll = entry.get("log_loss")

        print(
            f"- {target}: version={version} model={model} "
            f"acc={acc} f1={f1} log_loss={ll}"
        )


def cmd_infer(args: argparse.Namespace) -> None:
    if not SCRIPT_INFER.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT_INFER}")

    cmd = [
        sys.executable,
        str(SCRIPT_INFER),
        str(args.match_id),
        "--metric",
        args.metric,
    ]
    if args.force_version != "auto":
        cmd.extend(["--force-version", args.force_version])
    if args.no_fetch:
        cmd.append("--no-fetch")
    if getattr(args, "refresh", False):
        cmd.append("--refresh")
    if args.json:
        cmd.append("--json")

    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}"
        )


def cmd_calibrate(args: argparse.Namespace) -> None:
    if not SCRIPT_CALIBRATE.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT_CALIBRATE}")

    cmd = [
        sys.executable,
        str(SCRIPT_CALIBRATE),
        "--metric",
        args.metric,
        "--limit",
        str(args.limit),
        "--odds",
        str(args.odds),
        "--min-coverage",
        str(args.min_coverage),
    ]
    if args.json:
        cmd.append("--json")

    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}"
        )


def cmd_eval_date(args: argparse.Namespace) -> None:
    if not SCRIPT_MAIN_CLI.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT_MAIN_CLI}")

    cmd = [
        sys.executable,
        str(SCRIPT_MAIN_CLI),
        "eval-date",
        "--date",
        args.date,
        "--metric",
        args.metric,
        "--force-version",
        args.force_version,
        "--odds",
        str(args.odds),
    ]
    if args.limit_matches is not None:
        cmd.extend(["--limit-matches", str(args.limit_matches)])
    if args.result_tag:
        cmd.extend(["--result-tag", args.result_tag])
    if args.json:
        cmd.append("--json")

    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}"
        )


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _ask_metric(default: str = "f1") -> str:
    raw = input("Metric (accuracy|f1|log_loss) [f1]: ").strip().lower()
    if raw in ("accuracy", "f1", "log_loss"):
        return raw
    return default


def _interactive_menu() -> None:
    options = {
        "1": "eda",
        "2": "train-v1",
        "3": "train-v2",
        "4": "train-v3",
        "5": "train-v4",
        "6": "compare",
        "7": "summary",
        "8": "infer",
        "9": "all",
        "10": "calibrate",
        "11": "eval-date",
        "0": "exit",
    }

    while True:
        print("\n=== Model CLI Interactive Menu ===")
        print("1) EDA")
        print("2) Train V1")
        print("3) Train V2")
        print("4) Train V3 (dynamic live)")
        print("5) Train V4 (pressure/comeback features)")
        print("6) Compare V1 vs V2 vs V4")
        print("7) Summary")
        print("8) Infer by match_id")
        print("9) Run all")
        print("10) Calibrate gate thresholds")
        print("11) Evaluate one date")
        print("0) Exit")

        choice = input("Select option: ").strip()
        action = options.get(choice)
        if not action:
            print("Invalid option. Try again.")
            continue
        if action == "exit":
            print("Bye.")
            return

        try:
            if action == "eda":
                cmd_eda(argparse.Namespace())
            elif action == "train-v1":
                cmd_train_v1(argparse.Namespace())
            elif action == "train-v2":
                cmd_train_v2(argparse.Namespace())
            elif action == "train-v3":
                cmd_train_v3(argparse.Namespace())
            elif action == "train-v4":
                cmd_train_v4(argparse.Namespace())
            elif action == "compare":
                cmd_compare(argparse.Namespace())
            elif action == "summary":
                metric = _ask_metric()
                cmd_summary(argparse.Namespace(metric=metric))
            elif action == "infer":
                match_id = input("Match ID: ").strip()
                if not match_id:
                    print("Match ID is required.")
                    continue
                metric = _ask_metric()
                force_version = input(
                    "Force version (auto|v1|v2|v4|hybrid) [auto]: "
                ).strip().lower()
                if force_version not in (
                    "",
                    "auto",
                    "v1",
                    "v2",
                    "v4",
                    "hybrid",
                ):
                    print("Invalid force-version. Using auto.")
                    force_version = "auto"
                if not force_version:
                    force_version = "auto"
                no_fetch = _ask_yes_no("Disable auto-fetch if missing?", False)
                refresh = _ask_yes_no(
                    "Force refresh (re-fetch even if exists in DB)?", False
                )
                json_out = _ask_yes_no("Raw JSON output?", False)
                cmd_infer(
                    argparse.Namespace(
                        match_id=match_id,
                        metric=metric,
                        force_version=force_version,
                        no_fetch=no_fetch,
                        refresh=refresh,
                        json=json_out,
                    )
                )
            elif action == "all":
                skip_eda = _ask_yes_no("Skip EDA?", False)
                skip_v1 = _ask_yes_no("Skip V1?", False)
                skip_v2 = _ask_yes_no("Skip V2?", False)
                skip_v4 = _ask_yes_no("Skip V4?", False)
                cmd_all(
                    argparse.Namespace(
                        skip_eda=skip_eda,
                        skip_v1=skip_v1,
                        skip_v2=skip_v2,
                        skip_v4=skip_v4,
                    )
                )
            elif action == "calibrate":
                metric = _ask_metric()
                raw_limit = input("Limit matches [3000]: ").strip()
                raw_odds = input("Assumed decimal odds [1.91]: ").strip()
                raw_cov = input("Min coverage [0.08]: ").strip()
                json_out = _ask_yes_no("Raw JSON output?", False)

                limit = int(raw_limit) if raw_limit else 3000
                odds = float(raw_odds) if raw_odds else 1.91
                min_cov = float(raw_cov) if raw_cov else 0.08

                cmd_calibrate(
                    argparse.Namespace(
                        metric=metric,
                        limit=limit,
                        odds=odds,
                        min_coverage=min_cov,
                        json=json_out,
                    )
                )
            elif action == "eval-date":
                date_txt = input("Date (YYYY-MM-DD): ").strip()
                if not date_txt:
                    print("Date is required.")
                    continue
                metric = _ask_metric()
                force_version = input(
                    "Force version (auto|v1|v2|v4|hybrid) [hybrid]: "
                ).strip().lower()
                if not force_version:
                    force_version = "hybrid"
                if force_version not in (
                    "auto",
                    "v1",
                    "v2",
                    "v4",
                    "hybrid",
                ):
                    print("Invalid force-version. Using hybrid.")
                    force_version = "hybrid"

                raw_lim = input("Limit matches [none]: ").strip()
                raw_odds = input("Assumed decimal odds [1.91]: ").strip()
                result_tag = input(
                    "Result tag for DB columns [auto]: "
                ).strip()
                json_out = _ask_yes_no("Raw JSON output?", False)

                lim = int(raw_lim) if raw_lim else None
                odds = float(raw_odds) if raw_odds else 1.91
                if not result_tag:
                    result_tag = None

                cmd_eval_date(
                    argparse.Namespace(
                        date=date_txt,
                        metric=metric,
                        force_version=force_version,
                        limit_matches=lim,
                        odds=odds,
                        result_tag=result_tag,
                        json=json_out,
                    )
                )
        except Exception as exc:
            print(f"Error: {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified model workflow CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eda = sub.add_parser("eda", help="Run dataset EDA")
    p_eda.set_defaults(func=cmd_eda)

    p_v1 = sub.add_parser("train-v1", help="Train V1 models")
    p_v1.set_defaults(func=cmd_train_v1)

    p_v2 = sub.add_parser("train-v2", help="Train V2 models")
    p_v2.set_defaults(func=cmd_train_v2)

    p_v3 = sub.add_parser(
        "train-v3",
        help="Train V3 dynamic live models (minute-aware)",
    )
    p_v3.set_defaults(func=cmd_train_v3)

    p_v4 = sub.add_parser(
        "train-v4",
        help="Train V4 models with pressure/comeback features",
    )
    p_v4.set_defaults(func=cmd_train_v4)

    p_cmp = sub.add_parser("compare", help="Compare V1 vs V2 vs V4")
    p_cmp.set_defaults(func=cmd_compare)

    p_all = sub.add_parser(
        "all",
        help="Run EDA + V1 + V2 + V4 + compare",
    )
    p_all.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA step",
    )
    p_all.add_argument(
        "--skip-v1",
        action="store_true",
        help="Skip V1 training step",
    )
    p_all.add_argument(
        "--skip-v2",
        action="store_true",
        help="Skip V2 training step",
    )
    p_all.add_argument(
        "--skip-v4",
        action="store_true",
        help="Skip V4 training step",
    )
    p_all.set_defaults(func=cmd_all)

    p_sum = sub.add_parser(
        "summary",
        help="Show best model per target from comparison report",
    )
    p_sum.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric used to pick best model (default: f1)",
    )
    p_sum.set_defaults(func=cmd_summary)

    p_infer = sub.add_parser(
        "infer",
        help="Infer Q3/Q4 winner for one match_id (auto-fetch if missing)",
    )
    p_infer.add_argument("match_id", help="SofaScore match ID")
    p_infer.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric used to select the best model version",
    )
    p_infer.add_argument(
        "--force-version",
        choices=["auto", "v1", "v2", "v4", "hybrid"],
        default="auto",
        help="Override selected version (hybrid => q3=v2, q4=v4)",
    )
    p_infer.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch with scraper if match is missing in DB",
    )
    p_infer.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch and overwrite DB even if match already exists",
    )
    p_infer.add_argument(
        "--json",
        action="store_true",
        help="Return raw JSON output",
    )
    p_infer.set_defaults(func=cmd_infer)

    p_cal = sub.add_parser(
        "calibrate",
        help="Calibrate decision-gate thresholds from historical matches",
    )
    p_cal.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric used to select best model version",
    )
    p_cal.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Max recent matches to evaluate",
    )
    p_cal.add_argument(
        "--odds",
        type=float,
        default=1.91,
        help="Assumed decimal odds for ROI proxy",
    )
    p_cal.add_argument(
        "--min-coverage",
        type=float,
        default=0.08,
        help="Minimum desired BET coverage during optimization",
    )
    p_cal.add_argument(
        "--json",
        action="store_true",
        help="Return raw JSON output",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_eval = sub.add_parser(
        "eval-date",
        help="Discover+ingest+evaluate one date via root cli",
    )
    p_eval.add_argument(
        "--date",
        required=True,
        help="UTC date to evaluate (YYYY-MM-DD)",
    )
    p_eval.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric for model selection context",
    )
    p_eval.add_argument(
        "--force-version",
        choices=["auto", "v1", "v2", "v4", "hybrid"],
        default="hybrid",
        help="Inference policy",
    )
    p_eval.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Max finished matches to process from that date",
    )
    p_eval.add_argument(
        "--odds",
        type=float,
        default=1.91,
        help="Fixed decimal odds for ROI proxy",
    )
    p_eval.add_argument(
        "--result-tag",
        default=None,
        help=(
            "Tag for DB result columns "
            "(default: <force-version>_<metric>)"
        ),
    )
    p_eval.add_argument(
        "--json",
        action="store_true",
        help="Return raw JSON output",
    )
    p_eval.set_defaults(func=cmd_eval_date)

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        try:
            _interactive_menu()
            return
        except EOFError:
            # Non-interactive shell: fall through to argparse help.
            pass

    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
