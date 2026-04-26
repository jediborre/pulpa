"""
cli.py - Entrypoint unificado para V17.

Ejemplos:

  # Entrenar todo:
  python -m training.v17.cli train
  python -m training.v17.cli train --no-cache

  # Backtest sobre holdout con odds 1.40 (default):
  python -m training.v17.cli eval

  # Inferencia sobre un partido (JSON stdin):
  echo '{ "match_id": "abc", "target": "q3", "league": "NBA",
          "quarter_scores": {"q1_home": 20, "q1_away": 18,
                             "q2_home": 25, "q2_away": 26},
          "graph_points": [...], "pbp_events": [...] }' |
    python -m training.v17.cli infer

  # Listar ligas entrenadas y su ROI de holdout:
  python -m training.v17.cli leagues

  # Exportar config + overrides activos:
  python -m training.v17.cli config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.v17 import config, league_overrides


def _cmd_train(args):
    from training.v17 import train
    train.run_training(
        use_cache=not args.no_cache,
        verbose=args.verbose,
        train_days=args.train_days,
        val_days=args.val_days,
        cal_days=args.cal_days,
        holdout_days=args.holdout_days,
        min_samples_train=args.min_samples_train,
        min_samples_val=args.min_samples_val,
        active_days=args.active_days,
        train_end_date=args.train_end_date,
        val_end_date=args.val_end_date,
        cal_end_date=args.cal_end_date,
        use_league_activation=(False if args.no_league_activation else None),
    )


def _cmd_eval(args):
    from training.v17 import evaluate
    evaluate.run_backtest(
        use_cache=not args.no_cache,
        odds=args.odds,
        use_holdout=not args.full,
        snapshot_only_prod=not args.all_snapshots,
    )


def _cmd_infer(args):
    from training.v17.inference import V15Engine
    raw = sys.stdin.read()
    if not raw.strip():
        print("[error] se esperaba JSON por stdin", file=sys.stderr)
        sys.exit(2)
    payload = json.loads(raw)
    engine = V15Engine.load()
    pred = engine.predict(
        match_id=payload["match_id"],
        target=payload["target"],
        league=payload["league"],
        quarter_scores=payload.get("quarter_scores", {}),
        graph_points=payload.get("graph_points", []),
        pbp_events=payload.get("pbp_events", []),
    )
    print(pred.to_json())


def _cmd_leagues(args):
    from training.v17.inference import SUMMARY_PATH
    if not Path(SUMMARY_PATH).exists():
        print(f"[error] no existe {SUMMARY_PATH}. Corre `cli train` primero.",
              file=sys.stderr)
        sys.exit(1)
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    for m in summary.get("models", []):
        if m.get("skipped"):
            continue
        rows.append({
            "league": m["league"],
            "target": m["target"],
            "n_train": m["n_train"],
            "threshold": m["threshold"]["threshold"],
            "val_roi": m["threshold"]["roi"],
            "val_hit": m["threshold"]["hit_rate"],
            "val_n_bets": m["threshold"]["n_bets"],
            "holdout_roi": (m.get("holdout_betting") or {}).get("roi"),
            "holdout_hit": (m.get("holdout_betting") or {}).get("hit_rate"),
        })
    rows.sort(key=lambda r: (-(r["holdout_roi"] or -99), -(r["n_train"])))
    print(f"\n{'liga':30s} {'tgt':4s} {'n_tr':>6s} {'thr':>5s} "
          f"{'val_roi':>8s} {'val_hit':>8s} {'ho_roi':>8s} {'ho_hit':>8s}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['league'][:30]:30s} {r['target']:4s} {r['n_train']:>6d} "
            f"{r['threshold']:>5.2f} "
            f"{(r['val_roi'] or 0):>8.3f} {(r['val_hit'] or 0):>8.3f} "
            f"{(r['holdout_roi'] or 0):>8.3f} {(r['holdout_hit'] or 0):>8.3f}"
        )


def _cmd_plots(args):
    from training.v17 import plots
    plots.generate_all(
        use_cache=not args.no_cache,
        include_inference_based=not args.skip_inference,
    )


def _cmd_test_roi(args):
    from training.v17 import test_roi
    test_roi.run(
        odds=args.odds,
        top_n=args.top,
        min_bets=args.min_bets,
        initial_bank=args.initial_bank,
        bet_size=args.bet_size,
        emit_csv=not args.no_csv,
    )


def _cmd_config(args):
    snap = {
        k: getattr(config, k)
        for k in dir(config) if k.isupper() and not k.startswith("_")
    }
    print("CONFIG")
    print(json.dumps(snap, indent=2, default=str))
    print("\nLEAGUE OVERRIDES")
    print(json.dumps(league_overrides.LEAGUE_OVERRIDES, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="v17", description="V17 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Entrena modelos por liga")
    p_train.add_argument("--no-cache", action="store_true")
    p_train.add_argument("--verbose", action="store_true", default=True)
    p_train.add_argument("--train-days", type=int, default=None,
                         help="Override TRAIN_DAYS (default=config)")
    p_train.add_argument("--val-days", type=int, default=None,
                         help="Override VAL_DAYS (default=config)")
    p_train.add_argument("--cal-days", type=int, default=None,
                         help="Dias explicitos para calibration")
    p_train.add_argument("--holdout-days", type=int, default=None,
                         help="Dias de holdout; 0 = modo produccion (sin holdout)")
    p_train.add_argument("--min-samples-train", type=int, default=None,
                         help="Umbral de muestras por liga para entrenar")
    p_train.add_argument("--min-samples-val", type=int, default=None,
                         help="Umbral de muestras por liga para validacion")
    p_train.add_argument("--active-days", type=int, default=None,
                         help="Filtra a ligas con partidos en ultimos N dias")
    p_train.add_argument("--train-end-date", type=str, default=None,
                         help="Fecha exclusiva fin de train YYYY-MM-DD")
    p_train.add_argument("--val-end-date", type=str, default=None,
                         help="Fecha exclusiva fin de val YYYY-MM-DD")
    p_train.add_argument("--cal-end-date", type=str, default=None,
                         help="Fecha exclusiva fin de calibration YYYY-MM-DD")
    p_train.add_argument("--no-league-activation", action="store_true",
                         help="Desactiva el selector inteligente de ligas")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="Backtest sobre holdout")
    p_eval.add_argument("--no-cache", action="store_true")
    p_eval.add_argument("--odds", type=float, default=config.DEFAULT_ODDS)
    p_eval.add_argument("--full", action="store_true",
                        help="Evalua todo el dataset")
    p_eval.add_argument("--all-snapshots", action="store_true")
    p_eval.set_defaults(func=_cmd_eval)

    p_inf = sub.add_parser("infer", help="Inferencia live (JSON por stdin)")
    p_inf.set_defaults(func=_cmd_infer)

    p_lg = sub.add_parser("leagues", help="Lista ligas entrenadas")
    p_lg.set_defaults(func=_cmd_leagues)

    p_cfg = sub.add_parser("config", help="Muestra config y overrides activos")
    p_cfg.set_defaults(func=_cmd_config)

    p_plots = sub.add_parser("plots", help="Genera graficas de diagnostico")
    p_plots.add_argument("--no-cache", action="store_true")
    p_plots.add_argument("--skip-inference", action="store_true",
                         help="No corre inferencia para calibration/probas")
    p_plots.set_defaults(func=_cmd_plots)

    p_troi = sub.add_parser("test-roi",
                            help="Test rapido de ROI sobre holdout con resumen legible")
    p_troi.add_argument("--odds", type=float, default=config.DEFAULT_ODDS)
    p_troi.add_argument("--top", type=int, default=20)
    p_troi.add_argument("--min-bets", type=int, default=10,
                        help="Descarta ligas con menos apuestas en holdout")
    p_troi.add_argument("--initial-bank", type=float, default=1000.0)
    p_troi.add_argument("--bet-size", type=float, default=100.0)
    p_troi.add_argument("--no-csv", action="store_true")
    p_troi.set_defaults(func=_cmd_test_roi)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

