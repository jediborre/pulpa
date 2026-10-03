"""
Wrapper para test_roi que permite override de HOLDOUT_DAYS y ademas acepta
`--holdout-last-days N`, que fuerza el holdout a ser exactamente los
ultimos N dias del dataset (para comparaciones A/B justas entre runs con
distintos run_params).

Uso:
  python -m training.v16.test_roi_cli --holdout-last-days 7 --suffix BASELINE_h7
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from training.v16 import config, test_roi as troi
from training.v16 import dataset as ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", type=float, default=config.DEFAULT_ODDS)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-bets", type=int, default=3)
    ap.add_argument("--holdout-days", type=int, default=None)
    ap.add_argument("--holdout-last-days", type=int, default=None,
                    help="Fuerza holdout = ultimas N dias (cutoff > newest-N).")
    ap.add_argument("--train-days", type=int, default=None)
    ap.add_argument("--val-days", type=int, default=None)
    ap.add_argument("--cal-days", type=int, default=None)
    ap.add_argument("--suffix", default=None,
                    help="Sufijo para outputs (test_roi_v16_<suffix>.json)")
    args = ap.parse_args()

    # Monkey-patch split_temporal para forzar override
    _original = ds.split_temporal

    def _patched(samples, **kw):
        if args.holdout_last_days is not None:
            dated = [(s, ds._parse_date(s.date)) for s in samples]
            dated = [(s, d) for s, d in dated if d is not None]
            newest = max(d for _, d in dated)
            cutoff = newest - timedelta(days=args.holdout_last_days)
            buckets = {"train": [], "val": [], "cal": [], "holdout": []}
            for s, d in dated:
                if d > cutoff:
                    buckets["holdout"].append(s)
                else:
                    # reutilizamos la logica original para train/val/cal
                    buckets["cal"].append(s)  # placeholder, no se usa en test-roi
            return buckets
        kw.setdefault("train_days", args.train_days or config.TRAIN_DAYS)
        kw.setdefault("val_days", args.val_days or config.VAL_DAYS)
        kw.setdefault("cal_days", args.cal_days or config.CAL_DAYS)
        kw.setdefault("holdout_days", args.holdout_days if args.holdout_days is not None else config.HOLDOUT_DAYS)
        return _original(samples, **kw)

    ds.split_temporal = _patched
    try:
        troi.run(
            odds=args.odds, top_n=args.top,
            min_bets=args.min_bets, emit_csv=False,
        )
    finally:
        ds.split_temporal = _original

    # Opcionalmente renombrar outputs si se paso --suffix
    if args.suffix:
        src = Path(__file__).parent / "reports" / "test_roi_v16.json"
        dst = Path(__file__).parent / "reports" / f"test_roi_v16_{args.suffix}.json"
        if src.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"[test-roi-cli] copiado a {dst}")


if __name__ == "__main__":
    main()
