"""
Resumen de la corrida PROD V16.

Imprime:
  - run_params reales (lo que se uso, no lo del config_snapshot).
  - splits, ligas entrenadas/skipped, segundos de training.
  - Tabla por (liga, target) ordenada por val_roi descendente.
  - El train_val_gap se lee del campo nuevo `train_val_gap`; si no existe
    (corridas viejas) se calcula como f1(train) - f1(val).

Uso:
  python match/training/v16/reports/_summarize_prod.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE / ".." / "model_outputs" / "training_summary_v16.json"
s = json.loads(SUMMARY.read_text(encoding="utf-8"))


def _gap_for(m: dict) -> float | None:
    g = m.get("train_val_gap")
    if g is not None:
        return float(g)
    tm = (m.get("train_metrics") or {}).get("f1")
    vm = (m.get("val_metrics") or {}).get("f1")
    if tm is None or vm is None:
        return None
    return float(tm) - float(vm)


def _gap_marker(gap: float | None) -> str:
    if gap is None:
        return ""
    if gap > 0.20:
        return "  <- LEAK"
    if gap > 0.15:
        return "  <- overfit"
    return ""


print("=" * 92)
print("PRODUCCION V16")
print("=" * 92)
print(f"trained_at      : {s.get('trained_at')}  ({s.get('training_seconds', 0):.1f}s)")
print(f"run_params      : {json.dumps(s.get('run_params', {}))}")
print(f"splits          : {s.get('splits')}")
print(f"pace_thresholds : {s.get('pace_thresholds')}")
print(f"ligas entrenadas: {s.get('n_leagues_trained')}")
print(f"(liga, target) skipped: {s.get('n_leagues_skipped')}")
print()
print(f"{'liga':35s} {'tgt':4s} {'n_tr':>6s} {'n_vl':>5s} {'thr':>5s} "
      f"{'val_roi':>8s} {'val_hit':>8s} {'gap':>6s}  flag")
print("-" * 92)

models = sorted(
    [m for m in s["models"] if not m.get("skipped")],
    key=lambda m: (-(m.get("threshold", {}).get("roi") or 0), -m["n_train"]),
)
positive_roi = 0
for m in models:
    thr = m["threshold"]
    gap = _gap_for(m)
    if (thr.get("roi") or 0) > 0:
        positive_roi += 1
    print(
        f"{m['league'][:35]:35s} {m['target']:4s} {m['n_train']:>6d} "
        f"{m.get('n_val', 0):>5d} "
        f"{thr['threshold']:>5.2f} {thr['roi']:>+8.3f} {thr['hit_rate']:>8.3f} "
        f"{(gap if gap is not None else 0):>+6.3f}{_gap_marker(gap)}"
    )

print("-" * 92)
print(f"modelos con val_roi > 0: {positive_roi} / {len(models)}")

# Resumen agregado de skipped por reason
skipped_by_reason: dict[str, int] = {}
for sk in s.get("skipped", []):
    r = sk.get("reason", "?")
    skipped_by_reason[r] = skipped_by_reason.get(r, 0) + 1
print("\nskipped por motivo:")
for r, n in sorted(skipped_by_reason.items(), key=lambda x: -x[1]):
    print(f"  {r:30s} {n:>4d}")
