import json
from pathlib import Path

HERE = Path(__file__).parent
s = json.loads((HERE / ".." / "model_outputs" / "training_summary_v15.json").read_text(encoding="utf-8"))

print("PRODUCCION V15")
print("=" * 75)
print("run_params :", json.dumps(s.get("run_params", {})))
print("splits     :", s.get("splits"))
print("ligas entrenadas:", s.get("n_leagues_trained"))
print("(liga, target) skipped:", s.get("n_leagues_skipped"))
print()
print(f"{'liga':35s} {'tgt':4s} {'n_tr':>6s} {'n_vl':>5s} {'thr':>5s} "
      f"{'val_roi':>8s} {'val_hit':>8s} {'gap':>6s}")
print("-" * 85)
models = sorted(
    [m for m in s["models"] if not m.get("skipped")],
    key=lambda m: (-(m.get("threshold", {}).get("roi") or 0), -m["n_train"]),
)
for m in models:
    thr = m["threshold"]
    gap = m.get("train_val_gap")
    print(
        f"{m['league'][:35]:35s} {m['target']:4s} {m['n_train']:>6d} "
        f"{m.get('n_val', 0):>5d} "
        f"{thr['threshold']:>5.2f} {thr['roi']:>+8.3f} {thr['hit_rate']:>8.3f} "
        f"{(gap if gap is not None else 0):>6.3f}"
    )
