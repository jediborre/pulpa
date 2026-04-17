import json
from pathlib import Path
HERE = Path(__file__).parent
base = json.loads((HERE / "test_roi_v16_BASELINE.json").read_text(encoding="utf-8"))
new = json.loads((HERE / "test_roi_v16.json").read_text(encoding="utf-8"))

print("COMPARACION BASELINE -> MEJORADO (holdout)")
print("-" * 60)
for section in ("global", "portfolio"):
    b, n = base[section], new[section]
    print(f"[{section.upper()}]")
    print(f"  n_bets   : {b['n_bets']:>4}  ->  {n['n_bets']:>4}")
    print(f"  wins     : {b['wins']:>4}  ->  {n['wins']:>4}")
    print(f"  hit_rate : {b['hit_rate']:.4f} -> {n['hit_rate']:.4f}  (delta={n['hit_rate']-b['hit_rate']:+.4f})")
    print(f"  ROI      : {b['roi']:+.4f} -> {n['roi']:+.4f}  (delta={n['roi']-b['roi']:+.4f})")
    print(f"  P&L      : {b['pnl']:+.2f}u -> {n['pnl']:+.2f}u  (delta={n['pnl']-b['pnl']:+.2f}u)")
    print()

def stats(d):
    gs = d.get("groups", [])
    gaps = [g["train_val_gap"] for g in gs if g.get("train_val_gap") is not None]
    avg = sum(gaps) / max(len(gaps), 1)
    leaked = sum(1 for g in gaps if g > 0.15)
    return avg, leaked, len(gaps)

ba, bl, bt = stats(base)
na, nl, nt = stats(new)
print(f"train_val_gap promedio: {ba:.3f} -> {na:.3f}   (delta={na-ba:+.3f})")
print(f"modelos con leak (>0.15): {bl}/{bt} -> {nl}/{nt}")
