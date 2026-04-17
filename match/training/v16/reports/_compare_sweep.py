import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = ["A_baseline", "B_med_train", "C_big_train"]

rows = []
for name in RUNS:
    path = HERE / f"test_roi_v16_{name}.json"
    if not path.exists():
        print(f"[warn] missing {path}")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    g = data["global"]
    gs = data.get("groups", [])
    gaps = [x["train_val_gap"] for x in gs if x.get("train_val_gap") is not None]
    avg_gap = sum(gaps) / max(len(gaps), 1)
    leaks = sum(1 for x in gaps if x > 0.15)
    n_models = len(gaps)
    n_trained_leagues = len({x["league"] for x in gs if x.get("train_val_gap") is not None})
    rows.append({
        "name": name,
        "n_bets": g["n_bets"],
        "wins": g["wins"],
        "hit": g["hit_rate"],
        "roi": g["roi"],
        "pnl": g["pnl"],
        "leagues": n_trained_leagues,
        "models": n_models,
        "avg_gap": avg_gap,
        "leaks": leaks,
    })

print(f"{'run':<14} {'bets':>5} {'wins':>5} {'hit':>7} {'ROI':>8} "
      f"{'P&L':>7} {'lgs':>4} {'models':>7} {'avg_gap':>8} {'leaks':>6}")
print("-" * 85)
for r in rows:
    print(f"{r['name']:<14} {r['n_bets']:>5} {r['wins']:>5} "
          f"{r['hit']:>7.4f} {r['roi']:>+8.4f} {r['pnl']:>+7.2f}u "
          f"{r['leagues']:>4} {r['models']:>7} "
          f"{r['avg_gap']:>8.3f} {r['leaks']:>6}")

print()
if rows:
    best = max(rows, key=lambda r: r["roi"])
    print(f">>> MEJOR POR ROI: {best['name']}  "
          f"(ROI={best['roi']:+.4f}, hit={best['hit']:.4f}, "
          f"bets={best['n_bets']})")
