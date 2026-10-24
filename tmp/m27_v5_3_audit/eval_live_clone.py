import json, pickle, sqlite3, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\App\Desktop\pulpa")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3_1"))

# load new model
import pickle as pk
MD = ROOT / "models" / "m27_v5_3_1" / "model_outputs"
models = {
    "xgb": pk.load(open(MD / "xgb_model.pkl", "rb")),
    "hgb": pk.load(open(MD / "hgb_model.pkl", "rb")),
    "lr": pk.load(open(MD / "lr_model.pkl", "rb")),
    "reg": pk.load(open(MD / "reg_model.pkl", "rb")),
}
iso = pk.load(open(MD / "calibrator.pkl", "rb"))
from train_m27_v5_3_1_pipeline import p_sym, margin_sym
from models.m27_v5_3_1.features import mirror

# load dataset
D = pk.load(open(ROOT / "tmp" / "m27_v5_3_1" / "dataset_v5_3_1.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
anchor = np.array([m["anchor"] for m in meta])
mids = np.array([m["mid"] for m in meta])
lg = np.array([m["lg"] for m in meta])

# live bet match ids
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
live_rows = conn.execute(
    "SELECT l.match_id, l.signal_type, l.result, l.confidence, s.league "
    "FROM bet_monitor_log_v3 l LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id=l.match_id "
    "WHERE l.model_version='m27_v5_3' AND l.signal_type LIKE 'BET%'"
).fetchall()
conn.close()

live_ids = set(r["match_id"] for r in live_rows)
live_info = {r["match_id"]: dict(r) for r in live_rows}

# original model results for comparison
orig_results = {}
# (already in live_info: signal_type, result, confidence)

# For each live match, find anchor row in dataset, run new model
out = []
for mid in live_ids:
    idxs = np.where((mids == mid) & (anchor == True))[0]
    if len(idxs) == 0:
        # try any row
        idxs = np.where(mids == mid)[0]
    if len(idxs) == 0:
        continue
    i = idxs[0]
    Xi = X[i:i+1]
    p = float(iso.predict(p_sym(models, Xi))[0])
    m = float(margin_sym(models, Xi)[0])
    # bet decision
    bet = (p >= 0.75 and m >= 0) or (p <= 0.25 and m <= 0)
    pick_home = p >= 0.5
    # actual result
    info = live_info[mid]
    actual = info["result"]  # win/loss
    # determine if new model's bet would win: need actual home win in Q4
    # We have y for this row (1 if home won Q4)
    y_actual = int(y[i])
    if bet:
        won = (pick_home and y_actual == 1) or (not pick_home and y_actual == 0)
    else:
        won = None
    out.append({
        "match_id": mid,
        "league": info["league"],
        "orig_signal": info["signal_type"],
        "orig_result": actual,
        "orig_conf": info["confidence"],
        "new_p": round(p, 4),
        "new_margin": round(m, 3),
        "new_bet": bet,
        "new_pick_home": pick_home,
        "new_won": won,
        "actual_home_won": bool(y_actual == 1),
    })

# summary
n = len(out)
new_bets = [r for r in out if r["new_bet"]]
new_wins = [r for r in new_bets if r["new_won"]]
orig_bets = [r for r in out]  # all are bets in original
orig_wins = [r for r in out if r["orig_result"] == "win"]
orig_losses = [r for r in out if r["orig_result"] == "loss"]

print("Evaluated live matches:", n)
print("Original m27_v5_3: bets=%d wins=%d losses=%d WR=%.1f%%" % (
    len(orig_bets), len(orig_wins), len(orig_losses),
    100*len(orig_wins)/(len(orig_wins)+len(orig_losses))))
print("New m27_v5_3_1: bets=%d wins=%d losses=%d WR=%.1f%%" % (
    len(new_bets), len(new_wins), len(new_bets)-len(new_wins),
    100*len(new_wins)/len(new_bets) if new_bets else 0))

# save
import json
Path(ROOT / "tmp" / "m27_v5_3_audit" / "eval_live_clone.json").write_text(
    json.dumps({"summary": {
        "n_evaluated": n,
        "orig_bets": len(orig_bets), "orig_wins": len(orig_wins), "orig_losses": len(orig_losses),
        "new_bets": len(new_bets), "new_wins": len(new_wins), "new_losses": len(new_bets)-len(new_wins),
    }, "rows": out}, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved")
# print per-match
for r in sorted(out, key=lambda x: x["league"]):
    print("%-45s orig=%-4s conf=%.2f | new_p=%.3f new_bet=%s new_won=%s | actual_home_won=%s" % (
        r["league"][:45], r["orig_result"], r["orig_conf"], r["new_p"], r["new_bet"], r["new_won"], r["actual_home_won"]))
