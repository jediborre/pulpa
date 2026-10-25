import json, pickle, sqlite3, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\App\Desktop\pulpa")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3_3"))

MD = ROOT / "models" / "m27_v5_3_3" / "model_outputs"
models = {
    "xgb": pickle.load(open(MD / "xgb_model.pkl", "rb")),
    "hgb": pickle.load(open(MD / "hgb_model.pkl", "rb")),
    "lr": pickle.load(open(MD / "lr_model.pkl", "rb")),
    "reg": pickle.load(open(MD / "reg_model.pkl", "rb")),
}
iso = pickle.load(open(MD / "calibrator.pkl", "rb"))
from train_m27_v5_3_3_pipeline import p_sym, margin_sym

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3_3" / "dataset_v5_3_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
anchor = np.array([m["anchor"] for m in meta])
mids = np.array([m["mid"] for m in meta])

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
live_rows = conn.execute(
    "SELECT l.match_id, l.signal_type, l.result, l.confidence, s.league "
    "FROM bet_monitor_log_v3 l LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id=l.match_id "
    "WHERE l.model_version='m27_v5_3' AND l.signal_type LIKE 'BET%'"
).fetchall()
conn.close()
live_info = {r["match_id"]: dict(r) for r in live_rows}
live_ids = set(live_info.keys())

out = []
for mid in live_ids:
    idxs = np.where((mids == mid) & (anchor == True))[0]
    if len(idxs) == 0:
        idxs = np.where(mids == mid)[0]
    if len(idxs) == 0:
        continue
    i = idxs[0]
    Xi = X[i:i+1]
    p = float(iso.predict(p_sym(models, Xi))[0])
    m = float(margin_sym(models, Xi)[0])
    bet = (p >= 0.75 and m >= 0) or (p <= 0.25 and m <= 0)
    pick_home = p >= 0.5
    y_actual = int(y[i])
    won = ((pick_home and y_actual == 1) or (not pick_home and y_actual == 0)) if bet else None
    info = live_info[mid]
    out.append({
        "match_id": mid,
        "league": info["league"],
        "orig_signal": info["signal_type"],
        "orig_result": info["result"],
        "orig_conf": info["confidence"],
        "new_p": round(p, 4),
        "new_margin": round(m, 3),
        "new_bet": bet,
        "new_pick_home": pick_home,
        "new_won": won,
        "actual_home_won": bool(y_actual == 1),
    })

n = len(out)
new_bets = [r for r in out if r["new_bet"]]
new_wins = [r for r in new_bets if r["new_won"]]
orig_wins = [r for r in out if r["orig_result"] == "win"]
orig_losses = [r for r in out if r["orig_result"] == "loss"]
summary = {
    "n_evaluated": n,
    "orig_bets": len(out), "orig_wins": len(orig_wins), "orig_losses": len(orig_losses),
    "new_bets": len(new_bets), "new_wins": len(new_wins), "new_losses": len(new_bets)-len(new_wins),
}
Path(ROOT / "tmp" / "m27_v5_3_audit" / "eval_live_v5_3_3.json").write_text(
    json.dumps({"summary": summary, "rows": out}, indent=2, ensure_ascii=False), encoding="utf-8")
print("evaluated:", n, "orig WR:", round(100*len(orig_wins)/(len(orig_wins)+len(orig_losses)),1), "new WR:", round(100*len(new_wins)/len(new_bets),1) if new_bets else 0)
