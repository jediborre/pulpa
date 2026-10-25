import sqlite3, json
from datetime import datetime, timedelta
conn = sqlite3.connect(r"C:\Users\App\Desktop\pulpa\matches.db")
conn.row_factory = sqlite3.Row
# today's date
today = datetime.now().strftime("%Y-%m-%d")
out = []
out.append("=== Bets today (%s) ===" % today)
rows = conn.execute(
    "SELECT l.model_version, l.match_id, l.signal_type, l.picked_side, l.confidence, l.result, "
    "l.inference_minute, s.league, l.inference_json "
    "FROM bet_monitor_log_v3 l LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id=l.match_id "
    "WHERE date(l.created_at) = ? AND l.signal_type LIKE 'BET%' ORDER BY l.created_at",
    (today,)
).fetchall()
out.append("Total bets today: %d" % len(rows))
# group by result
from collections import Counter
rc = Counter(r["result"] for r in rows)
out.append("Results: %s" % dict(rc))
out.append("")
# losses detail
out.append("=== LOSSES ===")
for r in rows:
    if r["result"] != "loss":
        continue
    try:
        inf = json.loads(r["inference_json"] or "{}")
        raw = inf.get("raw_json", {})
    except Exception:
        raw = {}
    out.append("match=%s | model=%s | side=%s | conf=%.3f | min=%s | league=%s" % (
        r["match_id"], r["model_version"], r["picked_side"], r["confidence"],
        r["inference_minute"], r["league"]))
    out.append("   p_cal=%s margin=%s has_ml=%s toxic=%s stage=%s" % (
        raw.get("p_cal"), raw.get("margin"), raw.get("has_ml"), raw.get("toxic_league_blocked"),
        raw.get("snapshot_minute_used")))
    # features snippet
    feats = raw.get("features", {})
    if feats:
        out.append("   cur_diff=%s ht_diff=%s q3mom=%s recent5m=%s d_elo_ft=%s odds_ratio=%s" % (
            feats.get("cur_score_diff"), feats.get("ht_diff"), feats.get("q3_vs_ht_momentum"),
            feats.get("recent_pts_diff_5m"), feats.get("d_elo_ft"), feats.get("odds_log_ratio")))
conn.close()
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\today_bets.txt", "w", encoding="utf-8").write("\n".join(out))
print("done: %d bets" % len(rows))
