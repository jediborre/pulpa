# =====================================================================
# RESULTADOS EN VIVO DEL MONITOREO — m27_v5_3 (bet_monitor_log_v3)
# =====================================================================
import json, sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
OUT = Path(__file__).resolve().parent / "live_league_stats.json"

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
rows = conn.execute(
    """
    SELECT l.match_id, l.signal_type, l.result, l.confidence, l.inference_minute,
           s.league
    FROM bet_monitor_log_v3 l
    LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
    WHERE l.model_version = 'm27_v5_3' AND l.signal_type LIKE 'BET%'
    """
).fetchall()
conn.close()

agg = defaultdict(lambda: {"bets": 0, "wins": 0, "losses": 0, "pending": 0, "conf_sum": 0.0})
overall = {"bets": 0, "wins": 0, "losses": 0, "pending": 0, "conf_sum": 0.0}
for r in rows:
    lg = r["league"] or "?"
    res = r["result"]
    a = agg[lg]
    a["bets"] += 1
    a["conf_sum"] += float(r["confidence"] or 0)
    overall["bets"] += 1
    overall["conf_sum"] += float(r["confidence"] or 0)
    if res == "win":
        a["wins"] += 1; overall["wins"] += 1
    elif res == "loss":
        a["losses"] += 1; overall["losses"] += 1
    elif res == "pending":
        a["pending"] += 1; overall["pending"] += 1

leagues_out = []
for lg, a in agg.items():
    settled = a["wins"] + a["losses"]
    wr = (a["wins"] / settled * 100) if settled else None
    leagues_out.append({
        "league": lg,
        "bets": a["bets"],
        "wins": a["wins"],
        "losses": a["losses"],
        "pending": a["pending"],
        "settled": settled,
        "wr": round(wr, 1) if wr is not None else None,
        "avg_conf": round(a["conf_sum"] / a["bets"] * 100, 1) if a["bets"] else None,
    })
leagues_out.sort(key=lambda x: x["bets"], reverse=True)

settled_ov = overall["wins"] + overall["losses"]
result = {
    "model": "m27_v5_3",
    "source": "bet_monitor_log_v3 (monitoreo en vivo)",
    "overall": {
        "bets": overall["bets"],
        "wins": overall["wins"],
        "losses": overall["losses"],
        "pending": overall["pending"],
        "settled": settled_ov,
        "wr": round(overall["wins"] / settled_ov * 100, 1) if settled_ov else None,
        "avg_conf": round(overall["conf_sum"] / overall["bets"] * 100, 1) if overall["bets"] else None,
    },
    "n_leagues": len(leagues_out),
    "leagues": leagues_out,
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print("done: %d ligas, %d apuestas" % (len(leagues_out), overall["bets"]))
