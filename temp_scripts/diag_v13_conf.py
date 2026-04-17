"""Test V13 inference on a handful of April matches to see raw confidence values."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'match'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'match', 'training'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'match'))

import sqlite3
from pathlib import Path

db_path = Path("matches.db")
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

# Get matches that had actual results saved (q3/q4 scores exist)
rows = conn.execute("""
    SELECT m.match_id, m.home_team, m.away_team, m.league, m.date
    FROM matches m
    JOIN eval_match_results e ON e.match_id = m.match_id
    WHERE m.date BETWEEN '2026-04-01' AND '2026-04-05'
    LIMIT 10
""").fetchall()
conn.close()

print(f"Testing {len(rows)} matches...\n")
print(f"{'League':<40} {'Q3 Signal':<15} {'Q3 Conf':>8} {'Q4 Signal':<15} {'Q4 Conf':>8}  Reason")
print("-" * 110)

from training.v13.infer_match_v13 import run_inference

for row in rows:
    mid = row["match_id"]
    label = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
    league = row["league"] or ""

    for target in ("q3", "q4"):
        try:
            res = run_inference(mid, target, fetch_missing=False)
            if res.get("ok"):
                pred = res["prediction"]
                sig = pred.winner_signal
                conf = pred.winner_confidence
                reason = pred.reasoning[:60] if hasattr(pred, 'reasoning') else "-"
            elif res.get("available") is False:
                sig = "unavail"
                conf = 0.0
                reason = res.get("reason", "")[:60]
            else:
                sig = "error"
                conf = 0.0
                reason = str(res.get("reason", ""))[:60]
        except Exception as e:
            sig = "EXCEPTION"
            conf = 0.0
            reason = str(e)[:60]

        if target == "q3":
            q3s, q3c, q3r = sig, conf, reason
        else:
            print(f"{league[:38]:<40} {q3s:<15} {q3c:>8.3f} {sig:<15} {conf:>8.3f}  {q3r}")
