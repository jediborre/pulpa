import sqlite3
import time
import sys, io

# Forzar UTF-8 en stdout para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MATCH_ID = "16067319"

conn = sqlite3.connect("match/matches.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Check schedule_v2 row
cur.execute(
    "SELECT match_id, home_team, away_team, scheduled_utc_ts, status, skip_reason, final_fetched FROM bet_monitor_schedule_v2 WHERE match_id = ?",
    (MATCH_ID,)
)
row = cur.fetchone()
if row:
    d = dict(row)
    now = time.time()
    elapsed = now - d["scheduled_utc_ts"]
    d["elapsed_hours"] = round(elapsed / 3600, 2)
    print("Schedule row:", d)
else:
    print(f"NOT FOUND in schedule_v2 for match {MATCH_ID}")

# Check bet logs in bet_monitor_log_v2
cur.execute(
    "SELECT * FROM bet_monitor_log_v2 WHERE match_id = ? ORDER BY created_at DESC LIMIT 10",
    (MATCH_ID,)
)
rows = cur.fetchall()
if rows:
    print(f"\nBet logs (bet_monitor_log_v2):")
    for r in rows:
        print(" ", dict(r))
else:
    print(f"\nNo bet logs for match {MATCH_ID} in bet_monitor_log_v2")

# Print all active (non-done) matches
cur.execute(
    "SELECT match_id, home_team, away_team, scheduled_utc_ts, status FROM bet_monitor_schedule_v2 WHERE status NOT IN ('done', 'discarded') ORDER BY scheduled_utc_ts"
)
rows = cur.fetchall()
print("\nActive matches in schedule_v2:")
now = time.time()
for r in rows:
    d = dict(r)
    elapsed = int(now - d["scheduled_utc_ts"])
    home = d['home_team'].encode('ascii', errors='replace').decode()
    away = d['away_team'].encode('ascii', errors='replace').decode()
    print(f"  {d['match_id']} {home} vs {away} | status={d['status']} | elapsed={elapsed}s ({elapsed//3600}h {(elapsed%3600)//60}m)")
