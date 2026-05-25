import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import sqlite3, time

conn = sqlite3.connect('match/matches.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()
MATCH_ID = '16067319'

# Get the actual match schedule row
cur.execute(
    'SELECT match_id, home_team, away_team, scheduled_utc_ts, status, skip_reason, final_fetched FROM bet_monitor_schedule_v2 WHERE match_id = ?',
    (MATCH_ID,)
)
row = cur.fetchone()
if row:
    d = dict(row)
    now = time.time()
    elapsed = now - d['scheduled_utc_ts']
    print('Match:', d['home_team'], 'vs', d['away_team'])
    print('Status:', d['status'], '| skip_reason:', d['skip_reason'])
    print('Scheduled UTC ts:', d['scheduled_utc_ts'], '| Elapsed:', round(elapsed, 0), 's =', round(elapsed/3600, 2), 'h')
    print('3.5h threshold (12600s): elapsed > threshold =', elapsed > 12600)
    print('final_fetched:', d['final_fetched'])

# Check log_v2
cur.execute(
    'SELECT * FROM bet_monitor_log_v2 WHERE match_id = ? ORDER BY created_at DESC LIMIT 5',
    (MATCH_ID,)
)
rows = cur.fetchall()
if rows:
    for r in rows:
        print('log_v2:', dict(r))
else:
    print('No rows in bet_monitor_log_v2 for this match')
