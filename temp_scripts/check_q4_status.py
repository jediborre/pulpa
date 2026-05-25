import sqlite3
from datetime import datetime, timezone, timedelta

conn = sqlite3.connect('match/matches.db')
conn.row_factory = sqlite3.Row

names = ['BC TSU', 'Batumi', 'Juventus', 'Rytas', 'Arges', 'Sibiu', 'Dabrowa', 'Legia', 'Neptun', 'Gargzd', 'Amman', 'Faisaly', 'Filadel', 'IDEC', 'Engenharia', 'PAOK', 'Hashd', 'Degla']
like_clauses = ' OR '.join([f"(home_team LIKE '%{n}%' OR away_team LIKE '%{n}%')" for n in names])

rows = conn.execute(f'''
    SELECT match_id, home_team, away_team, event_date, scheduled_utc_ts,
           status, q3_checked, q4_checked, q4_signal, q4_notified
    FROM bet_monitor_schedule
    WHERE event_date = '2026-05-21' AND ({like_clauses})
    ORDER BY scheduled_utc_ts
''').fetchall()

for r in rows:
    sched = datetime.fromtimestamp(r['scheduled_utc_ts'], tz=timezone.utc) + timedelta(hours=-6)
    print(f"{sched.strftime('%H:%M')} | {r['home_team']} vs {r['away_team']} | status={r['status']} | q4_checked={r['q4_checked']} | q4_signal={r['q4_signal']} | notified={r['q4_notified']}")

conn.close()


for r in rows:
    sched = datetime.fromtimestamp(r['scheduled_utc_ts'], tz=timezone.utc) + timedelta(hours=-6)
    print(f"{sched.strftime('%H:%M')} | {r['home_team']} vs {r['away_team']} | status={r['status']} | q4_checked={r['q4_checked']} | q4_signal={r['q4_signal']} | q4_notified={r['q4_notified']}")

conn.close()
