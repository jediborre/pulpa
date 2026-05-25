import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import sqlite3

conn = sqlite3.connect('match/matches.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Find recent matches with FT data
cur.execute("""
    SELECT s.match_id, s.home_team, s.away_team, s.status,
           q.q4_home, q.q4_away,
           l.model_version, l.signal_type, l.actual_home_score, l.actual_away_score, l.result
    FROM bet_monitor_schedule_v2 s
    LEFT JOIN quarter_scores_v2 q ON q.match_id = s.match_id
    LEFT JOIN bet_monitor_log_v2 l ON l.match_id = s.match_id
    WHERE s.status = 'done' AND l.signal_type LIKE 'BET%'
    ORDER BY s.match_id DESC
    LIMIT 20
""")
rows = cur.fetchall()
print(f"{'Match ID':<12} {'Home':<30} {'Away':<30} {'Model':<10} {'Signal':<18} {'Q4 Home':<8} {'Q4 Away':<8} {'Bet H':<7} {'Bet A':<7} {'Result'}")
print("-" * 155)
for r in rows:
    d = dict(r)
    home = (d['home_team'] or '?')[:28]
    away = (d['away_team'] or '?')[:28]
    print(f"{d['match_id']:<12} {home:<30} {away:<30} {str(d['model_version']):<10} {str(d['signal_type']):<18} {str(d['q4_home']):<8} {str(d['q4_away']):<8} {str(d['actual_home_score']):<7} {str(d['actual_away_score']):<7} {str(d['result'])}")
