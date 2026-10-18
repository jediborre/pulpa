"""
tmp/m27_v5_2_audit/a4_events_probe.py
Semantica real de match_events (time_seconds, points) y cobertura de match_events/play_by_play por año,
para entender por que pbp_3pt_diff_q1_q3 vale 0 en 100% del train.
"""
import sqlite3, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=60)
m = c.execute("select match_id from matches where date>='2025-03-01' and status_type='finished' and match_id in (select distinct match_id from match_events) limit 1").fetchone()[0]
print("match", m)
df = pd.read_sql("select quarter,seq,time,time_seconds,incident_type,subtype,team,points,home_score,away_score from match_events where match_id=? order by id limit 400", c, params=(m,))
print(df.head(25)); print(df.groupby("quarter").time_seconds.agg(["min","max","count"]))
print(df.groupby(["incident_type"]).points.agg(["count", lambda s: s.notna().sum()]))
print(pd.read_sql("select points, count(*) n from match_events where match_id=? group by points", c, params=(m,)))
print(pd.read_sql("select team, points, count(*) n from match_events where match_id=? and time_seconds<=1800 and points in (1,2,3) group by team, points", c, params=(m,)))
print(pd.read_sql("select incident_type, subtype, count(*) n, sum(points is not null) npts from match_events where match_id=? group by 1,2", c, params=(m,)))
print("play_by_play sample:"); print(pd.read_sql("select quarter,seq,time,team,points,home_score,away_score from play_by_play where match_id=? limit 6", c, params=(m,)))
