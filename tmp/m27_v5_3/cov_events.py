"""
tmp/m27_v5_3/cov_events.py
Cobertura de match_events (goles con time_seconds) por cuarto y año, y de match_odds ML/spread/total, graph_points,
para decidir el diseño multi-snapshot de m27_v5_3 y el alcance del backfill.
"""
import sqlite3, sys, time, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=300)
t = time.time()
print(c.execute("select count(*) from match_events").fetchone(), time.time()-t)
print(pd.read_sql("select name from sqlite_master where type='index' and tbl_name in ('match_events','match_odds','graph_points','matches','quarter_scores')", c).name.tolist())
ev = pd.read_sql("select match_id, quarter, count(*) n, sum(time_seconds is null) nnull, sum(points>0) ngoal, max(time_seconds) mx from match_events where points is not null group by 1,2", c)
print("filas", len(ev), time.time()-t)
m = pd.read_sql("select match_id, substr(date,1,4) yr, league, status_type from matches", c)
p = ev.pivot_table(index="match_id", columns="quarter", values="ngoal", aggfunc="sum")
nn = ev.groupby("match_id").agg(nnull=("nnull","sum"), n=("n","sum"))
d = m.merge(p, left_on="match_id", right_index=True, how="left").merge(nn, left_on="match_id", right_index=True, how="left")
d = d[d.status_type=="finished"]
g = d.groupby("yr").agg(n=("match_id","size"), q1=("Q1", lambda s: (s>0).mean()), q2=("Q2", lambda s: (s>0).mean()), q3=("Q3", lambda s: (s>0).mean()), q4=("Q4", lambda s: (s>0).mean()),
    any_ev=("n", lambda s: s.notna().mean()), null_ts=("nnull", lambda s: (s>0).mean()))
print(g.round(3))
d["full3"] = (d.Q1>0)&(d.Q2>0)&(d.Q3>0)
print("partidos con Q1-Q3 con goles (2023+):", d[d.yr>='2023'].full3.sum(), "de", (d.yr>='2023').sum())
d.to_pickle("tmp/m27_v5_3/cov_matches.pkl")
