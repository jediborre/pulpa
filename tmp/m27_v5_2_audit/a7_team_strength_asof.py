"""
tmp/m27_v5_2_audit/a7_team_strength_asof.py
Verifica si team_strength (wins/losses/form) es un snapshot PRE-partido (as-of) o un snapshot tomado
al momento del backfill (fetched_at muy posterior al partido) -> look-ahead leakage en ts_delta_wp_bayes.
"""
import sys, sqlite3, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=60)
q = pd.read_sql("select t.match_id, t.team_name, t.wins, t.losses, t.form, t.fetched_at, m.date from team_strength t join matches m on m.match_id=t.match_id", c)
q["yr"] = q.date.str[:4]
q["fy"] = q.fetched_at.str[:7]
q["g"] = q.wins.fillna(0) + q.losses.fillna(0)
print(q.groupby("yr").agg(n=("g","size"), fetched_min=("fetched_at","min"), fetched_max=("fetched_at","max"), games_mean=("g","mean")).to_string())
print(q.groupby("fy").size().tail(15))
# mismo equipo, partidos de fechas distintas: cambia wins+losses con la fecha del partido? (as-of) o es constante (snapshot unico)?
t = q.groupby("team_name").filter(lambda g: g.date.nunique() >= 6 and g.fetched_at.str[:10].nunique() == 1).sort_values("date")
print("equipos con un solo fetched_at y >=6 fechas distintas:", t.team_name.nunique())
ex = t.team_name.iloc[0] if len(t) else None
if ex: print(t[t.team_name == ex][["date","wins","losses","form","fetched_at"]].head(12))
