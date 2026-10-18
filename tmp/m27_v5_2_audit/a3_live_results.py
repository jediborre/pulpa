"""
tmp/m27_v5_2_audit/a3_live_results.py
Rendimiento REAL en vivo (bet_monitor_log_v3 y v2) por modelo, minuto de inferencia y banda de confianza.
Hipotesis: el WR live de m27_v5_x es menor al 73% de backtest por skew train/live y por disparo multi-tick.
"""
import sqlite3, sys, json, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=30)
for t in ("bet_monitor_log_v3","bet_monitor_log_v2"):
    df = pd.read_sql(f"select * from {t}", c)
    print("\n=====", t, len(df), df.created_at.min(), df.created_at.max())
    print(df.groupby(["model_version","signal_type","result"], dropna=False).size().unstack(fill_value=0))
    print(df.target_quarter.value_counts().to_dict())
    d = df[df.model_version.str.startswith("m27_v5", na=False)].copy()
    d = d[d.result.isin(["win","loss","hit"])]
    d["win"] = d.result.isin(["win","hit"])
    print(d.groupby("model_version").agg(n=("win","size"), wr=("win","mean"), conf=("confidence","mean")).round(3))
    print(d.groupby(["model_version", d.inference_minute]).agg(n=("win","size"), wr=("win","mean")).round(3).head(40))
    d["cb"] = pd.cut(d.confidence, [0,.68,.72,.75,.8,1])
    print(d.groupby(["model_version","cb"], observed=True).agg(n=("win","size"), wr=("win","mean")).round(3))
df = pd.read_sql("select * from bet_monitor_log_v3 where model_version like 'm27_v5_2%' limit 2", c)
for _, r in df.iterrows():
    print(r.raw_json[:1500] if r.raw_json else None); print(r.inference_json[:1500] if r.inference_json else None)
