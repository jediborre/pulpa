"""
tmp/m27_v5_2_audit/a6_odds_semantics.py
Que representa match_odds historico: AUC de odds_log_ratio vs ganador FT y vs ganador Q4, y relacion con cur_diff.
Si el AUC FT es ~0.75-0.80 son odds pre-partido (closing). Si es mucho mayor, hay odds in-play/post-partido (leak).
Tambien: AUC de expected_ft_margin, y estructura de team_strength.fetched_at vs fecha del partido.
"""
import sys, pickle, sqlite3, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl","rb"))
X, y, meta, names = d["X"], d["y_class"], d["meta"], list(d["feature_names"])
cd = np.array([m["cur_diff"] for m in meta]); q4 = np.array([m["q4_diff"] for m in meta]); yr = np.array([m["yr"] for m in meta])
ft = cd + q4
olr = X[:, names.index("odds_log_ratio")]; has = olr != 0
elo = X[:, names.index("d_wr_bayes")]
ts = X[:, names.index("ts_delta_wp_bayes")]
print("n con odds", has.sum())
m = has & (ft != 0)
print("AUC odds_log_ratio vs FT win :", round(roc_auc_score(ft[m] > 0, olr[m]), 4))
print("AUC odds_log_ratio vs Q4 win :", round(roc_auc_score(q4[has] > 0, olr[has]), 4))
print("AUC odds_log_ratio vs sign(cur_diff) (HT+Q3):", round(roc_auc_score(cd[has & (cd!=0)] > 0, olr[has & (cd!=0)]), 4))
print("AUC d_wr_bayes (hist WR) vs FT win:", round(roc_auc_score(ft[m] > 0, elo[m]), 4))
t = ts != 0
mt = t & (ft != 0)
print("n con ts", t.sum(), "AUC ts_delta vs FT", round(roc_auc_score(ft[mt] > 0, ts[mt]), 4), "vs Q4", round(roc_auc_score(q4[t] > 0, ts[t]), 4), "vs cur_diff sign", round(roc_auc_score(cd[t & (cd != 0)] > 0, ts[t & (cd != 0)]), 4))
for y_ in (2023, 2024, 2025, 2026):
    mm = has & (yr == y_) & (ft != 0)
    print(y_, "odds AUC FT", round(roc_auc_score(ft[mm] > 0, olr[mm]), 4), "| Q4", round(roc_auc_score(q4[has & (yr==y_)] > 0, olr[has & (yr==y_)]), 4))
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=60)
print(pd.read_sql("select * from match_odds where odds_type='Home/Away' and market_period='Match' limit 5", c))
print(pd.read_sql("select substr(timestamp,1,10) d, count(*) n from match_odds group by 1 order by n desc limit 8", c))
print(pd.read_sql("select * from team_strength order by id desc limit 4", c))
q = pd.read_sql("select t.match_id, t.wins, t.losses, t.fetched_at, m.date from team_strength t join matches m on m.match_id=t.match_id limit 200000", c)
q["lag_days"] = (pd.to_datetime(q.fetched_at.str[:10], errors="coerce") - pd.to_datetime(q.date.str[:10], errors="coerce")).dt.days
print(q.lag_days.describe()); print(q.groupby(pd.cut(q.lag_days, [-9999,-1,0,1,7,30,9999])).size())
