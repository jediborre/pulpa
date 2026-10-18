"""
tmp/m27_v5_2_audit/a9_baselines_and_data.py
(1) Baselines simples al MISMO volumen de apuestas que m27_v5_2 (top-N por |expected_ft_margin|, |cur_diff|, |d_wr_bayes|, regla mixta).
(2) Techo teorico: SD del margen Q4 y P(win)=Phi(mu/sigma) -> que mu hace falta para P>=0.68.
(3) Inventario de datos explotables: incident_types en match_events, cobertura de Handicap/Over-Under en match_odds, team_statistics.
Hipotesis: el ensamble no supera a reglas simples de favorito/blowout al mismo volumen -> la señal esta en priors pre-partido, no en el estado de juego.
"""
import sys, pickle, sqlite3, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl","rb"))
X, y, yreg, meta, names = d["X"], d["y_class"], d["y_reg"], d["meta"], list(d["feature_names"])
TOXIC = ["korean basketball league","kbl","lmb, apertura","tipos sbl","albania super","georgia superli","kosovo superlig","armenia a-league","cyprus division","korisliiga, women","betsafe-lkl","nlb","1st dbbl","serie a2, women, group b","serie b, group b","israeli basketball super league"]
lg = np.array([m["lg"] for m in meta]); keep = np.array([not any(p in str(l).lower() for p in TOXIC) for l in lg])
X, y, yreg, lg = X[keep], y[keep], yreg[keep], lg[keep]; meta = [m for m, k in zip(meta, keep) if k]
KEPT = [0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,22,23,25,26,27,29,31,33,34,36]
s = int(len(y)*0.82)
f = lambda n: X[s:, names.index(n)]
yt = y[s:]; cd = f("cur_score_diff"); efm = f("expected_ft_margin"); wr = f("d_wr_bayes"); olr = f("odds_log_ratio")
hg = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42).fit(X[:s][:, KEPT], y[:s])
p = hg.predict_proba(X[s:][:, KEPT])[:, 1]
N = int(((p >= .68) | (p <= .32)).sum()); print("Volumen modelo (raw .68/.32):", N, "de", len(yt))
def topn(score, label):
    idx = np.argsort(-np.abs(score))[:N]; side = score[idx] > 0
    w = np.where(side, yt[idx] == 1, yt[idx] == 0).mean()*100
    print(f"{label:<46} N={N} WR={w:5.1f}%  ROI@1.40={(w*1.40/100-1)*100:+5.1f}%")
modelscore = p - 0.5
topn(modelscore, "MODELO HGB (27 feats)")
topn(efm, "regla: |expected_ft_margin| (favorito pre-partido)")
topn(wr, "regla: |d_wr_bayes| (win-rate historico)")
topn(cd, "regla: |cur_score_diff| (lider tras Q3)")
topn(np.sign(efm)*np.abs(efm) + 0.5*cd, "regla: efm + 0.5*cur_diff")
m = np.abs(cd) >= 20
for sd in (1, -1):
    sel = m & (np.sign(cd) == sd); print(f"lider +20: lado {'HOME' if sd>0 else 'AWAY'} n={sel.sum()} WR lider={np.mean(np.where(sd>0, yt[sel]==1, yt[sel]==0))*100:.1f}%")
big = m & (np.sign(cd) == np.sign(efm))
print(f"lider +20 Y favorito pre-partido: n={big.sum()} WR={np.mean(np.where(cd[big]>0, yt[big]==1, yt[big]==0))*100:.1f}%")
print("\n--- Techo teorico ---")
q4 = yreg[:]; print("SD margen Q4 global %.2f | media %.2f | por arch:" % (q4.std(), q4.mean()))
ar = np.array([m_["arch"] for m_ in meta]); print(pd.Series(q4).groupby(ar).agg(["std","mean","size"]).round(2))
for P in (0.60, 0.65, 0.68, 0.75):
    print(f"P={P}: mu requerido = {norm.ppf(P)*q4.std():.1f} pts de margen esperado en Q4")
print("AUC modelo por |cur_diff|:")
from sklearn.metrics import roc_auc_score
for lo, hi in [(0,3),(3,6),(6,10),(10,15),(15,20),(20,99)]:
    mm = (np.abs(cd) >= lo) & (np.abs(cd) < hi); print(f"  |cd| in [{lo},{hi}) n={mm.sum():>5} AUC={roc_auc_score(yt[mm], p[mm]):.3f}")
print("\n--- Inventario de datos ---")
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=120)
mx = c.execute("select max(id) from match_events").fetchone()[0]
print(pd.read_sql(f"select incident_type, subtype, count(*) n from match_events where id > {mx-400000} group by 1,2 order by n desc limit 25", c))
print(pd.read_sql("select odds_type, market_name, market_period, count(*) n, count(distinct match_id) m from match_odds group by 1,2,3 order by n desc limit 15", c))
print(pd.read_sql("select * from team_statistics limit 2", c).T.head(30))
