"""
tmp/m27_v5_2_audit/a11_new_features_experiment.py
Experimento: ¿mueven la aguja las stats por cuarto de team_statistics (faltas, timeouts, turnovers, 3PM/2PM/FTM acumulados Q1-Q3)
y el calendario (dias de descanso, back-to-back)? Compara HGB base(27) vs base+nuevas sobre las mismas filas (2024+ con stats Q3),
split cronologico 82/18, mismo hiperparametro. Solo evaluacion offline; las stats Q1-Q3 son conocibles al cierre de Q3 en vivo.
"""
import sys, pickle, sqlite3, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl","rb"))
X, y, meta, names = d["X"], d["y_class"], d["meta"], list(d["feature_names"])
TOXIC = ["korean basketball league","kbl","lmb, apertura","tipos sbl","albania super","georgia superli","kosovo superlig","armenia a-league","cyprus division","korisliiga, women","betsafe-lkl","nlb","1st dbbl","serie a2, women, group b","serie b, group b","israeli basketball super league"]
keepm = np.array([not any(p in str(m["lg"]).lower() for p in TOXIC) for m in meta])
X, y = X[keepm], y[keepm]; meta = [m for m, k in zip(meta, keepm) if k]
KEPT = [0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,22,23,25,26,27,29,31,33,34,36]
mid = np.array([m["mid"] for m in meta]); dt = pd.to_datetime([m["dt"] for m in meta])

# --- calendario: dias de descanso y partidos en 7 dias (desde el propio cache cronologico) ---
last = {}; hist = {}
rest_h = np.zeros(len(meta)); rest_a = np.zeros(len(meta)); g7_h = np.zeros(len(meta)); g7_a = np.zeros(len(meta))
for i, m in enumerate(meta):
    for t, r, g in ((m["ht"], rest_h, g7_h), (m["at"], rest_a, g7_a)):
        lt = last.get(t); r[i] = min((dt[i] - lt).days, 14) if lt is not None else 14
        g[i] = sum(1 for x in hist.get(t, [])[-6:] if (dt[i] - x).days <= 7)
    for t in (m["ht"], m["at"]):
        last[t] = dt[i]; hist.setdefault(t, []).append(dt[i])
sched = np.column_stack([rest_h - rest_a, np.minimum(rest_h,14), np.minimum(rest_a,14), (rest_h <= 1).astype(float) - (rest_a <= 1).astype(float), g7_h - g7_a])
sched_names = ["rest_diff","rest_h","rest_a","b2b_diff","g7_diff"]

# --- team_statistics acumuladas Q1..Q3 ---
sel = np.array([m["yr"] >= 2024 for m in meta])
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=300)
ts = pd.read_sql("""select match_id, period, stat_key, home_value h, away_value a from team_statistics
 where period in ('Q1','Q2','Q3','1Q','2Q','3Q') and stat_key in ('fouls','timeouts','turnovers','freeThrowsMade','threePointsMade','twoPointsMade','points')
 and match_id in (select match_id from matches where date>='2024-01-01')""", c)
ts["p"] = ts.period.str.extract(r"(\d)")[0]
ts = ts.drop_duplicates(["match_id","p","stat_key"])
piv = ts.pivot_table(index="match_id", columns=["stat_key","p"], values=["h","a"], aggfunc="first")
print("matches con stats:", len(piv))
def cum(side, key, ps):
    cols = [(side, key, p) for p in ps]
    cols = [c_ for c_ in cols if c_ in piv.columns]
    return piv[cols].sum(axis=1, min_count=len(ps)) if len(cols) == len(ps) else None
F = pd.DataFrame(index=piv.index)
for key, nm in [("fouls","fouls"),("timeouts","to_used"),("turnovers","tov"),("freeThrowsMade","ftm"),("threePointsMade","tpm"),("twoPointsMade","fgm2")]:
    h = cum("h", key, ["1","2","3"]); a = cum("a", key, ["1","2","3"])
    if h is None: continue
    F["d_"+nm] = h - a; F["h_"+nm] = h; F["a_"+nm] = a
    h3 = piv[("h", key, "3")] if ("h", key, "3") in piv.columns else None
    if h3 is not None: F["d_"+nm+"_q3"] = h3 - piv[("a", key, "3")]
F["d_3pt_share"] = (F["h_tpm"]*3/(F["h_tpm"]*3+F["h_fgm2"]*2+F["h_ftm"]+1)) - (F["a_tpm"]*3/(F["a_tpm"]*3+F["a_fgm2"]*2+F["a_ftm"]+1))
F["h_to_left"] = 7 - F["h_to_used"]; F["a_to_left"] = 7 - F["a_to_used"]   # aprox: proxy relativo
stat_names = list(F.columns)
Fm = F.reindex(mid).values
have = sel & ~np.isnan(Fm).any(axis=1)
print("filas 2024+ con stats Q1-Q3 completas:", have.sum(), "de", sel.sum())
base = X[:, KEPT]
# residual vs mercado
resid = (X[:, names.index("cur_score_diff")] - 0.75*X[:, names.index("expected_ft_margin")])[:, None]
Xa = base[have]; ya = y[have]; Sa = sched[have]; Fa = Fm[have]
s = int(len(ya)*0.82)
def run(M, label):
    m = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42).fit(M[:s], ya[:s])
    p = m.predict_proba(M[s:])[:, 1]; yt = ya[s:]
    b = (p >= .68) | (p <= .32); w = np.where(p >= .68, yt == 1, yt == 0)[b]
    print(f"{label:<44} AUC {roc_auc_score(yt,p):.4f} Brier {brier_score_loss(yt,p):.4f} | bets {b.sum():>4} WR {w.mean()*100 if b.sum() else 0:5.1f}%")
    return m
print("train", s, "test", len(ya)-s)
run(Xa, "BASE 27")
run(np.hstack([Xa, Sa]), "BASE + calendario")
run(np.hstack([Xa, Fa]), "BASE + stats Q1-Q3 (fouls/to/tov/3pm...)")
run(np.hstack([Xa, Fa, Sa]), "BASE + stats + calendario")
for nm_ in ["d_fouls","d_to_used","d_tov","d_tpm","d_3pt_share","d_ftm","d_fouls_q3"]:
    if nm_ in stat_names:
        j = stat_names.index(nm_); a_ = roc_auc_score(ya[s:], Fa[s:, j]); print(f"  AUC univariado {nm_:<14} {a_:.4f}  corr {np.corrcoef(Fa[s:, j], ya[s:])[0,1]:+.3f}")
# solo prior de mercado + stats (ver si stats aportan mas que score)
mk = Xa[:, [nm_i for nm_i, n in enumerate([names[i] for i in KEPT]) if n in ("expected_ft_margin","odds_log_ratio","d_wr_bayes","cur_score_diff","d_lineup_starters_elo_ft")]]
run(mk, "5 features top")
run(np.hstack([mk, Fa, Sa]), "5 top + stats + calendario")
