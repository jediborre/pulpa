"""
tmp/m27_v5_2_audit/a13_skew_impact.py
Cuantifica el impacto del train-serving skew: entrena el HGB base y evalua el test con las features
reemplazadas por lo que el extractor LIVE realmente produce (dead features en 0, rubber_band y white_flag con formula live,
d_wr/d_q4 reescalados desde ELO, etc.). Medicion incremental y acumulada.
"""
import sys, pickle, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.stdout.reconfigure(encoding="utf-8")
d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl","rb"))
X, y, meta, names = d["X"], d["y_class"], d["meta"], list(d["feature_names"])
TOX = ["korean basketball league","kbl","lmb, apertura","tipos sbl","albania super","georgia superli","kosovo superlig","armenia a-league","cyprus division","korisliiga, women","betsafe-lkl","nlb","1st dbbl","serie a2, women, group b","serie b, group b","israeli basketball super league"]
k = np.array([not any(p in str(m["lg"]).lower() for p in TOX) for m in meta]); X, y = X[k], y[k]
KEPT = [0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,22,23,25,26,27,29,31,33,34,36]
nm = [names[i] for i in KEPT]; B = X[:, KEPT]; n = len(y); s = int(n*.82)
m = HC(max_iter=140, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42).fit(B[:s], y[:s])
T = B[s:].copy(); yt = y[s:]; c = lambda f: nm.index(f)
def ev(M, label):
    p = m.predict_proba(M)[:, 1]; b = (p >= .68) | (p <= .32); w = np.where(p >= .68, yt == 1, yt == 0)[b]
    print(f"{label:<62} AUC {roc_auc_score(yt,p):.4f} Brier {brier_score_loss(yt,p):.4f} | bets {b.sum():>4} WR {w.mean()*100:5.1f}%")
ev(T, "0. test limpio (como backtest)")
M = T.copy()
for f in ["recent_2m_run_diff"]: M[:, c(f)] = 0
ev(M, "1. + recent_2m_run_diff=0 (bug clave 'run_diff')")
M[:, c("d_strat_q4_net")] = 0; ev(M, "2. + d_strat_q4_net=0 (state sin q_profile)")
M[:, c("h2h_bayes_ft_margin")] = 0; ev(M, "3. + h2h=0 (live pasa list, espera dict)")
M[:, c("ts_delta_wp_bayes")] = 0; ev(M, "4. + ts_delta_wp=0 (live pasa list, espera dict)")
cd = T[:, c("cur_score_diff")]
M[:, c("rubber_band_gap")] = np.where(np.abs(cd) >= 14, -cd*0.15, 0); ev(M, "5. + rubber_band formula live")
M[:, c("white_flag_risk_zone")] = np.where(cd >= 22, -1.0, np.where(cd <= -22, 1.0, 0)); ev(M, "6. + white_flag formula live")
elo_ft = T[:, c("d_wr_bayes")]*180/1.0  # aprox: wr_bayes live = d_elo_ft/180; train es WR ultimos 15 (escala distinta)
M[:, c("d_wr_bayes")] = T[:, c("d_wr_bayes")]*0.5; M[:, c("d_q4_bayes")] = T[:, c("d_q4_bayes")]*0.5
ev(M, "7. + d_wr/d_q4 con escala ELO/180,/35 (aprox x0.5)")
M2 = M.copy(); M2[:, c("odds_log_ratio")] = 0; M2[:, c("expected_ft_margin")] = np.clip(T[:, c("expected_ft_margin")]*0.6, -28, 28)
ev(M2, "8. + sin odds (live no las trae) y expected_ft solo ELO")
# live a min 27: cur_diff y gp con ~10% menos info
rng = np.random.default_rng(0)
M3 = M.copy(); M3[:, c("cur_score_diff")] = T[:, c("cur_score_diff")]*0.9 + rng.normal(0, 3.0, len(yt)); M3[:, c("abs_cur_score_diff")] = np.abs(M3[:, c("cur_score_diff")])
ev(M3, "9. 7 + marcador a min 27 (≈ -10% + ruido 3 pts)")
