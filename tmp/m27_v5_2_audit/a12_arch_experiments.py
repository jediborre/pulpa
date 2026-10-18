"""
tmp/m27_v5_2_audit/a12_arch_experiments.py
Experimentos arquitectonicos sobre el cache (mismo split 82/18, HGB base de m27_v5_2):
 A) identidad de liga (target encoding expanding, sin leakage) + volatilidad Q4 por liga
 B) features normalizadas: rubber_band real (cur - 0.75*efm), cur_diff/sqrt(pts), efm*0.25 vs cur
 C) simetria home/away (augmentation espejo + promedio) 
 D) cabeza gaussiana: regresor de margen Q4 + sigma por liga -> P=Phi(mu/sigma)
 E) bloque de interacciones minimo (efm, cd, d_wr, elo_starters) con LR para ver techo lineal
Hipotesis: la señal util esta en mercado+liga+estado normalizado; el resto (roster ELO, pace, etc.) es redundante.
"""
import sys, pickle, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.stdout.reconfigure(encoding="utf-8")
d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl","rb"))
X, y, yr_, meta, names = d["X"], d["y_class"], d["y_reg"], d["meta"], list(d["feature_names"])
TOX = ["korean basketball league","kbl","lmb, apertura","tipos sbl","albania super","georgia superli","kosovo superlig","armenia a-league","cyprus division","korisliiga, women","betsafe-lkl","nlb","1st dbbl","serie a2, women, group b","serie b, group b","israeli basketball super league"]
k = np.array([not any(p in str(m["lg"]).lower() for p in TOX) for m in meta])
X, y, yr_ = X[k], y[k], yr_[k]; meta = [m for m, kk in zip(meta, k) if kk]
KEPT = [0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,22,23,25,26,27,29,31,33,34,36]
nm = [names[i] for i in KEPT]; B = X[:, KEPT]
lg = np.array([m["lg"] for m in meta]); n = len(y); s = int(n*.82)
def col(a, c): return a[:, nm.index(c)]
def ev(p, label, yt=None):
    yt = y[s:] if yt is None else yt
    b = (p >= .68) | (p <= .32); w = np.where(p >= .68, yt == 1, yt == 0)[b]
    print(f"{label:<52} AUC {roc_auc_score(yt,p):.4f} Brier {brier_score_loss(yt,p):.4f} | bets {b.sum():>4} WR {w.mean()*100:5.1f}% ROI@1.40 {(w.mean()*1.4-1)*100:+5.1f}%")
def hgb(): return HC(max_iter=140, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42)
ev(hgb().fit(B[:s], y[:s]).predict_proba(B[s:])[:, 1], "BASE 27")

# A) target encoding expanding por liga (solo pasado)
cnt, sm, sq, ab = {}, {}, {}, {}
te = np.zeros(n); vol = np.zeros(n); bias = np.zeros(n); prior = y[:s].mean()
for i in range(n):
    l = lg[i]; c = cnt.get(l, 0)
    te[i] = (sm.get(l, 0) + 40*prior) / (c + 40); vol[i] = (ab.get(l, 0) + 40*6.0) / (c + 40); bias[i] = (sq.get(l, 0) + 40*0.5) / (c + 40)
    cnt[l] = c + 1; sm[l] = sm.get(l, 0) + y[i]; ab[l] = ab.get(l, 0) + abs(yr_[i]); sq[l] = sq.get(l, 0) + yr_[i]
A = np.column_stack([te, vol, bias])
ev(hgb().fit(np.hstack([B, A])[:s], y[:s]).predict_proba(np.hstack([B, A])[s:])[:, 1], "A) BASE + liga (home Q4 rate, vol, bias)")

# B) normalizadas
cd, efm = col(B, "cur_score_diff"), col(B, "expected_ft_margin"); pace = col(B, "pace_pts_per_min")
C = np.column_stack([cd - 0.75*efm, cd/np.sqrt(np.maximum(pace*30, 20)), cd/vol, efm/vol, (cd*0.25+efm*0.25)/vol])
ev(hgb().fit(np.hstack([B, C])[:s], y[:s]).predict_proba(np.hstack([B, C])[s:])[:, 1], "B) BASE + normalizadas por volatilidad liga")
BC = np.hstack([B, A, C])
ev(hgb().fit(BC[:s], y[:s]).predict_proba(BC[s:])[:, 1], "A+B) BASE + liga + normalizadas")

# C) simetria espejo
sym = {"abs_cur_score_diff","pace_pts_per_min","pace_vs_league_prior","white_flag_risk_zone","lg_avg_q4_prior"}
sg = np.array([1.0 if c not in sym else 0.0 for c in nm])
def mir(M): 
    M = M.copy(); M[:, :len(nm)] = np.where(sg == 1, -M[:, :len(nm)], M[:, :len(nm)]); return M
def sym_fit(M, label, rb_cols=None):
    Mtr = np.hstack([M[:s], np.ones((s,1))]); Mtr2 = np.hstack([mir(M[:s]), np.zeros((s,1))])
    m = hgb().fit(np.vstack([Mtr, Mtr2]), np.concatenate([y[:s], 1-y[:s]]))
    pa = m.predict_proba(np.hstack([M[s:], np.ones((n-s,1))]))[:, 1]; pb = 1 - m.predict_proba(np.hstack([mir(M[s:]), np.zeros((n-s,1))]))[:, 1]
    ev((pa+pb)/2, label); return (pa+pb)/2
# nota: en BC las extras de liga/normalizadas no se espejan salvo las primeras 27 cols; para C solo usar B
psym = sym_fit(B, "C) BASE simetrizado (espejo + promedio)")

# D) cabeza gaussiana
reg = HR(max_iter=200, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42).fit(BC[:s], yr_[:s])
mu = reg.predict(BC[s:]); res = yr_[:s] - reg.predict(BC[:s])
sdreg = HR(max_iter=100, max_leaf_nodes=10, min_samples_leaf=500, learning_rate=0.05, random_state=42).fit(BC[:s], np.abs(res))
sig = np.maximum(sdreg.predict(BC[s:]) * 1.2533, 3.0)
ev(norm.cdf(mu/sig), "D) gaussiana mu/sigma(heterocedastica) (A+B)")
ev(norm.cdf(mu/7.7), "D') gaussiana sigma fija 7.7")
# E) LR techo lineal
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(BC[:s], y[:s]); ev(lr.predict_proba(BC[s:])[:, 1], "E) LR sobre A+B (lineal)")
ev(hgb().fit(BC[:s], y[:s]).predict_proba(BC[s:])[:, 1]*0.5 + 0.5*norm.cdf(mu/sig), "F) promedio HGB(A+B) + gaussiana")
