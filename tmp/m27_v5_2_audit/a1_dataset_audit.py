"""
tmp/m27_v5_2_audit/a1_dataset_audit.py
Auditoria arquitectonica de m27_v5_2 sobre el cache de entrenamiento (tmp/m27_v5/dataset_v5_clean_cache.pkl).
Responde: (1) cobertura real de cada feature por año, (2) quien genera las apuestas verdes P>=0.68,
(3) cuanto aporta cada bloque de features (ablaciones con LR/HGB sobre el MISMO split 82/18),
(4) cuanto se pierde por calibrar isotonica in-sample vs out-of-time, (5) base rates por regimen.
Hipotesis: la señal real es debil (AUC~0.63) y esta concentrada en pocas features/regimenes; varias features
(odds, standings, h2h, lineups) tienen cobertura minima en train.
"""
import sys, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)

d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl", "rb"))
X, y, yreg, meta, names = d["X"], d["y_class"], d["y_reg"], d["meta"], d["feature_names"]
names = list(names)
KEPT = [0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,22,23,25,26,27,29,31,33,34,36]
TOXIC = ["korean basketball league","kbl","lmb, apertura","tipos sbl","albania super","georgia superli","kosovo superlig",
         "armenia a-league","cyprus division","korisliiga, women","betsafe-lkl","nlb","1st dbbl","serie a2, women, group b",
         "serie b, group b","israeli basketball super league"]
lg = np.array([m["lg"] for m in meta]); yr = np.array([m["yr"] for m in meta])
tox = np.array([any(p in str(l).lower() for p in TOXIC) for l in lg])
keep = ~tox
print(f"Total {len(y):,} | toxicas {tox.sum():,} | limpio {keep.sum():,}")
Xc, yc, yrc, lgc, yearc = X[keep][:, KEPT], y[keep], yreg[keep], lg[keep], yr[keep]
nm = [names[i] for i in KEPT]
print("Features:", nm)

print("\n=== 1. COBERTURA (fraccion != 0) por año en train ===")
cov = pd.DataFrame({n: (Xc[:, i] != 0) for i, n in enumerate(nm)})
cov["yr"] = yearc
print((cov.groupby("yr").mean().T * 100).round(1)[[y_ for y_ in sorted(set(yearc)) if y_ >= 2015]])

print("\n=== 2. Distribucion de apuestas: base rates por regimen de cur_diff (y=1 home gana Q4) ===")
cd = Xc[:, 0]
bins = [-100,-20,-14,-10,-6,-3,0,3,6,10,14,20,100]
df = pd.DataFrame({"cd": cd, "y": yc, "yr": yearc})
df["b"] = pd.cut(df.cd, bins)
print(df.groupby("b", observed=True).agg(n=("y","size"), home_q4_win=("y","mean")).round(3))

# split cronologico 82/18 como en produccion
n = len(yc); s = int(n*0.82)
Xtr, ytr, Xte, yte = Xc[:s], yc[:s], Xc[s:], yc[s:]
print(f"\nTrain {len(ytr):,} Test {len(yte):,}  (fechas test desde indice {s})")

def fit_eval(cols, label, model="lr"):
    idx = [nm.index(c) for c in cols]
    if model == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=300))
    else:
        m = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42)
    m.fit(Xtr[:, idx], ytr)
    p = m.predict_proba(Xte[:, idx])[:, 1]
    sel = (p >= 0.68) | (p <= 0.32)
    wins = ((p >= 0.68) & (yte == 1)) | ((p <= 0.32) & (yte == 0))
    wr = wins[sel].mean()*100 if sel.sum() else float("nan")
    print(f"{label:<58} AUC {roc_auc_score(yte,p):.4f} Brier {brier_score_loss(yte,p):.4f} | bets(P>=.68 raw) {sel.sum():>5} WR {wr:5.1f}%")
    return m

print("\n=== 3. ABLACIONES (mismo split; HGB) ===")
ALL = nm
score = ["cur_score_diff","abs_cur_score_diff","ht_diff","q1_diff","q2_diff","q3_eff_diff","q3_vs_ht_momentum","recent_2m_run_diff","pace_pts_per_min","pace_vs_league_prior","gp_val"]
elo = ["d_lineup_starters_elo_ft","d_lineup_starters_elo_q4","d_lineup_bench_elo_q4","d_lineup_top3_elo_q4","d_roster_drift_adv","d_strat_q4_net","d_q4_bayes","d_wr_bayes"]
mkt = ["odds_log_ratio","expected_ft_margin","h2h_bayes_ft_margin","ts_delta_wp_bayes"]
sit = ["pbp_3pt_diff_q1_q3","rubber_band_gap","white_flag_risk_zone","lg_avg_q4_prior"]
fit_eval(ALL, "TODAS (27)", "hgb")
fit_eval(score, "solo marcador/momentum (11)", "hgb")
fit_eval(elo, "solo ELO/roster/strat (8)", "hgb")
fit_eval(mkt, "solo mercado/h2h/standing (4)", "hgb")
fit_eval(sit, "solo situacional (4)", "hgb")
fit_eval(["expected_ft_margin"], "solo expected_ft_margin (1)", "hgb")
fit_eval(["cur_score_diff"], "solo cur_score_diff (1)", "hgb")
fit_eval(["cur_score_diff","abs_cur_score_diff"], "cur_diff + abs (2)", "hgb")
fit_eval(["cur_score_diff","expected_ft_margin"], "cur_diff + expected_ft_margin (2)", "hgb")
fit_eval(["cur_score_diff","expected_ft_margin","d_lineup_starters_elo_ft","d_wr_bayes","d_q4_bayes"], "5 features top", "hgb")
for drop_grp, gname in [(score,"marcador"),(elo,"elo"),(mkt,"mercado"),(sit,"situacional")]:
    fit_eval([c for c in ALL if c not in drop_grp], f"TODAS menos {gname}", "hgb")
for c in ["pbp_3pt_diff_q1_q3","d_lineup_starters_elo_ft","d_lineup_bench_elo_q4","d_lineup_top3_elo_q4","expected_ft_margin","ts_delta_wp_bayes","odds_log_ratio","rubber_band_gap","recent_2m_run_diff","gp_val"]:
    fit_eval([x for x in ALL if x != c], f"TODAS menos {c}", "hgb")

print("\n=== 4. AUC univariado (test) y correlacion con y ===")
rows = []
for i, c in enumerate(nm):
    a = roc_auc_score(yte, Xte[:, i]); rows.append((c, round(a,4), round(np.corrcoef(Xte[:, i], yte)[0,1],4)))
print(pd.DataFrame(rows, columns=["feat","auc_uni","corr"]).sort_values("auc_uni", key=lambda s: (s-0.5).abs(), ascending=False).to_string(index=False))

print("\n=== 5. Calibracion isotonica: in-sample (como produccion) vs fuera de muestra (val temporal) ===")
vs = int(len(Xtr)*0.85)
hg = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05, l2_regularization=40.0, random_state=42)
hg.fit(Xtr[:vs], ytr[:vs])
p_tr = hg.predict_proba(Xtr[:vs])[:, 1]; p_val = hg.predict_proba(Xtr[vs:])[:, 1]; p_te = hg.predict_proba(Xte)[:, 1]
iso_in = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(p_tr, ytr[:vs])
iso_oot = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(p_val, ytr[vs:])
for lab, iso in [("iso in-sample", iso_in), ("iso out-of-time", iso_oot)]:
    pc = iso.predict(p_te)
    print(f"{lab:<18} Brier {brier_score_loss(yte,pc):.4f} AUC {roc_auc_score(yte,pc):.4f} | n unicos {len(np.unique(pc))}")
print(f"raw                Brier {brier_score_loss(yte,p_te):.4f} AUC {roc_auc_score(yte,p_te):.4f}")

print("\n=== 6. WR por año del test segun P cruda (HGB) y por 'lado' ===")
pc = iso_oot.predict(p_te)
t = pd.DataFrame({"p": pc, "y": yte, "yr": yearc[s:], "cd": Xte[:, 0]})
t["side"] = np.where(t.p >= 0.68, "HOME", np.where(t.p <= 0.32, "AWAY", "NA"))
t["win"] = np.where(t.side == "HOME", t.y == 1, np.where(t.side == "AWAY", t.y == 0, np.nan))
b = t[t.side != "NA"]
print(b.groupby("side").agg(n=("win","size"), wr=("win","mean"), cd_mean=("cd","mean")).round(3))
b = b.assign(b=pd.cut(b.cd, bins))
print(b.groupby(["side","b"], observed=True).agg(n=("win","size"), wr=("win","mean")).round(3))
