"""
tmp/m27_v5_2/eval_leagues_and_features.py
Script auxiliar para evaluar el impacto de:
1. Mantener las 27 features vs 25 features vs 37 features.
2. Entrenar con poda de ligas toxicas vs con todas las ligas (198k).
3. Evaluar el rendimiento en el mismo conjunto de test holdout (35,717 partidos).
"""

import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")

# 25 Features de V5_1
INDICES_25 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 26, 27, 31, 33, 34, 36]

# 27 Features de V5_2 (incorporando 25: odds_log_ratio y 29: ts_delta_wp_bayes)
INDICES_27 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 25, 26, 27, 29, 31, 33, 34, 36]

TOXIC_LEAGUE_PATTERNS = [
    "korean basketball league", "kbl", "lmb, apertura", "tipos sbl",
    "albania super", "georgia superli", "kosovo superlig", "armenia a-league",
    "cyprus division", "korisliiga, women", "betsafe-lkl", "nlb", "1st dbbl",
    "serie a2, women, group b", "serie b, group b", "israeli basketball super league"
]

def is_toxic(lg_name):
    s = str(lg_name).lower()
    return any(p in s for p in TOXIC_LEAGUE_PATTERNS)

def train_and_eval(name, X_tr, y_tr, yr_tr, X_te, y_te, yr_te):
    t0 = time.time()
    
    clf_xgb = xgb.XGBClassifier(
        n_estimators=160, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=45.0,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    clf_xgb.fit(X_tr, y_tr)
    
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=15, min_samples_leaf=300,
        learning_rate=0.05, l2_regularization=40.0, random_state=42
    )
    clf_hgb.fit(X_tr, y_tr)
    
    clf_lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=300, random_state=42))
    clf_lr.fit(X_tr, y_tr)
    
    reg_hgb = HistGradientBoostingRegressor(
        max_iter=140, max_leaf_nodes=15, min_samples_leaf=300,
        learning_rate=0.05, l2_regularization=40.0, random_state=42
    )
    reg_hgb.fit(X_tr, yr_tr)
    
    p_tr_raw = 0.40 * clf_xgb.predict_proba(X_tr)[:, 1] + 0.35 * clf_hgb.predict_proba(X_tr)[:, 1] + 0.25 * clf_lr.predict_proba(X_tr)[:, 1]
    p_te_raw = 0.40 * clf_xgb.predict_proba(X_te)[:, 1] + 0.35 * clf_hgb.predict_proba(X_te)[:, 1] + 0.25 * clf_lr.predict_proba(X_te)[:, 1]
    
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calib.fit(p_tr_raw, y_tr)
    
    p_tr_cal = calib.predict(p_tr_raw)
    p_te_cal = calib.predict(p_te_raw)
    exp_margins_te = reg_hgb.predict(X_te)
    
    auc_tr = roc_auc_score(y_tr, p_tr_cal)
    auc_te = roc_auc_score(y_te, p_te_cal)
    gap = auc_tr - auc_te
    brier = brier_score_loss(y_te, p_te_cal)
    
    # Señales operativas Verdes en test
    b_home = (p_te_cal >= 0.68) & (clf_xgb.predict_proba(X_te)[:, 1] > 0.5) & (exp_margins_te >= 1.8)
    b_away = (p_te_cal <= 0.32) & (clf_xgb.predict_proba(X_te)[:, 1] < 0.5) & (exp_margins_te <= -1.8)
    n_b = int(np.sum(b_home) + np.sum(b_away))
    w_b = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
    wr = (w_b / n_b * 100.0) if n_b > 0 else 0.0
    roi_140 = ((w_b * 1.40 - n_b) / n_b * 100.0) if n_b > 0 else 0.0
    roi_183 = ((w_b * 1.83 - n_b) / n_b * 100.0) if n_b > 0 else 0.0
    
    dt = time.time() - t0
    print(f"[{name}] AUC Test: {auc_te:.4f} (Gap: {gap:+.4f}) | Brier: {brier:.4f} | Bets: {n_b:,} | WR: {wr:5.2f}% | ROI@1.40: {roi_140:+5.2f}% | Tiempo: {dt:.1f}s")
    return {
        "name": name, "auc_tr": auc_tr, "auc_te": auc_te, "gap": gap,
        "brier": brier, "bets": n_b, "wr": wr, "roi_140": roi_140, "roi_183": roi_183
    }

def main():
    print("Cargando dataset...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        
    raw_X = data["X"]
    y_class = data["y_class"]
    y_reg = data["y_reg"]
    meta = data["meta"]
    
    n_total = len(y_class)
    split_idx = int(n_total * 0.82)
    
    # Test set estricto idéntico para todos (los 35,717 partidos finales)
    y_te = y_class[split_idx:]
    yr_te = y_reg[split_idx:]
    meta_te = meta[split_idx:]
    
    # Experimento A: 27 Features con TODAS las ligas en train
    print("\n--- EXPERIMENTO A: 27 Features (Todas las ligas en train, 162k train / 35.7k test) ---")
    X_tr_27_all = raw_X[:split_idx, INDICES_27]
    X_te_27 = raw_X[split_idx:, INDICES_27]
    y_tr_all = y_class[:split_idx]
    yr_tr_all = y_reg[:split_idx]
    res_27_all = train_and_eval("27F_All_Leagues", X_tr_27_all, y_tr_all, yr_tr_all, X_te_27, y_te, yr_te)
    
    # Experimento B: 27 Features con PODA de ligas tóxicas en train
    print("\n--- EXPERIMENTO B: 27 Features (Poda de ligas tóxicas en train) ---")
    tr_meta = meta[:split_idx]
    clean_tr_indices = [i for i, m in enumerate(tr_meta) if not is_toxic(m.get("lg", ""))]
    clean_tr_indices = np.array(clean_tr_indices)
    
    X_tr_27_clean = X_tr_27_all[clean_tr_indices]
    y_tr_clean = y_tr_all[clean_tr_indices]
    yr_tr_clean = yr_tr_all[clean_tr_indices]
    res_27_clean = train_and_eval("27F_Clean_Train", X_tr_27_clean, y_tr_clean, yr_tr_clean, X_te_27, y_te, yr_te)
    
    # Experimento C: 25 Features de V5_1 sobre los mismos partidos para comparación
    print("\n--- EXPERIMENTO C: 25 Features V5_1 sobre los mismos partidos ---")
    X_tr_25_clean = raw_X[:split_idx, INDICES_25][clean_tr_indices]
    X_te_25 = raw_X[split_idx:, INDICES_25]
    res_25_clean = train_and_eval("25F_Clean_Train", X_tr_25_clean, y_tr_clean, yr_tr_clean, X_te_25, y_te, yr_te)

if __name__ == "__main__":
    main()
