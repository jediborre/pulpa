"""
tmp/m27_v5_2/compare_v5_v51_v52_same_matches.py
Comparación estricta y transparente de:
1. M27_V5 (37 Features, Original)
2. M27_V5_1 (25 Features, Lean Pruned)
3. M27_V5_2 (27 Features, Lean + Market Odds + Table WP)
Sobre los EXACTOS MISMOS PARTIDOS DE TEST (Holdout idéntico).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")
V5_DIR = Path("models/m27_v5/model_outputs")
V5_1_DIR = Path("models/m27_v5_1/model_outputs")
V5_2_DIR = Path("models/m27_v5_2/model_outputs")

INDICES_25 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 26, 27, 31, 33, 34, 36]
INDICES_27 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 25, 26, 27, 29, 31, 33, 34, 36]

def load_bundle(model_dir):
    with open(model_dir / "xgb_model.pkl", "rb") as f:
        xgb_m = pickle.load(f)
    with open(model_dir / "hgb_model.pkl", "rb") as f:
        hgb_m = pickle.load(f)
    with open(model_dir / "lr_model.pkl", "rb") as f:
        lr_m = pickle.load(f)
    with open(model_dir / "reg_model.pkl", "rb") as f:
        reg_m = pickle.load(f)
    with open(model_dir / "calibrator.pkl", "rb") as f:
        calib_m = pickle.load(f)
    return xgb_m, hgb_m, lr_m, reg_m, calib_m

def evaluate_model(name, models, X_te, y_te):
    xgb_m, hgb_m, lr_m, reg_m, calib_m = models
    p_xgb = xgb_m.predict_proba(X_te)[:, 1]
    p_hgb = hgb_m.predict_proba(X_te)[:, 1]
    p_lr = lr_m.predict_proba(X_te)[:, 1]
    p_raw = 0.40 * p_xgb + 0.35 * p_hgb + 0.25 * p_lr
    p_cal = calib_m.predict(p_raw)
    exp_margins = reg_m.predict(X_te)
    
    auc = roc_auc_score(y_te, p_cal)
    brier = brier_score_loss(y_te, p_cal)
    
    b_home = (p_cal >= 0.68) & (p_xgb > 0.5) & (exp_margins >= 1.8)
    b_away = (p_cal <= 0.32) & (p_xgb < 0.5) & (exp_margins <= -1.8)
    
    n_b = int(np.sum(b_home) + np.sum(b_away))
    w_b = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
    wr = (w_b / n_b * 100.0) if n_b > 0 else 0.0
    roi_140 = ((w_b * 1.40 - n_b) / n_b * 100.0) if n_b > 0 else 0.0
    roi_183 = ((w_b * 1.83 - n_b) / n_b * 100.0) if n_b > 0 else 0.0
    
    return {
        "Model": name,
        "AUC": round(auc, 4),
        "Brier": round(brier, 4),
        "Operative Bets": n_b,
        "Win Rate (%)": round(wr, 2),
        "ROI @ 1.40 (%)": round(roi_140, 2),
        "ROI @ 1.83 (%)": round(roi_183, 2),
    }

def main():
    print("Cargando dataset...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        
    X_all = data["X"]
    y_class = data["y_class"]
    y_reg = data["y_reg"]
    meta = data["meta"]
    
    split_idx = int(len(y_class) * 0.82)
    X_test_37 = X_all[split_idx:]
    X_test_25 = X_all[split_idx:, INDICES_25]
    X_test_27 = X_all[split_idx:, INDICES_27]
    y_test = y_class[split_idx:]
    
    print(f"Evaluando sobre los EXACTOS MISMOS {len(y_test):,} partidos de holdout...")
    
    m_v5 = load_bundle(V5_DIR)
    m_v5_1 = load_bundle(V5_1_DIR)
    m_v5_2 = load_bundle(V5_2_DIR)
    
    res_v5 = evaluate_model("M27_V5 (37 Features Original)", m_v5, X_test_37, y_test)
    res_v5_1 = evaluate_model("M27_V5_1 (25 Features Lean)", m_v5_1, X_test_25, y_test)
    res_v5_2 = evaluate_model("M27_V5_2 (27 Features Restored)", m_v5_2, X_test_27, y_test)
    
    df = pd.DataFrame([res_v5, res_v5_1, res_v5_2])
    print("\n" + "="*85)
    print("COMPARATIVA CARA A CARA SOBRE EXACTAMENTE EL MISMO HOLDOUT (35,717 PARTIDOS):")
    print("="*85)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
