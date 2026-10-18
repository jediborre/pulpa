"""
tmp/m27_v5_1/compare_v5_vs_v5_1_same_matches.py
Comparación estricta y transparente de M27_V5 vs M27_V5_1 sobre los EXACTOS MISMOS PARTIDOS.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")
V5_DIR = Path("models/m27_v5/model_outputs")
V5_1_DIR = Path("models/m27_v5_1/model_outputs")

KEPT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 26, 27, 31, 33, 34, 36]

def main():
    print("Cargando dataset...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        
    X_all = data["X"]
    y_class = data["y_class"]
    y_reg = data["y_reg"]
    meta = data["meta"]
    
    # Cargamos modelos de V5
    with open(V5_DIR / "xgb_model.pkl", "rb") as f:
        xgb_v5 = pickle.load(f)
    with open(V5_DIR / "hgb_model.pkl", "rb") as f:
        hgb_v5 = pickle.load(f)
    with open(V5_DIR / "lr_model.pkl", "rb") as f:
        lr_v5 = pickle.load(f)
    with open(V5_DIR / "reg_model.pkl", "rb") as f:
        reg_v5 = pickle.load(f)
    with open(V5_DIR / "calibrator.pkl", "rb") as f:
        calib_v5 = pickle.load(f)
        
    # Cargamos modelos de V5_1
    with open(V5_1_DIR / "xgb_model.pkl", "rb") as f:
        xgb_v5_1 = pickle.load(f)
    with open(V5_1_DIR / "hgb_model.pkl", "rb") as f:
        hgb_v5_1 = pickle.load(f)
    with open(V5_1_DIR / "lr_model.pkl", "rb") as f:
        lr_v5_1 = pickle.load(f)
    with open(V5_1_DIR / "reg_model.pkl", "rb") as f:
        reg_v5_1 = pickle.load(f)
    with open(V5_1_DIR / "calibrator.pkl", "rb") as f:
        calib_v5_1 = pickle.load(f)
        
    # Usamos exactamente el test split de V5 (los últimos 35,526 partidos)
    n_total = len(y_class)
    split_idx = int(n_total * 0.82)
    
    X_test_37 = X_all[split_idx:]
    X_test_25 = X_all[split_idx:, KEPT_INDICES]
    y_test = y_class[split_idx:]
    y_reg_test = y_reg[split_idx:]
    meta_test = meta[split_idx:]
    
    print(f"Total partidos evaluados cara a cara en test: {len(y_test):,}")
    
    # 1. Inferencia V5
    p_xgb_v5 = xgb_v5.predict_proba(X_test_37)[:, 1]
    p_hgb_v5 = hgb_v5.predict_proba(X_test_37)[:, 1]
    p_lr_v5 = lr_v5.predict_proba(X_test_37)[:, 1]
    p_ens_v5 = 0.40 * p_xgb_v5 + 0.35 * p_hgb_v5 + 0.25 * p_lr_v5
    p_cal_v5 = calib_v5.predict(p_ens_v5)
    margin_v5 = reg_v5.predict(X_test_37)
    
    # 2. Inferencia V5_1
    p_xgb_v5_1 = xgb_v5_1.predict_proba(X_test_25)[:, 1]
    p_hgb_v5_1 = hgb_v5_1.predict_proba(X_test_25)[:, 1]
    p_lr_v5_1 = lr_v5_1.predict_proba(X_test_25)[:, 1]
    p_ens_v5_1 = 0.40 * p_xgb_v5_1 + 0.35 * p_hgb_v5_1 + 0.25 * p_lr_v5_1
    p_cal_v5_1 = calib_v5_1.predict(p_ens_v5_1)
    margin_v5_1 = reg_v5_1.predict(X_test_25)
    
    auc_v5 = roc_auc_score(y_test, p_cal_v5)
    auc_v5_1 = roc_auc_score(y_test, p_cal_v5_1)
    
    brier_v5 = brier_score_loss(y_test, p_cal_v5)
    brier_v5_1 = brier_score_loss(y_test, p_cal_v5_1)
    
    print("\n" + "="*80)
    print("MÉTRICAS GLOBALES SOBRE LOS EXACTOS MISMOS 35,526 PARTIDOS DE TEST:")
    print("="*80)
    print(f"M27_V5   | AUC: {auc_v5:.4f} | Brier: {brier_v5:.4f}")
    print(f"M27_V5_1 | AUC: {auc_v5_1:.4f} | Brier: {brier_v5_1:.4f}")
    
    # Evaluar bajo las diferentes reglas de apuesta para entender el Win Rate
    print("\n" + "="*80)
    print("COMPARACIÓN BAJO DISTINTOS UMBRALES DE DISPARO:")
    print("="*80)
    
    for th in [0.65, 0.68, 0.70]:
        # V5
        b_h_v5 = (p_cal_v5 >= th) & (p_xgb_v5 > 0.5) & (margin_v5 >= 1.6)
        b_a_v5 = (p_cal_v5 <= (1-th)) & (p_xgb_v5 < 0.5) & (margin_v5 <= -1.6)
        n_v5 = int(b_h_v5.sum() + b_a_v5.sum())
        w_v5 = int(np.sum(y_test[b_h_v5] == 1) + np.sum(y_test[b_a_v5] == 0))
        wr_v5 = (w_v5 / n_v5 * 100) if n_v5 else 0
        roi140_v5 = (w_v5 * 1.40 - n_v5) / n_v5 * 100 if n_v5 else 0
        
        # V5_1
        b_h_v5_1 = (p_cal_v5_1 >= th) & (p_xgb_v5_1 > 0.5) & (margin_v5_1 >= 1.6)
        b_a_v5_1 = (p_cal_v5_1 <= (1-th)) & (p_xgb_v5_1 < 0.5) & (margin_v5_1 <= -1.6)
        n_v5_1 = int(b_h_v5_1.sum() + b_a_v5_1.sum())
        w_v5_1 = int(np.sum(y_test[b_h_v5_1] == 1) + np.sum(y_test[b_a_v5_1] == 0))
        wr_v5_1 = (w_v5_1 / n_v5_1 * 100) if n_v5_1 else 0
        roi140_v5_1 = (w_v5_1 * 1.40 - n_v5_1) / n_v5_1 * 100 if n_v5_1 else 0
        
        print(f"\n--- UMBRAL P >= {th:.2f} (con filtro de consenso) ---")
        print(f"M27_V5   -> Apuestas: {n_v5:5,d} ({n_v5/len(y_test)*100:.1f}%) | Win Rate: {wr_v5:5.2f}% | ROI@1.40: {roi140_v5:+6.2f}%")
        print(f"M27_V5_1 -> Apuestas: {n_v5_1:5,d} ({n_v5_1/len(y_test)*100:.1f}%) | Win Rate: {wr_v5_1:5.2f}% | ROI@1.40: {roi140_v5_1:+6.2f}%")

    # Comparación pura de probabilidad sin consenso
    print("\n--- COMPARACIÓN PURA (P >= 0.68 sin gates de margen) ---")
    b_h_pure_v5 = p_cal_v5 >= 0.68
    b_a_pure_v5 = p_cal_v5 <= 0.32
    n_p_v5 = b_h_pure_v5.sum() + b_a_pure_v5.sum()
    w_p_v5 = np.sum(y_test[b_h_pure_v5] == 1) + np.sum(y_test[b_a_pure_v5] == 0)
    print(f"V5 Pure   -> Apuestas: {n_p_v5:5,d} | WR: {w_p_v5/n_p_v5*100:5.2f}% | ROI@1.40: {(w_p_v5*1.40-n_p_v5)/n_p_v5*100:+6.2f}%")
    
    b_h_pure_v5_1 = p_cal_v5_1 >= 0.68
    b_a_pure_v5_1 = p_cal_v5_1 <= 0.32
    n_p_v5_1 = b_h_pure_v5_1.sum() + b_a_pure_v5_1.sum()
    w_p_v5_1 = np.sum(y_test[b_h_pure_v5_1] == 1) + np.sum(y_test[b_a_pure_v5_1] == 0)
    print(f"V5_1 Pure -> Apuestas: {n_p_v5_1:5,d} | WR: {w_p_v5_1/n_p_v5_1*100:5.2f}% | ROI@1.40: {(w_p_v5_1*1.40-n_p_v5_1)/n_p_v5_1*100:+6.2f}%")

if __name__ == "__main__":
    main()
