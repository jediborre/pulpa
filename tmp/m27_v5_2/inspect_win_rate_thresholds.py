"""
tmp/m27_v5_2/inspect_win_rate_thresholds.py
Analiza el Win Rate real de M27_V5_2 a diferentes umbrales de probabilidad y filtros.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")
V5_2_DIR = Path("models/m27_v5_2/model_outputs")
INDICES_27 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 25, 26, 27, 29, 31, 33, 34, 36]

def main():
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        
    X_all = data["X"]
    y_class = data["y_class"]
    meta = data["meta"]
    
    split_idx = int(len(y_class) * 0.82)
    X_te = X_all[split_idx:, INDICES_27]
    y_te = y_class[split_idx:]
    meta_te = meta[split_idx:]
    
    with open(V5_2_DIR / "xgb_model.pkl", "rb") as f: xgb_m = pickle.load(f)
    with open(V5_2_DIR / "hgb_model.pkl", "rb") as f: hgb_m = pickle.load(f)
    with open(V5_2_DIR / "lr_model.pkl", "rb") as f: lr_m = pickle.load(f)
    with open(V5_2_DIR / "reg_model.pkl", "rb") as f: reg_m = pickle.load(f)
    with open(V5_2_DIR / "calibrator.pkl", "rb") as f: calib_m = pickle.load(f)
    
    p_xgb = xgb_m.predict_proba(X_te)[:, 1]
    p_hgb = hgb_m.predict_proba(X_te)[:, 1]
    p_lr = lr_m.predict_proba(X_te)[:, 1]
    p_raw = 0.40 * p_xgb + 0.35 * p_hgb + 0.25 * p_lr
    p_cal = calib_m.predict(p_raw)
    exp_margins = reg_m.predict(X_te)
    
    # Evaluar por umbrales de P
    print("=" * 80)
    print("WIN RATE DE M27_V5_2 POR UMBRAL DE CONFIANZA P:")
    print("=" * 80)
    print(f"{'Umbral P':<12} | {'Apuestas':<10} | {'Win Rate':<10} | {'ROI @ 1.40':<12} | {'ROI @ 1.83':<12}")
    print("-" * 80)
    
    thresholds = [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]
    for th in thresholds:
        b_home = (p_cal >= th) & (p_xgb > 0.5) & (exp_margins >= 1.8)
        b_away = (p_cal <= (1.0 - th)) & (p_xgb < 0.5) & (exp_margins <= -1.8)
        n_b = int(np.sum(b_home) + np.sum(b_away))
        if n_b == 0: continue
        w_b = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
        wr = (w_b / n_b * 100.0)
        roi_140 = ((w_b * 1.40 - n_b) / n_b * 100.0)
        roi_183 = ((w_b * 1.83 - n_b) / n_b * 100.0)
        print(f"P >= {th:.2f}     | {n_b:<10,d} | {wr:6.2f}%    | {roi_140:+6.2f}%     | {roi_183:+6.2f}%")
        
    # Consenso estricto de los 3 modelos (XGB, HGB, LR)
    print("\n" + "=" * 80)
    print("IMPACTO DE CONSENSO TRIMODAL ESTRICTO (XGB > 0.60 & HGB > 0.60):")
    print("=" * 80)
    for th in [0.68, 0.70, 0.72]:
        b_home = (p_cal >= th) & (p_xgb >= 0.58) & (p_hgb >= 0.58) & (exp_margins >= 1.8)
        b_away = (p_cal <= (1.0 - th)) & (p_xgb <= 0.42) & (p_hgb <= 0.42) & (exp_margins <= -1.8)
        n_b = int(np.sum(b_home) + np.sum(b_away))
        if n_b == 0: continue
        w_b = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
        wr = (w_b / n_b * 100.0)
        roi_140 = ((w_b * 1.40 - n_b) / n_b * 100.0)
        roi_183 = ((w_b * 1.83 - n_b) / n_b * 100.0)
        print(f"Consenso P >= {th:.2f} | {n_b:<10,d} | {wr:6.2f}%    | {roi_140:+6.2f}%     | {roi_183:+6.2f}%")

if __name__ == "__main__":
    main()
