"""
Auditoría exhaustiva de sobreestimación por liga (Overconfidence / Overestimation Audit) para M27 V5.
Analiza si ciertas ligas presentan una probabilidad predicha inflada (Mean P) que no se traduce en Win Rate real.
Determina el umbral mínimo P requerido por cada liga para un escenario pesimista (cuota 1.40 -> WR >= 71.43%).
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_league_audit():
    cache_path = Path("tmp/m27_v5/dataset_v5_cache.pkl")
    if not cache_path.exists():
        print(f"Error: No se encontró {cache_path}")
        return

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    X = cache["X"]
    y = cache["y_class"]
    meta = cache["meta"]
    years = np.array([m["yr"] for m in meta])
    leagues = np.array([m["lg"] for m in meta])
    archs = np.array([m["arch"] for m in meta])

    with open("models/m27_v5/model_outputs/xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("models/m27_v5/model_outputs/hgb_model.pkl", "rb") as f:
        hgb = pickle.load(f)
    with open("models/m27_v5/model_outputs/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("models/m27_v5/model_outputs/calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    # Evaluamos los años 2024 a 2026 (muestra reciente de más de 116,000 partidos)
    test_mask = (years >= 2024)
    X_test = X[test_mask]
    y_test = y[test_mask]
    lg_test = leagues[test_mask]
    arch_test = archs[test_mask]

    print(f"Total partidos auditados (2024-2026): {len(y_test)}")

    p_xgb = xgb.predict_proba(X_test)[:, 1]
    p_hgb = hgb.predict_proba(X_test)[:, 1]
    p_lr = lr.predict_proba(X_test)[:, 1]
    p_ens = 0.40 * p_xgb + 0.40 * p_hgb + 0.20 * p_lr
    p_cal = calibrator.predict(p_ens.reshape(-1, 1))

    unique_lgs, counts = np.unique(lg_test, return_counts=True)
    valid_lgs = unique_lgs[counts >= 60]

    results = []

    for lg in valid_lgs:
        idx = np.where(lg_test == lg)[0]
        p_sub = p_cal[idx]
        y_sub = y_test[idx]
        px_sub = p_xgb[idx]
        ph_sub = p_hgb[idx]
        arch_lg = arch_test[idx][0]

        # Base green: th >= 0.58 con acuerdo de submodelos
        bet_h = (p_sub >= 0.58) & (px_sub > 0.5) & (ph_sub > 0.5)
        bet_a = (p_sub <= 0.42) & (px_sub < 0.5) & (ph_sub < 0.5)
        tb = int(np.sum(bet_h) + np.sum(bet_a))
        if tb < 15:
            continue
        w = int(np.sum(y_sub[bet_h] == 1) + np.sum(y_sub[bet_a] == 0))
        wr_058 = (w / tb) * 100

        # Confident: th >= 0.65
        bet_h65 = (p_sub >= 0.65) & (px_sub > 0.5) & (ph_sub > 0.5)
        bet_a65 = (p_sub <= 0.35) & (px_sub < 0.5) & (ph_sub < 0.5)
        tb65 = int(np.sum(bet_h65) + np.sum(bet_a65))
        w65 = int(np.sum(y_sub[bet_h65] == 1) + np.sum(y_sub[bet_a65] == 0)) if tb65 > 0 else 0
        wr_065 = (w65 / tb65) * 100 if tb65 > 0 else 0.0

        # High Confidence: th >= 0.70
        bet_h70 = (p_sub >= 0.70) & (px_sub > 0.5) & (ph_sub > 0.5)
        bet_a70 = (p_sub <= 0.30) & (px_sub < 0.5) & (ph_sub < 0.5)
        tb70 = int(np.sum(bet_h70) + np.sum(bet_a70))
        w70 = int(np.sum(y_sub[bet_h70] == 1) + np.sum(y_sub[bet_a70] == 0)) if tb70 > 0 else 0
        wr_070 = (w70 / tb70) * 100 if tb70 > 0 else 0.0

        # Promedio de P predicha al apostar
        p_win_pred_list = []
        if np.sum(bet_h) > 0:
            p_win_pred_list.extend(p_sub[bet_h])
        if np.sum(bet_a) > 0:
            p_win_pred_list.extend(1.0 - p_sub[bet_a])
        mean_p_pred = float(np.mean(p_win_pred_list)) * 100 if len(p_win_pred_list) > 0 else 0.0
        overestimation_gap = mean_p_pred - wr_058

        # Umbral mínimo P para alcanzar WR >= 71.43% (ROI > 0 a momio 1.40)
        min_th_140 = "Inviable"
        min_wr_achieved = 0.0
        for th_try in [0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]:
            bh = (p_sub >= th_try) & (px_sub > 0.5) & (ph_sub > 0.5)
            ba = (p_sub <= 1.0 - th_try) & (px_sub < 0.5) & (ph_sub < 0.5)
            tot = int(np.sum(bh) + np.sum(ba))
            if tot >= 8:
                wns = int(np.sum(y_sub[bh] == 1) + np.sum(y_sub[ba] == 0))
                achieved_wr = (wns / tot) * 100
                if achieved_wr >= 71.43:
                    min_th_140 = f"{th_try:.2f}"
                    min_wr_achieved = achieved_wr
                    break

        results.append({
            "league": lg,
            "arch": arch_lg,
            "matches": len(idx),
            "bets_058": tb,
            "wr_058": wr_058,
            "bets_065": tb65,
            "wr_065": wr_065,
            "bets_070": tb70,
            "wr_070": wr_070,
            "mean_p_pred": mean_p_pred,
            "overest_gap": overestimation_gap,
            "min_th_140": min_th_140,
            "min_wr_achieved": min_wr_achieved,
        })

    df_res = pd.DataFrame(results)

    print("\n" + "=" * 105)
    print("TOP 15 LIGAS MÁS SOBREESTIMADAS (TRAMPAS DE OVERCONFIDENCE: ALTO P PREDICHO PERO WR POBRE)")
    print("=" * 105)
    df_over = df_res[df_res["bets_058"] >= 30].sort_values(by="overest_gap", ascending=False).head(15)
    for _, r in df_over.iterrows():
        print(f"{r['league'][:30]:30s} | Arch: {r['arch']:14s} | Matches: {r['matches']:4d} | Bets@.58: {r['bets_058']:3d} | WR@.58: {r['wr_058']:5.1f}% | PredP: {r['mean_p_pred']:5.1f}% | Gap: {r['overest_gap']:+5.1f}% | WR@.65: {r['wr_065']:5.1f}% | MinP@1.4: {r['min_th_140']}")

    print("\n" + "=" * 105)
    print("TOP 15 LIGAS CON MAYOR RENTABILIDAD Y CALIDAD TÁCTICA (WR >= 71.43% A TH 0.58)")
    print("=" * 105)
    df_good = df_res[(df_res["bets_058"] >= 30) & (df_res["wr_058"] >= 71.43)].sort_values(by="wr_058", ascending=False).head(15)
    for _, r in df_good.iterrows():
        print(f"{r['league'][:30]:30s} | Arch: {r['arch']:14s} | Matches: {r['matches']:4d} | Bets@.58: {r['bets_058']:3d} | WR@.58: {r['wr_058']:5.1f}% | PredP: {r['mean_p_pred']:5.1f}% | Gap: {r['overest_gap']:+5.1f}% | WR@.65: {r['wr_065']:5.1f}% | MinP@1.4: {r['min_th_140']}")

    print("\n" + "=" * 105)
    print("ANÁLISIS DE LIGAS MAYORES POPULARES (NBA, EUROLEAGUE, ACB, BBL, PRO A, SERIE A)")
    print("=" * 105)
    target_lgs = ["NBA", "Euroleague", "Liga ACB", "BBL", "Pro A", "Serie A", "CBA", "NBL"]
    for t_lg in target_lgs:
        sub = df_res[df_res["league"].str.contains(t_lg, case=False, na=False)]
        for _, r in sub.head(2).iterrows():
            print(f"{r['league'][:30]:30s} | Arch: {r['arch']:14s} | Matches: {r['matches']:4d} | Bets@.58: {r['bets_058']:3d} | WR@.58: {r['wr_058']:5.1f}% | WR@.65: {r['wr_065']:5.1f}% | WR@.70: {r['wr_070']:5.1f}% | MinP@1.4: {r['min_th_140']}")

    # Guardar reporte completo en CSV para consulta
    out_csv = Path("tmp/m27_v5/league_audit_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nReporte completo guardado en: {out_csv}")

if __name__ == "__main__":
    run_league_audit()
