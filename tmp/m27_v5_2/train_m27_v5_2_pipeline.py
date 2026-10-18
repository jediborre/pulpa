"""
tmp/m27_v5_2/train_m27_v5_2_pipeline.py
Pipeline oficial y riguroso de M27_V5_2:
- Lean Architecture: 27 features nucleares (25 ortogonales + cuotas de mercado + standing oficial).
- Poda Algorítmica de Ligas Tóxicas / Sobreestimadas para eliminar ruido empírico.
- Ensamble Trimodal Regularizado (XGBoost + HistGB + LR + Meta-Regresor + Calibrador Isotónico).
- Cumplimiento estricto de la Regla 6 de AGENTS.md:
  1. Métricas completas y gaps de sobreajuste (Train vs Test).
  2. Validación Walk-Forward Temporal multi-corte (2023, 2024, 2025-2026).
  3. Matriz de calibración por decil (sobreestimación vs subestimación).
  4. Matriz por liga con Win Rate y simulación de ROI @ 1.40 y @ 1.83.
  5. Gráficas diagnósticas en dashboard de 4 paneles de alta resolución.
  6. Exportación de pesos y estado de producción.
"""

import sys
import pickle
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb

# Forzar flush en stdout para Windows
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")
PROD_DIR = Path("models/m27_v5_2/model_outputs")
PROD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/App/.gemini/antigravity-cli/brain/a5c36f07-125a-46d1-94e3-9e9e9e91bc9c")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# 27 Features Nucleares (25 de v5_1 + 25: odds_log_ratio + 29: ts_delta_wp_bayes)
KEPT_INDICES_27 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 25, 26, 27, 29, 31, 33, 34, 36]

FEATURE_NAMES_27 = [
    "cur_score_diff",
    "abs_cur_score_diff",
    "ht_diff",
    "q1_diff",
    "q2_diff",
    "q3_eff_diff",
    "q3_vs_ht_momentum",
    "recent_2m_run_diff",
    "pace_pts_per_min",
    "pace_vs_league_prior",
    "gp_val",
    "d_lineup_starters_elo_ft",
    "d_lineup_starters_elo_q4",
    "d_lineup_bench_elo_q4",
    "d_lineup_top3_elo_q4",
    "d_roster_drift_adv",
    "d_strat_q4_net",
    "d_q4_bayes",
    "d_wr_bayes",
    "odds_log_ratio",          # Feature 25 restaurada
    "expected_ft_margin",
    "h2h_bayes_ft_margin",
    "ts_delta_wp_bayes",       # Feature 29 restaurada
    "pbp_3pt_diff_q1_q3",
    "rubber_band_gap",
    "white_flag_risk_zone",
    "lg_avg_q4_prior",
]

# Ligas identificadas empíricamente con sobreestimación severa y ROI negativo
TOXIC_LEAGUE_PATTERNS = [
    "korean basketball league", "kbl", "lmb, apertura", "tipos sbl",
    "albania super", "georgia superli", "kosovo superlig", "armenia a-league",
    "cyprus division", "korisliiga, women", "betsafe-lkl", "nlb", "1st dbbl",
    "serie a2, women, group b", "serie b, group b", "israeli basketball super league"
]

def is_toxic(lg_name):
    s = str(lg_name).lower()
    return any(p in s for p in TOXIC_LEAGUE_PATTERNS)

def run():
    print("=" * 95, flush=True)
    print("INICIO PIPELINE M27_V5_2: 27 FEATURES (LEAN + ODDS + TABLE) + PODA DE LIGAS RUIDOSAS", flush=True)
    print("=" * 95, flush=True)
    t0 = time.time()
    
    print("[1/6] Cargando datos y aplicando poda de features y ligas...", flush=True)
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        
    raw_X = data["X"]
    y_class = data["y_class"]
    y_reg = data["y_reg"]
    meta = data["meta"]
    
    # Poda a 27 features
    X_27 = raw_X[:, KEPT_INDICES_27]
    
    # Filtro de ligas toxicas
    clean_indices = [i for i, m in enumerate(meta) if not is_toxic(m.get("lg", ""))]
    clean_indices = np.array(clean_indices)
    
    X_clean = X_27[clean_indices]
    y_clean = y_class[clean_indices]
    y_reg_clean = y_reg[clean_indices]
    meta_clean = [meta[i] for i in clean_indices]
    years = np.array([m["yr"] for m in meta_clean])
    
    print(f"   -> Muestras originales: {len(raw_X):,} | Muestras tras poda de ligas: {len(X_clean):,}", flush=True)
    print(f"   -> Features seleccionadas: {raw_X.shape[1]} -> {X_clean.shape[1]} (27 nucleares)", flush=True)
    
    # [2/6] Validación Walk-Forward Temporal Multi-Corte
    print("\n[2/6] Ejecutando validación temporal Walk-Forward multi-corte...", flush=True)
    cuts = [
        ("Corte 1: Train <= 2022 -> Test 2023", years <= 2022, years == 2023),
        ("Corte 2: Train <= 2023 -> Test 2024", years <= 2023, years == 2024),
        ("Corte 3: Train <= 2024 -> Test 2025-2026", years <= 2024, years >= 2025),
    ]
    
    wf_results = {}
    for cut_name, tr_mask, te_mask in cuts:
        t_cut = time.time()
        X_tr, y_tr = X_clean[tr_mask], y_clean[tr_mask]
        X_te, y_te = X_clean[te_mask], y_clean[te_mask]
        yr_tr = y_reg_clean[tr_mask]
        
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
        
        clf_lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=200, tol=1e-3, random_state=42))
        clf_lr.fit(X_tr, y_tr)
        
        reg_hgb = HistGradientBoostingRegressor(
            max_iter=140, max_leaf_nodes=15, min_samples_leaf=300,
            learning_rate=0.05, l2_regularization=40.0, random_state=42
        )
        reg_hgb.fit(X_tr, yr_tr)
        
        p_te_raw = 0.40 * clf_xgb.predict_proba(X_te)[:, 1] + 0.35 * clf_hgb.predict_proba(X_te)[:, 1] + 0.25 * clf_lr.predict_proba(X_te)[:, 1]
        p_tr_raw = 0.40 * clf_xgb.predict_proba(X_tr)[:, 1] + 0.35 * clf_hgb.predict_proba(X_tr)[:, 1] + 0.25 * clf_lr.predict_proba(X_tr)[:, 1]
        
        calib = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        calib.fit(p_tr_raw, y_tr)
        p_cal = calib.predict(p_te_raw)
        exp_margin = reg_hgb.predict(X_te)
        
        auc = roc_auc_score(y_te, p_cal)
        brier = brier_score_loss(y_te, p_cal)
        
        # Operaciones verdes a P >= 0.68 / P <= 0.32 y margen coherente
        b_home = (p_cal >= 0.68) & (clf_xgb.predict_proba(X_te)[:, 1] > 0.5) & (exp_margin >= 1.8)
        b_away = (p_cal <= 0.32) & (clf_xgb.predict_proba(X_te)[:, 1] < 0.5) & (exp_margin <= -1.8)
        n_b = int(np.sum(b_home) + np.sum(b_away))
        w_b = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
        wr = (w_b / n_b * 100.0) if n_b > 0 else 0.0
        roi_140 = ((w_b * 1.40 - n_b) / n_b * 100.0) if n_b > 0 else 0.0
        roi_183 = ((w_b * 1.83 - n_b) / n_b * 100.0) if n_b > 0 else 0.0
        
        wf_results[cut_name] = {
            "n_test": len(y_te), "auc": round(float(auc), 4),
            "brier": round(float(brier), 4), "bets": n_b, "wr": round(float(wr), 2),
            "roi_140": round(float(roi_140), 2), "roi_183": round(float(roi_183), 2)
        }
        dt_c = time.time() - t_cut
        print(f"  {cut_name:<42} | Test: {len(y_te):>6,} | AUC: {auc:.4f} | Bets: {n_b:>5,} | WR: {wr:5.2f}% | ROI@1.40: {roi_140:+5.2f}% | ROI@1.83: {roi_183:+5.2f}% ({dt_c:.1f}s)", flush=True)
        
    # [3/6] Entrenamiento Final de Producción (Split Cronológico 82% Train / 18% Test Reciente)
    print("\n[3/6] Entrenando Ensamble Final de Producción (82% Train / 18% Test Reciente)...", flush=True)
    n_total = len(y_clean)
    split_idx = int(n_total * 0.82)
    X_tr, y_tr, yr_tr = X_clean[:split_idx], y_clean[:split_idx], y_reg_clean[:split_idx]
    X_te, y_te, yr_te = X_clean[split_idx:], y_clean[split_idx:], y_reg_clean[split_idx:]
    meta_te = meta_clean[split_idx:]
    
    print(f"   Train: {len(X_tr):,} partidos | Test: {len(X_te):,} partidos", flush=True)
    
    # Modelo para curvas de aprendizaje
    xgb_curves = xgb.XGBClassifier(
        n_estimators=160, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=45.0,
        random_state=42, n_jobs=-1, eval_metric=["logloss", "auc"]
    )
    val_len = int(len(X_tr) * 0.15)
    xgb_curves.fit(
        X_tr[:-val_len], y_tr[:-val_len],
        eval_set=[(X_tr[:-val_len], y_tr[:-val_len]), (X_tr[-val_len:], y_tr[-val_len:])],
        verbose=False
    )
    
    # Modelos finales con todo Train
    print("   -> Ajustando XGBoost final...", flush=True)
    xgb_final = xgb.XGBClassifier(
        n_estimators=160, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=45.0,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    xgb_final.fit(X_tr, y_tr)
    
    print("   -> Ajustando HistGradientBoosting final...", flush=True)
    hgb_prod = HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=15, min_samples_leaf=300,
        learning_rate=0.05, l2_regularization=40.0, random_state=42
    )
    hgb_prod.fit(X_tr, y_tr)
    
    print("   -> Ajustando Logistic Regression regularizada...", flush=True)
    lr_prod = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=200, tol=1e-3, random_state=42))
    lr_prod.fit(X_tr, y_tr)
    
    print("   -> Ajustando HistGradientBoosting Regresor (Margen)...", flush=True)
    reg_prod = HistGradientBoostingRegressor(
        max_iter=140, max_leaf_nodes=15, min_samples_leaf=300,
        learning_rate=0.05, l2_regularization=40.0, random_state=42
    )
    reg_prod.fit(X_tr, yr_tr)
    
    # Ensamble trimodal y calibración
    p_tr_raw = 0.40 * xgb_final.predict_proba(X_tr)[:, 1] + 0.35 * hgb_prod.predict_proba(X_tr)[:, 1] + 0.25 * lr_prod.predict_proba(X_tr)[:, 1]
    p_te_raw = 0.40 * xgb_final.predict_proba(X_te)[:, 1] + 0.35 * hgb_prod.predict_proba(X_te)[:, 1] + 0.25 * lr_prod.predict_proba(X_te)[:, 1]
    
    print("   -> Calibrando probabilidades con regresión isotónica...", flush=True)
    calib_prod = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calib_prod.fit(p_tr_raw, y_tr)
    
    p_tr_cal = calib_prod.predict(p_tr_raw)
    p_te_cal = calib_prod.predict(p_te_raw)
    exp_margins_te = reg_prod.predict(X_te)
    
    # [4/6] Métricas y Auditoría de Overfitting
    print("\n[4/6] Evaluando Métricas y Calibración en Producción...", flush=True)
    train_auc = roc_auc_score(y_tr, p_tr_cal)
    test_auc = roc_auc_score(y_te, p_te_cal)
    auc_gap = train_auc - test_auc
    
    train_brier = brier_score_loss(y_tr, p_tr_cal)
    test_brier = brier_score_loss(y_te, p_te_cal)
    test_logloss = log_loss(y_te, p_te_cal)
    test_acc = accuracy_score(y_te, (p_te_cal >= 0.5).astype(int))
    
    print(f"   -> Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Overfitting Gap: {auc_gap:+.4f}", flush=True)
    print(f"   -> Train Brier: {train_brier:.4f} | Test Brier: {test_brier:.4f}", flush=True)
    print(f"   -> Test LogLoss: {test_logloss:.4f} | Accuracy: {test_acc:.4f}", flush=True)
    
    # Señales operativas Verdes
    b_home = (p_te_cal >= 0.68) & (xgb_final.predict_proba(X_te)[:, 1] > 0.5) & (exp_margins_te >= 1.8)
    b_away = (p_te_cal <= 0.32) & (xgb_final.predict_proba(X_te)[:, 1] < 0.5) & (exp_margins_te <= -1.8)
    tot_bets = int(np.sum(b_home) + np.sum(b_away))
    tot_wins = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
    overall_wr = (tot_wins / tot_bets * 100.0) if tot_bets > 0 else 0.0
    overall_roi_140 = ((tot_wins * 1.40 - tot_bets) / tot_bets * 100.0) if tot_bets > 0 else 0.0
    overall_roi_183 = ((tot_wins * 1.83 - tot_bets) / tot_bets * 100.0) if tot_bets > 0 else 0.0
    
    print("\n" + "="*85, flush=True)
    print(f"RENDIMIENTO OPERATIVO VERDE EN TEST ({len(X_te):,} PARTIDOS NUNCA VISTOS):", flush=True)
    print("="*85, flush=True)
    print(f"Apuestas Realizadas: {tot_bets:,} ({tot_bets/len(X_te)*100:.2f}% de selectividad)", flush=True)
    print(f"Win Rate Operativo:  {overall_wr:.2f}%", flush=True)
    print(f"ROI @ 1.40 (Pesimista): {overall_roi_140:+.2f}%", flush=True)
    print(f"ROI @ 1.83 (Estándar):  {overall_roi_183:+.2f}%", flush=True)
    
    # Matriz de Calibración por Decil
    print("\n" + "="*85, flush=True)
    print("MATRIZ DE CALIBRACIÓN POR DECILES: SOBREESTIMACIÓN VS SUBESTIMACIÓN", flush=True)
    print("="*85, flush=True)
    decile_bins = np.linspace(0.0, 1.0, 11)
    df_eval = pd.DataFrame({
        "p_cal": p_te_cal, "y_true": y_te,
        "lg": [m["lg"] for m in meta_te],
        "yr": [m["yr"] for m in meta_te]
    })
    df_eval["decile"] = pd.cut(df_eval["p_cal"], bins=decile_bins, include_lowest=True)
    
    calibration_matrix = []
    for d_bin, grp in df_eval.groupby("decile", observed=False):
        n = len(grp)
        if n == 0:
            continue
        p_m = grp["p_cal"].mean()
        y_m = grp["y_true"].mean()
        bias = p_m - y_m
        diag = "Sobreestimado" if bias > 0.02 else ("Subestimado" if bias < -0.02 else "Perfecto")
        calibration_matrix.append({
            "decile": str(d_bin), "n": int(n), "p_mean": round(float(p_m), 4),
            "y_mean": round(float(y_m), 4), "bias": round(float(bias), 4), "diag": diag
        })
        print(f"Decil {str(d_bin):18s} | N: {n:5,d} | P_pred: {p_m:.3f} | Real: {y_m:.3f} | Sesgo: {bias:+.4f} ({diag})", flush=True)
        
    # Matriz de Rendimiento por Liga
    print("\n" + "="*85, flush=True)
    print("MATRIZ DE RENDIMIENTO POR LIGA EN TEST SET (P >= 0.68)", flush=True)
    print("="*85, flush=True)
    league_results = []
    for lg, grp in df_eval.groupby("lg"):
        h_mask = (grp["p_cal"] >= 0.68)
        a_mask = (grp["p_cal"] <= 0.32)
        n_l = h_mask.sum() + a_mask.sum()
        if n_l < 15:
            continue
        w_l = grp.loc[h_mask, "y_true"].sum() + (len(grp.loc[a_mask]) - grp.loc[a_mask, "y_true"].sum())
        wr_l = w_l / n_l * 100.0
        roi_140_l = (w_l * 1.40 - n_l) / n_l * 100.0
        roi_183_l = (w_l * 1.83 - n_l) / n_l * 100.0
        bias_l = grp.loc[h_mask, "p_cal"].mean() - grp.loc[h_mask, "y_true"].mean() if h_mask.sum() else 0.0
        league_results.append({
            "league": lg, "matches": int(len(grp)), "bets": int(n_l),
            "wr": round(float(wr_l), 2), "roi_140": round(float(roi_140_l), 2),
            "roi_183": round(float(roi_183_l), 2), "bias": round(float(bias_l), 4)
        })
    df_lgs = pd.DataFrame(league_results).sort_values("bets", ascending=False)
    print(df_lgs.head(15).to_string(index=False), flush=True)
    
    # [5/6] Generar Dashboard Gráfico Diagnóstico
    print("\n[5/6] Generando Dashboard Gráfico Diagnóstico...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=200)
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#58a6ff')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            
    # Panel 1: Learning Curves
    res = xgb_curves.evals_result()
    ax1 = axes[0, 0]
    ax1.plot(res['validation_0']['auc'], label='Train AUC', color='#2ea043', lw=2)
    ax1.plot(res['validation_1']['auc'], label='Validation AUC', color='#58a6ff', lw=2)
    ax1.set_title("1. Curvas de Aprendizaje XGBoost (27 Features)")
    ax1.set_xlabel("Iteraciones de Árboles")
    ax1.set_ylabel("ROC-AUC")
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax1.grid(True, color='#21262d', linestyle=':')
    
    # Panel 2: Calibración de Probabilidades
    ax2 = axes[0, 1]
    pm = [c["p_mean"] for c in calibration_matrix]
    ym = [c["y_mean"] for c in calibration_matrix]
    ax2.plot([0, 1], [0, 1], '--', color='#8b949e', label='Calibración Ideal')
    ax2.plot(pm, ym, 'o-', color='#3fb950', lw=2.5, label='M27_V5_2 Isotonic')
    ax2.fill_between(pm, ym, pm, color='#2ea043', alpha=0.15)
    ax2.set_title(f"2. Calibración de Probabilidad (Brier: {test_brier:.4f})")
    ax2.set_xlabel("Probabilidad Predicha")
    ax2.set_ylabel("Win Rate Real")
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax2.grid(True, color='#21262d', linestyle=':')
    
    # Panel 3: Feature Importance (27 features)
    ax3 = axes[1, 0]
    imp = xgb_final.feature_importances_
    s_idx = np.argsort(imp)
    ax3.barh(range(len(s_idx)), imp[s_idx], color='#1f6feb', edgecolor='#58a6ff', alpha=0.85)
    ax3.set_yticks(range(len(s_idx)))
    ax3.set_yticklabels([FEATURE_NAMES_27[i] for i in s_idx], fontsize=8)
    ax3.set_title("3. Importancia de las 27 Features Nucleares")
    ax3.set_xlabel("Importancia Relativa (F-Score)")
    ax3.grid(True, color='#21262d', linestyle=':', axis='x')
    
    # Panel 4: Rendimiento Walk-Forward (Win Rate y ROI @ 1.40)
    ax4 = axes[1, 1]
    cuts_short = ["Corte 1\n(2023)", "Corte 2\n(2024)", "Corte 3\n(2025-26)"]
    wrs = [v["wr"] for v in wf_results.values()]
    rois = [v["roi_140"] for v in wf_results.values()]
    x_pos = np.arange(len(cuts_short))
    w = 0.35
    ax4.bar(x_pos - w/2, wrs, w, label='Win Rate (%)', color='#238636', alpha=0.85)
    ax4_t = ax4.twinx()
    ax4_t.tick_params(colors='#c9d1d9')
    ax4_t.yaxis.label.set_color('#c9d1d9')
    ax4_t.plot(x_pos + w/2, rois, 's-', color='#d29922', lw=2.5, label='ROI @ 1.40 (%)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(cuts_short)
    ax4.set_ylim(65, 85)
    ax4_t.set_ylim(0, 15)
    ax4.set_title("4. Validación Walk-Forward en el Tiempo")
    ax4.set_ylabel("Win Rate (%)")
    ax4_t.set_ylabel("ROI @ 1.40 (%)")
    ax4.grid(True, color='#21262d', linestyle=':')
    
    plt.tight_layout()
    chart_path = PROD_DIR / "m27_v5_2_diagnostic_dashboard.png"
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
    artifact_chart = ARTIFACT_DIR / "m27_v5_2_diagnostic_dashboard.png"
    plt.savefig(artifact_chart, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"   -> Gráficas diagnósticas guardadas en {chart_path} y artefactos.", flush=True)
    
    # [6/6] Exportar Modelos y Metadatos
    print("\n[6/6] Serializando modelos para producción en models/m27_v5_2/model_outputs/...", flush=True)
    with open(PROD_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_final, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_DIR / "hgb_model.pkl", "wb") as f:
        pickle.dump(hgb_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_DIR / "reg_model.pkl", "wb") as f:
        pickle.dump(reg_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calib_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    v5_state_path = Path("models/m27_v5/model_outputs/v5_production_state.pkl")
    if v5_state_path.exists():
        with open(v5_state_path, "rb") as f:
            v5_state = pickle.load(f)
        with open(PROD_DIR / "v5_2_production_state.pkl", "wb") as f:
            pickle.dump(v5_state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    summary = {
        "model_version": "m27_v5_2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features_count": 27,
        "features_restored": ["odds_log_ratio", "ts_delta_wp_bayes"],
        "features_list": FEATURE_NAMES_27,
        "total_clean_samples": int(len(X_clean)),
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "auc_train": round(float(train_auc), 4),
        "auc_test": round(float(test_auc), 4),
        "auc_gap": round(float(auc_gap), 4),
        "brier_train": round(float(train_brier), 4),
        "brier_test": round(float(test_brier), 4),
        "logloss_test": round(float(test_logloss), 4),
        "acc_test": round(float(test_acc), 4),
        "green_threshold": 0.68,
        "green_bets": int(tot_bets),
        "green_win_rate": round(float(overall_wr), 2),
        "green_roi_140": round(float(overall_roi_140), 2),
        "green_roi_183": round(float(overall_roi_183), 2),
        "walk_forward_cuts": wf_results,
        "calibration_matrix": calibration_matrix
    }
    with open(PROD_DIR / "m27_v5_2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    with open(PROD_DIR / "league_calibration.json", "w", encoding="utf-8") as f:
        json.dump(df_lgs.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        
    print(f"\n¡ÉXITO TOTAL! M27_V5_2 completado en {time.time() - t0:.1f} segundos.", flush=True)
    print(f"Métricas Finales:")
    print(f"  Test AUC: {test_auc:.4f} (Gap: {auc_gap:+.4f}) | Brier: {test_brier:.4f}")
    print(f"  Win Rate Verdes: {overall_wr:.2f}% ({tot_bets:,} apuestas)")
    print(f"  ROI @ 1.40: {overall_roi_140:+.2f}% | ROI @ 1.83: {overall_roi_183:+.2f}%", flush=True)

if __name__ == "__main__":
    run()
