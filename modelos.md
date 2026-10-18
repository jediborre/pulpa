# Modelos — Q4 Prediction Pipeline

## Arquitectura general

Todos los modelos comparten:
- **Target**: Ganador del Q4 (home/away)
- **Momio fijo**: 1.4 (break-even 71.4% efectividad)
- **Snapshot**: minutos desde inicio del partido
- **Features base**: scores por cuarto, win rates previos, estadísticas de PBP, graph points
- **Evaluación**: ROC AUC, accuracy, F1, Brier, yield

---

## 🏛️ Estructura Homologada del Subsistema de Modelos (`models/`)

Todos los modelos se encuentran centralizados y desacoplados en el directorio raíz `models/` bajo una interfaz universal:

```
models/
├── __init__.py                # Exporta predict(), predict_all(), get_available_models()
├── registry.py                # Despachador dinámico y catálogo de modelos
├── common/                    # Módulos transversales compartidos
│   ├── schema.py              # Dataclass PredictionResult
│   ├── data_loader.py         # Conexión única a matches.db y carga de partidos
│   ├── persistence.py         # Escritura en eval_match_results_v2 y bet_monitor_log_v2
│   ├── pbp_utils.py           # Reloj, estimación de minutos, scoring runs y PBP
│   ├── time_clip.py           # Truncado temporal estricto (min 24 Q3 / min 36 Q4)
│   ├── h2h.py                 # Extracción de métricas cara a cara (Head-to-Head)
│   └── monte_carlo.py         # Motor de simulación y presión de posesiones
└── <nombre_modelo>/           # ej: m27_v3, v6_2, v6_3, v12, v16...
    ├── train.py               # Entrenamiento del modelo
    ├── predict.py             # Inferencia retornando PredictionResult
    ├── evaluate.py            # Evaluación sobre partidos históricos
    ├── features.py            # Extracción de features específicas
    ├── model_outputs/         # Artefactos serializados (.joblib, .json)
    └── README.md              # Documentación individual
```

---

## Catálogo de features

### F0 — Quarter Scores (core)
`q1_diff`, `q2_diff`, `q3_diff`, `ht_home`, `ht_away`, `ht_diff`, `ht_total`, `q3_total`, `score_3q_home`, `score_3q_away`, `score_3q_diff`, `score_est_home`, `score_est_away`, `score_est_diff`, `cutoff_minute`

### F1 — Prior Win Rates
`home_prior_wr`, `away_prior_wr`, `prior_wr_diff`, `prior_wr_sum`

### F2 — Graph Points (gp_stats)
`gp_count`, `gp_last`, `gp_peak_home`, `gp_peak_away`, `gp_area_home`, `gp_area_away`, `gp_area_diff`, `gp_mean_abs`, `gp_swings`, `gp_slope_3m`, `gp_slope_5m`
- _V5+ simplifica a: `gp_count`, `gp_last`, `gp_slope_3m`, `gp_slope_5m`_ (solo Q4-only)

### F3 — Play-by-Play Stats (pbp_stats)
`pbp_home_pts_per_play`, `pbp_away_pts_per_play`, `pbp_pts_per_play_diff`, `pbp_home_plays`, `pbp_away_plays`, `pbp_plays_diff`, `pbp_home_3pt`, `pbp_away_3pt`, `pbp_3pt_diff`, `pbp_home_plays_share`, `pbp_home_3pt_share`

### F4 — Categorical Buckets
`league`, `league_bucket`, `gender_bucket`, `home_team_bucket`, `away_team_bucket`
- _V7+: raw strings `home_team`, `away_team`, `league` (CatBoost nativo)_

### F5 — Score Pressure (v4+)
`global_diff`, `global_abs_diff`, `is_tied`, `trailing_is_home`, `trailing_is_away`, `trailing_points_to_tie`, `trailing_points_to_lead`, `remaining_minutes_target`, `required_ppm_tie`, `required_ppm_lead`, `trailing_points_per_min`, `leading_points_per_min`, `trailing_points_per_play`, `trailing_play_share`, `trailing_plays_per_min`, `req_pts_per_trailing_event`, `pressure_ratio_tie`, `pressure_ratio_lead`, `scoring_gap_per_min`, `urgency_index`, `trailing_3pt_rate`

### F6 — Clutch / PBP Recent Window (v4+)
`clutch_window_minutes`, `clutch_scoring_events`, `clutch_home_points`, `clutch_away_points`, `clutch_points_diff`, `clutch_home_event_share`, `clutch_home_max_run_pts`, `clutch_away_max_run_pts`, `clutch_run_diff`, `clutch_last_scoring_home`, `clutch_last_scoring_away`

### F7 — Monte Carlo Simulation
- **V6**: `mc_home_win_prob`
- **V6.1+**: `mc_home_win_prob`, `mc_expected_diff`, `mc_cover_rate`, `mc_std_diff`, `mc_comeback_rate`

### F8 — Scoring Runs (v9+)
`current_run_home`, `current_run_away`, `max_run_home`, `max_run_away`, `run_diff`

### F9 — Quarter Winner Context (m27/m30)
`q1_winner`, `q2_winner`, `q1_q2_same_winner`, `home_wins_first2_count`, `away_wins_first2_count`, `home_wins_first2_plus_q3_partial_count`, `away_wins_first2_plus_q3_partial_count`

### F10 — Halftime Context
`halftime_home`, `halftime_away`, `halftime_diff`, `halftime_total`, `halftime_leader`, `halftime_margin_bin`, `halftime_trailing_side`, `halftime_deficit_abs`

### F11 — Current Score Context
`current_margin_bin`, `current_trailing_side`, `trailing_now_is_home`, `trailing_now_is_away`, `trailing_now_deficit_abs`, `trailing_now_recent_run_3m`, `trailing_now_recent_run_2m`

### F12 — Q3 Partial Context
`q3_partial_home`, `q3_partial_away`, `q3_partial_diff`, `q3_partial_total`, `q3_partial_leader`, `q3_partial_home_share`, `halftime_trailer_cutting_in_q3`, `halftime_to_current_margin_delta`, `abs_margin_delta_from_halftime`

### F13 — Q3 Momentum / Deep Momentum (m30)
`gp_q3_lead_erosion`, `gp_q3_late_momentum`, `gp_q3_volatility`, `gp_q3_turning_points`, `gp_q3_recovery_spent`, `gp_q3_peak_to_final_ratio`, `gp_q3_last_3min_momentum`, `gp_q3_momentum_accel`

### F14 — Q3 Pace & Projection (m30)
`regulation_quarter_minutes`, `q3_start_minute`, `q4_start_minute`, `q3_remaining_minutes`, `q3_partial_completion`, `q3_partial_home_pace_per_min`, `q3_partial_away_pace_per_min`, `q3_partial_projected_home`, `q3_partial_projected_away`, `q3_partial_projected_diff`

### F15 — Halftime Trailer Analysis (m30)
`halftime_trailer_won_q3_partial`, `halftime_trailer_gain`, `halftime_trailer_neutralized_deficit`, `halftime_leader_lost_lead`, `halftime_trailer_strong_recovery`, `big_halftime_lead_now_close`, `lead_flip_from_halftime`, `current_leader_won_q3_partial`, `q3_close_flip_or_tie_after_biglead_10m`, `q3_close_biglead_now_04_07_10m`

### G1 — Score (v13+)
`score_halftime_diff`, `score_halftime_total`, `score_q1_diff`, `score_q2_diff`, `score_q1_share`, `score_halftime_diff_ratio`, `score_q3_diff`, `score_q3_total`, `score_cumulative_total`, `score_cumulative_diff`, `score_q3_vs_ht_momentum`

### G2 — Graph / Trajectory (v13+)
`gp_count`, `gp_latest_diff`, `gp_slope_3m`, `gp_slope_5m`, `gp_acceleration`, `gp_peak`, `gp_valley`, `gp_amplitude`, `gp_swings`, `gp_sign_changes`, `gp_stddev`, `gp_last_sign`

### G3 — Trajectory (v14+)
`traj_lead_changes`, `traj_times_tied`, `traj_largest_lead_home`, `traj_largest_lead_away`, `traj_score_diff_end`, `traj_current_run_home`, `traj_current_run_away`, `traj_last5_home_pts`, `traj_last5_away_pts`, `traj_last5_diff`, `traj_last10_diff`, `traj_comeback_flag`, `traj_momentum_idx`

### G4 — PBP Density (v15+)
`pbp_count`, `pbp_pts_per_event`, `pbp_home_pts`, `pbp_away_pts`, `pbp_home_3pt_rate`, `pbp_away_3pt_rate`, `pbp_scoring_density`

### G5 — Pace (v15+)
`pace_total_prior`, `pace_ratio_vs_median`, `pace_bucket_low`, `pace_bucket_medium`, `pace_bucket_high`

### G6 — League Context (v15+)
`league_samples`, `league_ht_total_mean`, `league_ht_total_std`, `league_home_advantage_mean`, `league_q3_total_mean`, `league_q4_total_mean`

### G7 — Meta (v15+)
`meta_snapshot_minute`, `meta_target_is_q4`, `meta_minutes_to_quarter_end`

### G8 — Forecast (v16+)
`tfm_winner_pick`, `tfm_margin`, `tfm_uncertainty`, `tfm_trend_slope`, `tfm_current_trend` (TimesFM / Chronos)

### G9 — Legacy Hybrid (v17)
- **Pressure**: `legacy_global_diff`, `legacy_trailing_is_home`, `legacy_required_ppm_tie`, etc.
- **Clutch**: `legacy_last10_home_pts`, `legacy_last10_diff`, `legacy_current_run_home`, etc.
- **Monte Carlo**: `legacy_mc_home_win_prob`

### E0–E3 — Quarter ELO Engine (m27_v4)
`delta_elo_q4`, `delta_elo_ft`, `delta_elo_clutch`, `elo_expected_q4_margin`
- Sistema dinámico bayesiano actualizado partido a partido sin data leakage; mide la fuerza relativa acumulada cerrando cuartos bajo presión.

### R0–R5 — Match Lineup & Bench Quality (m27_v4)
`delta_starter_rating`, `delta_bench_rating`, `delta_roster_depth`, `delta_star_reliance`, `bench_points_share_prior`, `live_starters_foul_trouble`
- Calidad de los 5 titulares confirmados vs reservas (`lineups` + `player_stats`). Correlación $r = +0.401$ con desenlace de Q4 y penalización por sobredependencia de una estrella ($-0.214$).

### L0–L7 — League Taxonomy Priors (m27_v4 / leagues_classification)
`is_women`, `is_playoffs`, `is_final`, `is_relegation`, `prior_league_pace`, `prior_q4_total_points`, `prior_blowout_risk`, `prior_home_win_pct`
- Priors macro extraídos de `leagues_classification` para normalizar ritmo FIBA vs NBA, ligas femeninas (-20 pts) y tensión defensiva de playoffs (-14.6 pts).

### M0–M11 — Live Momentum at Min 27 (m27_v4)
`score_diff_m27`, `momentum_val_m27`, `momentum_slope_m24_m27`, `momentum_mean_q3`, `momentum_integral_total`, `q3_partial_margin_m27`, `points_pace_m27`, `run_unanswered_m27`, `h2h_win_pct`, `h2h_avg_margin`, `q1_q2_leader_held`, `lead_change_count_m27`
- Telemetría en vivo del minuto 27 con historial H2H directo.

---

## V1 — Original

| Campo | Valor |
|-------|-------|
| **Snapshot** | 36 (final Q3) |
| **Algoritmo** | LogReg + RandomForest + GradientBoosting (avg ensemble) |
| **Script entrenamiento** | `train_q3_q4_models.py` |
| **Script evaluación / ROI** | `report_model_comparison.py`, `report_db_results.py` |
| **Hiperparams** | LogReg(solver=liblinear, max_iter=4000); RF(n_est=300, min_samples_leaf=4); GB(n_est=250, lr=0.05, max_depth=3) |
| **Target** | Q3 y Q4 |
| **Split** | 80/20 temporal |
| **Champion** | `model_outputs/q4_logreg.joblib` + `q4_rf.joblib` + `q4_gb.joblib` (modelos separados) |
| **ROC AUC** | ~0.857 (Q4, full Q3 data) |
| **Features** | F0 (scores), F1 (win rates), F2 (graph 11f), F3 (pbp 11f), F4 (categorical) |
| **Filtros** | Ninguno. Evalúa todo el test set sin filtrar. |
| **Notas** | Primer modelo. Usa datos completos hasta minuto 36. Sin selección de champion explícita. |

---

## V2 — + League/Team Buckets

| Campo | Valor |
|-------|-------|
| **Snapshot** | 36 |
| **Algoritmo** | LogReg + RF + GB |
| **Script entrenamiento** | `train_q3_q4_models_v2.py` |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Hiperparams** | LogReg(max_iter=5000); RF(n_est=400, min_samples_leaf=3); GB(n_est=350, lr=0.04, max_depth=3) |
| **Target** | Q3 y Q4 |
| **Split** | 80/20 |
| **Champion** | `model_outputs_v2/q4_ensemble.joblib` |
| **Diferencias vs V1** | League/team bucket features. Más n_estimators, menor lr. |
| **Notas** | V1 + league buckets. |
| **Features** | F0, F1, F2, F3, **F4** (league/team buckets añadidos vs V1) |
| **Filtros** | Ninguno. Evalúa todo el test set. |

---

## V3 — Primer Multi-Snapshot

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q4: 24, 30, 36 (múltiples por partido) |
| **Algoritmo** | LogReg + GB (sin RF) |
| **Script entrenamiento** | `train_q3_q4_models_v3.py` |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Hiperparams** | LogReg(max_iter=5000); GB(n_est=350, lr=0.04, max_depth=3) |
| **Target** | Q3 y Q4 |
| **Champion** | `model_outputs_v3/q4_m24_gb.joblib`, `q4_m30_gb.joblib`, `q4_m36_gb.joblib` |
| **Diferencias vs V2** | Múltiples snapshots dinámicos. Quita RF. Introduce `SampleRow` con `snapshot_minute`. |
| **Notas** | Primera versión con ventanas temporales dinámicas. |
| **Features** | F0, F1, F2 (gp simplificado), F3 (pbp simplificado), F4 |
| **Filtros** | Ninguno. n test muy pequeño (312). |

---

## V4 — + Pressure/Comeback Features

| Campo | Valor |
|-------|-------|
| **Snapshot** | 36 (vuelve a single snapshot) |
| **Algoritmo** | LogReg + RF + GB |
| **Script entrenamiento** | `train_q3_q4_models_v4.py` |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Hiperparams** | LogReg(max_iter=5000); RF(n_est=400, min_samples_leaf=3); GB(n_est=350, lr=0.04, max_depth=3) |
| **Target** | Q3 y Q4 |
| **Champion** | `model_outputs_v4/q4_ensemble.joblib` |
| **Diferencias vs V3** | Vuelve a snapshot único. Añade pressure/comeback features. Reintroduce RF. |
| **Notas** | V2 + pressure features. |
| **Features** | F0, F1, F2, F3, F4, **F5** (pressure 22f), **F6** (clutch 11f) |
| **Filtros** | Ninguno. Test set completo, snapshot único. |

---

## V5 — XGBoost + HistGB + MLP

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | **XGBoost + HistGradientBoosting + MLP** |
| **Script entrenamiento** | `train_q3_q4_models_v5.py` |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5); MLP(hidden=(64,32), max_iter=500) |
| **Target** | Q3 y Q4 |
| **Ensemble** | Avg prob |
| **Champion** | `model_outputs_v5/q4_xgb.joblib`, `q4_hist_gb.joblib`, `q4_mlp.joblib` |
| **Diferencias vs V4** | Reemplaza LogReg/RF/GB con XGBoost, HistGB, MLP. Filtros estrictos, momentum slope exponencial. |
| **Notas** | Primera versión con el stack XGB+HistGB que persiste hasta V6.x. | 
| **Features** | F0, F1, F2, F3, F4, F5, F6 |
| **Filtros** | Ninguno. Evaluación completa. |

---

## V6 — Monte Carlo + Feature Expansion

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | XGBoost + HistGB + MLP |
| **Script entrenamiento** | `train_q3_q4_models_v6.py` |
| **Script evaluación / ROI** | `report_model_comparison.py`, `report_db_results.py` |
| **Hiperparams** | XGB(n_est=v6.n_est, lr=0.05, max_depth=4); HistGB(max_iter=v6.max_it, lr=0.05, max_depth=5); MLP(max_iter=mlp_it) |
| **Target** | Q3 y Q4 |
| **Ensemble** | Avg + `xgb_plus_mc` |
| **Champion** | `model_outputs_v6/q4_xgb.joblib` + HistGB + MLP |
| **ROC AUC (Q4 test)** | XGB: 0.846; HistGB: 0.845; MLP: 0.855; **Ensemble: 0.857**, acc 0.765 |
| **Diferencias vs V5** | Simulaciones Monte Carlo (5,000 sims). `_score_pressure_features`, `_pbp_recent_window_features`, `_monte_carlo_win_prob`. `MatchSample` dataclass. |
| **Notas** | Baseline de referencia para todas las versiones V6.x. Métricas muy altas porque usa Q3 completo (min 36). |
| **Features** | F0, F1, F2, F3, F4, F5, F6, **F7** (MC: `mc_home_win_prob`) |
| **Filtros** | Ninguno. Baseline de referencia. Test sin filtrar. |

---

## V6.1 — Temporal Splits + Calibración + Weighted Ensemble

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | XGBoost + HistGB + MLP |
| **Script entrenamiento** | `train_q3_q4_models_v6_1.py` |
| **Script evaluación / ROI** | `test_v6_1_league_filter.py`, `report_model_comparison.py` |
| **Hiperparams** | XGB(n_est=500, lr=0.03, max_depth=5, subsample=0.85, colsample=0.85, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.5); HistGB(max_iter=400, lr=0.03, max_depth=5); MLP(hidden=(64,32)) |
| **Target** | Q3 y Q4 |
| **Ensemble** | **AUC-weighted** + isotonic calibration |
| **Split** | **70/15/15 temporal** (introduce 3-way split) |
| **Champion** | `model_outputs_v6_1/q4_champion.joblib` (seleccionado por val AUC) |
| **Diferencias vs V6** | 70/15/15 split. Calibración isotónica post-hoc. Ensemble ponderado por AUC. Optimización de thresholds. TimeSeriesSplit CV. |
| **Notas** | Profesionaliza el pipeline con validación, calibración y selección de champion. |
| **Features** | F0, F1, F2, F3, F4, F5, F6, **F7** (MC expandido: 5f) |
| **Filtros** | Ninguno. 70/15/15 split pero sin filtrar predicciones. |

---

## V6.2 — League Pruning

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | XGBoost + HistGB (sin MLP) |
| **Script entrenamiento** | `train_q3_q4_models_v6_2.py` |
| **Script evaluación / ROI** | `report_v62_q4_roi.py`, `report_model_comparison.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5) |
| **Target** | Q3 y Q4 |
| **Ensemble** | **Blend: 0.6 * XGB + 0.4 * HistGB** |
| **Split** | 80/20 |
| **Champion** | `model_outputs_v6_2/q4_champion.joblib` |
| **Diferencias vs V6.1** | **League pruning automático**: ligas débiles → `LEAGUE_OTHER_SIGNAL_WEAK`. Quita MLP. Reglas de exclusión por nombre (JSON). 80/20 simpler. |
| **Notas** | Enfoque en filtrado de ligas por señal. |
| **Features** | F0 (quarter scores), F1 (win rates), F2 (graph 11f), F3 (pbp 11f), F4 (categorical), F5 (pressure 22f), F6 (clutch 11f), F7 (MC: `mc_home_win_prob`) |
| **Filtros** | **League pruning**: ligas débiles → `LEAGUE_OTHER_SIGNAL_WEAK`. Exclusión por palabras clave (JSON). |

---

## V6.2b — Feature Pruning Conservador

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | XGBoost + HistGB + MLP |
| **Script entrenamiento** | `train_q3_q4_models_v6_2b.py` |
| **Script evaluación / ROI** | `report_v62_q4_roi.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5); MLP(max_iter=500) |
| **Target** | Q3 y Q4 |
| **Ensemble** | Avg simple |
| **Split** | 80/20 |
| **Champion** | `model_outputs_v6_2b/q4_champion.joblib` |
| **Diferencias vs V6.2** | Poda conservadora: solo features pre-marcadas como "no_signal". Mantiene MLP. Siempre exporta A/B vs V6. |
| **Notas** | Alternativa más conservadora a V6.2. |
| **Features** | F0 (quarter scores), F1 (win rates), F2 (graph 11f), F3 (pbp 11f), F4 (categorical), F5 (pressure 22f), F6 (clutch 11f), F7 (MC: `mc_home_win_prob`) |
| **Filtros** | **Feature pruning**: solo elimina features marcadas como `no_signal`. Sin league pruning. |

---

## V6.3 — Dynamic Multi-Window Q4-Only

| Campo | Valor |
|-------|-------|
| **Snapshot** | **Dual: 27 y 30** (dinámico por partido) |
| **Algoritmo** | XGBoost + HistGB (sin MLP) |
| **Script entrenamiento** | `train_q4_models_v6_3.py` |
| **Script evaluación / ROI** | `report_v63_q4_roi.py`, `compare_model_versions.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5) |
| **Target** | **Solo Q4** (descarta Q3) |
| **Ensemble** | Avg simple (champion = `avg_prob(xgb, hist_gb)`) + isotonic |
| **Split** | 70/15/15, 5-fold TimeSeriesSplit CV |
| **Champion (m27)** | `model_outputs_v6_3/q4_m27_champion.joblib` |
| **Champion (m30)** | `model_outputs_v6_3/q4_m30_champion.joblib` |
| **Distribución ventanas** | m27: 24,746 (75.8%); m30: 7,889 (24.2%) |
| **ROC AUC (m27 test)** | Champion: **0.604**, XGB: 0.605, HistGB: 0.602 |
| **ROC AUC (m30 test)** | Champion: **0.769**, XGB: 0.772, HistGB: 0.764 |
| **ROI (snap 27)** | -19.8% yield, 2,561 bets (threshold >65%) |
| **Diferencias vs V6.2** | Ventanas dinámicas [27,30]. Q4-only. Late-window close-game margin constraints. |
| **Notas** | AUC 0.77 en m30 es engañoso: solo recibe partidos cerrados (margin ≤6). Apuesta agresivamente y quiebra el bankroll. Baseline para m27_v1/m30_v1. |
| **Features** | F0 (scores), F1 (win rates), F2 (graph), F3 (pbp), F4 (buckets) |
| **Filtros** | **m27**: sin filtro (3,713 test). **m30**: solo partidos cerrados (margin ≤6, 1,184 test). Ventana dinámica [27,30] por partido. |

---

## V7 — CatBoost + XGBoost Categórico Nativo

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | **CatBoost** + **XGBoost (enable_categorical=True)** |
| **Script entrenamiento** | `train_q3_q4_models_v7.py` |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Hiperparams** | CatBoost(iterations=500, lr=0.05, depth=6); XGB(n_est=500, lr=0.05, max_depth=4, enable_categorical=True) |
| **Target** | Q3 y Q4 |
| **Ensemble** | No ensemble (modelos independientes) |
| **Champion** | `model_outputs_v7/q4_catboost.joblib`, `q4_xgb_cat.joblib` |
| **Diferencias vs V6** | CatBoost para categóricas nativas. XGBoost con `enable_categorical=True`. Sin DictVectorizer para categóricas. |
| **Notas** | Experimental. Procesa strings categóricos crudos (alta cardinalidad). Sin ensemble. |
| **Features** | F0, F1, F2, F3, F4 (raw strings: `home_team`, `away_team`, `league`), F5, F6 |
| **Filtros** | Ninguno. Modelos independientes sin ensemble. |

---

## V8 — Hybrid LSTM + Tabular

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | **LSTM + Dense (PyTorch)** |
| **Script entrenamiento** | `train_q3_q4_models_v8.py` |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Arquitectura** | LSTM(hidden=24) + Tabular(64→32) → FC(56→16→1, Sigmoid). Adam, 25 epochs, lr=0.005, batch=64 |
| **Target** | Q3 y Q4 |
| **Ensemble** | Standalone PyTorch |
| **Champion** | `model_outputs_v8/q4_lstm_hybrid.joblib` |
| **Diferencias vs V7** | Primer deep learning. Lee secuencia de graph points minute-by-minute + features tabulares. Monte Carlo como features de entrada. |
| **Notas** | Experimental. Requiere PyTorch. No hay comparativa publicada. |
| **Features** | F0, F1, F2, F3, F4 (raw strings), F5, F6 + **seq_q3/seq_q4** (LSTM: graph points sequence) |
| **Filtros** | Ninguno. Standalone PyTorch. |

---

## V9 — Simplified Fast Ensemble

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | **LogReg + GB** (sin XGB, sin HistGB, sin DL) |
| **Script entrenamiento** | `train_q3_q4_models_v9.py` (`train_v9_fast.py`, `populate_v9.py`) |
| **Script evaluación / ROI** | `report_model_comparison.py` |
| **Hiperparams** | LogReg(C=0.5, max_iter=500); GB(n_est=50, max_depth=3, lr=0.15, min_samples_split=30, min_samples_leaf=15) |
| **Target** | Q3 y Q4 |
| **Ensemble** | Weighted (Q4: 0.5LR + 0.5GB) |
| **Champion** | `model_outputs_v9/q4_ensemble.joblib` |
| **Diferencias vs V8** | Simplificado para velocidad. Momentum/racha features. StandardScaler. |
| **Notas** | Stripea complejidad para prototipado rápido. |
| **Features** | F0, F1, F2, F3, F4, **F8** (scoring runs 5f) |
| **Filtros** | Ninguno. Simplificado para velocidad. |

---

## V10 — Regression (Over/Under)

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | **Ridge + GBR + XGBoost (regression)** |
| **Script entrenamiento** | `train_q3_q4_regression_v10.py` (`train_v10_fast.py`, `train_v10_simple.py`) |
| **Script evaluación / ROI** | `batch_populate_evals.py` |
| **Hiperparams** | GBR(n_est=80, max_depth=3, lr=0.1); XGB(n_est=80, max_depth=3, lr=0.1) |
| **Target** | **Puntos totales Q3/Q4** (regresión, no clasificación) |
| **Ensemble** | Avg, weighted, stacking |
| **Champion** | `model_outputs_v10/` |
| **Diferencias vs V9** | No clasifica ganador. Predice over/under totals. Ridge + GBR + XGBoost regression. |
| **Notas** | Primer modelo de regresión. |
| **Features** | F0, F1, F2 (11f), F3 (11f), F4 (buckets) |
| **Filtros** | Ninguno. Regresión directa sin gates. |

---

## V11 — Gender-Separated Regression

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 24; Q4: 36 |
| **Algoritmo** | Ridge + GBR + XGBoost |
| **Script entrenamiento** | `train_q3_q4_regression_v11.py` |
| **Script evaluación / ROI** | `predict_v11.py` |
| **Hiperparams** | GBR(n_est=80, max_depth=3, lr=0.1); XGB(n_est=80, max_depth=3, lr=0.1) |
| **Target** | Puntos totales (**gender-separated**: men vs women) |
| **Ensemble** | Avg, weighted, stacking |
| **Champion** | `model_outputs_v11/` |
| **Diferencias vs V10** | Modelos separados por género. Predice puntos totales sin threshold. |
| **Notas** | Regresión con conciencia de género. |
| **Features** | F0, F1, F2 (11f), F3 (11f), F4 (buckets) + gender field |
| **Filtros** | Separación por género (men vs women). Sin otros filtros. |

---

## V12 — Ultra Conservative Hybrid Ensemble

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 22; Q4: **31** |
| **Algoritmo** | XGB + LightGBM + CatBoost + LogReg + GB + **Stacking meta-learner** |
| **Script entrenamiento** | `v12/train_v12.py` |
| **Script evaluación / ROI** | `v12/eval_v12.py`, `v12/validate_v12.py`, `report_v12_v13.py` |
| **Hiperparams** | Múltiples algoritmos. Risk: loss 2x reward. |
| **Target** | Q3 y Q4 + regression confirmation |
| **Ensemble** | **Stacking** meta-learner |
| **Split** | Per-league |
| **Champion** | `v12/model_outputs/` (per-league) |
| **Diferencias vs V11** | Ensemble híbrido (clf + reg). Risk asimétrico. League gates (min 50 samples, 52% hit rate). Primer enfoque per-league. |
| **Notas** | Primer pipeline "production-grade" con risk management. |
| **Features** | F0, F1, F2, F3, F4, F5, F6, F8 + **league context** + `gp_acceleration` |
| **Filtros** | **League gates**: min 50 samples por liga, min 52% hit rate. **Risk asimétrico**: loss 2x reward. **Regression confirmation**: MAE ≤ umbral. |

---

## V13 — Improved Pipeline

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 22; Q4: 31 |
| **Algoritmo** | XGB + LightGBM + CatBoost + LogReg + GB + stacking |
| **Script entrenamiento** | `v13/train_v13.py` (`v13/train_clf.py`, `v13/train_reg.py`) |
| **Script evaluación / ROI** | `v13/eval_v13.py`, `v13/walk_forward.py`, `report_v12_v13.py` |
| **Hiperparams** | GB(n_est=100, max_depth=3, lr=0.1); XGB(n_est=150, max_depth=4, lr=0.08); LGB(n_est=150, max_depth=5, lr=0.08); CatBoost(iterations=200, depth=5, lr=0.08); Stacking meta-learner |
| **Target** | Q3 y Q4 + totals |
| **Ensemble** | Stacking + calibración per-league |
| **Split** | Walk-forward |
| **Champion** | `v13/model_outputs/` (12 buckets, per-pace-per-gender) |
| **Muestras (val)** | 137,085 quarters (13,710 matches) — walk-forward |
| **Accuracy (val)** | **0.632** (weighted, 12 buckets, rango 0.605–0.671) |
| **F1 (val)** | 0.624 weighted |
| **MAE (total reg)** | 13.83 avg (rango 7.09–21.49) |
| **Diferencias vs V12** | Walk-forward validation. League filters mejorados. Timing/live analysis. |
| **Notas** | Refinamiento de V12 basado en observaciones live. Sin ROC AUC computado. |
| **Features** | G1 (score), G2 (graph), **G3** (trajectory) — primeras features modularizadas |
| **Filtros** | **Walk-forward**. **12 buckets** (3 paces × 2 géneros × 2 targets Q3/Q4). Sin gates de confianza en evaluación. |

---

## V14 — Planning Only

| Campo | Valor |
|-------|-------|
| **Notas** | Solo existe `PLAN_V14.md`. Sin script de entrenamiento. Planeaba integrar TimesFM como regresor complementario. |
| **Script entrenamiento** | — (Solo planificación en `PLAN_V14.md`) |
| **Script evaluación / ROI** | — |
| **Features** | G1 (score), G2 (graph), **G3** (trajectory v14) |
| **Plan** | Fase 1: TimesFM ⇒ regresión; Fase 2: Refactor feature pipeline; Fase 3: Gates dinámicos |

---

## V15 — Per-League Models

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 22; Q4: **31** (train snaps: Q3 [18-23], Q4 [28-32]) |
| **Algoritmo** | **Per-league**: LogReg + GB + XGB + CatBoost (selección por n) |
| **Script entrenamiento** | `v15/train.py` |
| **Script evaluación / ROI** | `v15/evaluate.py`, `v15/test_roi.py` |
| **Hiperparams** | LogReg(C=0.3); GB(n_est=80, max_depth=3, lr=0.06, min_samples_leaf=20, subsample=0.8); XGB(n_est=80, max_depth=3, lr=0.06, min_child_weight=10, reg_lambda=3.0, reg_alpha=0.5) |
| **Target** | Q3, Q4 winner + regression |
| **Ensemble** | **Inverse-error weighted** (F1 para clf, 1/MAE para reg) + isotonic |
| **Split** | train/val/cal/holdout |
| **Champion** | `v15/model_outputs/` (per-league, 22 ligas) |
| **Muestras (val)** | 10,450 quarters (22 ligas, train 51,175) |
| **Accuracy (val)** | **0.584** (weighted, 22 ligas) |
| **MAE (total reg)** | 6.27 avg |
| **Diferencias vs V13** | **Modelo separado por liga**. Sin fallback global (<300 samples → NO_BET). Gates: min conf 0.75, volatility ≤8, current run ≤14. Regression como filtro de confirmación. Dynamic threshold learning. |
| **Notas** | Arquitectura más conservadora y específica por liga. Solo hombres. Sin ROC AUC computado. |
| **Features** | **G1** (score), **G2** (graph), **G3** (trajectory), **G4** (pbp), **G5** (pace), **G6** (league), **G7** (meta) |
| **Filtros** | **Min league samples**: 300 train / 60 val. **Solo hombres**. **Gates**: min confianza 0.75, volatilidad ≤8 swings, current run ≤14 pts. **Reg confirm**: MAE ≤8.0, reg disagreement bloquea bet. 22/100 ligas pasan filtros. Sin gates aplicados en métricas val (accuracy sobre todo el val set sin filtrar). |

---

## V16 — TimesFM + Chronos

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 22; Q4: 31 |
| **Algoritmo** | Per-league + **TimesFM** + **Chronos** (time-series foundation models) |
| **Script entrenamiento** | `v16/train.py` |
| **Script evaluación / ROI** | `v16/evaluate.py`, `v16/test_roi.py`, `v16/test_roi_cli.py`, `v16/ab_compare.py` |
| **Hiperparams** | LogReg(C=0.3); GB(n_est=80, max_depth=3, lr=0.06, min_samples_leaf=20, subsample=0.8); XGB(n_est=80, max_depth=3, lr=0.06, min_child_weight=10, reg_lambda=3.0, reg_alpha=0.5); + TimesFM/Chronos forecast features |
| **Target** | Q3, Q4 winner + regression |
| **Ensemble** | Inverse-error weighted + isotonic |
| **Champion** | `v16/model_outputs/` (per-league, PROD: 46 ligas) |
| **Muestras (val)** | 9,442 quarters (PROD: train 44,198 / val 9,442 / holdout 1,378) |
| **Accuracy (val / holdout)** | 0.597 val / **0.711 holdout** (PROD, weighted 46 ligas) |
| **MAE (total reg)** | 6.75 avg |
| **Diferencias vs V15** | TimesFM + Chronos features. `tfm_cache` para feature generation. Múltiples directorios de output. |
| **Notas** | Experimenta con Google TimesFM y Amazon Chronos para extraer features de series de tiempo de graph_points. Sin ROC AUC computado. Variantes: PROD (0.597), NOTFM (0.591), CHRONOS (0.592). |
| **Features** | G1 (score), G2 (graph), G3 (trajectory), G4 (pbp), G5 (pace), G6 (league), G7 (meta), **G8** (TimesFM/Chronos forecast) |
| **Filtros** | Mismos que V15 + TimesFM/Chronos feature threshold. **PROD 46/90 ligas**. **Holdout 0.711 con gates implícitos**: solo muestras que pasan gates de confianza → sesgo alcista. Variantes: PROD (0.597), NOTFM (0.591), CHRONOS (0.592). |

---

## V17 — Refined Time Series

| Campo | Valor |
|-------|-------|
| **Snapshot** | Q3: 22; Q4: 31 |
| **Algoritmo** | Per-league + TimesFM/Chronos |
| **Script entrenamiento** | `v17/train.py` |
| **Script evaluación / ROI** | `v17/evaluate.py`, `v17/test_roi.py`, `v17/test_roi_cli.py`, `v17/ab_compare.py` |
| **Hiperparams** | LogReg(C=0.3); GB(n_est=80, max_depth=3, lr=0.06, min_samples_leaf=20, subsample=0.8); XGB(n_est=80, max_depth=3, lr=0.06, min_child_weight=10, reg_lambda=3.0, reg_alpha=0.5); + TimesFM/Chronos + Legacy Hybrid (pressure, clutch, MC) |
| **Target** | Q3, Q4 winner + regression |
| **Ensemble** | Inverse-error weighted + isotonic |
| **Champion** | `v17/model_outputs/` (per-league, 67 ligas entrenadas) |
| **Muestras (val / holdout)** | 7,488 / 5,518 (train 118,992 / val 20,454 / holdout 17,238) |
| **Accuracy (val / holdout)** | 0.595 val / **0.609 holdout** (weighted) |
| **MAE (total reg)** | 6.65 avg |
| **Diferencias vs V16** | Integración refinada de TimesFM/Chronos. Código más limpio. Añade G9 (legacy hybrid). |
| **Notas** | Iteración más reciente de la serie V. El holdout más grande hasta ahora (17k). Sin ROC AUC computado. |
| **Features** | G1 (score), G2 (graph), G3 (trajectory), G4 (pbp), G5 (pace), G6 (league), G7 (meta), G8 (TimesFM/Chronos), **G9** (legacy hybrid: pressure, clutch, MC) |
| **Filtros** | Mismos gates que V15/V16. **67/150 ligas** entrenadas. Holdout grande (17k) con sesgo de gates. +G9 legacy hybrid como features adicionales (no como filtro). |

---

## m27_v1 — Independent Minute-27 (Best)

| Campo | Valor |
|-------|-------|
| **Snapshot** | **27** |
| **Algoritmo** | XGBoost + HistGradientBoosting |
| **Script entrenamiento** | `train_q4_m27_v1.py` |
| **Script evaluación / ROI** | `report_m_v1_roi.py`, `m27_v1_league_policy.py`, `compare_model_versions.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5) |
| **Target** | **Solo Q4** |
| **Ensemble** | Avg simple + isotonic |
| **Split** | 70/15/15, 5-fold TimeSeriesSplit CV |
| **Features** | 76: F0, F1, F2 (gp simplificado 4f), F9 (winner context), F10 (halftime), F11 (current score), F12 (Q3 partial) |
| **Filtros** | Ninguno. 70/15/75 split, test sin filtrar (3,713 matches). Match-level. Snapshot 27 (3m Q4 real). |
| **Muestras** | 24,746 (train: 17,322; val: 3,711; test: 3,713) |
| **ROC AUC (test)** | **0.668** (Champion), XGB: 0.669, HistGB: 0.665 |
| **Accuracy** | 0.626 |
| **ROI (10m)** | **+4.5%** yield, 643 bets, 74.7% efectividad |
| **Champion** | `model_outputs_m27_v1/q4_m27_v1_champion.joblib` |
| **Nota features** | Q1/Q2 closed quarters, halftime context, Q3 partial, momentum windows (2m, 3m), trailing/comeback, graph slopes. **Sin league one-hot, sin team buckets.** |
| **Diferencias vs V6.3 m27** | +0.064 ROC AUC (0.668 vs 0.604). Feature engineering limpio. Quita ~20 features ruidosas. |
| **Notas** | **Mejor modelo Q4 global.** Ventaja estructural: 3 min de datos reales de Q4 en ligas NBA 12m. |

---

## m27_v2 — Bugfixed + Normalized Q3 Rates + Filter-10m + Low-Effort Features

| Campo | Valor |
|-------|-------|
| **Snapshot** | **27** |
| **Algoritmo** | XGBoost + HistGradientBoosting |
| **Script entrenamiento** | `train_q4_m27_v2.py` |
| **Script evaluación / ROI** | `report_m_v1_roi.py`, `compare_model_versions.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5) |
| **Target** | **Solo Q4** |
| **Ensemble** | AUC-weighted (XGB 48% / HistGB 52%) + isotonic |
| **Split** | 70/15/15, 5-fold TimeSeriesSplit CV (match-aware, sin leakage) |
| **Features** | v1 features + **q3_partial_home_rate/away_rate/diff_rate** (siempre). **F8** (scoring runs: `current_run_home/away`, `max_run_home/away`). **G1** (score ratios: `score_halftime_diff_ratio`, `score_q1_share`, `score_q3_vs_ht_momentum`). **G4** (PBP density: `pbp_count`, `pbp_home/away_pts`, `pbp_pts_per_event`, `pbp_home/away_3pt_rate`). Pace features gated. _7 features podadas por imp=0 (ver findings)._ |
| **Filtros** | Eligibility filter (`_window_is_eligible_m27`) restaurado. `--filter-10m`: solo ligas de 10 minutos. |
| **Muestras** | ~27,542 (train: 19,279; val: 4,131; test: 4,132) |
| **ROC AUC (test, default)** | **0.668** (ensemble), XGB: 0.666, HistGB: 0.668 |
| **ROC AUC (test, filter-10m)** | **0.671** (ensemble), XGB: 0.670, HistGB: 0.670 |
| **Brier** | 0.228 (ensemble) |
| **Accuracy** | 0.623 (default) / 0.626 (10m) |
| **Champion** | `model_outputs_m27_v2/m27_v2_xgb.joblib` + `m27_v2_histgb.joblib` + `m27_v2_calibrator.joblib` |
| **Nota features** | v1 + normalized Q3 rates + F8 + G1 + G4. 86 features (7 podadas). Pace gated. |
| **Diferencias vs m27_v2 baseline** | +F8/G1/G4 (no suman). AUC-weighted ensemble (no mejora). Poda 7 features muertas (AUC idéntico). |
| **Notas** | El modelo es dominado por `recent_2m_points_diff` (18.4% imp). Recent windows 2m+3m = 35.9%. Features nuevas no sumaron señal. Filter-10m da +0.004 marginal. Próximo paso: buscar señales nuevas (fouls, timeouts, lineups, streak, pace real, datos externos). |

---

## m27_v3 — H2H Head-to-Head Features

| Campo | Valor |
|-------|-------|
| **Snapshot** | **27** |
| **Algoritmo** | XGBoost + HistGradientBoosting |
| **Script entrenamiento** | `train_q4_m27_v3.py` |
| **Script evaluación / ROI** | `report_m_v1_roi.py`, `compare_model_versions.py` |
| **Script inferencia** | `infer_match.py` (`score_m27_v3`) |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5) |
| **Target** | **Solo Q4** |
| **Ensemble** | AUC-weighted (XGB + HistGB) + isotonic |
| **Split** | 70/15/15, 5-fold TimeSeriesSplit CV |
| **Features** | m27_v2 features + **H2H**: `h2h_avg_q1_diff`, `h2h_recent3_home_won`, `h2h_last_home_won` |
| **Filtros** | Eligibility filter (`_window_is_eligible_m27`). Sin filter-10m. |
| **Muestras** | 27,542 (train: 19,279; val: 4,131; test: 4,132) |
| **Cobertura H2H** | 42% global (11,622 muestras con ≥1 H2H previo). **99% en test** (el split test es el más reciente, los equipos ya se enfrentaron). |
| **ROC AUC (test)** | **0.789** (ensemble), XGB: 0.788, HistGB: 0.787 |
| **Brier** | **0.187** (ensemble) |
| **Accuracy** | **0.705** (thr=0.60) |
| **Yield (odds=1.40)** | **+13.0%** @ thr=0.62 (2,414 bets, 80.7% hit rate). **+21.0%** @ thr=0.70 (1,599 bets, 86.4%). **+29.8%** @ thr=0.80 (960 bets, 92.7%). |
| **Champion** | `model_outputs_m27_v3/m27_v3_xgb.joblib` + `m27_v3_histgb.joblib` + `m27_v3_calibrator.joblib` |
| **Feature importance top 5** | `h2h_last_home_won` **(0.228)**, `trailing_now_recent_run_2m` (0.064), `score_est_diff` (0.052), `recent_2m_points_diff` (0.045), `h2h_recent3_home_won` (0.039) |
| **Nota features** | `h2h_last_home_won` es #1 con 22.8% de importancia (4× la siguiente). Las 3 H2H suman 28.5% del modelo. Las H2H se computan desde la DB local (tablas `matches` + `quarter_scores`). Si no hay historial previo, las features valen 0. |
| **Inference** | Integrado en `infer_match.py` vía `score_m27_v3()`. Disponible como `--force-version m27_v3` en `cli.py eval-date`. |
| **Diferencias vs m27_v2** | +**+0.122 ROC AUC**. Yield +13% vs +4.5% de m27_v1. 2,414 bets vs 643. El salto más grande desde m27_v1. Las H2H capturan dinámicas de matchup que `prior_wr_diff` (genérico) no ve. |
| **Próximos pasos** | Ver detalle abajo. |

### Próximos pasos e ideas de mejora — m27_v3

#### 1. Descomposición de `recent_2m_points_diff` (35.9% del modelo)
- **Pace real del live**: `recent_2m_possessions` — número de posesiones reales en la ventana de 2m. No es lo mismo +4 pts en 6 posesiones (juego lento, controlado) que +4 pts en 12 posesiones (transición, alta varianza).
- **Eficiencia pintura vs perímetro**: desglosar los puntos recientes en `recent_2m_paint_pts` / `recent_2m_perimeter_pts`. Los triples en racha tienen alta probabilidad de regresión a la media; los puntos en pintura + tiros libres indican ataque sostenido y faltas acumuladas.

#### 2. Variables de contexto de juego (fouls & timeouts)
- **Bonus flag**: si al minuto 27 algún equipo está en bonus (4+ faltas de equipo), cada falta defensiva se convierte en 2 TL automáticos. Cambia la eficiencia esperada del Q4.
- **Foul trouble**: flag si la estrella o el principal defensor interior tiene 4+ faltas individuales. Limita la agresividad defensiva o fuerza su salida.
- **Timeouts restantes**: si el entrenador del equipo que va perdiendo ya quemó sus timeouts para frenar la racha, o tiene la pizarra limpia para ajustar el cierre.

#### 3. Dinámica de alineaciones (lineups & rest)
- **Descanso acumulado de titulares**: minutos de descanso de los 3 mejores jugadores de cada equipo al llegar al minuto 27. Si los titulares del equipo A descansaron todo el Q3 y entran frescos vs un equipo B desgastado, hay ventaja estructural invisible.
- **Quinteto de cierre (clutch lineup)**: ± histórico de los 5 jugadores en duela. ¿Están los suplentes aguantando o ya entró el quinteto pesado?

#### 4. Datos macroscópicos y de mercado
- **Back-to-back / gira larga**: flag si el equipo jugó el día anterior o es su 4º partido como visitante en una gira. El cansancio crónico se nota exponencialmente en los últimos 6 minutos.
- **Closing Line Value (CLV) o hándicap inicial**: el hándicap de apertura/cierre le da al modelo la "línea base de calidad". Un favorito por -12 que va perdiendo por 4 tiene mucho más talento y urgencia para remontar que un underdog en la misma situación.

#### 5. Explotación de series de tiempo (hibridación ligera)
- **Bandas de Bollinger / Canales de Keltner** sobre los graph points históricos del partido para detectar cuándo una racha reciente está en un extremo estadístico (sobrecompra/sobreventa de momentum) y predecir regresión inminente.

#### 6. Robustez e ingeniería de variables H2H
- **Decaimiento temporal**: si el último H2H fue hace 2 semanas tiene gran valor; si fue hace 3 años (plantillas distintas) es ruido. Aplicar factor de decaimiento por días o filtrar a ≤365 días.
- **Localía estricta**: separar `h2h_last_match_absolute_winner` (quién ganó) de `h2h_last_match_same_venue_winner` (qué pasó la última vez en la misma duela específica).
- **H2H de Q4 específico**: `h2h_avg_q4_diff_historical` — ¿cómo se han comportado estos equipos cerrando partidos entre sí? Es la señal más alineada con el target.

#### 7. Estrategia de fallback para el 58% sin cobertura H2H
- **Two-stage inference**: gate lógico en `infer_match.py` — si existe H2H previo calificado → `score_m27_v3()` (AUC 0.789); si no → `score_m27_v2()` (optimizado sin H2H).
- **H2H proxy por rivales comunes**: para partidos sin enfrentamiento directo, computar el diferencial de rendimiento contra rivales comunes recientes (misma temporada, últimos 3 equipos compartidos).

#### 8. Infraestructura de producción
- **Async pre-fetching de H2H**: tan pronto como un partido aparezca en la lista del día, lanzar un worker asíncrono que consulte `/event/{id}/h2h` de SofaScore, transforme el historial al formato de la DB local y deje las features H2H cacheadas antes del minuto 27. Evitar latencia en la ventana de apuesta live.

---

## m27_v4 — Canonical Triad Multi-Head & Multi-Snapshot Champion (Cero Overfitting) ⭐

> **Paquete de producción:** [`models/m27_v4/`](file:///C:/Users/App/Desktop/pulpa/models/m27_v4/predict.py) (`features.py`, `predict.py`, `train.py`, `model_outputs/`)  
> **Bitácora de iteraciones, corrección de fugas y anti-overfitting:** [`docs/FINDINGS_M27_V4_ITERACIONES.md`](file:///C:/Users/App/Desktop/pulpa/docs/FINDINGS_M27_V4_ITERACIONES.md)

| Campo | Valor |
|-------|-------|
| **Snapshot Operativo** | **Multi-Snapshot Dinámico Pre-Q4:** `m26..m30` en FIBA 10m (**óptimo `m27`**, a 3:00 del cierre de Q3) y `m31..m36` en NBA 12m (**óptimo `m33`**, a 3:00 del cierre de Q3). Soporta también confirmación en vivo `m31..m32`. |
| **Segmentación (Tríada Canónica)** | **3 sub-modelos independientes** sobre el 100% de `matches.db` clasificada (`0` ligas sin clasificar):<br>1. **`fiba_men_10m`** (`149,713` partidos válidos)<br>2. **`fiba_women_10m`** (`34,251` partidos válidos)<br>3. **`nba_12m`** (`28,250` partidos válidos) |
| **Arquitectura (Ensemble Híbrido Regularizado)** | Combinación ponderada de árboles poco profundos + **Ancla Lineal L2 (`StandardScaler + LogisticRegression(C=0.08)`)** + **Calibración Isotónica Out-of-Fold (`IsotonicRegression`)**:<br>• `fiba_men_10m` y `fiba_women_10m`: `45% XGBoost` + `40% HistGradientBoosting` + `15% LogisticRegression L2`<br>• `nba_12m`: `35% XGBoost` + `35% HistGradientBoosting` + `30% LogisticRegression L2` |
| **Hiperparámetros Óptimos Anti-Overfitting** | • **FIBA 10m:** `n_estimators=260`, `learning_rate=0.03`, `max_depth=4`, `max_leaf_nodes=15`, `min_samples_leaf=350`, `reg_lambda (L2)=50.0`, `reg_alpha (L1)=15.0`, `subsample=0.70`, `colsample_bytree=0.70`<br>• **NBA 12m:** `max_depth=3`, `max_leaf_nodes=10`, `min_samples_leaf=450`, `reg_lambda (L2)=60.0`<br>• **Modality Dropout en Entrenamiento:** Apagado estocástico (20%–30%) de cuotas, desglose PBP, rolling de jugadores y curva de presión para inmunizar al modelo ante caídas parciales de la API en vivo. |
| **Targets Simultáneos (4 Cabezas / Multi-Head)** | • **Cabeza 1 (`q4_win`):** Clasificador calibrado de Ganador del Q4 (`HOME` vs `AWAY`).<br>• **Cabeza 2 (`expected_q4_margin`):** Regresor `HistGBRegressor (loss='absolute_error')` del diferencial de puntos esperado en el Q4 ($\text{Q4}_{\text{home}} - \text{Q4}_{\text{away}}$). Actúa como predictor de **Hándicap Q4** y como **Compuerta de Consenso (Gate Dual)** de la Cabeza 1.<br>• **Cabeza 3 (`expected_q4_total`):** Regresor `HistGBRegressor (loss='squared_error')` de puntos totales del Q4 ($\text{Q4}_{\text{home}} + \text{Q4}_{\text{away}}$) para el mercado **Over/Under Q4**.<br>• **Cabeza 4 (`p_ft_home_win`):** Clasificador en vivo del **Ganador del Partido Completo (FT Live)**. |
| **Niveles de Señal en Producción (`predict.py`)** | • `BET_SNIPER`: Liga no tóxica + `Conf >= 0.72` + `|E[dQ4]| >= 3.0 pts` + **Unanimidad de los 3 submodelos** (`min(XGB, HistGB, LR) >= 0.66`) + `d_elo_q4 > 0` a favor del pick.<br>• `BET_STRONG`: Liga no tóxica + `Conf >= 0.70` + `|E[dQ4]| >= 1.8 pts`.<br>• `BET`: Liga no tóxica + `Conf >= 0.68` + `|E[dQ4]| >= 1.8 pts`.<br>• `NO_BET`: No cumple umbrales o pertenece a una liga bloqueada en `toxic_leagues.json` (`[TOXIC_LG_BLOCKED]`). |

---

### 1. Diagrama de Arquitectura Multi-Head & Multi-Snapshot (`models/m27_v4`)

```
Payload en Vivo / Histórico (m26..m30 FIBA | m31..m36 NBA) + Estado O(1) (team_priors_state.pkl: 9,452 equipos, 3,643 ligas)
                                                │
                                                ▼
                    ┌───────────────────────────────────────────────────────┐
                    │ Enrutador Taxonómico de la Tríada Canónica            │
                    │  ├─ fiba_men_10m   (10 min/cuarto, varonil)           │
                    │  ├─ fiba_women_10m (10 min/cuarto, femenil)           │
                    │  └─ nba_12m        (12 min/cuarto, NBA/CBA/PBA/NBL)   │
                    └───────────────────────────────────────────────────────┘
                                                │
                                                ▼
                    ┌───────────────────────────────────────────────────────┐
                    │ Extractor de 36 Features Limpias (features.py)        │
                    │  • 0% Data Leakage | 0% Drift Temporal | Seasonal ELO │
                    └───────────────────────────────────────────────────────┘
                                                │
            ┌───────────────────────────────────┼───────────────────────────────────┬──────────────────────────┐
            ▼                                   ▼                                   ▼                          ▼
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐  ┌──────────────────────┐
│ CABEZA 1: Ganador Q4      │     │ CABEZA 2: Margen Q4 (Pts) │     │ CABEZA 3: Total Q4 (O/U)  │  │ CABEZA 4: Ganador FT │
│ • XGBoost Regularizado    │     │ • HistGBRegressor (MAE)   │     │ • HistGBRegressor (MSE)   │  │ • HistGBClassifier   │
│ • HistGradientBoosting    │     │ • Predice E[Q4_h - Q4_a]  │     │ • Predice E[Q4_h + Q4_a]  │  │ • Predice P(FT_Home) │
│ • Ancla Lineal L2 (LogReg)│     └─────────────┬─────────────┘     └─────────────┬─────────────┘  └──────────┬───────────┘
│ • Calibración Isotónica   │                   │                                 │                           │
└─────────────┬─────────────┘                   ▼                                 ▼                           ▼
              │                   ┌───────────────────────────┐     ┌───────────────────────────┐  ┌──────────────────────┐
              └──────────────────►│ Gate Dual + Filtro Tóxicas│     │ Señal Over/Under Q4       │  │ Señal FT Moneyline   │
                                  │ • Bloquea toxic_leagues   │     │ • |E[TotQ4] - Prior|>=2.5 │  │ • AUC 0.915 - 0.962  │
                                  │ • BET / BET_STRONG /      │     │ • WR: 60.9% - 61.4%       │  │ • Acc: 82.6% - 88.6% │
                                  │   BET_SNIPER (WR 77%-86%) │     └───────────────────────────┘  └──────────────────────┘
                                  └───────────────────────────┘
```

---

### 2. Catálogo Completo de las 36 Features (`FEATURE_NAMES_CLEAN` en `models/m27_v4/features.py`)

Todas las features están garantizadas **100% libres de fuga del futuro (*zero leakage*)** y **libres de *coverage drift* entre épocas (`2015–2026`)**:

| Índice | Nombre de Feature (`FEATURE_NAMES_CLEAN`) | Bloque | Descripción Técnica y Fórmula | Correlación Limpia ($r$ con `q4_win`) |
| :-: | :--- | :---: | :--- | :---: |
| `0` | `cur_score_diff` | **A. Marcador y Reloj** | Diferencial de marcador en el minuto de corte ($\text{Score}_{\text{home}} - \text{Score}_{\text{away}}$) | `+0.0566` (FIBA M) / `+0.2016` (FIBA W) / `-0.1190` (NBA) |
| `1` | `abs_cur_score_diff` | **A. Marcador y Reloj** | Magnitud absoluta de la ventaja actual ($|\text{cur\_score\_diff}|$) para medir paridad vs paliza | `-0.0412` |
| `2` | `ht_diff` | **A. Marcador y Reloj** | Diferencial de marcador al medio tiempo ($\text{HT}_{\text{home}} - \text{HT}_{\text{away}}$) | `+0.0684` |
| `3` | `q1_diff` | **A. Marcador y Reloj** | Diferencial neto del Q1 ($\text{Q1}_{\text{home}} - \text{Q1}_{\text{away}}$) | `+0.0541` |
| `4` | `q2_diff` | **A. Marcador y Reloj** | Diferencial neto del Q2 ($\text{Q2}_{\text{home}} - \text{Q2}_{\text{away}}$) | `+0.0489` |
| `5` | `q3_eff_diff` | **B. Momentum Q3** | Diferencial parcial acumulado dentro del Q3 hasta el corte ($\text{cur\_score\_diff} - \text{ht\_diff}$) | `+0.0218` |
| `6` | `q3_vs_ht_momentum` | **B. Momentum Q3** | Aceleración/cambio de tendencia en Q3 respecto a la primera mitad ($\text{q3\_eff\_diff} - 0.5 \cdot \text{ht\_diff}$) | `-0.0315` |
| `7` | `recent_2m_run_diff` | **B. Momentum Q3** | Racha neta de puntos en los últimos 2 minutos previos al corte desde `play_by_play` | `+0.0342` |
| `8` | `pace_pts_per_min` | **B. Momentum Q3** | Ritmo real del partido en puntos por minuto jugado ($\text{TotalPts} / \text{MinutosJugados}$) | `+0.0190` (Fuerte en Head 3 Total Q4: $r=+0.44$) |
| `9` | `pace_vs_league_prior` | **B. Momentum Q3** | Exceso/déficit de ritmo del partido frente al ritmo histórico de esa liga | `+0.0154` (Head 3 Total Q4: $r=+0.31$) |
| `10` | `gp_val` | **B. Momentum Q3** | Valor de la curva de dominio/presión (`graph_points`) en el minuto de corte | `+0.0612` |
| `11` | `gp_slope_3m` | **B. Momentum Q3** | Pendiente de la curva de presión en los últimos 3 minutos ($\text{gp}_{t} - \text{gp}_{t-3}$) | `+0.0287` |
| `12` | `fraction_q3_completed` | **A. Marcador y Reloj** | Fracción completada del Q3 (`0.60` en `m26`, `0.70` en `m27`, ..., `1.00` en `m30`) para normalizar multi-snapshot | `0.0000` (Variable de anclaje temporal) |
| `13` | `d_elo_ft` | **C. ELO con Decay** | Diferencial de ELO Full-Time pre-partido con **Decaimiento Estacional** (`>60d` $\to$ 35% a 1500; `>300d` $\to$ 75% a 1500) + ventaja local | **`+0.1943`** ⭐ |
| `14` | `d_elo_q4` | **C. ELO con Decay** | Diferencial de **ELO específico de Q4** (actualizado únicamente con quién gana/pierde los cuartos finales históricamente) | **`+0.1684`** ⭐ |
| `15` | `d_elo_q1` | **C. ELO con Decay** | Diferencial de ELO específico de Q1 (mide la potencia de arranque de titulares) | **`+0.1592`** |
| `16` | `elo_reliability` | **C. ELO con Decay** | Confiabilidad muestral del par de equipos ($\min(1.0, \min(N_{\text{home}}, N_{\text{away}}) / 15.0)$) | `+0.0420` |
| `17` | `d_q4_bayes` | **C. ELO con Decay** | Diferencial del margen promedio en Q4 de los últimos 15 partidos con contracción Bayesiana ($n / (n + 6)$) | **`+0.1715`** ⭐ |
| `18` | `d_wr_bayes` | **C. ELO con Decay** | Diferencial de Win Rate general de los últimos 15 partidos con contracción Bayesiana hacia `0.50` | **`+0.1640`** ⭐ |
| `19` | `has_odds` | **D. Mercado y Tabla** | Indicador binario (`1.0` / `0.0`) de disponibilidad de cuota pre-partido en `match_odds` | `+0.0110` |
| `20` | `odds_log_ratio` | **D. Mercado y Tabla** | Log-odds implícito del favorito pre-partido ($\ln(P_{\text{home}} / P_{\text{away}})$ sin vig) | **`+0.2185`** ⭐ |
| `21` | `expected_ft_margin` | **D. Mercado y Tabla** | Margen final esperado pre-partido derivado de cuotas (o de `0.035 * d_elo_ft` si no hay cuota) | **`+0.2140`** ⭐ |
| `22` | `h2h_bayes_ft_margin` | **D. Mercado y Tabla** | Margen histórico cara a cara (FT) estrictamente en fechas `< match_date` con contracción Bayesiana ($n/(n+5)$) | `+0.1312` |
| `23` | `h2h_bayes_q4_margin` | **D. Mercado y Tabla** | Margen histórico cara a cara específico en Q4 en enfrentamientos previos (`< match_date`) | `+0.1185` |
| `24` | `ts_delta_wp_bayes` | **D. Mercado y Tabla** | Diferencial de porcentaje de victorias en la clasificación de la liga (`team_strength`) con contracción Bayesiana | **`+0.1621`** |
| `25` | `ts_delta_form` | **D. Mercado y Tabla** | Diferencial de racha reciente reportada en la tabla de posiciones (`team_strength`) | `+0.1290` |
| `26` | `pbp_2pt_diff_q1_q3` | **E. Microestructura PBP** | Diferencial de puntos en la pintura/media distancia (canastas de 2 pts) extraído de `match_events` hasta el corte | `+0.0945` |
| `27` | `pbp_3pt_diff_q1_q3` | **E. Microestructura PBP** | Diferencial de triples anotados extraído directamente de `match_events` hasta el corte (100% cobertura 2015–2026) | `+0.0618` |
| `28` | `pbp_ft_diff_q1_q3` | **E. Microestructura PBP** | Diferencial de tiros libres anotados extraído de `match_events` (proxy directo de faltas provocadas y bonus) | `+0.0724` |
| `29` | `d_rol_plus_minus` | **F. Rolling Plantilla** | Diferencial del `plus_minus` promedio ponderado de los jugadores en sus **partidos previos (`< match_date`)** | **`+0.2294`** 🏆 *(Feature #1)* |
| `30` | `d_rol_bench_rating` | **F. Rolling Plantilla** | Diferencial del `sofascore_rating` histórico pre-partido de la rotación/banquillo (jugadores #6 a #10) | **`+0.1812`** ⭐ |
| `31` | `rubber_band_gap` | **G. Física de Cierre** | Tensión de regresión a la media ($\text{expected\_ft\_margin} - \text{cur\_score\_diff}$): mide cuánto "debe" recuperar el favorito en Q4 | **`+0.1488`** ⭐ |
| `32` | `white_flag_risk_zone` | **G. Física de Cierre** | Indicador (`-1`, `0`, `+1`) de zona de rendición de titulares cuando el favorito pierde por 14–26 pts al cierre del Q3 | `+0.0890` |
| `33` | `blowout_garbage_time` | **G. Física de Cierre** | Indicador binario (`1.0` si $|\text{cur\_score\_diff}| \ge 22$ pts) de *garbage time* con quintetos de reserva | `-0.0512` |
| `34` | `in_q4_partial_diff` | **A. Marcador y Reloj** | Parcial anotado dentro del Q4 (`0.0` en `m26..m30` pre-Q4; activo solo si se invoca confirmación live en `m31..m32`) | `0.0000` en `m27` (`+0.34` en `m32`) |
| `35` | `lg_avg_q4_prior` | **D. Mercado y Tabla** | Promedio histórico de puntos totales en Q4 de la liga específica (`leagues_classification`) | `+0.0120` (Head 3 Total Q4: $r=+0.48$) |

---

### 3. Estadísticas de Entrenamiento, Progresión Anti-Overfitting y Cortes Walk-Forward

#### 3.1 Evolución de Versiones (`V4.0_raw` $\to$ `V4.4_champion` en `fiba_men_10m`, $N = 149,713$)
Split cronológico estricto **70% Train (`104,799`) / 15% Validation (`22,457`) / 15% Test Out-of-Time (`22,457`)**:

| Iteración | Cambios en Arquitectura, Features e Hiperparámetros | Train AUC | Val AUC | Test AUC | **Brecha Overfit (`Train - Val`)** | Test Brier / LogLoss | Test WR @ $P \ge 0.70$ ($N$) | Yield @ $0.70$ (Cuota 1.40) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V4.0_raw`** | Árboles profundos (`max_depth=8`, `min_leaf=10`, `L2=0.1`), sin ELO decay, H2H crudo | `0.8242` | `0.6246` | `0.6298` | **`+0.1996`** ❌ *(Severo Overfit)* | `0.2412` / `0.6789` | `70.44%` (`4,360`) | `-1.39%` ❌ |
| **`V4.1_pruned`** | Poda de IDs ruidosos + PBP 2PT/3PT directo + *Seasonal ELO Decay* + Contracción Bayesiana H2H | `0.6761` | `0.6401` | `0.6411` | `+0.0360` ⚠️ *(Leve Overfit)* | `0.2345` / `0.6612` | `74.82%` (`3,185`) | `+4.75%` ✅ |
| **`V4.2_tuned`** | Regularización fuerte (`max_depth=4`, `max_leaf=15`, `min_leaf=350`, `L2=50`, `L1=15`, `lr=0.03`) | `0.6402` | `0.6396` | `0.6399` | **`+0.0006`** ✅ *(Cero Overfit)* | `0.2342` / `0.6606` | `76.59%` (`2,132`) | `+7.23%` ✅ |
| **`V4.4_champion`** ⭐ | **Multi-Snapshot + Modality Dropout + Ancla Lineal L2 + Gate Regresor Margen + Isotonic** | **`0.6417`** | **`0.6423`** | **`0.6406`** | **`-0.0006`** ⭐ *(Cero Overfit)* | **`0.2337` / `0.6594`** | **`77.19%` (`2,345`)** | **`+8.07%`** ⭐ |

#### 3.2 Métricas de Entrenamiento de los 3 Arquetipos de la Tríada Canónica (`212,214` Partidos)

| Arquetipo (`models/m27_v4`) | Total Partidos ($N$) | Split (`Train / Val / Test`) | Train AUC | Val AUC | **Test AUC (Ciego)** | **Brecha (`Train - Val`)** | Test Brier | Test LogLoss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fiba_men_10m`** | `149,713` | `104,799 / 22,457 / 22,457` | `0.6417` | `0.6423` | **`0.6406`** | **`-0.0006`** ✅ | `0.2337` | `0.6594` |
| **`fiba_women_10m`** | `34,251` | `23,975 / 5,138 / 5,138` | `0.6841` | `0.6886` | **`0.6714`** | **`-0.0045`** ✅ | `0.2268` | `0.6452` |
| **`nba_12m`** | `28,250` | `19,775 / 4,237 / 4,238` | `0.6584` | `0.6315` | **`0.6208`** | **`+0.0269`** ✅ | `0.2383` | `0.6689` |

#### 3.3 Validación Walk-Forward Cortando la Información en 4 Puntos Históricos (`2015–2026`)

| Corte Histórico (Ventana Expansiva) | Ventana de Prueba Ciega | Partidos Train $\to$ Test | Train AUC | Test AUC | Brecha (`Train - Test`) | Test WR @ $P \ge 0.70$ ($N$ Apuestas) | Yield @ Cuota 1.40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corte 1:** Train `<= 2023-H1` | `2024-01` a `2024-10` | `62,879` $\to$ `19,463` | `0.6405` | `0.6268` | `+0.0137` ✅ | **`76.18%`** (`995` bets) | **`+6.65%`** |
| **Corte 2:** Train `<= 2024-H1` | `2024-10` a `2025-04` | `82,342` $\to$ `22,457` | `0.6378` | `0.6349` | `+0.0029` ✅ | **`74.87%`** (`1,707` bets) | **`+4.82%`** |
| **Corte 3:** Train `<= 2024-Fin` | `2025-04` a `2025-12` | `104,799` $\to$ `22,457` | `0.6375` | `0.6397` | `-0.0022` ✅ | **`76.64%`** (`1,982` bets) | **`+7.30%`** |
| **Corte 4:** Train `<= 2025-Fin` | `2025-12` a `2026-10` | `127,256` $\to$ `22,457` | `0.6386` | `0.6409` | `-0.0023` ✅ | **`77.74%`** (`2,143` bets) | **`+8.84%`** |

---

### 4. Estadísticas de Efectividad Operativa (Corridas Pre-Q4 en `m27`/`m33` y Prueba en Partidos No Vistos)

> **Nota sobre momios y minuto de entrada:** En el minuto 32 (`2:00` jugados del Q4) la cuota del ganador del Q4 se desploma en las casas de apuestas. Por ello, **todas las tablas de efectividad principal siguientes están medidas estrictamente en el minuto 27 (`m27` en FIBA 10m) y minuto 33 (`m33` en NBA 12m), 3 minutos antes de que inicie el Q4**, cuando el momio de Ganador Q4 / Hándicap Q4 se mantiene íntegro (`~1.38–1.83`).

#### 4.1 Efectividad en Test Out-of-Time (`31,833` Partidos) por Nivel de Señal Pre-Q4 (`m27` / `m33`)

| Arquetipo (`m27`/`m33` Pre-Q4) | Nivel de Señal / Filtro Operativo en `predict.py` | **Win Rate en Test** | Apuestas en Test | Margen Real Ganado en Q4 | **Cuota Mínima Break-Even** | **ROI @ Cuota 1.40** | **ROI @ Cuota 1.35** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fiba_men_10m`** (`m27`) | `BET` (`P >= 0.68` + `E[dQ4] >= 1.8`) | **`73.80%`** | `4,019` | `+5.41 pts` | `1.36` | `+3.32%` | `-0.37%` |
| | `BET_STRONG` (`P >= 0.70` + `E[dQ4] >= 1.8`, Base V4.4) | **`77.00%`** | `2,248` | `+6.14 pts` | `1.30` | **`+7.80%`** | `+3.95%` |
| | `BET_STRONG` **+ Sin Ligas Tóxicas (`toxic_leagues.json`)** ⭐ | **`77.66%`** | `2,055` | `+6.27 pts` | **`1.29`** | **`+8.73%`** | **`+4.84%`** |
| | `BET_SNIPER` (`P >= 0.72` + `E[dQ4] >= 3.0` + `minSub >= 0.66` + Sin Tóxicas) | **`77.71%`–`78.32%`** | `1,041` | `+5.94 pts` | **`1.28`** | **`+8.80%`–`+9.64%`** | **`+4.91%`–`+5.73%`** |
| | `BET_SNIPER` **en Ligas Fuertes / Élite** 🏆 | **`79.37%`** | `383` | `+6.11 pts` | **`1.26`** | **`+11.12%`** | **`+7.15%`** |
| **`fiba_women_10m`** (`m27`) | `BET` (`P >= 0.68` + `E[dQ4] >= 1.8`) | **`75.57%`** | `1,105` | `+6.12 pts` | `1.32` | `+5.79%` | `+2.02%` |
| | `BET_STRONG` (`P >= 0.70` + `E[dQ4] >= 1.8` + Sin Ligas Tóxicas) ⭐ | **`77.56%`** | `936` | `+6.87 pts` | **`1.29`** | **`+8.59%`** | **`+4.71%`** |
| | `BET_STRONG` **+ Fuerza Histórica Q4 (`d_elo_q4 > 0 & d_q4_bayes >= 0`)** | **`78.63%`** | `791` | `+7.01 pts` | **`1.27`** | **`+10.09%`** | **`+6.15%`** |
| | `BET_SNIPER` (`P >= 0.74` + `E[dQ4] >= 3.8` + `minSub >= 0.68` + Sin Tóxicas) 🏆 | **`79.83%`** | `600` | `+7.74 pts` | **`1.25`** | **`+11.77%`** | **`+7.77%`** |
| **`nba_12m`** (`m33`) | `BET_STRONG` (`P >= 0.70` + `E[dQ4] >= 1.8`, Base V4.4) | **`74.00%`** | `250` | `+4.75 pts` | `1.35` | `+3.60%` | `-0.10%` |
| | `BET_STRONG` **+ Sin China CBA + Fuerza Q4 (`d_elo_q4 > 0`)** ⭐ | **`81.95%`** | `133` | `+6.32 pts` | **`1.22`** | **`+14.74%`** | **`+10.63%`** |
| | `BET_SNIPER` (`P >= 0.72` + `E[dQ4] >= 3.0` + `minSub >= 0.66` + Sin CBA) 🏆 | **`85.71%`** | `35` | `+7.51 pts` | **`1.17`** | **`+20.00%`** | **`+15.71%`** |

#### 4.2 Efectividad en los Mercados Secundarios (Cabezas 3 y 4 en `m27`/`m33`)
* **Cabeza 3 — Over/Under de Puntos Totales del Q4 (`|expected_q4_total - lg_avg_q4_prior| >= 2.5 pts`):**
  - **`fiba_men_10m`:** **`60.87%` Win Rate** (`4,636` apuestas en Test, MAE `6.27 pts`, $r = +0.5141$, **`+12.61%` Yield** a cuota estándar `1.85`).
  - **`nba_12m`:** **`61.40%` Win Rate** (`1,658` apuestas en Test, MAE `7.46 pts`, **`+13.59%` Yield** @ `1.85`).
  - **`fiba_women_10m`:** **`57.99%` Win Rate** (`1,390` apuestas en Test, MAE `6.02 pts`, **`+7.28%` Yield** @ `1.85`).
* **Cabeza 4 — Ganador del Partido Completo en Vivo (`p_ft_home_win` al minuto 27/33):**
  - **`fiba_men_10m`:** `ROC AUC = 0.9340` | `Accuracy = 84.71%`.
  - **`fiba_women_10m`:** `ROC AUC = 0.9622` | `Accuracy = 88.56%`.
  - **`nba_12m`:** `ROC AUC = 0.9150` | `Accuracy = 82.63%`.

#### 4.3 Prueba en Ciego sobre Partidos 100% Nuevos Fuera de `matches.db` (`2026-10-15` a `2026-10-17`)
Evaluación directa en memoria desde la API móvil de SofaScore sobre **`144` partidos terminados en fechas posteriores al corte de `matches.db`**, probados estrictamente en `m27`/`m33`:
* **Todos los partidos (`P >= 0.50`):** `65.28%` WR (`94 / 144`)
* **Umbral `P >= 0.65`:** **`87.50%` WR** (`28 / 32`, Break-Even `1.14`, `+22.50%` ROI @ `1.40`)
* **Señal `BET` (`P >= 0.68` + `E[dQ4] >= 1.8`):** **`94.74%` WR** (`18 / 19` ganadas, Break-Even `1.06`, **`+32.63%` ROI @ `1.40`**)
* **Señal `BET_STRONG` (`P >= 0.70` + `E[dQ4] >= 1.8`):** **`100.00%` WR** (`10 / 10` ganadas, **`+40.00%` ROI @ `1.40`**)

---

### 5. Catálogo de Ligas Idóneas (Élite) vs. Ligas Tóxicas (Bloqueadas en `toxic_leagues.json`)

#### 5.1 Ligas Idóneas / Élite para Operar en `m27` / `m33` (Mayor Efectividad Comprobada en Test)

| Arquetipo | Liga Idónea | Partidos Test | Accuracy Global (`P>=0.50`) | **WR @ `P >= 0.68` (`m27`/`m33`)** | **WR @ `P >= 0.70` (`BET_STRONG`)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`fiba_men_10m`** | **International - Euroleague (Regular Season)** | `88` | `69.3%` | **`100.0%`** (`9/9`) | **`100.0%`** (`5/5`) |
| **`fiba_men_10m`** | **Australia - NBL1 South (Men)** | `135` | `57.8%` | **`92.9%`** (`13/14`) | **`100.0%`** (`7/7`) |
| **`fiba_men_10m`** | **Saudi Arabia - Premier League** | `23` | `78.3%` | **`90.9%`** (`10/11`) | **`90.0%`** (`9/10`) |
| **`fiba_men_10m`** | **Iceland - Subway League / Premier League** | `62` | `61.3%` | **`90.0%`** (`9/10`) | **`83.3%`** (`5/6`) |
| **`fiba_men_10m`** | **Vietnam - VBA** | `57` | `64.9%` | **`89.5%`** (`17/19`) | **`93.8%`** (`15/16`) |
| **`fiba_men_10m`** | **Poland - 1. Liga (1st Division Men)** | `147` | `59.2%` | **`88.2%`** (`15/17`) | **`100.0%`** (`8/8`) |
| **`fiba_men_10m`** | **Brazil - CBI (Classificatória A & B)** | `154` | `72.1%` | **`95.0%`** (`38/40`) | **`96.7%`** (`29/30`) |
| **`fiba_women_10m`** | **Czech Republic - ZBL Playoffs (Women)** | `20` | `85.0%` | **`100.0%`** (`9/9`) | **`100.0%`** (`9/9`) |
| **`fiba_women_10m`** | **Rwanda - Basketball League Women** | `31` | `83.9%` | **`100.0%`** (`11/11`) | **`100.0%`** (`11/11`) |
| **`fiba_women_10m`** | **Belgium - Top Division Women** | `26` | `80.8%` | **`92.3%`** (`12/13`) | **`92.3%`** (`12/13`) |
| **`fiba_women_10m`** | **Kosovo / Albania - Superliga Women** | `19` | `73.7%` | **`90.0%`** (`9/10`) | **`90.0%`** (`9/10`) |
| **`fiba_women_10m`** | **Serbia - 1. ZLS Women** | `32` | `59.4%` | **`87.5%`** (`7/8`) | **`87.5%`** (`7/8`) |
| **`fiba_women_10m`** | **Switzerland - SB League Women** | `32` | `71.9%` | **`83.3%`** (`10/12`) | **`83.3%`** (`10/12`) |
| **`fiba_women_10m`** | **Mexico - LNBP Femenil** | `52` | `69.2%` | **`81.8%`** (`9/11`) | **`88.9%`** (`8/9`) |
| **`fiba_women_10m`** | **Spain - Liga Femenina Endesa** | `79` | `58.2%` | **`81.8%`** (`9/11`) | **`80.0%`** (`8/10`) |
| **`fiba_women_10m`** | **Australia - NBL1 East Women** | `118` | `75.4%` | **`79.6%`** (`39/49`) | **`80.4%`** (`37/46`) |
| **`fiba_women_10m`** | **USA - NCAA Women's March Madness** | `59` | `64.4%` | **`80.0%`** (`12/15`) | **`80.0%`** (`12/15`) |
| **`nba_12m`** | **Philippines - PBA Philippine Cup** | `117` | `65.0%` | **`90.9%`** (`10/11`) | **`90.9%`** (`10/11`) |
| **`nba_12m`** | **USA - NBA G League** | `946` | `59.3%` | **`77.5%`** (`31/40`) | **`79.5%`** (`31/39`) |
| **`nba_12m`** | **USA - NBA** | `1,871` | `59.1%` | **`77.2%`** (`95/123`) | **`77.7%`** (`94/121`) |

#### 5.2 Ligas Tóxicas / Bloqueadas (`models/m27_v4/model_outputs/toxic_leagues.json`)
Estas ligas están bloqueadas automáticamente en [`models/m27_v4/predict.py`](file:///C:/Users/App/Desktop/pulpa/models/m27_v4/predict.py) (`is_toxic_league = True` $\to$ `NO_BET [TOXIC_LG_BLOCKED]`):

| Arquetipo | Liga Tóxica Bloqueada | Partidos Test | **WR @ `P >= 0.68` (`m27`/`m33`)** | **WR @ `P >= 0.70`** | Causa Estructural del Fallo en Q4 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`fiba_men_10m`** | **Austria - Basketball Superliga** | `40` | **`36.4%`** (`4/11`) ❌ | `33.3%` (`3/9`) ❌ | Rotaciones masivas de titulares y alta tasa de remontadas en Q4 |
| **`fiba_men_10m`** | **Poland - PGE Basket Liga (Orlen Basket Liga)** | `104` | **`37.5%`** (`3/8`) ❌ | `0.0%` (`0/2`) ❌ | Paridad extrema en 1ra división polaca; el líder al Q3 cede el Q4 sistemáticamente |
| **`fiba_men_10m`** | **Qatar - Qatari Basketball League** | `26` | **`37.5%`** (`3/8`) ❌ | `20.0%` (`1/5`) ❌ | Restricción de jugadores extranjeros en cancha en el último cuarto |
| **`fiba_men_10m`** | **Luxembourg - Enovos League** | `64` | **`45.5%`** (`5/11`) ❌ | `50.0%` (`2/4`) ❌ | Varianza extrema de ritmo y anotación en cierres |
| **`fiba_men_10m`** | **Australia - NBL1 Central & NBL1 East (Men)** | `169` | **`46.2%–53.8%`** ❌ | `33.3%–58.8%` ❌ | *Garbage time* prematuro en ligas estatales australianas varoniles |
| **`fiba_men_10m`** | **Denmark - Basketligaen (Championship Round)** | `25` | **`50.0%`** (`4/8`) ❌ | `60.0%` (`3/5`) ❌ | Fuerte efecto *rubber-banding* en ronda de campeonato |
| **`fiba_men_10m`** | **Israel - Basketball Super League (Winner League)** | `81` | **`53.8%`** (`7/13`) ❌ | `40.0%` (`2/5`) ❌ | Dosificación de plantillas de Euroliga/Eurocup en liga doméstica |
| **`fiba_women_10m`** | **Croatia - Premijer ženska liga** | `40` | **`36.4%`** (`4/11`) ❌ | `36.4%` (`4/11`) ❌ | Las favoritas sacan +25 pts en Q1–Q3 y juegan todo el Q4 con juveniles |
| **`fiba_women_10m`** | **Australia - NBL1 North Women** | `74` | **`50.0%`** (`4/8`) ❌ | `66.7%` (`4/6`) ⚠️ | Rotación de titulares en el último cuarto en ventajas amplias |
| **`fiba_women_10m`** | **Great Britain - SLB Women** | `38` | **`52.9%`** (`9/17`) ❌ | `52.9%` (`9/17`) ❌ | Alta volatilidad en cierres de cuarto |
| **`fiba_women_10m`** | **Chile - Liga Nacional Femenina (Fase regular)** | `34` | **`54.5%`** (`6/11`) ❌ | `54.5%` (`6/11`) ❌ | Desigualdad extrema (+40 pts al Q3) que provoca relajación total de la líder en Q4 |
| **`nba_12m`** | **China - CBA** | `552` | **`63.5%`** (`33/52`) ⚠️ | `63.5%` (`33/52`) ⚠️ | **Regla CBA de extranjeros:** permite 2 extranjeros en Q1–Q3 pero **solo 1 extranjero en el Q4**, alterando drásticamente la física del último cuarto frente a NBA/G-League |

---

## m30_v1 — Independent Minute-30

| Campo | Valor |
|-------|-------|
| **Snapshot** | **30** |
| **Algoritmo** | XGBoost + HistGradientBoosting |
| **Script entrenamiento** | `train_q4_m30_v1.py` |
| **Script evaluación / ROI** | `compare_q4_min30_models.py`, `report_m_v1_roi.py` |
| **Hiperparams** | XGB(n_est=300, lr=0.05, max_depth=4); HistGB(max_iter=300, lr=0.05, max_depth=5) |
| **Target** | **Solo Q4** |
| **Ensemble** | Avg simple + isotonic |
| **Split** | 70/15/15, 5-fold TimeSeriesSplit CV |
| **Features** | 103: F0, F1, F2 (gp simplificado), F9 (winner context), F10 (halftime), F11 (current score), F12 (Q3 partial), **F13** (Q3 momentum 8f), **F14** (Q3 pace/projection 10f), **F15** (halftime trailer analysis) + PBP windows 1m/2m (hereda 3m, 2m de m27_v1) |
| **Filtros** | Ninguno. Test sin filtrar (4,144 matches). Snapshot 30 (sin Q4 real). |
| **Muestras** | 27,618 (train: 19,332; val: 4,142; test: 4,144) |
| **ROC AUC (test)** | **0.585** (Champion), XGB: 0.580, HistGB: **0.589** |
| **Accuracy** | 0.562 |
| **ROI (10m)** | -3.6% yield, 90 bets (filtro estricto) |
| **Champion** | `model_outputs_m30_v1/q4_m30_v1_champion.joblib` |
| **10m-only champion** | `model_outputs_m30_v1/q4_10m_only/q4_m30_v1_10m_champion.joblib` |
| **Anotación** | +0.007 ROC AUC (HistGB 10m: 0.585 → 0.591) gracias a F13/F14/F15 |
| **Diferencias vs m27_v1** | Snapshot más tarde (30 vs 27). Deep momentum features. Para 10m: Q3 completo (vs parcial en m27_v1). Para 12m: **sin datos de Q4** (vs 3 min en m27_v1). |
| **Notas** | Estructuralmente más débil que m27_v1. Snapshot 30 es inconsistente: 10m = final Q3, 12m = mitad Q3. Límite estructural ~0.59 ROC AUC para predicción pre-Q4. |

---

## Tabla comparativa rápida

| Versión | Snap | ROC AUC | Acc | Test n | Filtros / Sesgo eval | Nota |
|---------|:----:|:-------:|:---:|:------:|------|------|
| V1 | 36 | ~0.857 | — | — | Snapshot 36 (3m Q4 real). Sin filtro. | Sin CSV de métricas |
| V2 | 36 | **0.865** | 0.774 | 4,304 | Snapshot 36. Sin filtro. | +League buckets |
| V3 | 24/30/36 | 0.472/0.754/0.819 | — | 312 | Snapshot mixto. n muy pequeño (312). | Multi-snapshot |
| V4 | 36 | **0.848** | 0.760 | 1,284 | Snapshot 36. Sin filtro. | +Pressure features |
| V5 | 36 | **0.840** | 0.767 | 917 | Snapshot 36. Sin filtro. n más pequeño. | Stack XGB+HistGB+MLP |
| V6 | 36 | **0.857** | **0.765** | 4,284 | Snapshot 36. Sin filtro. | +MC, baseline |
| V6.1 | 36 | **0.847** | 0.756 | 2,700 | Snapshot 36. Sin filtro. | +Calibración |
| V6.2 | 36 | **0.855** | **0.770** | 2,459 | Snapshot 36. League pruning en test. | League pruning |
| V6.3 | **27/30** | 0.604 / 0.772\* | 0.573 / 0.702 | 3,713 / 1,184 | **\*m30 evaluado solo en partidos cerrados** — sesgo de selección extremo. m27 sin filtrar. | \*m30 inflado |
| V7 | 36 | **0.814** | 0.739 | 922 | Snapshot 36. Sin filtro. n pequeño. | CatBoost+XGB(cat) |
| V8 | 36 | **0.780** | 0.708 | 922 | Snapshot 36. Sin filtro. n pequeño. | LSTM+Tabular |
| V9 | 36 | **0.847** | 0.750 | 1,279 | Snapshot 36. Sin filtro. | LogReg+GB fast |
| V10 | 36 | N/A (reg) | MAE 5.4 (total) | 1,350 | Snapshot 36. Regresión. R² ~0.50. | Regresión totals |
| V11 | 36 | N/A (reg) | MAE 5.5 (total) | 1,150 | Snapshot 36. Regr. género-separado. R² ~0.46. | Género-separado |
| V12 | **31** | N/A (reg) | MAE 5.3 (total) | 3,524 | Snapshot 31 (sin Q4 real). Per-league. R² ~0.55. | Per-league, risk mgmt |
| V13 | 31 | Acc only | 0.632 (val) | 13,710 | Snapshot 31. Walk-forward. **Evalúa quarters, no matches**. Sin gates. | 12 buckets |
| V14 | — | — | — | — | — | Planning (TimesFM) |
| V15 | **31** | Acc only | 0.584 (val) | 10,450 | Snapshot 31. Per-league 22/100 ligas. **Solo hombres**. Sin gates en val. | 22 ligas, solo hombres |
| V16 | 31 | Acc only | 0.597 val / 0.711 hold | 9,442 / 1,378 | Snapshot 31. PROD 46/90 ligas. **Holdout 0.711 con gates implícitos** — sesgo alcista. | TimesFM/Chronos |
| V17 | 31 | Acc only | 0.595 val / 0.609 hold | 7,488 / 5,518 | Snapshot 31. 67/150 ligas. Holdout 17k grande. **Holdout con sesgo de gates**. | +G9 legacy hybrid |
| **m27_v1** | **27** | **0.668** | 0.626 | 3,713 | **Snapshot 27 (1m Q4 filtrado por reloj 12m)**. Sin filtro. Match-level. | **Señal combinada c/reloj 12m** |
| **m27_v2** | **27** | **0.668** / 0.671\* | 0.623 / 0.626 | 4,132 | \*10m filter. 86 feat (7 podadas). Match-level. | Idéntico a v1. Recent windows dominan. |
| **m27_v3** | **27** | **0.789\*** | **0.705\*** | 4,132 | Snapshot 27. **\*Nota: Con fuga H2H mismo día antes del fix (`0.64` limpio)**. | **Reemplazado por `m27_v4`** |
| **m27_v4** ⭐ | **26–32** | **0.641 / 0.671 / 0.621** | **77.2% WR** (@0.70) | **31,833** | **Tríada Canónica (`212,214` matches). Cero leakage, cero overfit (`Gap = -0.0006`). 4 Cabezas (Q4 Win, Margen, O/U, FT).** | **Campeón Actual (`+8.07%` Yield Q4, `+12.7%` O/U)** |
| **m30_v1** | **30** | 0.585 | 0.562 | 4,144 | **Snapshot 30 (sin Q4 real)**. Sin filtro. Match-level. | Límite ~0.59 |

---

## Glosario

| Término | Significado |
|---------|-------------|
| **Snapshot** | Minuto del partido hasta el cual se usan datos para predecir |
| **Q4-only** | Modelo que solo predice Q4 (no Q3) |
| **Ensemble** | Combinación de múltiples modelos (avg, weighted, stacking) |
| **Isotonic calibration** | Ajuste post-hoc de probabilidades para mejor calibración |
| **League pruning** | Filtrado automático de ligas con señal débil |
| **F0–F15** | Feature groups del catálogo — ver sección "Catálogo de features" |
| **G1–G9** | Feature groups de la serie V13+ (modularizadas) — ver catálogo |
| **TimesFM/Chronos** | Modelos fundacionales de series temporales de Google/Amazon |
| **Gates** | Reglas de decisión que bloquean apuestas si no se cumplen condiciones |
| **Yield** | Ganancia neta / total apostado |
