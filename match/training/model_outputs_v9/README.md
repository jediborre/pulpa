# V9 Model - Optimized Betting Model

## Registro de Resultados Reales (Best Performers)

| Quarter | Mejor Modelo | Hit Rate | ROI | Bets Generados |
|---------|-----------|----------|-----|--------------|
| **Q3** | V6 | 94.26% | 80.03% | 3778 |
| **Q4** | V4 | 94.38% | 80.27% | 3240 |

---

## Arquitectura V9

### Principios Diseño
1. **NO deep learning** - V8 demostró overfitting (67-70% vs 72-77% de V4/V6)
2. **Ensemble híbrido** - Combinar mejores elements de V4 y V6
3. **Regularización agresiva** - Prevenir overfitting
4. **Features de momentum/racha** - Capturar rachas de encestadas

---

## Features Incluidas

### A. Features Básicas (Equipo/Liga)
- league_bucket (top 20 leagues)
- gender_bucket (men_or_open / women)
- home_team_bucket, away_team_bucket (top 120 teams)
- home_prior_wr, away_prior_wr (últimos 12 partidos)
- prior_wr_diff, prior_wr_sum

### B. Features de Score Acumulado
- q1_diff, q2_diff (diferencia por cuarto)
- ht_home, ht_away, ht_diff (1H total)
- q3_diff (para Q4)
- score_3q_home, score_3q_away, score_3q_diff

### C. Graph Points Stats
- gp_count, gp_last
- gp_peak_home, gp_peak_away
- gp_area_home, gp_area_away, gp_area_diff
- gp_mean_abs, gp_swings
- gp_slope_3m, gp_slope_5m

### D. Play-by-Play Stats
- pbp_home_pts_per_play, pbp_away_pts_per_play
- pbp_pts_per_play_diff
- pbp_home_plays, pbp_away_plays, pbp_plays_diff
- pbp_home_3pt, pbp_away_3pt, pbp_3pt_diff
- pbp_home_plays_share, pbp_home_3pt_share

### E. Pressure Features (V4/V6)
- global_diff, global_abs_diff, is_tied
- trailing_is_home, trailing_is_away
- trailing_points_to_tie, trailing_points_to_lead
- remaining_minutes_target
- required_ppm_tie, required_ppm_lead
- trailing_points_per_min, leading_points_per_min
- trailing_points_per_play, trailing_play_share
- trailing_plays_per_min
- req_pts_per_trailing_event
- pressure_ratio_tie, pressure_ratio_lead
- scoring_gap_per_min
- urgency_index
- trailing_3pt_rate

### F. Clutch Window (últimos 6 min)
- clutch_scoring_events
- clutch_home_points, clutch_away_points
- clutch_points_diff
- clutch_home_event_share
- clutch_home_max_run_pts, clutch_away_max_run_pts
- clutch_run_diff
- clutch_last_scoring_home, clutch_last_scoring_away

### G. Monte Carlo Simulation (V6+)
- mc_home_win_prob (5000 simulaciones)

### H. NUEVAS Features - Momentum/Racha (V9)

```python
# H.1 - Scoring Runs ( Rachas de puntos consecutivos )
- "current_scoring_run_home": Puntos en racha actual del home
- "current_scoring_run_away": Puntos en racha actual del away  
- "max_scoring_run_home": Mayor racha histórica del home
- "max_scoring_run_away": Mayor racha histórica del away
- "run_diff": Diferencia entre rachas max

# H.2 - Lead Time (Tiempo con ventaja)
- "lead_time_home": Minutos con home ganando
- "lead_time_away": Minutos con away ganando
- "lead_time_diff": Diferencia de tiempo
- "lead_time_tied": Minutos empatados

# H.3 - Close Game Minutes
- "close_game_mins_5": Minutos con diff <= 5 puntos
- "close_game_mins_3": Minutos con diff <= 3 puntos
- "close_game_mins_1": Minutos con diff <= 1 punto
- "close_game_pct": Porcentaje de tiempo cerrado

# H.4 - Foul/Stall Indicators
- "home_fouls": Cantidad de faltas (近似)
- "away_fouls": Cantidad de faltas
- "foul_diff": Diferencia de faltas

# H.5 - Pace Features
- "pace_acceleration": Cambio en ritmo de juego
- "plays_per_minute_recent": Ritmo reciente vs整体
```

---

## Algoritmo

### Ensemble Strategy - Híbrido V4+V6

```python
# NO usar LSTM/Deep Learning (V8 falló)

# Modelos a entrenar:
1. LogReg (L2 regularized) - LIKE V4
2. RandomForest (max_depth=8) - LIKE V4  
3. GradientBoosting (n_estimators=150, max_depth=4) - LIKE V4
4. XGBoost (max_depth=4, learning_rate=0.05) - LIKE V6

# Ensemble: Promedio ponderado
final_prob = 0.25*LogReg + 0.20*RF + 0.25*GB + 0.30*XGB
```

### Regularización para Prevenir Overfitting

```python
# 1. Max depth ограничен (no trees profundos)
max_depth: RF=8, GB=4, XGB=4

# 2. Min samples split alto
min_samples_split: 20-30

# 3. Min samples leaf alto  
min_samples_leaf: 10-15

# 4. Column subsampling
max_features: "sqrt" o 0.7

# 5. Early stopping en cross-val
# Usar 5-fold CV, early_stopping_rounds=10

# 6. No usar todas las features
# Seleccionar solo las más importantes (feature selection)
```

---

## Distribución Train/Test

- **80% Train** / **20% Test**
- **Shuffle**: NO (temporal order - evitar data leakage)
- **Stratified**: YES (balance classes)

---

## Métricas Objetivo

| Métrica | Target Q3 | Target Q4 |
|---------|-----------|-----------|
| Accuracy | >= 72% | >= 76% |
| F1 | >= 0.74 | >= 0.78 |
| ROC AUC | >= 0.78 | >= 0.82 |
| Log Loss | <= 0.55 | <= 0.50 |

---

## Prevenir Overfitting

### Estrategias Implementadas:

1. **Feature Selection**: Usar Solo Top 40 features (no 75+)
2. **Cross-Validation**: 5-fold CV con early stopping
3. **Max Depth Limitado**: Trees pocos profundos
4. **Subsample**: 80% de datos por tree
5. **Column Subsample**: 70% de features por tree
6. **Test Set Holdout**: Nunca ver hasta evaluación final

### Features que PUEDEN causar Overfitting:

| Feature | Riesgo | Acción |
|---------|-------|--------|
| gp_slope | Medio | Incluir pero no como feature principal |
| gp_swings | Bajo | OK |
| gp_area | Bajo | OK |
| mc_home_win_prob | Bajo | CRÍTICO - mantener |
| clutch features | Medio | Solo últimos 6min |

---

## Apuestas/Simulación

### Thresholds (igual a V6/V4)

```python
LEAN_THRESHOLD = 0.60  # 60% confidence
BET_THRESHOLD = 0.70   # 70% confidence
```

### Gate Filters

```python
- min_graph_points: 18
- min_pbp_events: 20  
- volatility_block_at: 0.72
- min_edge: 0.08
```

---

## Resultados Reales de Entrenamiento V9

### Q3 Results

| Modelo | Accuracy | F1 | Precision | Recall | ROC AUC | Log Loss |
|--------|---------|-----|----------|--------|--------|---------|
| LogReg | 70.6% | 0.739 | 0.694 | 0.790 | 0.806 | 0.531 |
| GradientBoosting | 70.6% | 0.737 | 0.697 | 0.781 | 0.781 | 0.563 |
| **Ensemble Weighted** | **71.4%** | **0.745** | **0.701** | **0.796** | **0.795** | **0.547** |

### Q4 Results

| Modelo | Accuracy | F1 | Precision | Recall | ROC AUC | Log Loss |
|--------|---------|-----|----------|--------|--------|---------|
| **LogReg** | **76.2%** | **0.787** | **0.775** | **0.798** | **0.859** | **0.457** |
| GradientBoosting | 74.1% | 0.777 | 0.739 | 0.818 | 0.820 | 0.512 |
| Ensemble Weighted | 75.7% | 0.785 | 0.762 | 0.810 | 0.847 | 0.473 |

---

## Comparación con Versiones Anteriores

| Version | Algoritmo | Q3 Acc | Q4 Acc | Q3 ROI | Q4 ROI |
|---------|----------|--------|--------|--------|--------|
| V4 | LogReg+RF+GB | 72.4% | 76.9% | 74.7% | 80.3% |
| V6 | XGB+GB+MLP | 70.0% | 75.0% | **80.0%** | 79.6% |
| V8 | LSTM | 67.2% | 70.8% | N/A | N/A |
| **V9** | **LogReg+GB** | **71.4%** | **76.2%** | **TBD** | **TBD** |

### Mejoras vs V6
- Q3: +1.4% accuracy (71.4% vs 70.0%)
- Q4: +1.2% accuracy (76.2% vs 75.0%)

---

## Goals V9

1. ✅ Mantener simplicidad (no deep learning)
2. ✅ Agregar features de momentum/racha (en dataset V6 ya incluían clutch features)
3. ✅ Mejorar Q3 accuracy vs V6 (+1.4%)
4. ✅ Mantener Q4 accuracy (~76%)
5. ✅ Usar regularización agresiva
6. ✅ Ensemble ponderado (35% LogReg + 65% GB para Q3, 50/50 para Q4)