# M27_V2 Roadmap

## Estado actual (Mayo 2026)

| Métrica | Valor |
|---|---|
| ROC AUC (test) | **0.668** (ensemble XGB + HistGB) |
| ROC AUC (filter-10m) | 0.671 |
| Brier | 0.228 |
| Accuracy | 0.623 |
| Features | 86 (7 podadas del análisis de importancia) |
| Algoritmos | XGBoost + HistGradientBoosting, AUC-weighted ensemble + isotonic |
| Split | 70/15/15 temporal, 5-fold TimeSeries CV match-aware |

---

## Feature importance: el modelo en una foto

### Top 10 por XGB importance

| # | Feature | Imp | Corr |
|---|---------|:---:|:----:|
| 1 | `recent_2m_points_diff` | **0.184** | +0.240 |
| 2 | `prior_wr_diff` | 0.057 | +0.168 |
| 3 | `recent_2m_run_diff` | 0.036 | +0.239 |
| 4 | `score_halftime_diff_ratio` | 0.026 | +0.116 |
| 5 | `q3_partial_total` | 0.020 | -0.000 |
| 6 | `recent_2m_home_max_run` | 0.020 | +0.177 |
| 7 | `gp_last` | 0.018 | +0.115 |
| 8 | `recent_3m_away_points` | 0.017 | -0.132 |
| 9 | `halftime_margin_bin=04_07` | 0.017 | +0.001 |
| 10 | `current_run_away` | 0.015 | -0.180 |

### Por grupo de features

| Grupo | Importancia | Veredicto |
|-------|:-----------:|:---------:|
| **Recent windows (2m + 3m)** | **35.9%** | Domina el modelo |
| Q3 Partial | 9.8% | Señal estable |
| Win Rates | 8.9% | Único feature no-recent en top 5 |
| PBP Density (G4) | 5.9% | `pbp_scoring_density` muerta, el resto aporta |
| Score Ratios (G1) | 4.7% | `score_halftime_diff_ratio` es el mejor |
| Scoring Runs (F8) | 4.1% | `current_run_away` tiene corr -0.18 |
| Graph Points | 3.6% | `gp_count` muerta (constante), `gp_last` es la mejor |

---

## Historia de experimentos

### v2 baseline (corrección de bugs)
- 4 bugs corregidos vs v1: eligibility filter, target None skip, match-aware split, quality guards
- AUC: 0.667 (idéntico a v1)

### + Normalized Q3 rates
- `q3_partial_home_rate/away_rate/diff_rate` siempre activos
- Sin cambio medible en AUC

### + Low-effort features (F8 Scoring Runs, G1 Score Ratios, G4 PBP Density)
- Ninguna feature nueva rompió el top 3 de importancia
- AUC: 0.668 (+0.0009 vs baseline, ruido)

### AUC-weighted ensemble (antes avg simple)
- Pesos: ~48% XGB / ~52% HistGB
- Sin mejora vs avg simple (modelos rinden casi igual)

### + LightGBM
- LGB rindió peor (0.660) que XGB (0.666) e HistGB (0.668)
- Ensemble a 3 modelos: 0.666 (peor que 2 modelos)
- **Descartado**

### Poda de 7 features muertas
- `halftime_leader`, `halftime_trailing_side`, `q3_partial_leader` (categóricas completas, todos sus niveles con imp=0)
- `trailing_now_is_home/away` (redundante con `current_trailing_side`)
- `gp_count` (constante en snap 27)
- `pbp_scoring_density` (imp=0, corr=0.0003)
- AUC idéntico (0.668), 93 → 86 features

---

## Limitación fundamental

El modelo es esencialmente un predictor de `recent_2m_points_diff` (18.4% de importancia). Recent windows 2m+3m = 35.9% del árbol. Cualquier feature derivada de anotaciones va a correlacionar con esto.

Para mejorar hay que encontrar señales **ortogonales** al diferencial de puntos reciente.

---

## Próximas señales a explorar

### Grupo A: Desde datos existentes (bajo esfuerzo)

#### A1 — Volatilidad / Lead changes en Q3
- **Qué**: lead changes count, desviación estándar del score diff, turning points
- **De dónde**: graph_points de Q3
- **Por qué**: mide cuán "loco" fue el Q3, no correlaciona directo con points_diff
- **Riesgo**: correlación baja con target

#### A2 — Recovery spent
- **Qué**: cuánto del déficit del HT logró recuperar el perdedor en Q3
- **De dónde**: graph_points + HT scores
- **Por qué**: captura esfuerzo de remontada, no solo diff actual
- **Riesgo**: puede correlacionar con `halftime_trailer_cutting_in_q3` que ya existe

#### A3 — Momentum acceleration
- **Qué**: segunda derivada del score diff (cambio en la velocidad)
- **De dónde**: graph_points
- **Por qué**: captura aceleración/deceleración del momentum
- **Riesgo**: ruido, señal frágil

#### A4 — Team records parseados
- **Qué**: win% más granular que `prior_wr` (split home/away, last 10 games)
- **De dónde**: `home_record`/`away_record` en matches (string "W-L")
- **Por qué**: prior_wr es global, no captura forma reciente
- **Riesgo**: puede solaparse con `prior_wr`

#### A5 — Sequías de anotación en Q3
- **Qué**: tiempo máximo sin anotar de cada equipo en Q3
- **De dónde**: PBP events
- **Por qué**: sequía larga → equipo frío → menos probable que gane Q4
- **Riesgo**: señal débil

#### A6 — Ritmo de posesiones en Q3
- **Qué**: eventos por minuto, no puntos por evento
- **De dónde**: PBP count / minutos transcurridos
- **Por qué**: pace alto favorece ciertos estilos
- **Riesgo**: ya tenemos `pbp_count` y `pbp_scoring_density`

### Grupo B: Fuera del scope actual (esfuerzo alto)

#### B1 — Fouls / Faltas
- **Qué**: conteo de faltas por equipo, bonus situation
- **De dónde**: **NO está en la DB actual**. Requeriría scraping adicional.
- **Impacto potencial**: alto — foul trouble afecta rotaciones y ritmo

#### B2 — Timeouts
- **Qué**: timeouts restantes, momentum post-timeout
- **De dónde**: **NO está en la DB actual**
- **Impacto potencial**: medio-alto — timeouts cambian momentum

#### B3 — Lineups / Sustituciones
- **Qué**: quién juega, descanso de estrellas, profundidad de banca
- **De dónde**: **NO está en la DB actual**
- **Impacto potencial**: alto — star player resting cambia todo

#### B4 — Datos externos
- **Qué**: Vegas lines, Elo ratings, power rankings
- **De dónde**: APIs externas
- **Impacto potencial**: alto — el mercado tiene información que nosotros no

---

## Orden sugerido de implementación

1. **A1 + A2 + A3** (volatilidad Q3 + recovery + momentum acceleration) — mismo pase de graph_points, bajo esfuerzo
2. **A4** (team records) — parsear strings, esfuerzo mínimo
3. **A5** (sequías) — un pase de PBP events
4. **A6** (ritmo) — ya casi lo tenemos
5. **B1-B4** — requieren nueva data

Cada batch se prueba con: implementar → rebuild cache → train → comparar AUC contra baseline de 0.668.

---

## Referencias

- `findings.md` — hallazgos generales
- `modelos.md` — tabla de modelos y features
- `M27_FEATURES_COMPARISON.md` — comparación de features entre versiones
- `m27_v2_feature_importance.csv` — importancia completa (86 features)
- `train_q4_m27_v2.py` — script de entrenamiento
