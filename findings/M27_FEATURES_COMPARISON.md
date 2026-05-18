# M27 Features Comparison

Comparacion entre el snapshot `v6.3 m27` original y la variante independiente `m27_v1`.

## Objetivo

- `v6.3 m27`: baseline actual dentro del pipeline V6.3.
- `m27_v1`: variante aislada para experimentar con un set de features mas limpio y orientado a minuto 27.

## Fuente de cada set

- `v6.3 m27`: `train_q4_models_v6_3.py` -> `_build_q4_features_at_window(...)`
- `m27_v1`: `train_q4_m27_v1.py` -> `_build_m27_v1_features(...)`

## V6.3 m27

### Features incluidas

#### Contexto categorico / buckets
- `league`
- `league_bucket`
- `gender_bucket`
- `home_team_bucket`
- `away_team_bucket`

#### Priors de equipo
- `home_prior_wr`
- `away_prior_wr`
- `prior_wr_diff`
- `prior_wr_sum`

#### Estado por cuartos cerrados
- `q1_diff`
- `q2_diff`

#### Estado del marcador al snapshot
- `cutoff_minute`
- `score_est_home`
- `score_est_away`
- `score_est_diff`

#### Estado parcial del Q3
- `q3_diff`
- `q3_total`

#### Score acumulado 3Q estimado
- `score_3q_home`
- `score_3q_away`
- `score_3q_diff`

#### Graph features
- `gp_count`
- `gp_last`
- `gp_peak_home`
- `gp_peak_away`
- `gp_area_home`
- `gp_area_away`
- `gp_area_diff`
- `gp_mean_abs`
- `gp_swings`
- `gp_slope_3m`
- `gp_slope_5m`

#### PBP features hasta el minuto
- `pbp_home_plays`
- `pbp_away_plays`
- `pbp_plays_diff`
- `pbp_home_3pt`
- `pbp_away_3pt`
- `pbp_3pt_diff`
- `pbp_home_plays_share`
- `pbp_home_3pt_share`

### Debilidades detectadas

- Usa `league` cruda como one-hot, que en m27 parece meter bastante ruido.
- Usa `home_team_bucket` / `away_team_bucket`, tambien potencialmente ruidosos en snapshot temprano.
- Tiene duplicacion exacta de informacion:
  - `score_est_home` y `score_3q_home`
  - `score_est_away` y `score_3q_away`
  - `score_est_diff` y `score_3q_diff`
- Incluye varias features de paridad global del graph que no mostraron senal fuerte en el analisis frio para m27.

## M27_V1

### Features incluidas

#### Contexto general conservado
- `gender_bucket`
- `home_prior_wr`
- `away_prior_wr`
- `prior_wr_diff`
- `prior_wr_sum`

#### Q1 / Q2 cerrados
- `q1_diff`
- `q2_diff`
- `q1_winner`
- `q2_winner`
- `q1_q2_same_winner`
- `home_wins_first2_count`
- `away_wins_first2_count`

#### Halftime / presion base
- `halftime_home`
- `halftime_away`
- `halftime_diff`
- `halftime_total`
- `halftime_leader`
- `halftime_margin_bin`
- `halftime_trailing_side`
- `halftime_deficit_abs`

#### Estado global al minuto 27
- `score_est_home`
- `score_est_away`
- `score_est_diff`
- `current_margin_bin`
- `current_trailing_side`
- `trailing_now_is_home`
- `trailing_now_is_away`
- `trailing_now_deficit_abs`

#### Q3 parcial al minuto 27
- `q3_partial_home`
- `q3_partial_away`
- `q3_partial_diff`
- `q3_partial_total`
- `q3_partial_leader`
- `q3_partial_home_share`
- `halftime_trailer_cutting_in_q3`

#### Momentum reciente / runs
- `recent_3m_home_points`
- `recent_3m_away_points`
- `recent_3m_points_diff`
- `recent_3m_home_event_share`
- `recent_3m_home_max_run`
- `recent_3m_away_max_run`
- `recent_3m_run_diff`
- `recent_3m_last_scoring_home`
- `recent_3m_last_scoring_away`
- `recent_2m_home_points`
- `recent_2m_away_points`
- `recent_2m_points_diff`
- `recent_2m_home_event_share`
- `recent_2m_home_max_run`
- `recent_2m_away_max_run`
- `recent_2m_run_diff`
- `recent_2m_last_scoring_home`
- `recent_2m_last_scoring_away`
- `trailing_now_recent_run_3m`
- `trailing_now_recent_run_2m`

#### Graph features retenidas
- `gp_count`
- `gp_last`
- `gp_slope_3m`
- `gp_slope_5m`

### Features eliminadas respecto a v6.3 m27

#### Categoricas con mayor riesgo de ruido temprano
- `league`
- `league_bucket`
- `home_team_bucket`
- `away_team_bucket`

#### Duplicadas o redundantes
- `cutoff_minute`
- `score_3q_home`
- `score_3q_away`
- `score_3q_diff`

#### Graph global de baja senal fria para m27
- `gp_peak_home`
- `gp_peak_away`
- `gp_area_home`
- `gp_area_away`
- `gp_area_diff`
- `gp_mean_abs`
- `gp_swings`

#### PBP agregadas generales reemplazadas por ventanas recientes
- `pbp_home_plays`
- `pbp_away_plays`
- `pbp_plays_diff`
- `pbp_home_3pt`
- `pbp_away_3pt`
- `pbp_3pt_diff`
- `pbp_home_plays_share`
- `pbp_home_3pt_share`

## Principio de diseño de M27_V1

`m27_v1` intenta modelar mejor lo observable y utilizable a minuto 27:

- informacion cerrada de Q1 y Q2
- estado parcial real del Q3
- quien va abajo y por cuanto
- si el que va abajo viene recortando
- momentum reciente en ventanas cortas
- tendencia reciente del margen

Y evita depender de:

- identificadores categoricos demasiado finos
- agregados globales del partido que diluyen lo reciente
- informacion duplicada del marcador

## Interpretacion rapida

- `v6.3 m27` es mas amplio y generico.
- `m27_v1` es mas especifico para minuto 27 y mas orientado a presion + remontada + run reciente.
- `m27_v1` sacrifica cobertura categorica a cambio de intentar ganar senal causal y reducir ruido.
