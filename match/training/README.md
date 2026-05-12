# Training Workspace

Este directorio contiene el flujo de entrenamiento para predecir ganador de Q3 y Q4.

## Objetivo

- Entrenar modelos para `Q3 home win` y `Q4 home win`.
- Usar solo partidos con informacion completa.
- Probar varias familias de modelos y un ensamble para robustez.
- Mantener versionado (`v1`, `v2`) para comparar y volver a una version anterior.

## Requisitos

Desde la raiz del proyecto (`match/`):

```bash
pip install -r requirements.txt
```

Dependencias clave para entrenamiento:

- scikit-learn
- joblib

## Criterio de data completa

Un partido se usa para entrenamiento solo si tiene:

- Q1, Q2, Q3, Q4 en `quarter_scores`
- `graph_points`
- `play_by_play`

Adicionalmente, para cada target:

- Q3 target: se excluyen empates de Q3.
- Q4 target: se excluyen empates de Q4.

## Versiones

## CLI unificado (recomendado)

Para no recordar comandos sueltos, usa:

```bash
python training/model_cli.py --help
```

Comandos principales:

```bash
python training/model_cli.py eda
python training/model_cli.py train-v1
python training/model_cli.py train-v2
python training/model_cli.py train-v4
python training/model_cli.py compare
python training/model_cli.py summary --metric f1
python training/model_cli.py infer 14442355 --metric f1
python training/model_cli.py calibrate --metric f1
python training/model_cli.py eval-date --date 2026-03-23 --force-version hybrid
python training/model_cli.py all
```

Inferencia por partido (`match_id`):

- Si existe en DB: usa data local.
- Si no existe en DB: usa scraper de SofaScore, guarda en DB y luego predice.

```bash
python training/model_cli.py infer <match_id> --metric f1
python training/model_cli.py infer <match_id> --metric accuracy
python training/model_cli.py infer <match_id> --metric log_loss
```

Para desactivar auto-fetch cuando no existe en DB:

```bash
python training/model_cli.py infer <match_id> --no-fetch
```

Para forzar re-extraccion aunque el partido ya este en DB (datos stale, Q3 faltante):

```bash
python training/model_cli.py infer <match_id> --refresh
python training/model_cli.py infer <match_id> --metric f1 --force-version hybrid --refresh
```

Para comparar versiones en el mismo partido:

```bash
python training/model_cli.py infer <match_id> --metric f1 --force-version v2
python training/model_cli.py infer <match_id> --metric f1 --force-version v4
```

Opciones de force-version:

- auto (default): usa la mejor version segun compare.
- v1, v2, v4: fuerza version estatica para comparar head-to-head.
- hybrid: fuerza q3=v2 y q4=v4.

Evaluacion diaria por fecha (descubre/ingesta/evalua):

```bash
python training/model_cli.py eval-date --date 2026-03-23 --force-version hybrid
python training/model_cli.py eval-date --date 2026-03-23 --force-version hybrid --limit-matches 50 --json
python training/model_cli.py eval-date --date 2026-03-23 --force-version hybrid --result-tag hybrid_f1_v1
```

Notas:

- Muestra barra de progreso en ingesta y evaluacion.
- En ingesta muestra dos barras: progreso y errores.
- Si un partido ya esta FT completo en DB (Q1-Q4), no lo re-descarga.
- Los errores de ingesta se agregan y se muestran en resumen para evitar spam de log.
- Muestra preview por partido mientras evalua:
  - local, visitante, q3 (pick + signal), q3_gano, q4 (pick + signal), q4_gano.
- Guarda resultados en DB en `eval_match_results` con PK (`event_date`, `match_id`).
- Cada iteracion/modelo usa columnas dinamicas por `--result-tag`.
  - Si no pasas `--result-tag`, usa `<force-version>_<metric>`.
- Si no hay BET, tambien se guarda como `NO_BET`.

Cada corrida guarda historial automaticamente en:

- `training/model_comparison/daily_eval_<YYYY-MM-DD>_<policy>_<metric>.json`

Pipeline completo sin repetir todo:

```bash
python training/model_cli.py all --skip-eda
```

## Calibracion del gate de riesgo

Comando recomendado:

```bash
python training/model_cli.py calibrate --metric f1 --limit 3000
```

Parametros utiles:

```bash
python training/model_cli.py calibrate --metric f1 --limit 5000 --odds 1.91 --min-coverage 0.08
```

Que hace:

- Evalua historico FT con split temporal (train/validacion).
- Ajusta umbrales del gate por target (`q3`, `q4`) para balancear ROI y cobertura.
- Guarda configuracion usada por inferencia automaticamente.

Salida:

- `training/model_outputs_v2/gate_config.json`
- `training/model_outputs_v2/gate_calibration_report.json`

Uso en inferencia:

- `training/infer_match.py` carga `gate_config.json` si existe.
- Si no existe, usa umbrales por defecto hardcodeados.

## V1

Script base:

```bash
python training/train_q3_q4_models.py
```

Salida en:

- `training/model_outputs/`

## V2

Script mejorado:

```bash
python training/train_q3_q4_models_v2.py
```

Mejoras de V2:

- Segmentacion de ligas en buckets (`league_bucket`) por top volumen.
- Encoding de equipos por buckets (`home_team_bucket`, `away_team_bucket`) por top volumen.
- Features de fuerza historica por equipo (ventana rolling).
- Metricas extendidas para evaluar efectividad:
  - accuracy
  - f1
  - precision
  - recall
  - roc_auc
  - log_loss
  - brier

Salida en:

- `training/model_outputs_v2/`

## V3 (dinamico por minuto para live)

Script:

```bash
python training/train_q3_q4_models_v3.py
```

O desde CLI unificado:

```bash
python training/model_cli.py train-v3
```

Que hace V3:

- Entrena snapshots temporales para evitar fuga temporal en live.
- Q3: snapshot minuto 24 (halftime).
- Q4: snapshots minuto 24, 30 y 36.
- Entrena dos modelos base por snapshot:
  - logistic regression
  - gradient boosting
- Calcula ensemble por promedio de probabilidad.

Seleccion dinamica en inferencia live:

- Q3:
  - minuto < 24: no disponible (aun no hay snapshot de halftime)
  - minuto >= 24: usa `q3_m24`
- Q4:
  - 24 <= minuto < 30: usa `q4_m24`
  - 30 <= minuto < 36: usa `q4_m30`
  - minuto >= 36: usa `q4_m36`

Que NO hace V3:

- No modela overtime por separado.
- No incorpora odds/cuotas del mercado.
- No reemplaza evaluacion walk-forward estricta por fecha (se mantiene split temporal simple 80/20).

En que minuto apostar (guia practica):

- Q3 winner: minuto 24 (halftime).
- Q4 winner:
  - agresivo: minuto 30 (mitad Q3)
  - conservador: minuto 36 (inicio Q4)
  - temprano (mas riesgo): minuto 24

## V4 (pressure/comeback features)

Script:

```bash
python training/train_q3_q4_models_v4.py
```

O desde CLI unificado:

```bash
python training/model_cli.py train-v4
```

Que agrega V4 respecto a V2:

- Features de marcador global y contexto de remontada al inicio del cuarto objetivo.
- Cuantos puntos faltan para empatar y para ponerse arriba.
- Intensidad requerida de anotacion para remontar (puntos por minuto requeridos).
- Proxy de ritmo desde PBP para estimar eventos restantes y presion ofensiva.
- Ratios de presion (ritmo requerido vs ritmo anotador observado).
- Features clutch/momentum en ventana reciente de 6 minutos antes del cuarto objetivo.
- Deteccion de rachas maximas por equipo (max run points).
- Ultimo equipo en anotar dentro de ventana y share de eventos anotadores.

Ejemplos de features nuevas:

- `trailing_points_to_tie`
- `trailing_points_to_lead`
- `required_ppm_tie`
- `required_ppm_lead`
- `pressure_ratio_tie`
- `pressure_ratio_lead`
- `urgency_index`
- `req_pts_per_trailing_event`
- `clutch_home_points`
- `clutch_away_points`
- `clutch_points_diff`
- `clutch_home_max_run_pts`
- `clutch_away_max_run_pts`
- `clutch_run_diff`
- `clutch_last_scoring_home`
- `clutch_last_scoring_away`

Salida en:

- `training/model_outputs_v4/`

Salida V3:

- `training/model_outputs_v3/v3_metrics.csv`
- `training/model_outputs_v3/v3_summary.json`
- modelos `.joblib` por target/snapshot/modelo

## V11 (Over/Under - Total Points Regression)

Script:

```bash
python training/train_q3_q4_regression_v11.py
```

O desde CLI unificado (si está configurado):

```bash
python training/model_cli.py train-v11
```

### Que hace V11

- **Predice el total de puntos** del Q3 o Q4 (ej: 45 pts) - no threshold fijo
- **Modelos separados por género**: `men_or_open` vs `women`
- **12 modelos totales**: 2 géneros × 3 targets (home/away/total) × 2 cuartos (Q3/Q4)
- **Sin umbrales hardcodeados**: la línea de la casa se pasa como parámetro en predicción

### Features utilizados

- `league_bucket`, `home_team_bucket`, `away_team_bucket`: equipos/ligas en buckets
- `gender_bucket`: inferido del nombre de la liga/equipo
- Score previo (Q1+Q2 para Q3, Q1+Q2+Q3 para Q4)
- Diferencias de score por cuarto (`q1_diff`, `q2_diff`, `q3_diff`)
- **Graph stats**: `gp_count`, `gp_last`, `gp_peak_home`, `gp_peak_away`, `gp_area_home`, `gp_area_away`, `gp_area_diff`, `gp_mean_abs`, `gp_swings`, `gp_slope_3m`, `gp_slope_5m`
- **PBP stats**: `pbp_home_pts_per_play`, `pbp_away_pts_per_play`, `pbp_pts_per_play_diff`, `pbp_home_plays`, `pbp_away_plays`, `pbp_plays_diff`, `pbp_home_3pt`, `pbp_away_3pt`, `pbp_3pt_diff`, `pbp_home_plays_share`, `pbp_home_3pt_share`
- Win rate histórico del equipo (ventana de 12 partidos)

### Algoritmo

1. **Entrenamiento**: 80/20 split temporal
2. **Modelos base**: Ridge + GradientBoosting + XGBoost
3. **Ensemble**: promedio ponderado (Ridge 0.20, GB 0.35, XGB 0.45)
4. **Predicción**: promedio del ensemble

### Evaluación (betting simulation)

Script:

```bash
python training/evaluate_v11_betting.py --line 45
python training/evaluate_v11_betting.py --line 55 --margin 5 --min-edge 3
```

Parámetros:
- `--line`: línea de la sportsbook (default: 45)
- `--margin`: margen que agrega la sportsbook (default: 5)
- `--min-edge`: mínimo edge para apostar (default: 3)

Lógica de apuesta (V2):

```
prediction = modelo predice (ej: 24 pts)
suggested_line = prediction + MARGIN (ej: 29 pts)
edge = suggested_line - sportsbook_line (ej: 29 - 55 = -26)

if edge > MIN_EDGE: bet OVER (valor subestimado)
elif edge < -MIN_EDGE: bet UNDER (valor sobreestimado)
else: NO_BET (edge insuficiente)
```

Hit: `actual > line` (OVER gana) o `actual < line` (UNDER gana)

### Resultados (línea 55, margin 5, min-edge 3)

| Target | Bet | Count | Hits | Rate | Profit |
|--------|-----|-------|------|------|--------|
| q3_total | UNDER | 1590 | 1492 | 93.8% | $1,260 |
| q4_total | UNDER | 1590 | 1486 | 93.5% | $1,248 |

**OVER only**: 0 bets (modelo siempre subestima)
**UNDER: 3180 bets, 2978 hits (93.6%), ROI: 78.9%, Profit: $2,508**

### Análisis de error

- Predicción media: 23.8 pts
- Actual media: 39.7 pts
- Error sistemático: **-15.9 pts** (modelo subestima ~16 puntos)

Esto es porque el modelo está entrenado con features de ritmo de Q1+Q2 y no captura bien el ritmo de Q3/Q4 que puede variar.

**Nota importante**: El alto % de acierto de UNDER es engañoso. El modelo subestima sistemáticamente, y con líneas reales de 45-65 pts, siempre apostamos UNDER y ganamos. En la práctica, las sportsbooks ajustan la línea durante el partido, por lo que este backtest no refleja la realidad.

### Archivos de salida

- `training/model_outputs_v11/`
  - `q3_total_men_or_open_gb.joblib` - modelo Q3 total hombres
  - `q3_total_women_gb.joblib` - modelo Q3 total mujeres
  - `q4_total_men_or_open_gb.joblib` - modelo Q4 total hombres
  - `q4_total_women_gb.joblib` - modelo Q4 total mujeres
  - `q3_total_*_ridge.joblib`, `q3_total_*_xgb.joblib` - otros modelos
  - `metrics.csv` - métricas de entrenamiento (MAE, RMSE, R2)
  - `betting_results_*.csv` - resultados de betting simulation
  - `betting_report_v11.txt` - reporte completo
  - `betting_report_v11_over_only.txt` - reporte solo OVER

### Posibles mejoras

1. **Calibrar predicción**: el modelo subestima ~16 pts, agregar corrección (+16) a la predicción
2. **Usar features de ritmo en tiempo real**: usar graph_points para calcular ritmo actual del partido
3. **Entrenar por separado por género y league**: aumentar granularidad del modelo
4. **Feature engineering avanzado**: distancia de viajes, fatigue, B2B games
5. **Incorporar odds del mercado**: como feature o para calibrar líneas
6. **Live betting**: ajustar línea durante el partido based en ritmo actual (Q3/Q4)

### Uso en producción

**Opción 1: Desde match en DB**

```bash
python training/predict_v11.py --match-id 15556170 --line 55.5
```

Salida:
```
============================================================
MATCH: BMS Herlev U13 vs SKM-Mustangs U13
Target: Q3_TOTAL | Gender: men_or_open
============================================================
Q1+Q2 Score: 27-28 (total: 55)
Model Prediction: 25.4 pts
Suggested Line: 30.4 pts (prediction + 5.0)
Sportsbook Line: 55.5
Edge: -25.1
Bet: UNDER
Recommendation: APOSTAR UNDER
Odds: 1.91
If bet $100: Win $91 if hit, Lose $100 if miss
```

**Opción 2: Live betting (manual params)**

```bash
python training/predict_v11.py --league "NBA" --home "Lakers" --away "Celtics" --q1h 28 --q2h 27 --q1a 25 --q2a 20 --line 55.5
```

Salida:
```
============================================================
LIVE PREDICTION
============================================================
League: NBA
Match: Lakers vs Celtics
Q1+Q2: 28-25 + 27-20 = 55-45 (total: 100)
Gender: men_or_open
Sportsbook Line: 55.5
Margin: 5.0, Min Edge: 3.0

Q3_TOTAL:
  Prediction: 37.5 | Suggested: 42.5 | Edge: -13.0
  -> UNDER

Q4_TOTAL:
  Prediction: 27.6 | Suggested: 32.6 | Edge: -22.9
  -> UNDER
```

**Parámetros:**
- `--line`: línea de la sportsbook (obligatorio)
- `--margin`: margen (default: 5)
- `--min-edge`: edge mínimo para apostar (default: 3)
- Para live: `--league`, `--home`, `--away`, `--q1h`, `--q2h`, `--q1a`, `--q2a`

---

## Comparacion entre versiones

```bash
python training/compare_model_versions.py
```

Genera (comparando V1, V2 y V4):

- `training/model_comparison/version_comparison.csv`
- `training/model_comparison/version_comparison.json`

## Como volver entre versiones

No se sobreescribe la version anterior. Cada version tiene su carpeta.

- V1: `training/model_outputs/`
- V2: `training/model_outputs_v2/`

Para usar una u otra, apunta tu inferencia/carga de modelos a la carpeta deseada.

## Artefactos esperados por version

- `q3_dataset.csv`
- `q4_dataset.csv`
- `q3_metrics.csv`
- `q4_metrics.csv`
- `q3_consensus.json`
- `q4_consensus.json`
- modelos `*.joblib`

## Interpretacion rapida de efectividad

- `accuracy`: acierto global.
- `f1`: balance precision/recall (mejor cuando clases no estan perfectamente balanceadas).
- `precision`: calidad de picks positivos.
- `recall`: cobertura de picks positivos.
- `roc_auc`: calidad de ranking probabilistico.
- `log_loss` y `brier`: calidad de calibracion de probabilidades.

Para apuestas, no basta con accuracy; la calibracion (`log_loss`, `brier`) y la consistencia por segmento son clave.

## V6.2 (league pruning + name exclusion)

Script:

```bash
python training/train_q3_q4_models_v6_2.py
```

Que agrega V6.2 respecto a V6:

- Filtrado automatico de ligas por signal (signal strength pruning por abs_effect_vs_global).
- Exclusion de ligas por patron de nombre (categorias: obscure, amateur, youth, etc.) via `v6_2_league_name_exclusions.json`.
- Elimina MLP del ensemble; usa xgb y hist_gb.
- Champion Q4: 0.6*xgb + 0.4*hist_gb.
- Champion Q3: xgb solo.
- Exporta A/B vs V6 baseline.

Salida en:

- `training/model_outputs_v6_2/`

## V6.3 (ventanas dinamicas, Q4 only, 70/15/15, calibracion isotonica)

Script:

```bash
python training/train_q3_q4_models_v6_3.py
```

### Cambios respecto a V6.2

| Aspecto | V6.2 | V6.3 |
|---|---|---|
| Targets | Q3 + Q4 | **Q4 solo** (Q3 descartado) |
| Ventanas de features | Estaticas (36 min) | **Dinamicas** [27, 30, 33, 36] min |
| Split | 80/20 temporal | **70/15/15** (train/val/test) temporal |
| Calibracion | Sin calibracion | **Isotonica post-hoc** (fit en val) |
| Cross-validation | Sin CV | **TimeSeriesSplit 5 folds** |
| Champion Q4 | 0.6*xgb + 0.4*hist_gb | avg(xgb, hist_gb) |
| Baseline A/B | vs V6 | vs V6 + vs V6.2 |

### Ventanas dinamicas

El monitor de apuestas puede llegar a diferentes minutos del partido (minuto 27 al 38 para Q4). En lugar de entrenar solo con features del minuto 36, V6.3 genera un sample por cada ventana de snapshot para cada partido:

- **snapshot 27**: inicio de Q3 + primeros minutos (baja completitud de Q3)
- **snapshot 30**: mitad de Q3 (50% completitud)
- **snapshot 33**: final de Q3 (alta completitud)
- **snapshot 36**: Q3 completo (100% completitud, cutoff original)

Esto multiplica el dataset por 4 (una fila por partido por ventana) y agrega `snapshot_minute`, `snapshot_minutes_before_q4` y `snapshot_q3_completeness` como features numericas. El modelo aprende el sesgo de cada momento de inferencia.

### Calibracion

Se entrena una calibracion isotonica post-hoc sobre el split de validacion (15%) y se evalua en el test (15%). Los artefactos exportados incluyen:

- `xgb` (raw), `hist_gb` (raw): modelos sin calibrar
- `xgb_cal`, `hist_gb_cal`: probabilidades calibradas en test
- `champion_q4_ensemble_avg`: avg raw
- `champion_q4_ensemble_avg_cal`: avg calibrado

### Cross-Validation Temporal

Se ejecuta `TimeSeriesSplit(n_splits=5)` sobre **todos los datos** para verificar estabilidad en el tiempo y detectar degradacion del modelo con nuevos partidos. Los resultados se exportan en `q4_cv_metrics.csv` con media y desviacion por fold.

### Actualizacion del modelo con nuevos partidos

Cuando llegan nuevos partidos, ejecutar de nuevo el script re-entrena desde cero sobre toda la historia disponible (enfoque estateless, sin fine-tuning). La CV temporal muestra si la efectividad se mantiene en los folds mas recientes.

### Filtro de ligas

Mantiene la misma logica de V6.2:

- Exclusion por nombre: `v6_2_league_name_exclusions.json` (67 patrones)
- Exclusion por signal: `LEAGUE_MIN_TRAIN_ROWS=30`, `LEAGUE_MIN_EFFECT_ABS_DIFF=0.015`

### Salida en `model_outputs_v6_3/`

| Archivo | Descripcion |
|---|---|
| `q4_metrics.csv` | Metricas holdout por modelo y split |
| `q4_cv_metrics.csv` | Metricas por fold de CV temporal |
| `ab_comparison_v6_vs_v6_3.csv` | Delta vs V6 baseline |
| `ab_comparison_v6_2_vs_v6_3.csv` | Delta vs V6.2 |
| `league_signal_q4.csv` | Signal por liga en train |
| `league_name_exclusions_q4.csv` | Partidos excluidos por patron de nombre |
| `window_distribution.csv` | Distribucion de samples por ventana |
| `q4_champion.joblib` | Artefacto del modelo champion (xgb+hist_gb) |
| `q4_xgb.joblib` | Modelo xgb raw |
| `q4_hist_gb.joblib` | Modelo hist_gb raw |
| `run_summary.json` | Resumen de config y resultados |
| `V6_3_REPORT.md` | Reporte completo con tablas A/B y CV |
