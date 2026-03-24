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
```

Notas:

- Muestra barra de progreso en ingesta y evaluacion.
- Si un partido ya esta FT completo en DB (Q1-Q4), no lo re-descarga.

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
