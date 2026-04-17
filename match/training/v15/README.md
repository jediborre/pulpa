# V15 - Modelo live de predicci�n de ganador por cuarto (Q3 / Q4) en basketball

Sistema de machine learning para decidir apuestas de "ganador del cuarto" en partidos de baloncesto masculino, tomando la decisi�n **antes** de que arranque el cuarto objetivo. Optimizado para odds pesimistas (**1.40**) donde el break-even matem�tico es del 71.43% y cada punto porcentual de hit rate es dinero real.

> Documentaci�n relacionada
> - [RECOMENDACIONES.md](RECOMENDACIONES.md)   c�mo operar semana a semana, kill-switches, overrides.
> - [ROADMAP.md](ROADMAP.md)   pr�ximos pasos, deuda t�cnica de v14, opini�n honesta sobre el modelo.

---

## Tabla de contenidos

1. [Resumen ejecutivo](#resumen-ejecutivo)
2. [Arquitectura](#arquitectura)
3. [Datos y preprocesamiento](#datos-y-preprocesamiento)
4. [Feature engineering](#feature-engineering)
5. [Entrenamiento](#entrenamiento)
6. [Evaluaci�n y criterio de selecci�n](#evaluaci�n-y-criterio-de-selecci�n)
7. [Resultados del modelo activo](#resultados-del-modelo-activo)
8. [Uso](#uso)
9. [Integraci�n](#integraci�n)
10. [Estructura del proyecto](#estructura-del-proyecto)

---

## Resumen ejecutivo

**Tarea**: dado un partido de baloncesto en curso, justo antes del inicio del Q3 (minuto 22) o Q4 (minuto 31), predecir qu� equipo anotar� m�s puntos en ese pr�ximo cuarto, **con suficiente confianza para que una apuesta a odds 1.40 sea rentable en el largo plazo**.

**Pipeline**:

```
DB SQLite  ->  Samples (din�mico por snapshot)  ->  Features (50+)
                                                          |
                 Walk-forward league stats  <--->  Ensemble por liga
                                                          |
                                                 Threshold �ptimo (val)
                                                          |
                                           Gates (volatilidad, liga, QA)
                                                          |
                                                    BET / NO_BET
```

**Principios rectores**

| Principio | C�mo se aplica |
|---|---|
| Un modelo **por liga** | No hay fallback global. Si no hay datos suficientes para una liga -> `NO_BET`. |
| **Pesimismo** en odds | Toda la pipeline se optimiza para `odds=1.40` (break-even 71.43%). |
| **Anti-leakage** temporal | Split por fechas, features computadas con solo datos previos al snapshot. |
| **Calibraci�n** expl�cita | `CalibratedClassifierCV` sobre el ensemble para que `predict_proba` refleje probabilidades reales. |
| **Gates** antes de apostar | 7 gates secuenciales (modelo, data quality, historia, confianza, volatilidad, racha, regresi�n). |
| **Cobertura > precisi�n marginal** | Preferimos 85 bets a 75% que 20 bets a 85%; ambos son rentables pero el primero es operable. |

**Estado actual**: el modelo activo tiene **22 (liga, target) entrenados sobre 11 ligas activas**, con volumen de 51k muestras de entrenamiento. Las configuraciones se validaron mediante un barrido contra holdout de 7 d�as y la ganadora se entren� con todo el dataset disponible (modo producci�n).

---

## Arquitectura

### Por qu� un modelo por liga

Las din�micas son muy diferentes entre ligas: la NBA tiene 48 minutos, Euroleague 40, reglas distintas de faltas, ritmo de scoring radicalmente diferente (B2 League scorea 180+, Euroleague 150). Un modelo global sub-aprende el detalle de cada liga; un modelo por liga captura la granularidad.

El **costo** es que ligas con poca data no entrenan bien; decidimos asumir ese costo y simplemente bloquear apuestas en ligas sin muestras suficientes (ver `config.LEAGUE_MIN_SAMPLES_TRAIN`).

### Ensemble por liga

Cada (liga, target) entrena un **stack de 4 algoritmos** (si hay `>=800` muestras) o subset:

- `LogisticRegression` (regularizada, estable, baseline)
- `GradientBoostingClassifier`
- `XGBClassifier`
- `CatBoostClassifier`

Se combinan por **promedio ponderado por F1 de validaci�n**. Encima del ensemble se aplica `CalibratedClassifierCV` (isot�nica, `cv=3`) para que las probabilidades reflejen frecuencias reales.

**Paralelamente** un ensemble de regresi�n (`Ridge + GB + XGB + CatBoost`) predice el total de puntos del cuarto como **confirmaci�n**. Si el clasificador y la regresi�n discrepan por m�s de `REG_DISAGREEMENT_BLOCK_PTS = 4.0` puntos, la apuesta se bloquea.

Los hiperpar�metros fueron **reducidos en capacidad** tras detectar overfitting en la versi�n inicial de v15:

```15:35:training/v15/models.py
def _make_clf(algo: str):
    if algo == "logreg":
        return LogisticRegression(C=0.3, max_iter=2000, random_state=42)
    if algo == "gb":
        return GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )
    if algo == "xgb":
        return xgb.XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            min_child_weight=10, reg_lambda=3.0, reg_alpha=0.5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", n_jobs=1,
        )
    ...
```

### Threshold aprendido por liga

Cada liga aprende su **propio threshold de confianza** que maximiza ROI en validaci�n:

```python
for thr in np.arange(0.55, 0.92, 0.01):
    bets = [p for p in probs if p >= thr]
    roi = (hits * (odds - 1) - losses) / n_bets
    ...
# Se guarda el threshold con mejor ROI en training_summary_v15.json
```

Esto significa que algunas ligas operan con 0.75, otras con 0.82. El piso m�nimo es `MIN_CONFIDENCE_BASE = 0.75`.

### Gates (filtros de decisi�n)

En inferencia, para que un partido pase de `NO_BET` a `BET` debe superar secuencialmente:

| # | Gate | Qu� verifica |
|---|---|---|
| 1 | `model_exists` | Existe modelo entrenado para esta (liga, target). |
| 2 | `league_force_nobet` | No est� bloqueada por `league_overrides.py` (full o por target). |
| 3 | `data_quality` | `graph_points >= MIN_GP_*` y `pbp_events >= MIN_PBP_*`. |
| 4 | `league_history` | La liga tiene `>= 15` partidos hist�ricos (`LEAGUE_MIN_HISTORY_FOR_INFERENCE`). |
| 5 | `confidence` | Probabilidad calibrada `>= threshold` aprendido por liga. |
| 6 | `volatility` | `traj_lead_changes <= MAX_VOLATILITY_SWINGS (8)`. |
| 7 | `current_run` | `max(run_home, run_away) <= MAX_CURRENT_RUN_PTS (14)`. |
| 8 | `regression_agreement` | El regressor no contradice al clasificador con spread > 4 pts. |

Cada gate emite un registro `(name, passed, reason)` incluido en el payload de debug para auditor�a.

---

## Datos y preprocesamiento

### Fuente

SQLite en `matches.db` con tablas principales:

- `matches`: una fila por partido (league, date, quarter scores, final score).
- `graph_points`: serie temporal por partido con `(minute, value)`.
- `play_by_play`: eventos con `(quarter, minute, team, points, home_score, away_score)`.

### Volumen actual (90 d�as)

- **Span**: 2026-01-17 �! 2026-04-17 (90 d�as).
- **Total matches**: ~135k filas; masculinos: ~16.6k tras filtrar mujeres y NCAA terminada.
- **Temporada bajando**: las �ltimas semanas tienen 400-850 partidos/semana vs. pico de 2600 en W07. NCAA divisiones I/II/III ya terminaron.
- **Ligas activas** (con partidos en �ltimos 7 d�as): NBA, B1 League, China CBA, Argentina LN, Poland 1st, B2 League, Brazil NBB, Euroleague, Serie B, Elite 2, Super League, The Basketball League, CIBACOPA, RKL Division A.

### Filtros aplicados en modo producci�n

- `ALLOWED_GENDERS = ("men",)`   se excluyen femeninas por bajo volumen (se reactivar�n cuando haya e"500 muestras/liga).
- `active_days = 14`   solo ligas con partidos en los �ltimos 14 d�as.
- `min_samples_train = 200`   liga necesita e"200 muestras (� 2 targets = 400) para que ambos targets entren al training.

### Split temporal

Se ordena por fecha y se divide secuencialmente (nunca aleatorio):

```
| -------- train (72d) -------- | -- val (15d) -- | cal (3d) |  holdout (0d) |
```

En modo producci�n `holdout = 0`: todos los datos entran al modelo porque el "holdout real" son los partidos nuevos que llegan cada d�a. Esto se valid� previamente con un barrido comparando contra holdout de 7 d�as (ver secci�n [Evaluaci�n](#evaluaci�n-y-criterio-de-selecci�n)).

### Walk-forward league stats

Las estad�sticas hist�ricas de cada liga (totales promedio de Q3/Q4, ventaja local, etc.) se calculan **walk-forward**: para cada partido, s�lo se usan partidos anteriores a su `date`. Esto garantiza que no haya leakage incluso cuando la misma feature depende de estad�sticas agregadas de la liga.

Ver `dataset.compute_league_stats_walkforward`.

### Snapshots din�micos en training

Aunque en inferencia siempre se usa un cutoff fijo (Q3 = minuto 22, Q4 = minuto 31), en training se generan m�ltiples versiones del mismo partido con cutoffs cercanos (18, 20, 21, 22, 23 para Q3) para aumentar robustez ante peque�as diferencias de timestamp entre feeds.

```50:52:training/v15/config.py
Q3_TRAIN_SNAPSHOTS = (18, 20, 21, 22, 23)
Q4_TRAIN_SNAPSHOTS = (28, 29, 30, 31, 32)
```

---

## Feature engineering

50+ features agrupadas:

| Familia | Ejemplos | Fuente |
|---|---|---|
| **Graph** | `graph_slope`, `graph_acceleration`, `graph_range`, `graph_mean_last_5` | serie `graph_points` |
| **PBP agregado** | `pbp_points_per_event`, `pbp_3pt_rate`, `pbp_event_count` | `play_by_play` |
| **Score por cuarto** | `halftime_diff`, `halftime_total`, `q1_diff`, `q2_diff` | `matches` |
| **Trajectory** (heredada de v14) | `traj_lead_changes`, `traj_times_tied`, `traj_largest_lead_home/away`, `traj_current_run_home/away`, `traj_last5_home_pts`, `traj_last10_diff`, `traj_comeback_flag`, `traj_momentum_idx` | `play_by_play` |
| **League stats (walk-forward)** | `league_avg_q3_total`, `league_home_advantage`, `league_q4_std`, `league_samples_count` | agregado hist�rico |
| **Pace bucket** | `pace_low_flag`, `pace_medium_flag`, `pace_high_flag` | `halftime_total` vs. `pace_thresholds` |
| **Snapshot minute** | `snapshot_minute` como feature expl�cita | metadata |

Las **trajectory features** son una mejora directa tra�da de v14 (v13 no las ten�a). Capturan din�mica discreta que `graph_points` no refleja por ser demasiado "suave".

---

## Entrenamiento

### Comando producci�n

```bash
python -m training.v15.cli train \
    --train-days 72 --val-days 15 --cal-days 3 --holdout-days 0 \
    --min-samples-train 200 --active-days 14
```

O desde el **CLI interactivo**:

```bash
python -m training.v15.cli_menu
# -> [1] Entrenamiento -> [1] PRODUCCION
```

### Pipeline

1. `ds.build_samples()`   genera muestras por partido � snapshot (con cache).
2. Filtro de **ligas activas** si `active_days > 0`.
3. `ds.split_temporal()`   split por fechas.
4. `ds.calculate_pace_thresholds()`   percentiles de `halftime_total` sobre el set de train.
5. `ds.compute_league_stats_walkforward()`   stats por liga en modo rolling.
6. Para cada liga con `n_train >= 200 � 2`:
    - Para cada target `q3` / `q4`:
        - Vectorizar train/val/cal.
        - Entrenar `ClassifierEnsemble` + `RegressorEnsemble`.
        - Calibrar clasificador sobre `cal` (si `len(cal) >= 30`).
        - Aprender threshold �ptimo por ROI sobre `val`.
        - Persistir modelo en `model_outputs/{clf,reg}_{league}_{target}.joblib`.
7. Persistir `training_summary_v15.json` con metadata, splits, run_params, config snapshot y resultados por liga.
8. Generar 9 gr�ficas de diagn�stico (ver `plots.py`).

### Tiempo

- Con cache: ~90 segundos end-to-end (90 d�as, 22 modelos).
- Sin cache (reconstruyendo samples desde DB): ~3 minutos.

### Producto del training

```
training/v15/model_outputs/
%%% clf_NBA_q3.joblib             # ~11 MB por modelo
%%% reg_NBA_q3.joblib
%%% clf_Euroleague_q4.joblib
%%% ...
%%% training_summary_v15.json     # metadata completa de la corrida
%%% plots/
    %%% 01_train_val_gap.png      # leak detection
    %%% 02_temporal_distribution.png
    %%% 03_roi_by_league.png      # �! elegir portfolio
    %%% 04_threshold_curves.png
    %%% 05_calibration_curves.png # diagonal = calibraci�n perfecta
    %%% 06_probability_distribution.png
    %%% 07_feature_importance.png
    %%% 08_samples_per_league.png
    %%% 09_coverage_vs_roi.png    # elegir threshold manualmente
```

---

## Evaluaci�n y criterio de selecci�n

### M�tricas reportadas

- **Hit rate global**: porcentaje de aciertos sobre todas las apuestas emitidas.
- **ROI**: `(hits � (odds - 1) - losses) / n_bets`.
- **P&L en unidades**: ganancias acumuladas con stake = 1 unidad por apuesta.
- **Cobertura**: fracci�n de holdout donde se emiti� se�al.
- **train_val_gap**: `accuracy(train) - accuracy(val)`. Si > 0.15, sospechar overfitting.
- **Coverage vs ROI**: curva para elegir threshold manualmente si hace falta.

### Barrido de configuraciones (c�mo se eligi� la config PROD)

Antes de comprometer la config de producci�n, se corrieron 3 variantes con el **mismo holdout de 7 d�as** (los �ltimos 7 d�as de la DB, donde s� hay ligas activas):

| Config | train | val | min | active | bets | hit | ROI | P&L |
|---|---|---|---|---|---|---|---|---|
| A (baseline) | 50 | 20 | 300 |   | 57 | 78.9% | +10.5% | +6.00u |
| **B (ganadora)** | **65** | **15** | **200** | **14d** | **85** | **75.3%** | **+5.4%** | **+4.60u** |
| C (pocos bets) | 73 | 8 | 200 | 14d | 18 | 83.3% | +16.7% | +3.00u |

**Criterio de selecci�n**:

- A tiene mejor ROI/bet pero cobertura baja (solo 57 bets en 7 d�as). En 1 mes ~245 bets, demasiado selectivo.
- **B es el balance �ptimo**: ~1.5� m�s bets que A, 75% hit rate (por encima del break-even 71.4%), portfolio curado 50 bets @ 80% hit +12% ROI.
- C tiene val demasiado chico (8 d�as), s�lo 4 modelos sobreviven. Estad�sticamente fr�gil.

La configuraci�n final para producci�n es una extensi�n de B donde `holdout=0` (usando todos los datos disponibles): `train=72 val=15 cal=3 holdout=0 min=200 active=14`.

### Test continuo sobre datos nuevos

El operativo real es:

1. Resultado del d�a N llega a DB (proceso externo).
2. D�a N+1: re-entrenar.
3. Ese d�a los partidos nuevos son el "test set real".

Para medir drift entre re-entrenamientos, correr:

```bash
python -m training.v15.cli test-roi --odds 1.40 --min-bets 5
```

El reporte incluye secci�n **LEAK CHECK** que avisa si alg�n modelo est� mostrando `train_val_gap > 0.15`, se�al temprana de overfitting.

---

## Resultados del modelo activo

**Entrenado con**: `train=72 val=15 cal=3 holdout=0 min=200 active=14`.
**Modelos activos**: 22 (liga, target) / 11 ligas.

### Ligas y targets limpios (val_roi > 0, ordenados por ROI)

| liga | target | val ROI | val hit | n_train | n_val |
|---|---|---|---|---|---|
| LF Challenge Regular Season | Q4 | +26.8% | 90.6% | 335 | 60 |
| Euroleague | Q4 | +23.0% | 87.9% | 590 | 60 |
| B1 League | Q4 | +18.5% | 84.6% | 840 | 175 |
| B2 League | Q3 | +15.3% | 82.4% | 620 | 100 |
| Elite 2 | Q4 | +12.3% | 80.2% | 410 | 95 |
| Euroleague | Q3 | +9.4% | 78.1% | 590 | 65 |
| B1 League | Q3 | +8.2% | 77.3% | 840 | 175 |
| Superliga | Q4 | +7.7% | 76.9% | 285 | 70 |
| China CBA | Q4 | +7.4% | 76.7% | 735 | 105 |
| Argentina Liga Nacional | Q3 | +4.4% | 74.6% | 550 | 60 |
| NBA | Q3 | +3.2% | 73.7% | 2495 | 550 |
| LF Challenge | Q3 | +2.4% | 73.2% | 335 | 60 |

### Ligas y targets bloqueados (val_roi < 0 -> `force_nobet`)

Ver [RECOMENDACIONES.md](RECOMENDACIONES.md#overrides-activos) para detalles y justificaciones. Se bloquean por target en `league_overrides.py` con `force_nobet` (completo) o `force_nobet_q3`/`force_nobet_q4` (por cuarto).

---

## Uso

### 1. CLI interactivo (recomendado para exploraci�n)

```bash
python -m training.v15.cli_menu
```

Muestra men�s navegables para:

- Entrenar (producci�n / custom / baseline / barrido)
- Evaluar (test-roi / eval / listar ligas / resumen / plots)
- Inferencia (payload de ejemplo / desde archivo JSON)
- Config y utilidades (mostrar config, scan de DB, abrir docs)

El banner superior indica el estado del modelo activo:

```
[modelo activo] train=72d val=15d cal=3d holdout=0d min=200  |  22 targets activos / 78 skipped
```

### 2. CLI directo (scripts y automatizaci�n)

```bash
# Entrenar
python -m training.v15.cli train [--train-days N] [--val-days N] [--cal-days N] \
    [--holdout-days N] [--min-samples-train N] [--active-days N] [--no-cache]

# Evaluar
python -m training.v15.cli test-roi --odds 1.40 --min-bets 5 --top 30
python -m training.v15.cli eval --odds 1.40 [--full] [--all-snapshots]
python -m training.v15.cli plots
python -m training.v15.cli leagues
python -m training.v15.cli config

# Inferencia (JSON por stdin)
echo '{
  "match_id": "abc",
  "target": "q3",
  "league": "B1 League",
  "quarter_scores": {"q1_home":20,"q1_away":18,"q2_home":25,"q2_away":26},
  "graph_points": [...],
  "pbp_events": [...]
}' | python -m training.v15.cli infer
```

### 3. API program�tica (para tu backend)

```python
from training.v15.inference import V15Engine

engine = V15Engine.load()  # carga todos los modelos una vez

pred = engine.predict(
    match_id="abc",
    target="q3",
    league="B1 League",
    quarter_scores={"q1_home": 20, "q1_away": 18,
                    "q2_home": 25, "q2_away": 26},
    graph_points=[...],
    pbp_events=[...],
)

print(pred.signal)           # "BET" | "NO_BET"
print(pred.winner)           # "home" | "away" (solo si BET)
print(pred.probability)      # probabilidad calibrada
print(pred.debug["gates"])   # lista auditada de gates
print(pred.debug["features"]) # top features usadas
print(pred.to_json())         # serializable
```

---

## Integraci�n

### Inputs esperados (inferencia)

```json
{
  "match_id": "unique_string",
  "target": "q3" | "q4",
  "league": "nombre exacto de DB",
  "quarter_scores": {
    "q1_home": int, "q1_away": int,
    "q2_home": int, "q2_away": int,
    "q3_home": int, "q3_away": int   // solo si target=q4
  },
  "graph_points": [
    {"minute": int, "value": int}, ...
  ],
  "pbp_events": [
    {"quarter": "q1", "minute": float, "team": "home" | "away",
     "points": int, "home_score": int, "away_score": int},
    ...
  ]
}
```

### Outputs (con debug completo)

```json
{
  "match_id": "abc",
  "target": "q3",
  "league": "B1 League",
  "signal": "BET",
  "winner": "home",
  "probability": 0.812,
  "raw_probability": 0.798,
  "regression_prediction": 52.3,
  "debug": {
    "gates": [
      {"name": "model_exists", "passed": true},
      {"name": "data_quality", "passed": true},
      {"name": "confidence", "passed": true, "threshold": 0.81},
      ...
    ],
    "features": {
      "top": [["traj_current_run_home", 0.34], ["halftime_diff", 0.21], ...]
    },
    "league_stats": {"samples": 423, "avg_q3_total": 52.1},
    "data_quality": {"gp_count": 18, "pbp_count": 36},
    "league_history_roi": {"hit_rate": 0.77, "roi": 0.082, "n": 175}
  }
}
```

El campo `debug` est� pensado para alimentar un dashboard o gr�ficos de exploraci�n: auditar por qu� el modelo decidi� lo que decidi�, flexibilizar gates por liga, etc.

### Latencia

- Inicializaci�n (`V15Engine.load`): ~2-3 s (cargar joblib + stats walk-forward).
- Predicci�n individual: ~150-300 ms (incluyendo calibraci�n y gates).

Para alta concurrencia, mantener una instancia de `V15Engine` singleton.

---

## Estructura del proyecto

```
training/v15/
%%% __init__.py
%%% config.py              # umbrales globales
%%% league_overrides.py    # overrides por (liga, target)
%%% dataset.py             # build_samples, split_temporal, walk-forward stats
%%% features.py            # 50+ features
%%% models.py              # ClassifierEnsemble, RegressorEnsemble, calibraci�n
%%% gates.py               # 8 gates secuenciales
%%% train.py               # pipeline de entrenamiento
%%% inference.py           # V15Engine
%%% evaluate.py            # backtest completo
%%% test_roi.py            # resumen ROI r�pido
%%% plots.py               # 9 gr�ficas de diagn�stico
%%% cli.py                 # CLI por argumentos
%%% cli_menu.py            # CLI interactivo (men�s)
%
%%% README.md              # este archivo
%%% RECOMENDACIONES.md     # operaci�n semanal y overrides
%%% ROADMAP.md             # roadmap, deuda de v14, opini�n honesta
%
%%% model_outputs/
%   %%% clf_*.joblib
%   %%% reg_*.joblib
%   %%% training_summary_v15.json
%   %%% training_summary_v15_PROD.json
%   %%% plots/
%
%%% reports/
    %%% test_roi_v15.json
    %%% test_roi_v15.csv
    %%% test_roi_v15_A/B/C.json        # barrido
    %%% _scan_db.py                    # volumen por liga/semana
    %%% _sweep.ps1                     # barrido automatizado
    %%% _compare_sweep.py              # comparativa de barrido
    %%% _summarize_prod.py             # tabla del modelo PROD
```

---

## Dependencias

```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.6       # necesita FrozenEstimator para CalibratedClassifierCV
xgboost>=2.0
catboost>=1.2
joblib
matplotlib              # solo para plots.py
```

Instalaci�n t�pica:

```bash
pip install numpy pandas scikit-learn xgboost catboost joblib matplotlib
```

No requiere GPU. Entrena en CPU en < 3 min con el dataset actual.

---

## Pr�ximos pasos y deuda

Ver [ROADMAP.md](ROADMAP.md) para an�lisis detallado de qu� falta implementar de v14 (TimesFM, integraci�n Telegram), mejoras de modelado pendientes, y una opini�n honesta sobre el nivel de madurez actual del modelo y sus riesgos.

Para operaci�n diaria, ver [RECOMENDACIONES.md](RECOMENDACIONES.md).
