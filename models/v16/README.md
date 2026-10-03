# V16 - Engine de prediccion de cuarto ganador

Ultima actualizacion: **18-abr-2026**. Corrida vigente:
`reports/training_run_pruned.log` (866s, 50 pares entrenados, TimesFM
real + **feature pruning activo 52 features** post-audit). Ver A/B completo
en `reports/AB_FEATURE_PRUNING.md`.

Baseline TFM anterior (64 features, 46 pares, 991s) respaldado en
`model_outputs_tfm_baseline/`.

## Que es v16

Motor de prediccion del ganador de los cuartos 3 y 4 en basquet, entrenado
por liga. Cada (liga, target) produce un clasificador calibrado + 3 regresores
(home, away, total) que alimentan filtros secuenciales en inference live.

Diferencias clave respecto a v15:

- **Selector inteligente de ligas** (`dataset_analyzer.py`). En lugar de un
  unico umbral global, cada liga recibe un score 0-100 basado en volumen,
  regularidad, actividad reciente y ruido del pbp. Solo las ligas con
  `decision=activate` entran al catalogo (68 de 177 en la ultima corrida).
- **Live multi-snapshot**. El clasificador entrena con 5 snapshots Q3
  (minutos 17/19/21/22/23) y 6 snapshots Q4 (26/28/30/31/32/33). El bot
  puede disparar apuestas en cualquier minuto cuando cruza umbral.
- **Regresion como feature**. La salida del regresor de totales se inyecta
  como feature del clasificador (`TOTAL_REG_AS_FEATURE=True`).
- **Forecasting de la serie diff** con **TimesFM (Google Research)** como
  backend principal (~200M params). La serie `home_score - away_score` se
  proyecta 20 eventos adelante y se agregan 5 features:
  `tfm_winner_pick`, `tfm_margin`, `tfm_uncertainty`, `tfm_trend_slope`,
  `tfm_current_trend`. Ver `timesfm_features.py`.
- **Quantile regression** para intervalos de confianza del total;
  si `p90 - p10 > REG_QUANTILE_UNCERTAINTY_GATE`, la regresion no filtra.
- **Versionado semanal** de modelos (`model_outputs/weekly/...`).
- **Log de predicciones** a SQLite (`logs/predictions.db`) para el detector
  de drift y el feedback loop de pnl en Telegram.
- **Drift detection**: alerta si `hit_rate` rolling (50 apuestas) cae por
  debajo de `DRIFT_HIT_RATE_ALERT=0.70`, y si el gap train-val de una liga
  supera `DRIFT_TRAIN_VAL_GAP_ALERT=0.15`.
- **Bug fix train_val_gap**: ahora se guarda como campo explicito en el
  summary (`f1_train - f1_val`) por (liga, target). Antes estaba en `0.000`
  en todos los reportes porque `_summarize_prod.py` leia una clave inexistente.
- **Bug fix HOLDOUT_DAYS**: el default del config ahora es `HOLDOUT_DAYS=0`
  (modo PROD), coherente con la invocacion real del CLI. La variante con
  holdout se mantiene en `HOLDOUT_DAYS_SWEEP=7` para barridos comparativos.

## Estructura del paquete

```
training/v16/
  config.py                 Umbrales unicos; editar aqui, no en los modulos.
  dataset.py                Build de samples, split temporal, cache JSON.
  dataset_analyzer.py       Selector inteligente de ligas (scoring 0-100).
  features.py               Feature engineering (G1..G8). Punto de entrada:
                              build_features_for_sample(...).
  timesfm_features.py       Wrapper TimesFM (Google). Tiene batch + cache
                              en disco. Fallback Holt si el paquete falta.
  chronos_features.py       Backend alternativo (Amazon Chronos). EN HOLD.
                              Se puede activar con FORECAST_BACKEND="chronos".
  models.py                 Ensembles de XGBoost, CatBoost, HistGB, LogReg.
  train.py                  Orquestador de training. Persiste summary JSON.
  evaluate.py               Backtest sobre holdout (odds 1.40).
  test_roi.py               Test rapido ROI con diagnostico de leak.
  inference.py              V15Engine (sic, nombre heredado). Carga modelos
                              y expone predict(...) con gates secuenciales.
  gates.py                  Gates configurables por liga.
  league_overrides.py       Overrides por liga (autogenerado).
  plots.py                  9 graficas de diagnostico.
  cli.py                    Entrada unificada: train / eval / infer / ...
  cli_menu.py               Menu interactivo.
  model_outputs/            joblib + summaries + cache de TimesFM.
  reports/                  CSV, JSON, log y scripts auxiliares.
```

## Setup rapido

La dependencia `timesfm` (Google Research) solo soporta Python 3.10 y 3.11.
El repo mantiene un venv paralelo en `.venv311/` para TRAINING, separado del
Python 3.13 del sistema (que se usa para la BD/scrapers/telegram).

```bash
# Crear venv 3.11 (una vez)
py -V:3.11 -m venv training/v16/.venv311
training/v16/.venv311/Scripts/python.exe -m pip install --upgrade pip wheel setuptools
training/v16/.venv311/Scripts/python.exe -m pip install numpy pandas scikit-learn xgboost catboost joblib tqdm matplotlib
training/v16/.venv311/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
training/v16/.venv311/Scripts/python.exe -m pip install "timesfm[torch]"
```

## Como entrenar

```powershell
cd match
$env:PYTHONIOENCODING = "utf-8"
./training/v16/.venv311/Scripts/python.exe -u -m training.v16.cli train `
  --train-days 72 --val-days 15 --cal-days 3 --holdout-days 7 `
  --min-samples-train 200 --min-samples-val 60 --active-days 14
```

El primer run descarga el checkpoint de TimesFM (~500 MB a `~/.cache/huggingface/hub/`).
Runs posteriores reutilizan el cache de features en disco (`model_outputs/timesfm_features_cache.json`).

## Resultados del ultimo run (17-abr-2026, TimesFM real)

- Samples cacheados: **149 108**
- Tras selector de ligas: **74 289** (79 activas / 177 dormidas)
- Tras filtro de ligas activas (14 dias): **56 371** (68 ligas)
- Split temporal (72/15/3/7):
  - train = 44 198, val = 9 442, cal = 1 353, holdout = 1 378
- Ligas candidatas: **45** (de 66 con >=1 muestra en train)
- (liga, target) entrenados: **46**
- (liga, target) skipped: **44** (43 por `insufficient_val`, 1 por `insufficient_train`)
- Modelos con `val_roi > 0`: **20 / 46**
- Tiempo total de training: **991 s**

### Test-ROI sobre holdout (odds 1.40)

```
global    : n_bets=11  wins=8   hit=72.73%  ROI=+1.82%  PnL=+0.20u
portfolio : n_bets=0   wins=0   (sin ligas con >=5 apuestas)
leak check: 19 modelos con gap train-val > 0.15
```

Comparativa vs corrida ANTERIOR (pre-TimesFM, holdout=0, sin bugs fix):

| Metrica            | Pre-TimesFM       | Con TimesFM real  |
|--------------------|-------------------|-------------------|
| global ROI         | **-9.05%**        | **+1.82%**        |
| global hit         | 64.96%            | 72.73%            |
| global n_bets      | 137               | 11                |
| portfolio ROI      | -8.24%            | n/a (0 ligas)     |
| train_val_gap reportado | 0.000 (bug) | valores reales    |

Interpretacion: TimesFM endurece el filtro (mucha menos cobertura) pero sube
la precision por encima del break-even. El hit_rate del 72.7% supera el
minimo aceptable (72%) por 1 punto. Es un resultado fragil por volumen bajo,
hace falta acumular 50-100 apuestas reales para validarlo.

### A/B TimesFM vs baseline Holt (17-abr-2026)

Se re-entreno v16 con `TIMESFM_ENABLED=False` usando los **mismos CLI args,
mismo split temporal y mismas 375 muestras de holdout** que la variante con
TimesFM. Reporte completo en `reports/AB_TIMESFM_VS_BASELINE.md` y script
reusable en `training/v16/ab_compare.py`.

| Metrica             | TimesFM ON | TimesFM OFF | ? (ON - OFF) |
|---------------------|-----------:|------------:|-------------:|
| apuestas            |       11   |        6    |        +5    |
| wins                |        8   |        5    |        +3    |
| hit rate            |   72.73%   |  **83.33%** | -10.61 pp    |
| ROI                 |   +1.82%   |  **+16.67%**| -14.85 pp    |
| modelos entrenados  |      46    |       87    |       -41    |
| tiempo total train  |    991 s   |     852 s   |     +139 s   |

**Conclusiones:**

1. **En este holdout el baseline sin TimesFM rinde mejor en ROI puro**, pero
   con n=6 vs n=11 la diferencia **no es estadisticamente significativa**
   (IC 95% del hit_rate se solapa casi por completo: ON ? [43%, 90%],
   OFF ? [44%, 99%]).
2. TimesFM no mejora la precision; lo que hace es **habilitar apuestas en
   ligas marginales** (Argentina Liga Nacional q4, China CBA q4, Euroleague q4,
   Superliga q3) que sin TimesFM quedan bloqueadas por kill-switch. Eso
   explica el +5 en bets pero -10pp en hit rate.
3. En la unica liga donde ambas variantes coinciden (B1 League q3), los dos
   backends dan 100% hit y +40% ROI con 3 bets identicos. TimesFM agrega 1
   bet extra que tambien acierta.
4. **TimesFM reduce la cobertura de modelos**: 46 entrenados vs 87 sin
   TimesFM. El filtro de "muestras con pbp suficiente para forecast" deja
   fuera ligas con volumen bajo.

**Decision operativa:** se mantienen los modelos con TimesFM como vigentes.
El A/B no es concluyente con n=6-11 bets. Ver tambien el reporte 3-way
con Chronos abajo.

Backups del experimento:

- `model_outputs_tfm_backup/`  -> modelos TFM (= los vigentes)
- `model_outputs_notfm_backup/` -> modelos sin TimesFM del A/B
- `model_outputs_chronos/`     -> modelos Chronos Bolt del A/B/C
- `reports/test_roi_v16_TFM.json`, `_NOTFM.json`, `_CHRONOS.json`

### A/B/C 3-way: TimesFM vs Chronos Bolt vs baseline Holt (17-abr-2026)

Mismo holdout de 375 muestras no vistas. Reporte completo en
`reports/AB_TIMESFM_VS_BASELINE.md` y script `ab_compare.py`.

| Metrica            | TimesFM ON | Chronos Bolt | Holt baseline |
|--------------------|----------:|-------------:|--------------:|
| apuestas           |        11  |          8   |           6   |
| wins               |         8  |          6   |           5   |
| hit rate           |    72.73%  |     **75.00%**|      83.33%  |
| ROI                |    +1.82%  |     **+5.00%**|     +16.67%  |
| modelos entrenados |        46  |         87   |          87   |
| tiempo train       |     991 s  |      3417 s  |         852 s |

**Lectura por liga (donde apostaron):**

| Liga                    | TFM        | Chronos    | Holt       |
|-------------------------|------------|------------|------------|
| B1 League q3            | 4b 100%+40%| 4b 100%+40%| 3b 100%+40%|
| Argentina LN q4         | 2b 100%+40%| -          | -          |
| Euroleague q4           | 2b 100%+40%| -          | -          |
| Superliga q3            | 2b 0%-100% | -          | -          |
| China CBA q4            | 1b 0%-100% | -          | -          |
| U21 Espoirs q4          | -          | 1b 100%+40%| 1b 100%+40%|
| Prva Liga q4            | -          | 1b 100%+40%| 1b 100%+40%|
| Germany BBL q4          | -          | 1b 0%-100% | -          |
| LF Challenge q4         | -          | 1b 0%-100% | 1b 0%-100% |

**Conclusiones:**

1. **En ROI crudo, Holt gana. En hit rate, Chronos gana.** Pero con n<=11
   bets los IC 95% se solapan en las tres variantes: diferencia no significativa.
2. **Chronos ocupa el lugar intermedio** (ROI +5%, hit 75%) entre TimesFM
   (+1.82%) y Holt (+16.67%). Entrena los mismos 87 pares que Holt pero
   tarda 4× mas (3417 s vs 852 s, principalmente por la precomputacion).
3. **TimesFM usa mas bets en ligas marginales** (Argentina LN, Euroleague,
   Superliga, China CBA) que las otras variantes bloquean. Eso infla sus bets
   pero diluye el hit rate (2 de 4 apuestas extra son perdedoras).
4. **B1 League q3 es la unica liga consistente** en las 3 variantes:
   100% hit y +40% ROI. Es la mas fiable del portfolio actual.
5. **Chronos Bolt-tiny (8M)** se comporta mas conservador que TimesFM (200M),
   apostando menos en ligas marginales. Podria ser mejor match para el perfil
   "pocos bets, alta precision".

**Decision vigente:** mantener TimesFM como vigente. Acumular 30 dias de
produccion real y re-correr `ab_compare.py` con el holdout largo. Si Chronos
sigue en ROI superior, evaluar cambio definitivo de backend.

### Ligas apostables (no bloqueadas en `league_overrides.py`)

De las 45 ligas que llegaron a training, 26 quedaron bloqueadas (20 full, 5
solo q3, 1 solo q4). Quedan apostables:

| Liga                         | Target | val_roi  | val_hit | n_tr |
|------------------------------|--------|----------|---------|------|
| U21 Espoirs Elite            | q4     | +0.400   | 1.000   | 348  |
| Segunda FEB, Group East      | q4     | +0.295   | 0.925   | 270  |
| Meridianbet KLS              | q4     | +0.235   | 0.882   | 396  |
| B2 League                    | q3     | +0.174   | 0.839   | 590  |
| LF Challenge, Regular Season | q4     | +0.173   | 0.838   | 354  |
| B1 League                    | q3     | +0.145   | 0.818   | 780  |
| B2 League                    | q4     | +0.129   | 0.806   | 708  |
| CIBACOPA, Primera Vuelta     | q4     | +0.120   | 0.800   | 546  |

Nota: varias de estas (B1, B2, LF Challenge, CIBACOPA) tienen gap alto. Hay
que vigilarlas con `test-roi` semanalmente; si el val_hit real cae por
debajo de 0.72 o el gap vuela > 0.5, bloquearlas.

## Comandos utiles

```bash
# Resumen legible del ultimo training (usa el campo train_val_gap ahora real)
./training/v16/.venv311/Scripts/python.exe training/v16/reports/_summarize_prod.py

# Test rapido de ROI con diagnostico leak
./training/v16/.venv311/Scripts/python.exe -m training.v16.cli test-roi --odds 1.40 --min-bets 5 --top 30

# Regenerar league_overrides a partir del summary
./training/v16/.venv311/Scripts/python.exe -m training.v16.reports._auto_overrides

# Backtest completo en holdout
./training/v16/.venv311/Scripts/python.exe -m training.v16.cli eval
```

## Flags importantes en `config.py`

```python
FORECAST_BACKEND = "timesfm"        # "timesfm" | "chronos" | "holt"
TIMESFM_ENABLED = True
TIMESFM_FORECAST_HORIZON = 20
TIMESFM_UNCERTAINTY_GATE = 15.0

TRAIN_DAYS = 72                     # default = modo PROD
VAL_DAYS = 15
CAL_DAYS = 3
HOLDOUT_DAYS = 0                    # default PROD; para sweep usar 7
HOLDOUT_DAYS_SWEEP = 7

LEAGUE_MIN_SAMPLES_TRAIN = 200
LEAGUE_MIN_SAMPLES_VAL = 60
LEAGUE_ACTIVATION_MIN_SCORE = 40

DRIFT_HIT_RATE_ALERT = 0.70
DRIFT_TRAIN_VAL_GAP_ALERT = 0.15
```

Ver `RECOMENDACIONES.md` para el flujo operativo diario y `ROADMAP.md` para
la deuda tecnica pendiente.
