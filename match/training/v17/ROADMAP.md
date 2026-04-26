# Roadmap V17

Ultima actualizacion: 18-abr-2026.

Este archivo captura deuda tecnica, hipotesis aun sin probar, y el orden
sugerido de trabajo. Se revisa cada vez que se re-entrena en PROD.

## Estado actual

- **Corrida de referencia**: 17-abr-2026, `training_run_tfm.log`.
991 s, 46 (liga, target) entrenados, 20 con val_roi > 0.
- **Holdout (TimesFM ON, vigente)**: ROI +1.82%, hit 72.73%, 11 apuestas.
  Fragil por volumen bajo; hay que acumular minimo 50 apuestas reales.
- **A/B TimesFM ON vs OFF (17-abr-2026)**: sobre el mismo holdout de 375
  muestras, TimesFM OFF rinde mejor en ROI (+16.67% vs +1.82%) pero con
  menos volumen (6 bets vs 11). Diferencia **no significativa** (n<<30).
  Ver `reports/AB_TIMESFM_VS_BASELINE.md`. Decision: mantener TimesFM ON y
  repetir el A/B con holdout >= 30 dias.
- **TimesFM**: activo (Google Research, modelo PyTorch 200M parametros) en
venv 3.11 paralelo. Runtime adicional: ~16 min vs ~14 min sin TimesFM.
- **Chronos Bolt**: A/B/C ejecutado (17-abr-2026). ROI +5%, hit 75%, 8 bets.
  Mejor ROI que TimesFM; no significativo con n=8. Cache y batch integrados
  en `chronos_features.py`. Modelos archivados en `model_outputs_chronos/`.
- **Documentacion**: README + RECOMENDACIONES + ROADMAP actualizados y
alineados con la corrida actual + A/B findings.

## Bugs RESUELTOS en esta entrega (17-abr-2026)

- `train_val_gap` siempre figuraba como `0.000` en `_summarize_prod.py`.
Era un bug de lectura (la clave no existia). Ahora `train.py` lo guarda
explicitamente como `f1(train) - f1(val)` y el reporte lo lee directo.
`_auto_overrides.py` tambien prefiere el campo cuando existe.
- Discrepancia `HOLDOUT_DAYS`: el default del config era 7 pero la
invocacion real del CLI lo forzaba a 0. Se alineo a `HOLDOUT_DAYS=0` en
`config.py` (modo PROD puro) con `HOLDOUT_DAYS_SWEEP=7` para barridos.
- Features TimesFM sin batching. `timesfm_features.extract_forecast_features`
hacia una inferencia por sample (~50 ms). Se agrego
`extract_forecast_features_batch` (batch_size=32) + cache en disco por
hash de serie. Reduce el coste total de ~15 ms/sample a ~5 ms/sample.
- `_pbp_before_target` era privada en `features.py` y `train.py` la
accedia con underscore. Se renombro a `pbp_before_target` con alias
privado para compat.
- **A/B TimesFM vs baseline Holt ejecutado (17-abr-2026).** Mismo split,
mismo holdout. Resultado: ROI crudo favorece al baseline (+16.67% vs
+1.82%) pero la diferencia no es significativa (n=6 vs n=11). Script
reusable `training/v17/ab_compare.py` y reporte
`reports/AB_TIMESFM_VS_BASELINE.md` quedan para futuras corridas.

## Bugs / deudas ABIERTAS

### Alta prioridad

- **[DONE] Probar Chronos Bolt como backend de forecasting** (17-abr-2026).
  Resultados: ROI +5.00%, hit 75%, 8 bets (vs 11 de TFM y 6 de Holt).
  Mejor ROI que TimesFM; no significativo con n=8. `chronos_features.py`
  implementado con cache en disco + batch nativo. Modelos en `model_outputs_chronos/`.
  Veredicto: **mantener TimesFM como vigente por ahora**. Re-evaluar con 30 dias.
- **[NEXT] A/B/C con holdout largo (>= 30 dias).** Las tres corridas del
  17-abr-2026 usaron el mismo holdout de 7 dias (n<=11 bets). Re-ejecutar
  `ab_compare.py` cuando el holdout tenga al menos 30 dias de produccion real.
  Criterio de decision: ROI global y IC 95% no solapados.
  Comando: `./training/v17/.venv311/Scripts/python.exe training/v17/ab_compare.py`.
- **Gaps train-val sistematicamente altos.** 19 modelos con gap > 0.15 y
varios con gap > 0.50. Hipotesis a investigar, por orden de probabilidad:
  1. `val` muy chico (n_val 60-100) -> F1 val muy ruidoso -> gap
    artificialmente grande. Prueba: subir `VAL_DAYS` de 15 a 21 dias
     y ver si baja.
  2. Features de forecasting permiten memorizar la trayectoria en train.
    **Probado con Chronos Bolt**: gap sistematico se mantiene (25 modelos
    con gap>0.15 con Chronos vs 19 con TimesFM), por lo que el forecasting
    no es la causa principal. Hipotesis mas probable: n_val chico.
  3. Ensemble demasiado expresivo. Prueba: desactivar CatBoost y dejar
    solo XGBoost+LogReg en ligas con < 600 muestras.
- **Portfolio vacio en test-roi.** Con `min_bets=5` no pasa ninguna
liga. Bajar a `min_bets=3` como default del CLI, o correr test-roi sobre
14 dias en vez de 7.
- **Cobertura muy baja.** 11 apuestas en 7 dias. El TimesFM endurece el
filtro; hay que decidir si:
  - Se mantiene la seleccion y se apuesta por calidad (pocas, buenas).
  - Se relaja `MIN_CONFIDENCE_BASE` de 0.75 a 0.72 para duplicar volumen.
- **TimesFM reduce cobertura de modelos entrenados.** TFM entrena 46 pares
  vs 87 de Chronos y Holt (porque filtra muestras sin pbp suficiente para
  hacer forecast). Investigar si el filtro se puede relajar permitiendo
  `tfm_*` a cero cuando el contexto pbp sea menor al `TIMESFM_CONTEXT_MIN`.

### Plan de mejoras estructurales (18-abr-2026)

Diagnostico motivador (corrida vigente `training_summary_v17.json`):

- Rango real entrenado: **2026-01-10 -> 2026-04-17** (97 dias, 72 efectivos
  de train) sobre 46 pares (liga, target), total 24 697 muestras train.
- **30 de 46 ligas tienen < 500 muestras train** (65% del portfolio).
- **12 ligas tienen < 300 muestras train** (27%). En casi todas, gap
  train-val >= 0.4 y F1_val < 0.55: el modelo memoriza train y val es ruido.
- Redundancia en features: 7 derivados `score_*` casi equivalentes
  (`score_q1_share`, `score_q2_total`, `score_q2_diff`,
  `score_halftime_diff_ratio`, `score_cumulative_diff`, `score_q3_diff`,
  `score_q3_total`) + 5-6 `traj_*` + agregados `league_*`. En ligas con 250
  muestras, eso **es ruido**, no senal.

Mejoras ordenadas por impacto / esfuerzo:

**[DONE 18-abr-2026] 1. Feature pruning por correlacion** (alto impacto / bajo esfuerzo)
- Implementado: `training/v17/feature_audit.py`, `config.FEATURE_BLACKLIST`
  (12 features), filtro en `features.py::build_features_for_sample`.
- Resultado: **64 -> 52 features (-18.75%)**. Training -12.6%. 50 pares
  entrenados vs 46 del baseline.
- Reportes: `reports/AB_FEATURE_PRUNING.md`, `reports/FEATURE_AUDIT*.md`,
  `reports/feature_audit.json`, `reports/test_roi_v17_PRUNED.json`.
- **ROI no mejoro** con el holdout actual: baseline portfolio +40% (4 bets,
  100% hit) vs pruned +12% (5 bets, 80% hit). Portfolio es B1 League q3 en
  ambos casos. Con n=4-5 la varianza domina; no hay evidencia de
  degradacion estadisticamente significativa.
- **Modelos vigentes**: PRUNED_TFM. Baseline respaldado en
  `model_outputs_tfm_baseline/` para revertir si se observan regresiones.
- **Proxima accion**: re-evaluar una vez finalizado el historico extendido
  (#4). Un holdout con >= 30 bets por portfolio cerraria el bucle.

**[NEXT] 2. Elevar `LEAGUE_MIN_SAMPLES_TRAIN` de 200 a 400** (alto impacto /
bajo esfuerzo)
- 12 ligas caen fuera, todas con gap >= 0.4 y F1_val <= 0.55. No generaban
  ROI positivo sostenido.
- Implementacion: cambio de 1 linea en `config.py` + re-entrenar.
- KPI: n_models baja de 46 a ~34 pero el gap medio del portfolio cae.

**[NEXT] 3. Regularizacion adaptativa al tamano de liga** (alto / medio)
- Hoy todos los LightGBM/XGBoost usan la misma configuracion. Plan:
  mapping por `n_train` en `models.py`:
  - `n_train < 400`: `min_child_samples=20`, `num_leaves=15`,
    `reg_lambda=1.0`, solo `XGBoost + LogReg` (sin CatBoost).
  - `400 <= n_train < 1000`: valores intermedios.
  - `n_train >= 1000`: config actual.
- KPI: gap train-val < 0.3 en el 80% de las ligas chicas.

**[NEXT] 4. Extender la ventana historica `TRAIN_DAYS` 72 -> 180** (alto /
medio)
- `matches.db` tiene partidos desde 2024. Mas data = mas ligas superando
  el umbral de samples.
- Requisito: auditar que el walk-forward de `league_*` sigue limpio
  (no-leakage) al operar sobre una ventana mayor.
- KPI: n_train global >= 50 000, ligas entrenables >= 55.

**[LATER] 5. Ensemble TimesFM + Chronos como features paralelas** (medio /
medio)
- Hoy son mutuamente excluyentes (`FORECAST_BACKEND`). Plan:
  `FORECAST_MODE = "ensemble"` que genera 10 claves:
  `tfm_*` (TimesFM) + `chr_*` (Chronos Bolt).
- Ademas 1 feature meta: `forecast_agreement = 1 - |tfm_wp - chr_wp|`.
  Alto valor = ambos forecasters coinciden en el ganador. Candidata a
  gate adicional (apostar solo si `agreement > 0.8`).
- Solo sumar en ligas con `n_train >= 1000` (NBA, B1, B2, Euroleague,
  China CBA, Brazil NBB). En ligas chicas el ratio features/muestras ya
  esta ajustado y el ensemble empeora.
- KPI: ROI incremental > +2% en holdout largo sobre esas 6 ligas.

**[LATER] 6. Modelo jerarquico / multi-task para ligas chicas** (alto / alto)

Respuesta directa a la duda recurrente: **no es "otro modelo distinto
por liga chica"** (eso es lo que ya tenemos y es justamente el problema).
La solucion correcta es **un modelo compartido que aprende patrones
globales + ajuste por liga**:

- Arquitectura: un solo GBM / red neuronal con `league_id` como feature
  categorica (target encoding o embedding). Entrena sobre **todas las
  31k muestras juntas**, no una-por-liga.
- Ligas chicas heredan el conocimiento de las grandes (`NBA`, `Euroleague`,
  `B1 League`) y solo ajustan el offset especifico. Polish q4 hoy tiene
  306 muestras; con este esquema "ve" las 31 236 y aporta solo su sesgo.
- Implementacion posible:
  1. Baseline simple: `LightGBM` global + feature `league_mean_encode`
     (tasa historica de hits en la liga).
  2. Paso intermedio: `CatBoost` con `cat_features=['league_id']` + target
     encoding ordenado (evita leakage).
  3. Version avanzada: red neuronal con `nn.Embedding(n_leagues, 8)` +
     stacking sobre el LightGBM global para ligas grandes.
- Importante: este modelo jerarquico **reemplaza** al modelo-por-liga
  solo en ligas con n_train < 1000. Las grandes siguen con su modelo
  dedicado (que ya funciona bien).
- KPI: gap train-val medio en ligas chicas baja de 0.45 a <0.20 y al
  menos 5 ligas chicas generan ROI > 0 sostenido.

**[LATER] 7. Permutation importance post-train** (medio / bajo)
- Agregar paso al final de `train.py`: para cada liga, correr
  `sklearn.inspection.permutation_importance` en val y guardar ranking.
- Features que no mueven F1 mas de 0.005 se marcan como candidatas a
  drop en el proximo ciclo. Reporte en `reports/FEATURE_AUDIT.md`.

Orden sugerido de ejecucion: **1 -> 2 -> 3 -> 4 -> 7 -> 5 -> 6**.
Primero limpiar y consolidar el portfolio actual, luego expandir.

### Media prioridad

- **Detector de drift automatizado.** Existe flag `DRIFT_HIT_RATE_ALERT`
pero no hay cron que lo revise. Programar un job diario que:
  1. Lea las ultimas 50 apuestas del `predictions.db`.
  2. Calcule hit_rate.
  3. Si baja de 0.70, dispare alerta a Telegram y marque las ligas
    recientes como candidatas a bloqueo.
- **Promocion semanal automatizada.** El versionado semanal
(`WEEKLY_MODELS_DIR`) existe pero no hay un script que mueva los modelos
de `model_outputs/` a `model_outputs/weekly/YYYY-WW/` y que el engine
de inference apunte al `latest/`. Agregar a `cli.py` un subcomando
`promote --week YYYY-WW`.
- **Paneles en canvas.** Actualmente `_summarize_prod.py` imprime en
consola. Portar a un dashboard Canvas con las 5 metricas clave (ROI,
hit, n_bets, gap medio, modelos > 0 ROI) y drill-down por liga.
- **Tests de regresion.** No hay tests unitarios sobre `features.py`
ni sobre `gates.py`. Agregar al menos:
  - Test de anti-leakage: fijar un sample, avanzar el snapshot_minute y
  asegurar que `pbp_before_target` no filtra eventos posteriores.
  - Test de contrato TimesFM: `extract_forecast_features` devuelve las 5
  claves esperadas con los 3 backends (timesfm/chronos/holt).

### Baja prioridad

- **Migrar todo a Python 3.11.** Si se invierte en un venv 3.11 completo
para todo el proyecto, se elimina la friccion de los dos interpretes.
Requisito: que `python-telegram-bot`, `playwright` y otros scrapers sigan
funcionando en 3.11 (presumiblemente si, pero probar).
- **Probar TimesFM 2.0 si sale.** Master branch en GitHub lo usa pero
todavia no esta en PyPI para Python 3.12/3.13.
- ~~**A/B test Chronos vs TimesFM.**~~ **COMPLETADO** (17-abr-2026). Ver
  seccion "Bugs / deudas ABIERTAS > Alta prioridad > [DONE]" y README.md.
- **Revisar si `score_halftime_total`/`score_q1_total` generan leak en
q3.** En teoria estan disponibles antes del cutoff pero habria que
auditar que `features.py` no mire el q3 en sample.target='q3'.

## Hipotesis a probar (fuera del alcance del sprint actual)

- **Inclusion de odds del libro como feature.** Si tenemos el precio del
mercado al momento del snapshot, se puede agregar `book_implied_prob`.
Ojo: puede meter ruido o leak segun cuando se capturen las odds.
- **Features de estilo de juego por equipo.** Rolling de ultimos 20 partidos
(posesiones, tempo, eFG%). Hoy solo tenemos stats de liga agregadas.
- **Ensemble sobre ligas similares.** Clusterizar ligas por perfil estadistico
y entrenar un modelo "transfer" para las que tienen poca data.

## Criterios para declarar V17 "listo para PROD full"

1. Global ROI >= +3% en 30 dias de holdout reales.
2. >= 60 apuestas en 30 dias (cobertura suficiente para inferir ROI).
3. Ninguna liga apostable con gap > 0.30 sostenido.
4. Drift detector corriendo automaticamente con alerta a Telegram.
5. Documentacion (README, RECOMENDACIONES, ROADMAP) alineada con la
  ultima corrida (ya esta en verde).
