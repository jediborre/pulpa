# Recomendaciones operativas V16

Ultima actualizacion: 17-abr-2026 (corrida TimesFM real + A/B/C TimesFM vs
Chronos Bolt vs baseline Holt ejecutado).

Este documento es la guia viva para operar, monitorear y ajustar el motor v16.
Cada seccion incluye numeros actuales y los umbrales que disparan accion.

## 1. Numeros base que debes conocer de memoria

- Break-even a odds 1.40: **71.43%** de hit rate.
- Hit rate minimo aceptable en validacion: **72.0%** (`MIN_ACCEPTABLE_HIT_RATE`).
- Hit rate objetivo en produccion: **75.0%** (`TARGET_HIT_RATE`).
- Gap train-val tolerable: **0.15**. Entre 0.15 y 0.25 = vigilar. Entre 0.25
y 0.30 = considerar bloqueo. Mayor a 0.30 = bloqueo automatico por
`_auto_overrides`.
- Drift alerta si hit_rate rolling (50 apuestas) < 0.70.
- Ultimos numeros globales (17-abr-2026, holdout 7d):
  - **TimesFM ON (vigente):**    `ROI=+1.82%   hit=72.73%  n_bets=11`.
  - **Chronos Bolt (A/B):**      `ROI=+5.00%   hit=75.00%  n_bets=8`.
  - **Holt baseline (A/B):**     `ROI=+16.67%  hit=83.33%  n_bets=6`.
  - Diferencia no significativa con n<12. Ver `reports/ab_compare_report.txt`
    y seccion 5.
  - Portfolio vacio en las 3 variantes (holdout de 7 dias demasiado corto).

## 2. Flujo operativo semanal

1. **Lunes por la manana (antes del arranque de la semana competitiva)**:
  1. Actualizar la base de datos con los partidos de la semana anterior
    (scrapers de v15 / v16, lo que este activo).
  2. Desde `match/`:
    ```powershell
     cd match
     $env:PYTHONIOENCODING = "utf-8"
     ./training/v16/.venv311/Scripts/python.exe -u -m training.v16.cli train `
       --train-days 72 --val-days 15 --cal-days 3 --holdout-days 7 `
       --min-samples-train 200 --min-samples-val 60 --active-days 14
    ```
     Notas:
    - El venv 3.11 es obligatorio para usar TimesFM; el Python 3.13 del
    sistema no lo soporta.
    - Duracion esperada: ~16 min (primer run descarga checkpoint TimesFM,
    ~500 MB). Runs siguientes reutilizan cache en disco
    (`model_outputs/timesfm_features_cache.json`).
  3. Revisar `reports/training_run_tfm.log`. Buscar:
    - Linea `[v16/timesfm] backend=timesfm cache_loaded=N`. Si dice
     `holt_fallback` es que TimesFM no cargo (ver seccion 6).
    - `trained N (league,target) pairs`. Esperar ~45-50.
2. **Evaluar el estado del run**:
  ```powershell
   ./training/v16/.venv311/Scripts/python.exe training/v16/reports/_summarize_prod.py
  ```
   Revisar la tabla de val_roi descendente. Banderas:
  - `<- LEAK`: gap > 0.20. Investigar si es una liga nueva con poco val.
  - `<- overfit`: gap > 0.15.
  - Ligas con `val_hit = 1.000` y `n_vl < 100` son sospechosas (pocas val,
  el threshold optimo se encontro con solo un punado de aciertos).
3. **Test de ROI en holdout**:
  ```powershell
   ./training/v16/.venv311/Scripts/python.exe -m training.v16.cli test-roi `
     --odds 1.40 --min-bets 5 --top 30
  ```
   Aceptar el modelo para produccion si:
  - `global ROI >= 0.02` (+2%) con `n_bets >= 10` en holdout, o
  - `portfolio ROI >= 0.05` con `n_bets >= 5` en al menos 3 ligas.
   Si no, NO poner en produccion y regresar a bloquear ligas malas.
4. **Regenerar overrides**:
  ```powershell
   ./training/v16/.venv311/Scripts/python.exe -m training.v16.reports._auto_overrides
  ```
   Esto reescribe `league_overrides.py` con los bloqueos nuevos (criterios:
   val_roi<0, val_hit<0.72, gap>0.30, o gap>0.25 AND val_hit<0.75).
   **Nunca editar manualmente sin anotar en este archivo el motivo.** Si
   una liga aparece como bloqueada y crees que no lo deberia, agrega un
   override que la destrabe PERO documenta aqui el porque.
5. **Verificar ligas apostables**:
  - Cross-check con la tabla en `README.md` -> `Ligas apostables`.
  - Si la lista encoge por debajo de 5 ligas efectivas, se debe decidir
  entre: bajar `MIN_CONFIDENCE_BASE` (de 0.75 a 0.72), o relajar
  `LEAGUE_MIN_SAMPLES_VAL` (de 60 a 40) y re-entrenar.

## 3. Como interpretar el dashboard de plots

Carpeta: `model_outputs/plots/`. Generados en cada training.

- `01_train_val_gap.png`: barras por liga ordenadas. Cualquier barra > 0.20
es un candidato a bloqueo. Si > 40% de las barras pasan 0.20, el
problema es sistemico, no de una liga. Revisar feature leakage en
`features.py` (usualmente en `_trajectory_features` y
`_pbp_summary_features` cuando el cutoff no filtra bien).
- `02_temporal_distribution.png`: muestras por dia. Huecos = ligas en pausa
o falla de scraper.
- `03_roi_by_league.png`: ROI en val ordenado. Zona verde = apostables;
zona roja = bloqueadas.
- `04_threshold_curves.png`: ROI vs threshold por liga. Permite ver si el
pico esta lejos de `MIN_CONFIDENCE_BASE=0.75`. Si una liga pide 0.88
como optimo, tiene poco volumen (threshold alto = pocas apuestas).
- `05_calibration_curves.png`: reliability diagrams. Si la curva se despega
de la diagonal, la calibracion isotonica no esta funcionando (suele ser
por poca data en `cal`, <30 muestras -> se desactiva).
- `06_probability_distribution.png`: histograma de probas. Si la moda
esta en 0.5, el clasificador no aprendio (clase dominante o features
ruidosas).
- `08_samples_per_league.png`: ligas con volumen.
- `09_coverage_vs_roi.png`: scatterplot. El ideal es el cuadrante
sup-der: alta cobertura y ROI positivo. Las ligas en sup-izq son
rentables pero poco frecuentes; las de inf-der son trampas de volumen.

## 4. Que hacer ante cada tipo de sintoma


| Sintoma                                       | Revisar                                     | Accion                                                                                                                                                             |
| --------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `global ROI < 0` en test-roi                  | `03_roi_by_league.png`                      | Bloquear manualmente ligas con ROI < -0.10, re-correr test-roi. Si sigue negativo, rebajar threshold o aumentar TRAIN_DAYS.                                        |
| `hit_rate < 0.70` rolling en produccion       | `logs/predictions.db` (drift detector)      | Ejecutar `cli train` con los ultimos 7 dias como holdout y comparar. Probable drift de mercado (odds o equipos nuevos).                                            |
| Gap > 0.50 en liga A                          | `train_metrics` vs `val_metrics` en summary | Inspeccionar features. Suele ser que `n_val < 80` y el F1 val es ruidoso. Alternativamente bajar capacidad del modelo (cambiar `select_clf_algos` a medium/small). |
| Todas las apuestas se concentran en 1 liga    | `portfolio.leagues` en `test_roi_v16.json`  | Diversificar: relajar thresholds en ligas con val_roi > 0 pero n_bets bajo.                                                                                        |
| `TimesFM` no cargo (log dice `holt_fallback`) | `training_run_tfm.log`                      | Reinstalar en venv 3.11 (ver README). Si `torch` falla, puede ser DLL de VC++; instalar `Microsoft Visual C++ Redistributable`.                                    |


## 4.5 Feature pruning (mejora #1 del ROADMAP, 18-abr-2026)

A/B ejecutado entre el baseline TFM (64 features) y una variante con
feature pruning por correlacion (52 features, -18.75%). Ver
`reports/AB_FEATURE_PRUNING.md` para detalle.

| Metrica | BASELINE_TFM | PRUNED_TFM | Delta |
|---|---:|---:|---:|
| Features totales | 64 | **52** | -12 |
| Modelos entrenados | 46 | **50** | +4 |
| Tiempo training | 991 s | **866 s** | -12.6% |
| Holdout samples | 565 | 375 | (split distinto) |
| Portfolio ROI (B1L q3) | +40.00% | +12.00% | -28 pp |
| Portfolio hit rate | 100% (4/4) | 80% (4/5) | -20 pp |

**Conclusion operativa**: con n<=5 bets en el portfolio la varianza domina;
**no hay evidencia estadisticamente significativa** de que el pruning
mejore ni empeore el ROI. Los beneficios mecanicos (training mas rapido,
modelos mas livianos, menos vectorizers) si se materializaron.

**Estado vigente**: modelos PRUNED_TFM en produccion con
`config.FEATURE_PRUNING_ENABLED=True`. Baseline respaldado en
`training/v16/model_outputs_tfm_baseline/` por si se detectan regresiones.
Reversion 1-comando:

```powershell
Move-Item training/v16/model_outputs training/v16/model_outputs_pruned
Move-Item training/v16/model_outputs_tfm_baseline training/v16/model_outputs
# editar config.py: FEATURE_PRUNING_ENABLED = False
```

Re-evaluar cuando finalice la descarga de historico extendido (mejora #4
del ROADMAP): el experimento requiere >= 30 bets/portfolio para IC95%.

## 5. A/B de backends de forecasting

### 5.1 Resumen A/B/C: TimesFM vs Chronos Bolt vs baseline Holt (17-abr-2026)

Experimento controlado: mismo split, mismo holdout (375 samples no vistas),
mismos thresholds. Reporte completo: `reports/AB_TIMESFM_VS_BASELINE.md`.

| Metrica            | TimesFM ON | Chronos Bolt | Holt baseline |
|--------------------|----------:|-------------:|--------------:|
| apuestas (bets)    |        11  |          8   |           6   |
| wins               |         8  |          6   |           5   |
| hit rate           |    72.73%  |     **75.00%**|      83.33%  |
| ROI                |    +1.82%  |     **+5.00%**|     +16.67%  |
| modelos entrenados |        46  |         87   |          87   |
| tiempo train       |     991 s  |      3417 s  |         852 s |

**Veredicto:** en ROI crudo, Holt gana en este holdout; en hit rate,
Chronos es el mejor de los DL. Sin embargo, con n<=11 bets los IC 95% se
solapan entre las 3 variantes; **ninguna diferencia es estadisticamente
significativa**. Chronos Bolt ocupa un punto intermedio prometedor (mejor
ROI que TimesFM, mas conservador que Holt, misma cobertura de modelos).

**Decision vigente:** mantener TimesFM activo. Acumular 30 dias de produccion
y re-correr `ab_compare.py` con el holdout largo para decidir. Si Chronos
mantiene ROI > TimesFM en ese holdout, evaluar cambio de backend.

### 5.2 Como re-correr el A/B TimesFM on/off

```powershell
cd match

# Variante ON (la actual)
# -> ya esta guardada en reports/test_roi_v16_TFM.json y model_outputs_tfm_backup/

# Variante OFF
# 1. Editar config.py: TIMESFM_ENABLED = False
# 2. Apartar modelos actuales:
Move-Item training/v16/model_outputs/*.joblib training/v16/model_outputs_tfm_backup/
# 3. Re-entrenar:
./training/v16/.venv311/Scripts/python.exe -u -m training.v16.cli train `
  --train-days 72 --val-days 15 --cal-days 3 --holdout-days 7 `
  --min-samples-train 150 --min-samples-val 40
# 4. Evaluar:
./training/v16/.venv311/Scripts/python.exe -u -m training.v16.cli test-roi
# 5. Guardar como NOTFM y generar comparativo:
Copy-Item training/v16/reports/test_roi_v16.json training/v16/reports/test_roi_v16_NOTFM.json
./training/v16/.venv311/Scripts/python.exe training/v16/ab_compare.py > training/v16/reports/ab_compare_report.txt
# 6. Restaurar modelos TFM como vigentes y reactivar TIMESFM_ENABLED=True.
```

### 5.3 Como activar Chronos (backend alternativo, YA PROBADO)

Chronos (Amazon) ya fue probado en A/B/C (17-abr-2026). Resultados: ROI
+5.00%, hit 75.00%, 8 bets — mejor ROI que TimesFM en este holdout pero
no significativo (n<12). Modelo por defecto: `chronos-bolt-tiny` (8M params,
~4× mas rapido que TimesFM en batch, 4× mas lento en precompute por tener
mas series distintas). Integrado en `chronos_features.py`.

```bash
# Instalar en el venv 3.11
./training/v16/.venv311/Scripts/python.exe -m pip install chronos-forecasting
```

Editar `config.py`:

```python
FORECAST_BACKEND = "chronos"        # antes: "timesfm"
TIMESFM_ENABLED = False             # apaga el pipeline TimesFM
CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"   # o "-base" si hay RAM
```

Ejecutar `cli train` normal. **Importante**: las features `tfm_*` que genera
Chronos no son numericamente iguales a las de TimesFM, aunque lleven el
mismo nombre (mismo contrato). Por eso es obligatorio RE-ENTRENAR los
clasificadores; no basta con cambiar la inferencia.

Flujo A/B Chronos (replicar exactamente el esquema de 5.2):

1. Editar `config.py`: `FORECAST_BACKEND="chronos"`.
2. Entrenar con mismo CLI -> los resultados se guardan en `model_outputs/`.
3. `test-roi` -> guardar como `test_roi_v16_CHRONOS.json`.
4. Correr `ab_compare.py` (detecta automaticamente los 3 JSONs si existen).
5. Decidir segun ROI en holdout acumulado >= 30 dias.

Nota: el precompute de Chronos (66k series) tarda ~38 min en la primera
corrida. Runs posteriores usan `model_outputs/chronos_features_cache.json`.

## 6. Problemas conocidos

1. **TimesFM + Python 3.13 no son compatibles.** El paquete `timesfm` pide
  `>=3.10, <3.12`. Por eso existe el venv paralelo `.venv311/`. No instalar
   ese requirement en el venv principal 3.13.
2. **Gaps altos en ligas con `n_val` chico.** Es en parte falso-positivo:
  F1 con ~60 muestras tiene intervalo de confianza amplio. El auto-override
   lo considera igual porque la senal agregada (val_roi < 0 + val_hit < 0.72)
   tambien se cumple. Si quieres recuperar alguna liga grande como Liga ACB o
   Germany BBL, baja `--min-samples-val` a 40-50 y re-entrena. Eso permitira
   que entren al catalogo, aunque con val mas ruidoso.
3. **Portfolio vacio en test-roi.** Pasa cuando el holdout de 7 dias tiene
  pocos partidos por liga. No es un bug, es baja cobertura. Valorar el
   global y esperar 1-2 semanas mas de holdout.
4. **Cache TimesFM puede crecer.** El archivo
  `model_outputs/timesfm_features_cache.json` agrega entradas con cada
   training. Si supera 500 MB, borrarlo: se repoblara en el siguiente run.
5. **config_snapshot vs run_params en summary.** El `config_snapshot` lee los
  valores del modulo `config` al momento del training; los `run_params`
   guardan lo que efectivamente se uso (incluyendo overrides de CLI). Si
   difieren es esperable, no un bug.

## 7. Checklist antes de poner modelos en produccion

- `_summarize_prod.py` muestra al menos 15 modelos con `val_roi > 0`.
- `test-roi` muestra `global ROI >= 0` y `hit_rate >= 0.72`.
- `_auto_overrides` escribio <= 30 bloqueos (si es mas, algo esta mal).
- `training_run_tfm.log` confirma `backend=timesfm`.
- No hay excepciones sin capturar en el log.
- Copiar `training_summary_v16.json` a
`training_summary_v16_PROD_YYYYMMDD.json` como snapshot de referencia.
- Ejecutar `cli plots` y revisar `01_train_val_gap.png`: ninguna
barra > 0.60 (eso seria una liga con leak grave no detectada).

