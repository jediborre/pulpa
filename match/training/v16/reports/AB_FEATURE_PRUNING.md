# A/B Feature Pruning (mejora #1 del ROADMAP)

Fecha de experimento: 2026-04-17.
Mejora evaluada: **#1 - Feature pruning por correlacion** (ver `ROADMAP.md`,
seccion "Plan de mejoras estructurales (18-abr-2026)").

## Objetivo

Validar la hipotesis de que reducir la dimensionalidad del feature set
(eliminando features con correlacion Pearson |r| >= 0.90 y redundantes por
construccion matematica) mejora el overfitting en ligas chicas y no degrada
el ROI de las grandes.

## Metodologia

1. **Audit de features** (`training/v16/feature_audit.py`):
   - Muestreo aleatorio de 8000 samples de `train + val` filtradas a las 27
     ligas entrenables.
   - Matriz de correlacion Pearson por target (q3 / q4).
   - Clustering union-find con umbral |r| >= 0.90.
   - Cruce con `top_features` del summary vigente para conservar las mas
     usadas por el modelo.
   - Features `tfm_*` protegidas por diseno.

2. **Blacklist curada** (`config.FEATURE_BLACKLIST`, 12 items):

   ```python
   FEATURE_BLACKLIST = (
       "meta_target_is_q4",               # constante dentro de cada target
       "meta_minutes_to_quarter_end",     # funcion deterministica de snapshot_minute
       "pace_total_prior",                # = halftime_total (q3) / = cumulative (q4)
       "pace_ratio_vs_median",            # derivada de pace_total_prior
       "score_cumulative_total",          # = halftime_total + q3_total
       "score_cumulative_diff",           # = halftime_diff + q3_diff
       "score_halftime_diff_ratio",       # reescalado de halftime_diff
       "score_q3_vs_ht_momentum",         # = q3_diff - halftime_diff
       "league_q3_total_mean",            # ~ league_ht_total_mean
       "league_q4_total_mean",            # ~ league_ht_total_mean
       "gp_amplitude",                    # ~ gp_stddev
       "gp_valley",                       # ~ traj_largest_lead_away (signo)
   )
   ```

3. **Entrenamiento** de ambas variantes con identico config (odds=1.40,
   train_days=72, val_days=15, cal_days=3, holdout_days=7,
   min_samples_train=200, min_samples_val=60):

   | Variante | Features | Modelos entrenados | Tiempo training |
   |---|---:|---:|---:|
   | BASELINE_TFM (vigente pre-17-abr) | **64** | 46 pares (liga, target) | 991 s |
   | PRUNED_TFM (post feature audit) | **52** (-18.75%) | 50 pares (liga, target) | 866 s (-12.6%) |

4. **Evaluacion de ROI sobre holdout** (odds 1.40, break-even 71.43%):
   - BASELINE: 565 muestras de holdout (HOLDOUT_DAYS=0 = remanente completo).
   - PRUNED: 375 muestras de holdout (ventana estricta de 7 dias post-cal).

## Resultados

### GLOBAL (todas las ligas, post-gates y thresholds por liga)

| Metrica | BASELINE_TFM | PRUNED_TFM | Delta |
|---|---:|---:|---:|
| Predicciones totales | 565 | 375 | - |
| Apuestas (post-gates) | 7 | 7 | 0 |
| Wins | 6 | 5 | -1 |
| **Hit rate** | **85.71%** | **71.43%** | -14.28 pp |
| **ROI** | **+20.00%** | **-0.00%** | -20.00 pp |
| P&L (unidades) | +1.40 u | -0.00 u | -1.40 u |

### PORTFOLIO (ligas con ROI > 0 y >= 3 bets)

| Metrica | BASELINE_TFM | PRUNED_TFM | Delta |
|---|---:|---:|---:|
| Ligas en portfolio | 1 | 1 | 0 |
| Apuestas | 4 | 5 | +1 |
| **Hit rate** | **100.00%** | **80.00%** | -20.00 pp |
| **ROI** | **+40.00%** | **+12.00%** | -28.00 pp |
| P&L (unidades) | +1.60 u | +0.60 u | -1.00 u |
| Liga protagonica | B1 League q3 | B1 League q3 | - |

### Detalle de la liga portfolio (B1 League q3)

| Metrica | BASELINE | PRUNED |
|---|---:|---:|
| Predicciones | 13 | 13 |
| Apuestas | 4 | 5 |
| Wins | 4 | 4 |
| Hit rate | 100% | 80% |
| ROI | +40% | +12% |
| Cobertura | 30.8% | 38.5% |
| Gap train-val | +0.297 | +0.296 |

## Interpretacion

### 1. El pruning **NO mejoro el ROI** observado

Contraintuitivo al diagnostico del audit. Posibles razones (ordenadas por
plausibilidad):

1. **Varianza muestral gigante**: el portfolio tiene 4-5 bets. Con odds 1.40
   la diferencia entre "100% hit" y "80% hit" es **1 resultado**. El
   intervalo de confianza al 90% para 4 bets con p=1.0 incluye valores
   por debajo de 0.50. Necesitamos n >= 30-40 bets para concluir algo.

2. **Diferencia de holdout** (565 vs 375 muestras): el split no es
   equivalente. El baseline tuvo mas oportunidades de bet. No es A/B puro.

3. **Over-pruning en features sutilmente utiles**: `gp_amplitude` y
   `gp_valley` tienen correlaciones bajas con target (< 0.07) pero aportan
   senal sobre el caracter del partido (cuanto oscilo, cuanta ventaja tuvo
   el perdedor). Al pruning eliminarlas, el modelo perdio matiz.

4. **Efecto del "selector inteligente de ligas"**: en la corrida pruned se
   activaron 50 modelos vs 46. Algunas ligas nuevas agregaron ruido.

### 2. El pruning **SI mejoro** aspectos mecanicos

- **Reduccion del feature set de 64 -> 52** (-18.75%): modelos mas livianos,
  inferencia mas rapida, vectorizer mas chico.
- **Tiempo de training -12.6%** (866s vs 991s): menos features = menos
  splits de GBM = menor tiempo.
- **Gap train-val** practicamente identico: no se redujo el overfitting
  (contraintuitivo al objetivo del audit).

### 3. Datos no suficientes para validar la hipotesis

El holdout (4-5 bets efectivas por portfolio) **no permite concluir nada**.
Necesitamos:

- Al menos **30 bets del portfolio** para IC 95% de +/-10 pp en ROI.
- Con 1 bet/dia por liga en portfolio, eso son **~30 dias de holdout**, no 7.
- Esto se logra con la **mejora #4 del ROADMAP**: extender `TRAIN_DAYS` de 72
  a 180 sobre la DB historica (requisito: que termine la descarga de matches).

## Estado actual

- **Modelos vigentes en produccion**: PRUNED_TFM (50 pares, 52 features).
- **Modelos baseline respaldados** en `training/v16/model_outputs_tfm_baseline/`.
- `config.FEATURE_PRUNING_ENABLED = True` (consistencia train/inference).
- `config.FEATURE_BLACKLIST` documentada con las 12 features drop.

## Proxima iteracion

Cuando el usuario finalice la descarga de historico extendido:

1. Re-ejecutar `feature_audit.py` con dataset ampliado (mas muestras =
   clusters mas estables).
2. Re-entrenar **AMBAS** variantes (baseline + pruned) con mismo
   `holdout_days=14` para A/B con >= 30 bets/portfolio.
3. Aplicar simultaneamente las mejoras #2 y #3 del ROADMAP (subir
   MIN_SAMPLES_TRAIN a 400, regularizacion adaptativa) sobre el dataset
   extendido para aislar el efecto del pruning.

## Reproducibilidad

```powershell
# 1. Audit
python -m training.v16.feature_audit --corr 0.90 --sample 8000 --quick

# 2. Training pruned (ya ejecutado)
python -m training.v16.cli train --holdout-days 7 --train-days 72 `
  --val-days 15 --cal-days 3

# 3. Test ROI sobre holdout
python -m training.v16.cli test-roi --min-bets 3 --no-csv
```

Reports generados:
- `reports/FEATURE_AUDIT.md` - combined q3+q4
- `reports/FEATURE_AUDIT_q3.md` - detalle target q3
- `reports/FEATURE_AUDIT_q4.md` - detalle target q4
- `reports/feature_audit.json` - datos estructurados
- `reports/test_roi_v16_PRUNED.json` - ROI test set pruned
- `reports/test_roi_v16_BASELINE_h7.json` - ROI test set baseline
- `reports/training_run_pruned.log` - log completo del re-train
