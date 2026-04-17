# V16 · A/B Test TimesFM (Google) vs baseline Holt

**Fecha:** 2026-04-17
**Objetivo:** medir si la adición del algoritmo de Google (TimesFM) mejora el rendimiento
del modelo v16 sobre **datos no vistos** (holdout de 7 días).

---

## Diseño experimental

Experimento **controlado** (mismo split temporal, misma cache de samples, mismos thresholds
por liga) con dos variantes únicamente diferenciadas por el flag `TIMESFM_ENABLED`:

| Parámetro | Valor |
|---|---|
| `TRAIN_DAYS` | 72 |
| `VAL_DAYS` | 15 |
| `CAL_DAYS` | 3 |
| `HOLDOUT_DAYS` | 7 (mismas 375 muestras no vistas en ambas variantes) |
| `MIN_SAMPLES_TRAIN` | 150 |
| `MIN_SAMPLES_VAL` | 40 |
| odds simuladas | 1.40 (break-even = 71.43%) |

Fuentes:
- `reports/test_roi_v16_TFM.json`   (TimesFM ON, 1 corrida)
- `reports/test_roi_v16_NOTFM.json` (TimesFM OFF, 1 corrida)
- `model_outputs/training_summary_v16_TFM.json` (summary TFM ON)
- `model_outputs/training_summary_v16_NOTFM.json` (summary TFM OFF)
- `reports/ab_compare_report.txt` (reporte auto-generado por `ab_compare.py`)

---

## Resultados globales sobre el mismo holdout

| Métrica | TimesFM ON | TimesFM OFF | Δ (ON - OFF) |
|---|---:|---:|---:|
| n_predictions | 375 | 375 | 0 |
| apuestas (bets) | **11** | **6** | +5 |
| wins | 8 | 5 | +3 |
| hit rate | 72.73% | **83.33%** | -10.61 pp |
| ROI | +1.82% | **+16.67%** | -14.85 pp |
| edge vs break-even | +1.30 pp | +11.90 pp | -10.60 pp |
| modelos entrenados | **46** | **87** | -41 |

**Portfolio recomendado (≥10 bets/liga con ROI > 0):** 0 ligas en ambas variantes.
El holdout es demasiado corto para separar portfolio, así que se compara el bloque GLOBAL.

---

## Detalle por liga (donde alguna variante apostó)

| Liga | Target | Bets TFM/OFF | Hit TFM/OFF | ROI TFM/OFF |
|---|---|---:|---:|---:|
| Argentina Liga Nacional | q4 | 2 / 0 | 100% / - | +40.00% / - |
| B1 League | q3 | 4 / 3 | 100% / 100% | **+40.00%** / **+40.00%** |
| China CBA | q4 | 1 / 0 | 0% / - | -100.00% / - |
| Euroleague | q4 | 2 / 0 | 100% / - | +40.00% / - |
| LF Challenge | q4 | 0 / 1 | - / 0% | - / -100.00% |
| Prva Liga | q4 | 0 / 1 | - / 100% | - / +40.00% |
| Superliga | q3 | 2 / 0 | 0% / - | -100.00% / - |
| U21 Espoirs Elite | q4 | 0 / 1 | - / 100% | - / +40.00% |

En la **única** liga donde ambas variantes apuestan simultáneamente (B1 League / q3),
los resultados son **idénticos** (100% hit, +40% ROI, 3 bets comunes). TimesFM añade una
cuarta apuesta que también acierta; el baseline Holt se la pierde.

---

## Interpretación

### 1) ROI sobre datos no vistos: TimesFM NO mejora (en esta ventana)
Con 375 muestras de holdout, TimesFM OFF rinde un ROI de **+16.67%** y TimesFM ON rinde
**+1.82%**. Medido cara a cara, **el baseline sin TimesFM gana en ROI en este holdout**.

### 2) Pero la muestra es minúscula y NO concluyente
- TFM ON: 11 bets. TFM OFF: 6 bets.
- Con n=6 o n=11 la varianza del hit rate es enorme. Un solo partido desplaza el ROI
  más de 10 pp.
- Intervalo de confianza al 95% para hit rate:
  - ON (8/11)  ≈ [43%, 90%]
  - OFF (5/6)  ≈ [44%, 99%]
- Los intervalos se superponen casi por completo → la diferencia
  **no es estadísticamente significativa**.

### 3) TimesFM cambia fuertemente la distribución de apuestas, no solo la calidad
- TFM ON entrenó **46** pares (liga, target). TFM OFF entrenó **87**.
  La razón es que TimesFM excluye muestras sin histórico play-by-play suficiente, lo que
  reduce el tamaño efectivo de train y hace que muchas ligas no superen `MIN_SAMPLES_VAL`.
- En el holdout, TFM ON apuesta en Argentina Liga Nacional q4, China CBA q4, Euroleague q4
  y Superliga q3 — ligas que TFM OFF directamente bloquea por kill-switch.
  Es decir, TimesFM **habilita más exposure en ligas marginales**. Eso explica por qué
  TFM ON hace más bets (11 vs 6) pero con peor hit rate.

### 4) Coste operativo de TimesFM es alto
| | TFM ON | TFM OFF |
|---|---:|---:|
| Modelos entrenados | 46 | 87 |
| Tiempo total de entrenamiento | ~991 s (16.5 min) | ~852 s (14.2 min) |
| Dependencias extra | Python 3.11, torch, timesfm | ninguna |
| Feature cache | `timesfm_features_cache.json` (obligatoria) | no aplica |

---

## Veredicto

En el holdout evaluado **no hay evidencia de que TimesFM aporte valor**. Al contrario,
en esta ventana el modelo sin TimesFM rinde +14.85 pp más de ROI y 10.60 pp más de hit rate.

Pero:
1. La diferencia **no es estadísticamente significativa** con n=11 y n=6.
2. TimesFM **habilita más ligas para apostar** (a costa de entrenar menos modelos por
   selección severa). Esto puede ser ventaja en el largo plazo si esas ligas tienen edge,
   o desventaja si la señal tfm_* mete ruido.
3. La señal verdadera de TimesFM se vería con un holdout de 30-90 días, no 7.

### Recomendación técnica

**Corto plazo (hoy):** mantener TimesFM activo como está (modelos actuales) para no perder
cobertura de ligas marginales, pero **monitorear en producción** el ROI real por liga y
compararlo contra `reports/test_roi_v16_NOTFM.json`. Si tras 30 días de tracking el ROI
real sigue por debajo del NOTFM, apagar TimesFM.

**Medio plazo:** repetir este A/B con holdout ≥30 días para obtener resultados
estadísticamente significativos. El script `ab_compare.py` es rerunnable.

**Largo plazo:** considerar Chronos (ya hay stub en `chronos_features.py`) como
alternativa — puede ser que el problema no sea TimesFM en sí sino el tamaño pequeño
del prompt (25 eventos pbp previos) que alimenta a TimesFM.

---

## Archivos generados por este experimento

```
match/training/v16/
  ab_compare.py                                # script reusable
  reports/
    test_roi_v16.json                          # variante vigente (TimesFM ON)
    test_roi_v16_TFM.json                      # backup variante TFM
    test_roi_v16_NOTFM.json                    # variante sin TimesFM
    test_roi_v16.csv                           # CSV vigente
    test_roi_v16_TFM.csv                       # CSV TFM
    test_roi_v16_NOTFM.csv                     # CSV NOTFM
    ab_compare_report.txt                      # este reporte en texto plano
  model_outputs/                               # modelos VIGENTES (TimesFM ON)
  model_outputs_tfm_backup/                    # backup de seguridad TFM
  model_outputs_notfm_backup/                  # modelos del experimento sin TimesFM
  training_notfm.log                           # log del entrenamiento sin TimesFM
```

**Estado actual:** los modelos en `model_outputs/` son los TimesFM ON (restaurados tras
el experimento). `TIMESFM_ENABLED=True` en `config.py`.
