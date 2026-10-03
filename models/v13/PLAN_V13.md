# V13 — Plan de diseño y mejoras

Fecha base de análisis: 2026-04-15  
Motivación principal: el modelo V12 fue diseñado para operar sobre snapshots fijos  
(minuto 24 para Q3, minuto 36 para Q4), pero en uso **live** el dato llega de forma  
incremental y con retardo, provocando discrepancias entre la señal ideal y la real.

---

## 1. Diagnóstico raíz — el desajuste entre entrenamiento y operación live

### 1.1 El snapshot de entrenamiento no existe en producción

Durante el entrenamiento, cada muestra se construye con **todos los datos disponibles**  
al final del Q2 (para Q3) o al final del Q3 (para Q4): marcador completo, play-by-play  
completo, graph_points hasta el minuto de corte.  
En operación live **ese estado nunca existe como instante discreto**:

```
Entrenamiento asume:          Producción recibe:
────────────────────          ──────────────────
min=24 → snapshot "fijo"      min=20 → aún llega juego
                              min=22 → jugadas acumulando
                              min=24 → Q2 termina (silencio si halftime)
                              min=24+ → bot_monitor despierta y scrape
                              min=25-26 → inferencia ejecutada
```

Para Q3 el descanso de medio tiempo actúa como buffer natural: cuando el bot  
despierta hay ~2-4 min reales de pausa, el dato está completo y la señal es estable.

Para Q4 **no hay descanso**. El Q3 termina y el Q4 arranca de inmediato.  
El bot despierta en el minuto real de juego ~32 (= 36 − 4 de margen anticipado),  
scrape tarda ~10s, y cuando llega la inferencia el juego ya está en minuto ~33-34.  
El modelo recibe un snapshot **parcial** del Q3, aunque técnicamente Q3 está  
completo (tiene su marcador final), los graph_points solo llegan hasta ~min 33  
en vez de hasta ~min 36.

### 1.2 Consecuencia directa: señal inestable en Q4

Porque el modelo fue entrenado con graph_points hasta min 36 pero live recibe  
graph_points hasta min ~32-33, el vector de features se parece a un "partido a  
mitad de Q3" en la vista del modelo. Eso provoca:

- Confianza artificialmente baja en el primer check → señal NO_BET
- En el siguiente tick (~100s después) ya hay más graph_points → señal cambia a BET
- Pero el partido ya lleva 3-4 min de Q4 jugados → tarde para apostar

### 1.3 El flip NO_BET → BET (caso 15736636 y similares)

El mecanismo `NO_BET_CONFIRM_TICKS = 2` fue diseñado para evitar notificaciones  
prematuras inciertas. Tiene sentido en Q3 (hay pausa para esperar). En Q4 la  
penalidad de esperar 2 × 100s = 3.3 min es inaceptable porque el cuarto dura ~12 min  
y los primeros 3-4 min son los más apostables.

---

## 2. Problemas adicionales identificados

### 2.1 Desajuste de umbrales entre capas

| Capa              | Q3 min_gp | Q4 min_gp | Q4 min_pbp |
|-------------------|-----------|-----------|-----------|
| `bet_monitor.py`  | 16        | 26        | —         |
| `infer_match_v12` | 20 (→16)  | 28 (→22)  | 26 (→20)  |

El bet_monitor pasaba el partido a V12 con 26 graph_points, V12 rechazaba con  
"Insufficient: 26 < 28" → mensaje "puntos insuficientes" aunque visualmente el  
partido tenía Q4 avanzado.  
**Parche aplicado en V12**: umbrales bajados a 16/22/20 para alinearse.  
**V13 debe definir estos umbrales una sola vez en un lugar centralizado.**

### 2.2 Modelo demasiado restrictivo para uso live

`MIN_CONFIDENCE_TO_BET = 0.65` fue calibrado sobre datos de test estático.  
En live, la confianza es estructuralmente menor porque el snapshot es parcial.  
El resultado es que prácticamente **siempre da NO_BET en Q4 live**.  
El modelo funciona mejor post-match que en live.

### 2.3 Ligas no apostables por muestras insuficientes

El gate de liga rechaza partidos con < 30 muestras históricas. Muchas ligas  
internacionales tienen pocas muestras en la DB porque son recientes en el dataset,  
no porque sean impredecibles. Esto reduce cobertura útil en producción.

### 2.4 Datos de play-by-play inconsistentes en Q4 live

El PBP de Q3 a veces llega incompleto en el scrape del inicio de Q4 (latencia  
del proveedor). El modelo penaliza fuerte por `pbp_count < min_pbp` aunque el  
partido tenga scores de Q3 correctos. Para Q4 el PBP de Q1+Q2+Q3 puede tomar  
varios scrapes hasta estar completo.

---

## 3. Propuesta de diseño V13

### 3.1 Separar el modelo por momento de captura (cutoff dinámico)

En vez de entrenar con "todos los datos hasta min X", entrenar con datos capturados  
en distintos momentos y que el modelo sepa **en qué minuto real fue tomada la snapshot**:

```
Q3 model:
  training samples: snapshot at {min 18, 20, 22, 24}  ← aleatorizado por partido
  feature extra: "snapshot_minute" (18–24)
  objetivo: predecir ganador Q3

Q4 model:
  training samples: snapshot at {min 28, 30, 32, 34}  ← antes de que Q3 termine
  feature extra: "snapshot_minute" (28–34)
  objetivo: predecir ganador Q4
```

Este diseño hace que el modelo sea **robusto a snapshots parciales** porque  
las ha visto durante entrenamiento. El modelo aprende que con min=30 hay  
menos graph_points Q3 disponibles y ajusta su confianza de forma correcta.

### 3.2 Cutoff de features antes del fin del cuarto previo

| Target | Cutoff actual | Cutoff propuesto | Justificación                        |
|--------|---------------|------------------|--------------------------------------|
| Q3     | min 24        | min 21-23        | 1-3 min antes del fin de Q2 (buffer) |
| Q4     | min 36        | min 29-33        | Antes del fin de Q3; no depende de gp Q3 tardíos |

Para Q4 especialmente, construir features hasta min **29-31** (fin de Q3 = min 36,  
pero el bot va a mirar en min ~32 real). Si entrenamos hasta min 31, el modelo  
ya conoce ese estado y puede operar justo cuando el Q3 termina sin esperar más gp.

### 3.3 Calibración de confianza vía Platt/Isotonic post-entrenamiento

Añadir una capa de calibración de probabilidades usando datos de validación  
(no de entrenamiento) para que el `winner_confidence` refleje frecuencias reales.  
Actualmente la confianza raw del ensemble no está calibrada y puede estar  
sistemáticamente desplazada.

### 3.4 Umbral de confidence separado para Q3 y Q4

```python
# V12 (actual):
MIN_CONFIDENCE_TO_BET = 0.65  # igual para ambos

# V13 propuesto:
MIN_CONFIDENCE_Q3 = 0.62   # Q3 tiene mejor calidad de datos → se puede exigir más
MIN_CONFIDENCE_Q4 = 0.55   # Q4 llega con snapshot parcial → umbral más bajo
```

Alinear el umbral con la calidad real de los datos en cada escenario live.

### 3.5 Segmentación por ritmo de anotación (alto vs bajo) — NUEVA DIMENSIÓN

V12 ya separa modelos por **género** y aplica contexto de **liga**. V13 agrega una  
dimensión igual de importante: el **ritmo de anotación** del partido.

#### Por qué importa

Un partido de alto anotaje (total Q3 ≥ 50 pts) y uno de bajo anotaje (total Q3 ≤ 32 pts)  
tienen dinámicas completamente distintas:

| Aspecto                    | Alto anotaje (≥50 pts Q3)          | Bajo anotaje (≤32 pts Q3)          |
|----------------------------|------------------------------------|------------------------------------|
| Señal de graph_points       | Oscila entre +30 y −30             | Oscila entre +10 y −10             |
| PBP eventos / tiempo        | Muchas jugadas → más señal         | Pocas jugadas → más ruido          |
| Momentum cambia rápido      | Sí (+-10 pts en pocos minutos)     | No (diferencia estable por largos períodos) |
| Predictibilidad del Q4      | Depende del ritmo continuo         | Una racha de 3 jugadas cambia todo |
| MAE de regresión            | ~6-8 pts                           | ~3-4 pts (menos varianza absoluta) |

Si se mezclan ambos tipos en el mismo modelo, las features de graph_points tienen  
escalas completamente distintas y el modelo no aprende bien ninguno de los dos.

#### Cómo definir el bucket de ritmo anotador

```python
# Basado en el total de puntos de los 2 primeros cuartos (para Q3)
# o de los 3 primeros cuartos (para Q4)
def scoring_pace_bucket(total_pts_so_far: int, target: str) -> str:
    if target == "q3":
        # total_pts_so_far = Q1home + Q1away + Q2home + Q2away (halftime total)
        if total_pts_so_far >= 54:   return "high"    # ~27+ pts por equipo en Q1+Q2
        if total_pts_so_far >= 42:   return "medium"  # ~21-27 pts por equipo
        return "low"                                   # <21 pts por equipo
    else:  # q4
        # total_pts_so_far = suma de Q1+Q2+Q3 de ambos equipos
        if total_pts_so_far >= 78:   return "high"    # ~26+ pts por equipo
        if total_pts_so_far >= 63:   return "medium"
        return "low"
```

Estos rangos deben calibrarse con la distribución real del DB antes de entrenar.  
La idea es que cada bucket tenga al menos **200 muestras** para que el modelo sea confiable.

#### Estrategia de modelos separados por ritmo

```
Por cada (target, gender, pace_bucket) se entrena un modelo independiente:

  q3_high_men_clf.joblib     q3_medium_men_clf.joblib     q3_low_men_clf.joblib
  q3_high_women_clf.joblib   q3_medium_women_clf.joblib   q3_low_women_clf.joblib
  q4_high_men_clf.joblib     q4_medium_men_clf.joblib     q4_low_men_clf.joblib
  q4_high_women_clf.joblib   q4_medium_women_clf.joblib   q4_low_women_clf.joblib

Total: 12 modelos de clasificación (vs 2 en V12)
+ 12 × 3 modelos de regresión (home, away, total) = 36
Total: 48 modelos (manejable con joblib serialization)
```

**Fallback**: si el bucket tiene < 100 muestras en el DB, usar el modelo `medium`  
como fallback para ese bucket.

#### Features adicionales por ritmo

Al incluir el ritmo como dimensión de segmentación, también se pueden agregar  
features específicas que antes eran ruidosas al mezclar:

- `pts_per_possession_home` / `pts_per_possession_away` (requiere PBP)
- `pace_normalized_diff`: diferencia de marcador normalizada por el total de puntos  
  (`ht_diff / ht_total`) — sin normalizar, +8 con 50 pts jugados es muy diferente  
  a +8 con 26 pts jugados
- `graph_amplitude`: rango peak−valley de graph_points (directamente relacionado con ritmo)

### 3.6 Gate de liga mejorado

Reemplazar el hard-gate (< 30 muestras → no apostable) por un gate penalizado:

```python
# En vez de bloquear, reducir la confianza máxima según muestras:
def league_confidence_cap(samples):
    if samples < 15:   return 0.0   # bloqueo total
    if samples < 30:   return 0.50  # máx 50% confianza
    if samples < 60:   return 0.70  # máx 70% confianza
    return 1.0                       # sin cap
```

### 3.7 Centralizar umbrales en un único archivo de configuración

Crear `v13/config.py` que sea importado por `train_v13.py`, `infer_match_v13.py`  
y `bet_monitor.py`. Eliminar los valores duplicados en distintos archivos.

```python
# v13/config.py
Q3_GRAPH_CUTOFF    = 22    # minuto máximo de graph_points para features Q3
Q4_GRAPH_CUTOFF    = 31    # minuto máximo de graph_points para features Q4
MIN_GP_Q3          = 14    # gate mínimo graph_points Q3 (bet_monitor y modelo)
MIN_GP_Q4          = 18    # gate mínimo graph_points Q4
MIN_PBP_Q3         = 12
MIN_PBP_Q4         = 16
MIN_CONF_Q3        = 0.62
MIN_CONF_Q4        = 0.55
MAX_VOLATILITY     = 0.70
```

### 3.8 Eliminar NO_BET_CONFIRM_TICKS para Q4 (o reducir a 0)

El mecanismo de doble tick tiene sentido en Q3 (hay pausa). En Q4, si el modelo  
dice BET en el primer check, publicar de inmediato. Si dice NO_BET en el primer  
check y en el slot de Q4 aún queda tiempo (p.ej. menos del 30% del cuarto jugado),  
hacer una segunda verificación; si ya pasó el 30% del cuarto, no esperar más.

```python
# Propuesto en bet_monitor:
NO_BET_CONFIRM_TICKS_Q3 = 2   # mantiene el comportamiento actual
NO_BET_CONFIRM_TICKS_Q4 = 0   # sin espera: primer resultado es final
```

---

## 4. Impacto esperado de cada mejora

| Mejora                                  | Problema que resuelve                              | Prioridad |
|-----------------------------------------|---------------------------------------------------|-----------|
| Cutoff dinámico / features parciales    | Señal inestable Q4, flip NO_BET→BET               | ALTA      |
| MIN_CONF_Q4 = 0.55                      | Modelo nunca apuesta Q4 en live                    | ALTA      |
| NO_BET_CONFIRM_TICKS_Q4 = 0             | Notificación tardía match 15736636                 | ALTA      |
| Segmentación por ritmo anotador         | Modelo confundido por escalas distintas, falsos "puntos insuficientes" en partidos defensivos | ALTA |
| Centralizar umbrales config.py          | Desajuste capas (puntos insuficientes falsos)      | MEDIA     |
| Metadata de entrenamiento en output     | Opacidad del modelo, imposible auditar rangos      | MEDIA     |
| Calibración Platt/Isotonic              | Confianza no representativa                        | MEDIA     |
| Gate de liga penalizado                 | Baja cobertura ligas poco muestreadas              | BAJA      |
| Cutoff Q3 un poco antes (min 22)        | Dato llega antes del descanso, menor latencia      | BAJA      |

---

## 5. Estrategia de construcción de dataset para V13

### 5.1 Muestras Q3 (sin cambio grande)

```
Por cada partido en DB con Q3 finalizado:
  - Tomar graph_points, pbp, marcadores Q1+Q2
  - snapshot_minute ∈ {18, 20, 21, 22, 23}  (elegido aleatoriamente o por turno)
  - Label: ganador real de Q3
```

### 5.2 Muestras Q4 (cambio crítico)

```
Por cada partido en DB con Q4 finalizado:
  - Tomar graph_points hasta min 29-31 (NO hasta min 36)
  - pbp de Q1+Q2+Q3 COMPLETO (Q3 tiene marcador final)
  - Marcador al fin de Q3 (score_3q_*)
  - snapshot_minute ∈ {28, 29, 30, 31, 32}
  - Label: ganador real de Q4

Justificación: el bot siempre mira en ~min 32 (= 36 - WAKE_BEFORE=4).
Si entrenamos hasta min 31, el modelo ve datos realistas de cuando se llama.
```

### 5.3 Buckets de ritmo anotador — construcción y balanceo

```
Antes de construir el dataset:

1. Calcular ht_total (Q1+Q2 ambos equipos) para Q3
   o score_3q_total (Q1+Q2+Q3 ambos equipos) para Q4

2. Determinar percentiles 33 y 66 del total en el DB completo:
   → umbral_low = percentil 33     (ej: ~42 pts para Q3)
   → umbral_high = percentil 66    (ej: ~54 pts para Q3)

3. Asignar bucket: low / medium / high

4. Registrar en training_summary.json:
   pace_thresholds: {
     q3: {low_upper: 42, high_lower: 54},
     q4: {low_upper: 63, high_lower: 78}
   }
   (estos valores son del DB en el momento del entrenamiento y deben incluirse
    en el modelo para poder asignar el bucket al momento de inferencia live)
```

### 5.4 Walk-forward validation (igual que V12)

Mantener la misma metodología de validación temporal para evitar data leakage  
en las league_stats. Calcular league_stats solo con partidos anteriores al del test.

---

## 6. Output de entrenamiento — metadata obligatoria

Cada vez que se ejecute `train_v13.py`, debe generarse un archivo  
`model_outputs/training_summary.json` con la siguiente información.  
Esto permite auditar el modelo sin leer código y compartir resultados reproducibles.

```json
{
  "version": "v13",
  "trained_at": "2026-04-15T20:33:00",

  "dataset": {
    "total_matches": 4812,
    "date_range": {
      "oldest": "2024-08-01",
      "newest": "2026-04-14"
    },
    "train_range": {
      "oldest": "2024-08-01",
      "newest": "2026-01-31",
      "matches": 3850
    },
    "test_range": {
      "oldest": "2026-02-01",
      "newest": "2026-03-31",
      "matches": 680
    },
    "calibration_range": {
      "oldest": "2026-04-01",
      "newest": "2026-04-14",
      "matches": 282
    }
  },

  "pace_thresholds": {
    "q3": { "low_upper": 42, "medium_upper": 54, "unit": "pts_halftime_total" },
    "q4": { "low_upper": 63, "medium_upper": 78, "unit": "pts_3quarters_total" }
  },

  "models_trained": [
    { "key": "q3_high_men",    "samples": 941, "val_accuracy": 0.643, "val_f1": 0.631 },
    { "key": "q3_medium_men",  "samples": 1204, "val_accuracy": 0.671, "val_f1": 0.659 },
    { "key": "q3_low_men",     "samples": 728, "val_accuracy": 0.598, "val_f1": 0.581 },
    { "key": "q3_high_women",  "samples": 403, "val_accuracy": 0.662, "val_f1": 0.648 },
    "..."
  ],

  "regression_mae": {
    "q3_home_high": 3.2, "q3_home_medium": 3.6, "q3_home_low": 2.8,
    "q3_away_high": 3.1, "q3_away_medium": 3.5, "q3_away_low": 2.7,
    "q4_home_high": 3.4, "q4_home_medium": 3.8, "q4_home_low": 3.0,
    "..."
  },

  "gates": {
    "q3_graph_cutoff_min": 22,
    "q4_graph_cutoff_min": 31,
    "min_gp_q3": 14,
    "min_gp_q4": 16,
    "min_pbp_q3": 12,
    "min_pbp_q4": 14,
    "min_confidence_q3": 0.62,
    "min_confidence_q4": 0.55,
    "max_volatility": 0.70,
    "league_samples_block": 15,
    "league_samples_penalize": 30
  },

  "fallback_mae": {
    "note": "Usado cuando el modelo regresivo no está disponible o falló",
    "q3_total": 5.0, "q3_home": 3.4, "q3_away": 3.3,
    "q4_total": 5.2, "q4_home": 3.5, "q4_away": 3.4
  },

  "inference_debug_fields": {
    "note": "Campos que infer_match_v13.py debe incluir en cada objeto predicción",
    "por_prediccion": [
      "predicted_total", "predicted_home", "predicted_away",
      "mae", "mae_home", "mae_away",
      "league_quality",    // strong / moderate / weak / unknown
      "league_bettable",   // bool
      "volatility_index",  // 0.0–1.0
      "data_quality",      // excellent / good / poor
      "reasoning"          // texto libre del gate/decisión
    ],
    "uso": "Estos campos son leídos por telegram_bot._render_inference_debug para el panel de debug"
  }
}
```

Este archivo debe ser **leído por `infer_match_v13.py`** para:
- Cargar los umbrales de pace en inferencia (no hardcodearlos)
- Usar los MAE correctos en la proyección del Telegram
- Mostrar en el bot de qué fecha es el modelo activo

---

## 7. Revisión de algoritmos de V12 para V13

### Algoritmos actuales en V12

| Algoritmo      | Rol en V12        | ¿Válido para V13? | Observación |
|----------------|-------------------|-------------------|-------------|
| LogisticRegression | Base ensemble clf | ✅ Sí            | Útil como baseline, rápido, calibrable |
| GradientBoosting   | Clf + Reg         | ✅ Sí            | Robusto, buen desempeño con tabular |
| XGBoost            | Clf + Reg         | ✅ Sí            | Mejor opción para tabular con <5k muestras |
| LightGBM           | Clf + **Reg**     | ✅ Sí (condicional) | Añadido a regresión en V12 backport (2026-04-15). Útil con >300 muestras; reducir peso o eliminar en buckets pequeños |
| CatBoost           | Clf + Reg         | ✅ Sí            | Excelente con features categóricas (liga, género, pace) |
| Ridge Regression   | Reg               | ✅ Sí            | Mantener como fallback/regularizador |

### Cambios sugeridos para V13

**LightGBM en regresión** fue añadido como backport en V12 (train_v12.py + infer_match_v12.py),  
corriendo con pesos iguales a GradientBoosting. Para V13 mantener en regresión **solo si el  
subset tiene ≥ 300 muestras**; en subsets menores LGBM puede overfittear más rápido que XGBoost.  
En clasificación aplicar la misma regla: quitar LGBM si el bucket (pace×gender) tiene < 300 muestras.

**Añadir calibración obligatoria** (Platt scaling = `CalibratedClassifierCV`) sobre  
el ensemble final para todos los clasificadores. Sin calibración, `predict_proba` de  
XGBoost y LGBM no refleja frecuencias reales → la `winner_confidence` no es confiable.

**Para regresión en partidos de bajo anotaje**, considerar `QuantileRegressor`  
adicionalmente al ensemble existente. Los partidos defensivos tienen distribución  
más sesgada → predicción puntual clásica tiende a subestimar la varianza.

```python
# V13: selección de algoritmos por tamaño de subset
def select_algorithms(n_samples: int) -> list[str]:
    if n_samples >= 500:
        return ["logreg", "gb", "xgb", "lgbm", "catboost"]
    elif n_samples >= 200:
        return ["logreg", "xgb", "catboost"]
    else:
        return ["logreg", "xgb"]  # mínimo viable; advertencia en training_summary
```

---

## 8. Búsqueda de hiperparámetros por submodelo

### 8.1 Por qué hacerlo por submodelo

Cada combinación `(target, gender, pace_bucket)` tiene diferente número de muestras,  
distribución de targets y relación señal/ruido. Los hiperparámetros óptimos de  
CatBoost para `q3_high_men` (>900 muestras, muchos gp) son distintos de los de  
`q4_low_women` (<200 muestras, pocas jugadas). Usar los mismos parámetros fijos  
para todos los submodelos como hacía V12 sacrifica calidad en los subsets extremos.

### 8.2 Estrategia: Optuna por submodelo

Usar **Optuna** (o `sklearn.model_selection.RandomizedSearchCV` como alternativa  
más simple) para cada submodelo independientemente. El espacio de búsqueda debe  
ser acotado para que el tuning no tarde más que el entrenamiento base.

```python
# hparam_search_v13.py (o función dentro de train_v13.py)
import optuna

HPARAM_STUDY_TRIALS = 40   # máximo de trials por submodelo
HPARAM_TIMEOUT_SECS = 120  # máx 2 min por submodelo; si timeout → usar defaults

def tune_xgb_clf(x_train, y_train, x_val, y_val, n_trials=HPARAM_STUDY_TRIALS):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
            "max_depth":         trial.suggest_int("max_depth", 2, 6),
            "learning_rate":     trial.suggest_float("lr", 0.03, 0.20, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 3, 15),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")
        model.fit(x_train, y_train)
        return float(f1_score(y_val, model.predict(x_val)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=HPARAM_TIMEOUT_SECS)
    return study.best_params
```

Los mismos principios aplican a CatBoost (depth, iterations, l2_leaf_reg, learning_rate)  
y LightGBM (num_leaves, min_child_samples, learning_rate).

### 8.3 Parámetros críticos por algoritmo

| Algoritmo   | Parámetros a tunear                                  | Rango sugerido       |
|-------------|------------------------------------------------------|----------------------|
| XGBoost clf | `max_depth`, `n_estimators`, `min_child_weight`      | depth 2–6, est 50–300 |
| XGBoost reg | igual + `subsample`, `colsample_bytree`              | subsample 0.6–1.0    |
| CatBoost    | `depth`, `iterations`, `l2_leaf_reg`                 | depth 3–7, iter 100–400 |
| LightGBM    | `num_leaves`, `min_child_samples`, `learning_rate`   | leaves 15–63, child 10–40 |
| GradientBoosting | `max_depth`, `min_samples_leaf`, `n_estimators` | depth 2–5, leaf 5–25 |
| Ridge reg   | `alpha`                                              | 0.01–10 (log scale)  |

### 8.4 Guardar parámetros óptimos en `training_summary.json`

```json
"best_hparams": {
  "q3_high_men": {
    "xgb":  {"max_depth": 4, "n_estimators": 180, "min_child_weight": 7, ...},
    "cat":  {"depth": 5, "iterations": 250, "l2_leaf_reg": 2},
    "lgbm": {"num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.08}
  },
  "q4_low_women": {
    "xgb":  {"max_depth": 3, "n_estimators": 80, ...},
    "note": "LGBM omitido: n=142 < 300"
  }
}
```

Esto permite reproducir el entrenamiento exacto y comparar parámetros entre versiones.

### 8.5 Flag `--skip-tuning` para entrenamientos rápidos

```bash
python train_v13.py                    # entrenamiento completo con tuning
python train_v13.py --skip-tuning      # usa defaults de config.py (más rápido)
python train_v13.py --subset q4_high   # entrena solo ese submodelo
```

---

## 9. Gráficas de diagnóstico del entrenamiento

### 9.1 Gráficas obligatorias por submodelo

Para cada `(target, gender, pace_bucket)` el script `train_v13.py` debe guardar  
en `model_outputs/plots/{key}/`:

#### a) Curvas de aprendizaje (learning curves)
```
Eje X: tamaño del training set (10%, 20%, … 100%)
Eje Y: métrica (F1 o MAE)
Líneas: train_score vs val_score
```
Si la curva de validación sigue subiendo al 100% → más datos ayudarían.  
Si las curvas no convergen → underfitting (modelo demasiado simple).  
Si val_score es mucho menor que train_score → **señal de leakage o overfitting**.

#### b) Curva de validación temporal (walk-forward)
```
Eje X: fecha del test split (meses)
Eje Y: F1 del modelo en ese mes
Referencia: baseline (predicción por clase mayoritaria)
```
Si la curva cae en determinados meses → identificar qué ligas o condiciones  
cambiaron. Si el modelo no mejora sobre baseline en ningún mes → señal de  
data leakage temporal (el modelo aprendió la fecha, no el partido).

#### c) Distribución de probabilidades predichas (calibración)
```
Histogram: predicted_proba de clase 1 vs clase 0
Calibration curve (reliability diagram): P(real) vs P(predicted)
```
Antes y después de Platt calibration para validar que la calibración mejora.

#### d) Importancia de features (top 20)
```
Bar chart: feature importance (gain para XGBoost, permutation importance para ensemble)
```
Si `event_date` o `match_id` aparece con alta importancia → leakage directo.  
Si `league_bucket` domina todo → el modelo memoriza ligas, no aprende dinámica.

### 9.2 Gráfica global de comparación de submodelos

Una gráfica resumen al final del entrenamiento:

```
Grid 2×3 (targets × pace_buckets):
  Cada celda: val_F1 por género (hombres=azul, mujeres=rojo)
  Referencia: línea horizontal = baseline (clase mayoritaria)
```

Permite detectar de un vistazo qué submodelos son débiles (val_F1 < baseline)  
y cuáles tienen potencial. Los submodelos débiles deben generar una advertencia  
en `training_summary.json` y no deben activarse en producción.

### 9.3 Generación en `train_v13.py`

```python
# Al final de cada submodelo:
from training.v13.plots import save_diagnostic_plots

save_diagnostic_plots(
    key=model_key,            # ej: "q3_high_men"
    x_all=x_all,
    y_all=y_all,
    timestamps=timestamps,    # para walk-forward
    model=best_model,
    feature_names=vec.get_feature_names_out(),
    output_dir=OUT_DIR / "plots" / model_key,
)
```

Donde `plots.py` es un módulo auxiliar que encapsula toda la lógica de matplotlib.

---

## 10. Estructura y separación del código de entrenamiento

### 10.1 Problema de V12 (`train_v12.py`)

`train_v12.py` tiene ~1000 líneas en un solo archivo que mezcla:
- Construcción del dataset
- Definición de features
- Lógica de ligas
- Entrenamiento de clasificadores
- Entrenamiento de regresores
- Guardado de modelos
- Generación de métricas

Esto hace imposible reutilizar partes individualmente (por ejemplo, re-entrenar  
solo los regresores sin reconstruir el dataset, o cambiar la construcción de features  
sin tocar el loop de entrenamiento).

### 10.2 Estructura modular propuesta para V13

```
training/v13/
├── config.py               # umbrales y constantes (ya existe)
├── dataset.py              # construcción del dataset + pace buckets
├── features.py             # _build_features(), _graph_stats(), _pbp_stats()
├── hparam_search.py        # tune_*() functions usando Optuna
├── train_clf.py            # train_classifier(samples, key) → ensemble
├── train_reg.py            # train_regressor(samples, key, type) → ensemble
├── calibrate.py            # calibrate_ensemble(clf, x_val, y_val) → calibrated
├── plots.py                # save_diagnostic_plots() + summary grid
├── train_v13.py            # orquestador principal (loop de submodelos)
├── infer_match_v13.py      # inferencia live
├── eval_v13.py             # evaluación con simulación de timing
├── PLAN_V13.md
├── TIMING_LIVE_ANALYSIS.md
└── model_outputs/
    ├── training_summary.json
    ├── best_hparams.json    # parámetros óptimos de la última búsqueda
    ├── q3_*_clf_ensemble.joblib
    ├── q4_*_reg_ensemble.joblib
    └── plots/
        ├── q3_high_men/
        │   ├── learning_curve.png
        │   ├── calibration.png
        │   ├── feature_importance.png
        │   └── walkforward.png
        └── ...
```

### 10.3 Responsabilidades por módulo

| Módulo           | Responsabilidad                                                        |
|------------------|------------------------------------------------------------------------|
| `config.py`      | Constantes: cutoffs, gates, umbrales de confianza, PACE_*, ALGO_*     |
| `dataset.py`     | `load_samples()`, `scoring_pace_bucket()`, división train/val/cal     |
| `features.py`    | `build_features(data, target, cutoff)`, `graph_stats_upto()`, `pbp_stats_upto()` |
| `hparam_search.py` | `tune_classifier(x, y, algo)`, `tune_regressor(x, y, algo)`         |
| `train_clf.py`   | `train_classifier(samples, key, best_params)` → guarda ensemble       |
| `train_reg.py`   | `train_regressor(samples, key, type, best_params)` → guarda ensemble  |
| `calibrate.py`   | `calibrate_classifier(clf, x_val, y_val)` → CalibratedClassifierCV    |
| `plots.py`       | Todas las gráficas matplotlib; sin lógica de negocio                  |
| `train_v13.py`   | Orquestador: itera submodelos, llama módulos, escribe training_summary |
| `infer_match_v13.py` | Solo inferencia live; importa features.py y config.py             |
| `eval_v13.py`    | Solo evaluación; simula timing live sobre datos históricos             |

### 10.4 `train_v13.py` como orquestador limpio

```python
# train_v13.py — estructura de alto nivel
from training.v13 import config, dataset, features, hparam_search, \
                          train_clf, train_reg, calibrate, plots

def main(skip_tuning: bool = False, subset: str | None = None):
    samples = dataset.load_samples()
    summary = {"version": "v13", "trained_at": ..., "models_trained": []}

    for key in dataset.iter_model_keys(samples, subset=subset):
        sub = dataset.get_subset(samples, key)
        x_train, y_train, x_val, y_val, x_cal, y_cal = dataset.split(sub)

        if not skip_tuning:
            best_p = hparam_search.tune_all(x_train, y_train, x_val, y_val, key)
        else:
            best_p = config.DEFAULT_HPARAMS

        clf = train_clf.train_classifier(x_train, y_train, key, best_p)
        clf = calibrate.calibrate_classifier(clf, x_cal, y_cal)

        reg = {t: train_reg.train_regressor(x_train, y_train, key, t, best_p)
               for t in ["home", "away", "total"]}

        plots.save_diagnostic_plots(key, x_train, y_train, x_val, y_val, clf)
        summary["models_trained"].append({"key": key, "n": len(sub), ...})

    dataset.write_training_summary(summary)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--subset", default=None)
    args = p.parse_args()
    main(skip_tuning=args.skip_tuning, subset=args.subset)
```

---

## 11. Archivos a crear para V13

```
training/v13/
├── config.py               # umbrales (ya existe)
├── dataset.py              # construcción + pace buckets (nuevo)
├── features.py             # build_features + helpers (nuevo)
├── hparam_search.py        # búsqueda Optuna por submodelo (nuevo)
├── train_clf.py            # entrenador de clasificadores (nuevo)
├── train_reg.py            # entrenador de regresores (nuevo)
├── calibrate.py            # Platt/Isotonic calibration (nuevo)
├── plots.py                # gráficas diagnóstico (nuevo)
├── train_v13.py            # orquestador principal (nuevo)
├── infer_match_v13.py      # inferencia live (nuevo)
├── eval_v13.py             # evaluación con simulación timing (nuevo)
├── PLAN_V13.md             # este documento
├── TIMING_LIVE_ANALYSIS.md # análisis timing Q3 vs Q4 (ya existe)
└── model_outputs/
    ├── training_summary.json
    ├── best_hparams.json
    ├── q3_*_clf_ensemble.joblib
    ├── q4_*_clf_ensemble.joblib
    ├── q3_*_reg_ensemble.joblib
    ├── q4_*_reg_ensemble.joblib
    └── plots/
        └── {key}/
            ├── learning_curve.png
            ├── calibration.png
            ├── feature_importance.png
            └── walkforward.png
```

---

## 12. Cambios en bet_monitor.py para usar V13

```python
# Constantes a modificar:
Q3_MINUTE = 22               # bajar de 24 a 22 (obtener datos antes)
Q4_MINUTE = 31               # bajar de 36 a 31 (antes de que Q3 termine)
WAKE_BEFORE_MINUTES = 2      # reducir de 4 a 2 (menos tiempo muerto)
NO_BET_CONFIRM_TICKS = 0     # para Q4: enviar primer resultado
                             # (hacer configurable por cuarto)
```

> **Nota importante**: bajar Q4_MINUTE de 36 a 31 significa que el bot buscará  
> datos mientras Q3 aún puede estar en curso (si el partido tarda). Se necesita  
> verificar que `_has_scores(data, "Q1","Q2","Q3")` siga siendo la condición de  
> habilitación — el Q3 debe estar terminado aunque el minuto de clock sea 31.

---

## 13. Observaciones de sesión (2026-04-15)

- Match **15736636**: Q3, señal NO_BET → BET por flip. Partido ganado pero  
  notificación llegó tarde (ligera ventaja ya establecida). Causa documentada  
  en §1.3.
- **Puntos insuficientes Q4 falsos**: el mensaje aparecía porque V12 exigía 28 gp  
  pero el monitor solo verificaba ≥ 26. Parche aplicado en V12 (16/22). V13  
  resuelve esto definitivamente con `config.py`.
- **Q4 siempre da NO_BET en live**: confirmado por múltiples partidos. El modelo  
  opera con datos de calidad "poor" en live porque el snapshot es parcial.  
  La única solución estructural es reentrenar con snapshots parciales (§3.1, §5.2).
- **Flip Q4**: al menos un partido reportado donde el modelo dijo SÍ en Q4 y  
  luego cambió de opinión. Mismo mecanismo de flip, agravado porque NO_BET_CONFIRM  
  en Q4 lleva 200s de retraso con el cuarto ya avanzado.

### Mejoras aplicadas en sesión del 2026-04-15 (backport V12 + prep V13)

- **LightGBM en regresión** (backport V12): `train_v12.py` y `infer_match_v12.py` actualizados.  
  Pesos ensemble 5 modelos: ridge=10%, gb=20%, xgb=25%, lgbm=20%, cat=25%.  
  V13 debe respetar gate de muestras (≥300) para incluir LGBM.
- **Debug inference ampliado** (`telegram_bot._render_inference_debug`): el panel ahora muestra  
  por cada check: `data_quality`, `volatility_index`, `liga: quality + ❌ si no apostable`,  
  `pred: total (home/away) ± MAE`. El nombre de la liga y su calidad se muestran en el header.
- **`inference_debug_log.inference_json` enriquecido** (`bet_monitor._run_inference_sync`):  
  los campos `league_quality`, `league_bettable`, `volatility_index`, `data_quality` se  
  persisten ahora en el JSON para todos los checks futuros.
- **Spec formal de campos de debug** añadida en §6 (`inference_debug_fields`) para que  
  `infer_match_v13.py` exponga los mismos campos desde el día 1.

---

## 14. Integración de V13 en el sistema live (bot + monitor)

### 14.1 Cómo funciona el selector de versión actual

El selector de modelo del bot de Telegram opera mediante dos diccionarios en  
`telegram_bot.py`:

```python
# telegram_bot.py – variables globales
MODEL_CONFIG:         {"q3": "v4", "q4": "v4"}  # predicciones manuales en el bot
MONITOR_MODEL_CONFIG: {"q3": "v4", "q4": "v4"}  # modelo usado por el daemon
AVAILABLE_MODELS:     ["v2", "v4", "v6", "v9", "v12"]
```

El usuario cambia el modelo desde el panel del monitor → botón "⚙️ Modelos" →  
se ejecuta `monmodel:set:q4:v12` → llama a `bet_monitor.set_model_config({"q4": "v12"})`  
y persiste en la BD con `db_mod.set_setting(conn, "monitor_model_q4", "v12")`.  

El valor se restaura al arrancar el bot leyendo la BD:

```python
# telegram_bot.py startup
mv = db_mod.get_setting(conn, "monitor_model_q3")
if mv and mv in AVAILABLE_MODELS:
    MONITOR_MODEL_CONFIG["q3"] = mv
```

Cuando el monitor ejecuta una inferencia, `_run_inference_sync` lee  
`_model_config.get(target, "v4")` y despacha al módulo correcto.  
Para V12 ya existe la rama `if version == "v12":`; V13 necesita la misma.

### 14.2 Cambios necesarios para activar V13

#### Paso 1 — Añadir "v13" a `AVAILABLE_MODELS`

```python
# telegram_bot.py línea ~102
AVAILABLE_MODELS: list[str] = ["v2", "v4", "v6", "v9", "v12", "v13"]
```

#### Paso 2 — Añadir rama V13 en `bet_monitor._run_inference_sync`

```python
# bet_monitor.py – función _run_inference_sync
if version == "v13":
    try:
        v13_mod = importlib.import_module("training.v13.infer_match_v13")
        pred = v13_mod.run_inference(
            match_id=match_id,
            target=target,
            fetch_missing=False,
        )
        if isinstance(pred, dict) and not pred.get("ok", True):
            return {"ok": False, "reason": pred.get("reason", "V13 failed")}

        def _attr(obj, name, default=None):
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        return {
            "ok": True,
            "predictions": {
                target: {
                    "available": True,
                    "predicted_winner": _attr(pred, "winner_pick"),
                    "confidence":       _attr(pred, "winner_confidence"),
                    "bet_signal":       _attr(pred, "winner_signal"),
                    "final_recommendation": _attr(pred, "final_signal"),
                    "predicted_total":  _attr(pred, "predicted_total"),
                    "predicted_home":   _attr(pred, "predicted_home"),
                    "predicted_away":   _attr(pred, "predicted_away"),
                    "reasoning":        _attr(pred, "reasoning"),
                    "mae":              _attr(pred, "mae"),
                    "mae_home":         _attr(pred, "mae_home"),
                    "mae_away":         _attr(pred, "mae_away"),
                    "league_quality":   _attr(pred, "league_quality"),
                    "league_bettable":  _attr(pred, "league_bettable"),
                    "volatility_index": _attr(pred, "volatility_index"),
                    "data_quality":     _attr(pred, "data_quality"),
                }
            },
        }
    except Exception as exc:
        return {"ok": False, "reason": f"V13 error: {exc}"}
```

> Nota: incluye todos los campos de debug (§6.inference_debug_fields) para  
> que el panel de Telegram los muestre automáticamente sin cambios adicionales.

#### Paso 3 — Actualizar defaults al arrancar en V13

Cuando V13 sea estable, cambiar los defaults en `telegram_bot.py`:

```python
MODEL_CONFIG:         {"q3": "v13", "q4": "v13"}
MONITOR_MODEL_CONFIG: {"q3": "v13", "q4": "v13"}
```

Y el fallback en `bet_monitor.py`:

```python
_model_config: dict[str, str] = {"q3": "v13", "q4": "v13"}
```

### 14.3 Flujo de activación sin reiniciar el bot

```
Usuario en Telegram:
  /monitor → ⚙️ Modelos
  → Q4: [v2] [v4] [v9] [v12] [v13]   ← elegir v13
  → callback: monmodel:set:q4:v13
  → set_model_config({"q4": "v13"})   ← efecto inmediato
  → se guarda en DB con set_setting()  ← persiste en siguiente arranque
```

No se requiere reiniciar el bot ni el monitor para cambiar de versión.  
El cambio aplica al próximo partido que el monitor procese (el que ya está  
en seguimiento no cambia su configuración — finaliza con la versión asignada  
al momento de ser agendado).

### 14.4 Criterio de activación (cuándo pasar a V13 en producción)

Antes de poner V13 como default en `AVAILABLE_MODELS`, verificar:

| Criterio                                    | Cómo comprobarlo                        |
|---------------------------------------------|-----------------------------------------|
| `model_outputs/training_summary.json` existe | `ls training/v13/model_outputs/`        |
| MAE Q3 total < 5.5                           | `training_summary.json.regression_mae`  |
| Val accuracy Q3 > 63%                        | `training_summary.json.models_trained`  |
| Val accuracy Q4 > 58%                        | ídem                                    |
| Sin errores en `eval_v13.py` con datos reales | `python eval_v13.py --live-sim`        |
| Prueba manual con 5 partidos recientes       | `python infer_match_v13.py <match_id>`  |
