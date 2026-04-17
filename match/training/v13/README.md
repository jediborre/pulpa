# V13 — Basketball Live Prediction Model

## 📖 Descripción General

V13 es un sistema de predicción de resultados de cuartos de baloncesto (Q3 y Q4) diseñado específicamente para **operar en entorno live** con datos que llegan de forma incremental y con latencia.

### Motivación Principal

El modelo V12 fue diseñado para operar sobre **snapshots fijos** (minuto 24 para Q3, minuto 36 para Q4), pero en producción live el dato llega de forma **incremental y con retardo**, provocando:

- Discrepancias entre la señal ideal y la real
- Q4: el modelo nunca recibe datos completos → **siempre NO_BET**
- Flip de señales: NO_BET → BET cuando ya es tarde para apostar

V13 resuelve esto entrenando con **snapshots dinámicos** y segmentando por **ritmo de anotación** del partido.

---

## 🎯 Arquitectura

### Modelo Híbrido

```
┌─────────────────────────────────────────────────────────────┐
│                    V13 PREDICTION PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Datos Live → Features → Clasificador → Ganador Q3/Q4      │
│                      → Regresor → Puntos esperados          │
│                      → Gates → Señal final (BET/NO_BET)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Segmentación por Ritmo de Anotación

Cada partido se clasifica en un bucket de ritmo según los puntos totales acumulados:

**Para Q3** (puntos de Q1+Q2 de ambos equipos):
- **Low**: ≤72 pts (partidos defensivos)
- **Medium**: 72-85 pts (ritmo normal)
- **High**: ≥85 pts (partidos de alto anotaje)

**Para Q4** (puntos de Q1+Q2+Q3 de ambos equipos):
- **Low**: ≤108 pts
- **Medium**: 108-126 pts
- **High**: ≥126 pts

Estos umbrales se calculan automáticamente con los percentiles 33 y 66 del dataset en cada entrenamiento.

### Estructura de Modelos

Por cada combinación `(target, gender, pace_bucket)` se entrena un modelo independiente:

```
Clasificación (12 modelos):
  q3_low_men, q3_medium_men, q3_high_men
  q3_low_women, q3_medium_women, q3_high_women
  q4_low_men, q4_medium_men, q4_high_men
  q4_low_women, q4_medium_women, q4_high_women

Regresión (36 modelos):
  Para cada clf anterior: home, away, total
  12 × 3 = 36 modelos regresivos

Total: 48 modelos
```

**Fallback automático**: Si un bucket tiene <100 muestras, se usa el modelo `medium` del mismo género y target.

---

## 🧠 Features

### Features de Graph Points (hasta el minuto de corte)

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `gp_count` | Número de puntos de gráfica disponibles | 18-24 |
| `gp_diff` | Diferencia actual home - away | +5 |
| `gp_slope_3m` | Pendiente en últimos 3 min | +0.67 pts/min |
| `gp_slope_5m` | Pendiente en últimos 5 min | +0.40 pts/min |
| `gp_acceleration` | Cambio de pendiente (aceleración) | +0.27 |
| `gp_peak` | Valor máximo alcanzado | +12 |
| `gp_valley` | Valor mínimo alcanzado | -8 |
| `gp_amplitude` | Rango peak-valley | 20 |
| `gp_swings` | Número de cambios de dirección | 7 |
| `snapshot_minute` | Minuto real de captura | 18-32 |

### Features de Play-by-Play

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `pbp_count` | Número de eventos PBP | 45 |
| `pbp_pts_per_event` | Promedio de puntos por evento | 1.2 |
| `pbp_home_pts` | Puntos totales del home | 52 |
| `pbp_away_pts` | Puntos totales del away | 48 |
| `pbp_home_3pt` | Triples del home | 6 |
| `pbp_away_3pt` | Triples del away | 5 |

### Features de Marcador

| Feature | Descripción | Ejemplo |
|---------|-------------|---------|
| `q1_diff` | Diferencia al final de Q1 | +3 |
| `q2_diff` | Diferencia al final de Q2 | -2 |
| `q3_diff` | Diferencia al final de Q3 (para Q4) | +4 |
| `q3_total` | Puntos totales en Q3 | 45 |

### Contexto de Liga (Walk-Forward)

| Feature | Descripción | Cálculo |
|---------|-------------|---------|
| `league_home_advantage` | Ventaja histórica de local | Rolling avg de partidos anteriores |
| `league_avg_total_points` | Promedio de puntos de liga | Rolling avg de partidos anteriores |
| `league_std_total_points` | Desviación de puntos de liga | Rolling std de partidos anteriores |
| `league_samples` | Número de partidos históricos | Count de partidos anteriores |

**Importante**: Todas las stats de liga se calculan con **walk-forward validation**, usando SOLO partidos anteriores al partido actual. Esto evita **data leakage** de futuro.

---

## 🔧 Algoritmos

### Clasificación (Ganador del Cuarto)

| Algoritmo | Rol | Cuándo se usa |
|-----------|-----|---------------|
| Logistic Regression | Base lineal | Siempre (≥50 muestras) |
| Gradient Boosting | No-lineal robusto | Siempre (≥200 muestras) |
| XGBoost | Boosting avanzado | Siempre (≥200 muestras) |
| CatBoost | Categóricas nativas | Siempre (≥200 muestras) |

**Selección automática por tamaño de subset**:
- **≥500 muestras**: `[logreg, gb, xgb, catboost]`
- **200-499 muestras**: `[logreg, xgb, catboost]`
- **50-199 muestras**: `[logreg, xgb]`
- **<50 muestras**: Se omite (insuficientes)

**Ensemble**: Ponderado por **F1 score de validación**. Mayor peso al modelo con mejor F1.

### Regresión (Puntos del Cuarto)

| Algoritmo | Rol | Cuándo se usa |
|-----------|-----|---------------|
| Ridge Regression | Base lineal regularizada | Siempre (≥50 muestras) |
| Gradient Boosting | No-lineal robusto | Siempre (≥200 muestras) |
| XGBoost | Boosting avanzado | Siempre (≥200 muestras) |
| CatBoost | Categóricas nativas | Siempre (≥200 muestras) |

**Ensemble**: Ponderado por **inverso del MAE**. Menor MAE → mayor peso.

---

## 🚫 Gates de Decisión

El modelo aplica filtros secuenciales antes de emitir una señal de apuesta:

### 1. Gate de Confianza

```python
if target == 'q3' and confidence < 0.62: → NO_BET
if target == 'q4' and confidence < 0.55: → NO_BET
```

**Racional**: Q4 llega con snapshot parcial en live → umbral más bajo (0.55 vs 0.62).

### 2. Gate de Volatilidad

```python
if volatility_index > 0.70: → NO_BET
```

Partidos con cambios bruscos de ritmo son impredecibles.

### 3. Gate de Liga

```python
if league_samples < 15: → NO_BET (bloqueo total)
if 15 ≤ league_samples < 30: → max_confidence = 0.50
if 30 ≤ league_samples < 60: → max_confidence = 0.70
if league_samples ≥ 60: → sin penalización
```

**Racional**: Ligas con pocas muestras históricas son menos confiables, pero no se bloquean completamente (gate degradado vs hard-gate de V12).

### 4. Gate de Calidad de Datos

```python
if gp_count < MIN_GP: → NO_BET
if pbp_count < MIN_PBP: → NO_BET
if data_quality == 'poor': → NO_BET
```

Umbrales mínimos:
- **Q3**: min 14 graph points, min 12 PBP events
- **Q4**: min 16 graph points, min 14 PBP events

---

## ⏱️ Timing Live

### Cutoff Dinámico de Features

En vez de entrenar con "todos los datos hasta fin de cuarto", V13 entrena con **snapshots en distintos momentos**:

**Q3**: snapshot_minute ∈ {18, 20, 21, 22, 23}
- Fin real de Q2 = min 24
- Entrenamos 1-6 min antes → modelo robusto a datos "parciales"

**Q4**: snapshot_minute ∈ {28, 29, 30, 31, 32}
- Fin real de Q3 = min 36
- El bot despierta en min ~32 en producción
- Si entrenamos hasta min 32, el modelo ve datos realistas

### Feature `snapshot_minute`

El modelo recibe explícitamente el minuto de captura como feature:

```python
features['snapshot_minute'] = 30  # Ejemplo
```

Esto permite que el modelo aprenda que con min=28 hay menos graph_points disponibles y ajuste su confianza accordingly.

### Consecuencia en Producción

```
Producción real:
  min 32 → bot despierta
  scrape → datos hasta min 32
  modelo V13 entrenado con min 28-32 → reconoce este estado
  → confianza correcta (NO_BET artificial bajo como en V12)
```

---

## 📊 Dataset

### Fuentes de Datos

| Tabla | Contenido | Filas aprox. |
|-------|-----------|--------------|
| `matches` | Metadata (fecha, liga, equipos) | ~19,300 |
| `quarter_scores` | Puntos por cuarto (Q1-Q4) | ~71,400 |
| `graph_points` | Puntos de gráfica por minuto | ~536,000 |
| `play_by_play` | Eventos de jugadas | ~1.2M |

### Estadísticas del Dataset (2026-04-15)

| Métrica | Valor |
|---------|-------|
| Partidos completados | 16,400 |
| Rango de fechas | 2026-01-22 a 2026-04-15 |
| Ligas distintas | 762 |
| Género: hombres | 13,064 (80%) |
| Género: mujeres | 3,336 (20%) |

### Distribución por Pace Bucket

| Bucket | Q3 (men) | Q4 (men) | Q3 (women) | Q4 (women) |
|--------|----------|----------|------------|------------|
| Low | ~4,354 | ~4,353 | ~1,112 | ~1,111 |
| Medium | ~4,354 | ~4,353 | ~1,112 | ~1,111 |
| High | ~4,354 | ~4,353 | ~1,112 | ~1,111 |

**Conclusión**: Todos los buckets tienen ≥1,100 muestras → suficiente para entrenamiento robusto.

### Split Temporal

| Conjunto | Rango de fechas | Uso |
|----------|-----------------|-----|
| **Train** | Hasta 2026-01-31 | Entrenamiento de modelos |
| **Val** | 2026-02-01 a 2026-03-31 | Validación y selección |
| **Cal** | 2026-04-01 a 2026-04-15 | Calibración de probabilidades |

---

## 🛡️ Walk-Forward Validation

### Problema en V12 (Data Leakage)

En V12, las `league_stats` se calculaban usando **TODOS** los partidos del DB, incluyendo los del test set:

```python
# V12 (INCORRECTO):
league_stats = compute_league_stats(conn)  # Usa TODOS los partidos
```

Esto provocaba:
- Accuracy inflada: 74-89% (debería ser 50-55%)
- ROI imposible: 853-181,398%
- El modelo "conocía" el rendimiento real de cada liga de antemano

### Solución en V13

```python
# V13 (CORRECTO):
for match in sorted_matches_by_date:
    # Calcular stats SOLO con partidos ANTERIORES
    league_stats = compute_stats_from_history(league, matches_before(match.date))
```

**Resultado**: El modelo solo ve información histórica disponible en el momento real de predicción.

### Implementación

Ver `walk_forward.py`:
- `compute_league_stats_walkforward()`: Calcula stats acumulativas por liga
- Para cada partido, usa SOLO partidos con fecha anterior
- Guarda historial para partidos futuros

---

## 🔮 Inferencia Live

### Flujo de Predicción

```
1. Cargar datos del match desde DB
   ├── Metadata (fecha, liga, género)
   ├── Quarter scores (Q1-Q4)
   ├── Graph points (hasta minuto actual)
   └── Play-by-play events

2. Determinar pace bucket
   └── Según puntos totales acumulados

3. Cargar modelo específico
   └── {target}_{pace}_{gender}_clf_ensemble.joblib

4. Construir features
   ├── Graph features hasta snapshot_minute
   ├── PBP features del cuarto objetivo
   └── Score features de cuartos anteriores

5. Predecir ganador
   └── Ensemble ponderado por F1

6. Predecir puntos
   └── Ensembles de home, away, total

7. Aplicar gates
   ├── Confianza (0.62 Q3, 0.55 Q4)
   ├── Volatilidad (<0.70)
   ├── Liga (≥15 muestras)
   └── Calidad de datos (gp_count, pbp_count)

8. Emitir señal
   └── BET_HOME, BET_AWAY, o NO_BET
```

### Fallback de Modelos

Si el modelo específico no existe (pocas muestras):

```
q4_low_women no encontrado
  → Probar q4_medium_women
    → No existe → Probar q4_low_men (gender-agnostic)
      → No existe → Usar MAE fallback (promedio histórico)
```

---

## 📁 Estructura de Archivos

```
training/v13/
├── config.py                    # Umbrales y constantes centralizados
├── dataset.py                   # Construcción de dataset + pace buckets
├── features.py                  # Feature engineering (graph + PBP)
├── walk_forward.py              # Walk-forward validation para league stats
├── train_clf.py                 # Entrenamiento de clasificadores
├── train_reg.py                 # Entrenamiento de regresores
├── train_v13.py                 # Orquestador principal
├── infer_match_v13.py           # Motor de inferencia live
├── PLAN_V13.md                  # Plan de diseño original
├── TIMING_LIVE_ANALYSIS.md      # Análisis de timing Q3 vs Q4
├── ANALISIS_V12_V13.md          # Análisis comparativo V12 vs V13
├── IMPLEMENTACION.md            # Guía de implementación
├── README.md                    # Este documento
└── model_outputs/               # Modelos entrenados
    ├── training_summary.json    # Metadata del entrenamiento
    ├── q3_*_clf_ensemble.joblib # Clasificadores
    ├── q4_*_clf_ensemble.joblib
    ├── q3_*_reg_ensemble.joblib # Regresores
    ├── q4_*_reg_ensemble.joblib
    ├── *_vectorizer.joblib      # Vectorizadores
    └── *_scaler.joblib          # Escaladores
```

---

## 🚀 Uso

### Entrenamiento

```bash
# Entrenamiento completo
cd match
python training/v13/train_v13.py

# Sin hyperparameter tuning (más rápido)
python training/v13/train_v13.py --skip-tuning

# Solo un subset específico
python training/v13/train_v13.py --subset q3_high

# Solo un género
python training/v13/train_v13.py --subset q3_women
```

### Inferencia

```bash
# Predecir un match
python training/v13/infer_match_v13.py 15736636

# Output JSON
python training/v13/infer_match_v13.py 15736636 --json
```

### Output de Entrenamiento

Cada entrenamiento genera `model_outputs/training_summary.json`:

```json
{
  "version": "v13",
  "trained_at": "2026-04-15T20:33:00",
  "dataset": {
    "total_matches": 16400,
    "pace_thresholds": {
      "q3_low_upper": 72,
      "q3_high_lower": 85,
      "q4_low_upper": 108,
      "q4_high_lower": 126
    }
  },
  "models_trained": [
    {"key": "q3_low_men", "samples": 4354, "val_accuracy": 0.643, "val_f1": 0.631}
  ],
  "gates": {
    "min_confidence_q3": 0.62,
    "min_confidence_q4": 0.55
  }
}
```

---

## 📊 Métricas Esperadas

### Con Data Leakage (V12 - NO REALISTA)
| Métrica | Valor |
|---------|-------|
| Accuracy | 74-89% |
| Hit Rate | 83-89% |
| ROI | 853-181,398% |

### Sin Leakage (V13 - REALISTA)
| Métrica | Valor Esperado |
|---------|---------------|
| Accuracy | 52-58% |
| Hit Rate | 52-56% |
| ROI | 0-5% por bet |
| Coverage | 30-50% (vs 20-30% en V12) |

**Nota**: Un ROI de 2-5% por bet es nivel profesional. No buscar números mágjicos.

---

## 📈 Gráficas de Diagnóstico

Cada entrenamiento genera automáticamente gráficas en `model_outputs/plots/` para auditar el modelo.

### 1. Learning Curves (`learning_curve.png`)

Muestra el rendimiento del modelo en train vs validation a diferentes tamaños de dataset.

**Qué buscar:**
- ✅ Train y val convergen → Modelo saludable
- ⚠️ Gap > 0.10 → Posible overfitting moderado
- 🚨 Gap > 0.15 → **Posible data leakage**
- 📈 Val score sigue subiendo al 100% → Más datos ayudarían

### 2. Detección de Leakage (`leakage_detection.png`)

Compara train vs validation scores de TODOS los modelos en una gráfica.

**Criterios:**
- **PASS**: Gap promedio < 0.10 → Sin leakage sistemático
- **WARNING**: Gap 0.10-0.15 → Revisar modelos individuales
- **FAIL**: Gap > 0.15 → Leakage probable, NO usar modelo

**Automático**: Si se detecta leakage, el training_summary incluye `"assessment": "FAIL"`.

### 3. Feature Importance (`feature_importance.png`)

Top 20 features por importancia para el modelo.

**Qué buscar:**
- ✅ Features de juego (gp_*, pbp_*, score_*) dominan
- 🚨 Features de fecha (`date`, `match_id`, `event`) → **Leakage directo**
- ⚠️ Una sola feature domina todo → Modelo memoriza, no aprende

### 4. Calibration Curve (`calibration_curve.png`)

Muestra si las probabilidades predichas reflejan frecuencias reales.

**Qué buscar:**
- ✅ Línea del modelo cerca de diagonal → Bien calibrado
- ⚠️ Curva por encima → Sobre-confianza
- ⚠️ Curva por debajo → Sub-confianza
- Error de calibración < 0.05 → Excelente

### 5. Dataset Summary (`dataset_summary.png`)

4 paneles con la distribución del dataset:
- Split temporal (train/val/cal)
- Buckets de pace por género
- Distribución Q3 vs Q4
- Top 10 ligas por muestras

### 6. Walk-Forward Over Time (`walkforward_over_time.png`)

Rendimiento del modelo a lo largo del tiempo (meses).

**Qué buscar:**
- ✅ Línea estable → Modelo consistente
- 📉 Degradación en meses recientes → Concept drift
- 📈 Mejora continua → Modelo aprendiendo bien

---

## 📋 Metadata del Dataset

Cada entrenamiento genera `training_summary.json` con metadata completa:

### Información del Dataset Utilizado

```json
{
  "dataset": {
    "total_samples": 82000,
    "total_matches": 16400,
    "date_range": {
      "oldest": "2026-01-22",
      "newest": "2026-04-15"
    },
    "splits": {
      "train": {
        "samples": 57400,
        "matches": 11480,
        "date_range": {
          "oldest": "2026-01-22",
          "newest": "2026-01-31"
        }
      },
      "validation": {
        "samples": 16400,
        "matches": 3280,
        "date_range": {
          "oldest": "2026-02-01",
          "newest": "2026-03-31"
        }
      },
      "calibration": {
        "samples": 8200,
        "matches": 1640,
        "date_range": {
          "oldest": "2026-04-01",
          "newest": "2026-04-15"
        }
      }
    },
    "by_target": {"q3": 41000, "q4": 41000},
    "by_gender": {"men": 65600, "women": 16400},
    "by_pace": {"low": 27333, "medium": 27333, "high": 27334},
    "by_model_key": {
      "q3_low_men": 6833,
      "q3_medium_men": 6833,
      "...": "..."
    },
    "top_leagues": {
      "NCAA Women's Division I": 5645,
      "NCAA Men's Division I": 5320,
      "...": "..."
    }
  }
}
```

### Información por Modelo Entrenado

```json
{
  "models_trained": [
    {
      "key": "q3_high_men",
      "samples_train": 6833,
      "samples_val": 1366,
      "samples_cal": 683,
      "val_accuracy": 0.643,
      "val_f1": 0.631,
      "train_f1": 0.651,
      "train_val_gap": 0.020,
      "algorithms": ["logreg", "gb", "xgb", "catboost"],
      "weights": {"logreg": 0.15, "gb": 0.25, "xgb": 0.35, "catboost": 0.25},
      "regression_mae": {"home": 3.2, "away": 3.1, "total": 5.0}
    }
  ]
}
```

### Detección de Leakage

```json
{
  "leakage_detection": {
    "average_train_val_gap": 0.023,
    "max_gap": 0.045,
    "model_gaps": {
      "q3_high_men": {"train": 0.651, "val": 0.631, "gap": 0.020},
      "...": "..."
    },
    "assessment": "PASS",
    "note": "Gap < 0.10 is healthy, 0.10-0.15 needs review, > 0.15 indicates possible leakage"
  }
}
```

---

## ⚠️ Advertencias Importantes

### 1. NO Apostar Dinero Real Inmediatamente

V13 resuelve el data leakage de V12, pero necesita:
- ✅ Validación con datos holdout reales
- ✅ Paper trading 100+ bets
- ✅ Comparación con líneas de sportsbook

### 2. Expectativas Realistas

- Hit rate de 52-56% es **excelente** para betting deportivo
- ROI de 2-5% por bet es nivel profesional
- Se necesitan 1000+ bets para demostrar edge consistente

### 3. Varianza

- Rachas de 5-8 pérdidas consecutivas son normales
- No confundir varianza con modelo defectuoso
- Kelly criterion para sizing de stakes

---

## 🔍 Comparación V12 vs V13

| Aspecto | V12 | V13 |
|---------|-----|-----|
| **League stats** | Usa TODO (leakage) | Walk-forward (solo pasado) |
| **Snapshots** | Fijos (min 24/36) | Dinámicos (18-23, 28-32) |
| **Modelos** | 2 clf + 12 reg | 12 clf + 36 reg (por pace) |
| **Config** | Duplicada en archivos | Centralizada en config.py |
| **Confianza** | 0.65 (igual para ambos) | 0.62 Q3, 0.55 Q4 |
| **Q4 live** | Siempre NO_BET | Snapshot realista → BET posible |
| **Pace buckets** | No existe | low/medium/high por percentiles |
| **Fallback** | Hard gate | Degradado + fallback automático |
| **Algoritmos** | Fijos | Automáticos por tamaño de subset |
| **Metadata** | Poca | training_summary.json completo |

---

## 📚 Dependencias

### Requeridas
```
scikit-learn
numpy
joblib
tqdm
```

### Opcionales (mejoran performance)
```
xgboost
catboost
```

Instalar todas:
```bash
pip install scikit-learn numpy joblib tqdm xgboost catboost
```

---

## 🛠️ Troubleshooting

### "Model not found"
Ejecutar entrenamiento primero: `python training/v13/train_v13.py`

### "Insufficient samples"
El bucket tiene <50 muestras. El modelo usará fallback a "medium" automáticamente.

### Poor prediction quality
- Verificar calidad de datos (gp_count, pbp_count)
- Revisar si el match tiene pace bucket válido
- Confirmar que league_stats se calcularon correctamente

### Training too slow
Usar `--skip-tuning` para omitir búsqueda de hiperparámetros.

---

## 📝 Notas de Desarrollo

### Decisions de Diseño

1. **Segmentación por pace**: Mezclar partidos de alto y bajo anotaje confunde al modelo. Separarlos mejora la predictibilidad.

2. **Cutoff dinámico**: El modelo debe ver snapshots parciales durante entrenamiento para ser robusto en producción.

3. **Walk-forward obligatorio**: Sin esto, las league stats contaminan el modelo con datos de futuro.

4. **Config centralizado**: Elimina bugs de desalineación entre bet_monitor, infer_match, y train scripts.

5. **Fallback automático**: Algunos buckets de mujeres pueden tener <200 muestras. El fallback a "medium" evita errores.

### Futuras Mejoras

- [ ] Optuna tuning por submodelo (estructura lista)
- [ ] Calibración Platt para probabilidades reales
- [ ] Gráficas de diagnóstico (learning curves, feature importance)
- [ ] Features de rest_days (back-to-back)
- [ ] Home/away splits en league stats
- [ ] Detección de anomalías en datos live
- [ ] Integración completa con telegram_bot.py

---

## 📞 Contacto y Soporte

- **Plan original**: Ver `PLAN_V13.md`
- **Análisis comparativo**: Ver `ANALISIS_V12_V13.md`
- **Timing analysis**: Ver `TIMING_LIVE_ANALYSIS.md`
- **Guía de implementación**: Ver `IMPLEMENTACION.md`

---

**Versión**: v13.0  
**Fecha**: 2026-04-15  
**Estado**: ✅ Listo para entrenamiento y validación  
**Autor**: Implementado basado en análisis de datos reales y lecciones de V12
