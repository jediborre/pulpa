# V12 → V13: Análisis Exhaustivo y Conclusiones

## 📊 Estado Actual de la Base de Datos

**Datos disponibles (2026-04-15):**
- Total de partidos: **19,290**
- Partidos completados: **14,699** (76.2%)
- Ligas distintas: **762**
- Rango de fechas: 2026-01-22 a 2026-04-15
- Distribución por género:
  - Hombres: **11,823** (80.4%)
  - Mujeres: **2,876** (19.6%)

**Top 5 ligas:**
1. NCAA Women: 1,129 partidos
2. NCAA Men: 1,064 partidos
3. NCAA D2 Men: 529 partidos
4. NBA: 446 partidos
5. NCAA (general): 385 partidos

---

## 🔍 Análisis de V12

### Arquitectura
V12 es un **ensemble ultra-conservador** que combina:
- **Clasificación** (ganador Q3/Q4): LogReg + GB + XGBoost + LightGBM + CatBoost
- **Regresión** (puntos): Ridge + GB + XGBoost + LightGBM + CatBoost
- **70+ features**: graph_points, PBP, pressure, momentum, Monte Carlo, league context
- **Gender-separated**: modelos diferentes para hombres/mujeres

### Problemas Críticos Identificados

#### 1. **DATA LEAKAGE** (CRÍTICO)
- Las `league_stats` se calculan con TODOS los partidos (incluyendo test set)
- Esto infla métricas de forma artificial:
  - Accuracy reportada: **74-89%** → Realidad esperada: **50-55%**
  - ROI simulado: **853-181,398%** → Realidad esperada: **-10% a +10%**

**Conclusión honesta**: V12 NO está listo para producción con dinero real.

#### 2. **Desajuste Training vs Live (TIMING)**
- V12 entrena con snapshots "fijos" al final de cada cuarto
- En live, los datos llegan con latencia y de forma incremental
- **Q4 es el peor afectado**: no hay pausa entre Q3→Q4
  - Modelo entrena con gp hasta min 36
  - Live recibe gp hasta min ~32-33
  - Resultado: modelo nunca confía → **siempre NO_BET en Q4 live**

#### 3. **Umbrales duplicados y desalineados**
- `bet_monitor.py` tiene sus propios umbrales
- `infer_match_v12.py` tiene otros diferentes
- Resultado: "puntos insuficientes" falsos positivos

#### 4. **Modelo demasiado restrictivo**
- `MIN_CONFIDENCE_TO_BET = 0.65` (igual para Q3 y Q4)
- En live, la confianza es estructuralmente menor
- **Coverage**: solo 20-30% de partidos son apostables

### Lo Que V12 Hace BIEN

✅ Arquitectura de ensemble multi-algoritmo  
✅ Feature engineering (70+ features bien diseñadas)  
✅ Risk management framework (gates conservadores)  
✅ Gender separation (reconocer diferencias hombres/mujeres)  
✅ Monte Carlo simulation (agrega señal real)  
✅ League context (buena idea, mala implementación)  

### Lo Que V12 Hace MAL

❌ Data leakage en league stats  
❌ Overfitting (números demasiado buenos)  
❌ Evaluación con test set contaminado  
❌ Sin walk-forward validation  
❌ Timing misalignment training vs live  

---

## 🎯 Propuesta V13: Mejoras Clave

### Mejora 1: **Cutoff Dinámico de Features** ⭐ ALTA

**Problema**: V12 entrena con datos completos de Q3 (min 24) y Q4 (min 36), pero en live recibe datos parciales.

**Solución V13**:
```
Q3: entrenar con snapshots en min {18, 20, 21, 22, 23}
Q4: entrenar con snapshots en min {28, 29, 30, 31, 32}
+ feature "snapshot_minute" para que modelo sepa el contexto
```

**Evaluación**: ✅ **VIABLE Y NECESARIO**
- El DB tiene 14,699 partidos completados → suficientes para ~4-5 snapshots por partido
- Esto multiplica el dataset efectivamente pero no crea data leakage (son vistas del mismo partido)
- **Crítico para Q4**: resolvería el problema de "siempre NO_BET"

---

### Mejora 2: **Segmentación por Ritmo de Anotación** ⭐ ALTA

**Problema**: Mezclar partidos de alto anotaje (Q3 total ≥50) con bajo anotaje (Q3 total ≤32) confunde al modelo.

**Solución V13**:
```python
# Buckets por percentiles
Q3: low ≤42 pts, medium 42-54 pts, high ≥54 pts
Q4: low ≤63 pts, medium 63-78 pts, high ≥78 pts

# 12 modelos clf + 36 modelos reg = 48 total
q3_high_men, q3_medium_men, q3_low_men, ...
```

**Evaluación**: ⚠️ **VIABLE CON RESERVAS**

**Análisis de viabilidad con datos actuales:**

Si asumimos que de los 14,699 partidos completados:
- ~60% tienen datos de Q3 completos (~8,800)
- ~50% tienen datos de Q4 completos (~7,300)
- Split gender: 80% men, 20% women

Distribución estimada por bucket:
```
Q3 Men (80% de 8,800 = 7,040):
  high: ~2,350 (33%)
  medium: ~2,350 (33%)
  low: ~2,340 (33%)

Q3 Women (20% de 8,800 = 1,760):
  high: ~587
  medium: ~587
  low: ~586

Q4 Men (80% de 7,300 = 5,840):
  high: ~1,947
  medium: ~1,947
  low: ~1,946

Q4 Women (20% de 7,300 = 1,460):
  high: ~487
  medium: ~487
  low: ~486
```

**Veredicto**:
- ✅ Buckets "high" y "medium" de hombres: >1,900 muestras → **OK**
- ⚠️ Buckets "low" de mujeres: ~486-586 muestras → **MARGINAL** (necesita ≥300 para LGBM)
- ❌ Algunos buckets de women Q4 podrían tener <200 muestras → **PROBLEMA**

**Recomendación**: 
- Usar segmentación por ritmo SÍ, pero con **fallback automático a "medium"** si el bucket tiene <200 muestras
- Considerar solo 2 buckets (alto/bajo) en vez de 3 para subsets pequeños
- **Validar con consulta real al DB** para confirmar distribución exacta de puntos

---

### Mejora 3: **Umbrales de Confianza Separados** ⭐ ALTA

**Problema**: `MIN_CONFIDENCE = 0.65` es igual para Q3 y Q4, pero Q4 tiene datos peores en live.

**Solución V13**:
```python
MIN_CONFIDENCE_Q3 = 0.62  # Q3 mejor dato → exigir más
MIN_CONFIDENCE_Q4 = 0.55  # Q4 snapshot parcial → umbral menor
```

**Evaluación**: ✅ **SENSATO Y NECESARIO**
- Resuelve problema de "Q4 nunca apuesta"
- Pero necesita **calibración post-entrenamiento** para que 0.55 signifique algo real
- Sin calibración, es solo un número arbitrario

---

### Mejora 4: **Centralizar Configuración** ⭐ MEDIA

**Problema**: Umbrales duplicados en bet_monitor.py, infer_match_v12.py, etc.

**Solución V13**: `config.py` como fuente única de verdad

**Evaluación**: ✅ **SIMPLE Y MUY NECESARIO**
- Ya existe `v13/config.py` bien diseñado
- Elimina bugs de desalineación de capas
- **Implementación inmediata sin riesgo**

---

### Mejora 5: **NO_BET_CONFIRM_TICKS_Q4 = 0** ⭐ ALTA

**Problema**: En Q4, esperar 2 ticks (200s) hace que la señal llegue cuando ya pasó 30-40% del cuarto.

**Solución V13**: Sin espera en Q4, mantener 1-2 ticks en Q3

**Evaluación**: ✅ **CRÍTICO PARA Q4**
- Q4 dura ~12 min; 200s = 28% del cuarto
- En Q3 hay pausa de ~15 min → 200s no importa
- **Implementación simple, impacto alto**

---

### Mejora 6: **Calibración Platt/Isotonic** ⭐ MEDIA

**Problema**: `predict_proba` de XGBoost/LightGBM no refleja frecuencias reales.

**Solución V13**: `CalibratedClassifierCV` post-entrenamiento

**Evaluación**: ✅ **RECOMENDABLE**
- Permite que confianza = probabilidad real
- Necesario para que `MIN_CONFIDENCE_Q4 = 0.55` tenga significado
- Requiere set de calibración separado (no usar train ni val)

---

### Mejora 7: **Gate de Liga Penalizado** ⭐ BAJA

**Problema**: Gate actual bloquea ligas con <30 muestras (hard gate).

**Solución V13**: Gate degradado
```python
< 15 muestras → bloqueo total
15-29 muestras → máx 50% confianza
30-59 muestras → máx 70% confianza
≥ 60 muestras → sin cap
```

**Evaluación**: ⚠️ **INTERESANTE PERO COMPLEJO**
- 762 ligas distintas pero muchas con <10 partidos
- Con 14,699 partidos / 762 ligas = **promedio 19 partidos/liga**
- La mayoría de ligas quedarían penalizadas
- **Mejor enfoque**: filtrar top 50-100 ligas y solo apostar ahí

---

### Mejora 8: **Hiperparámetros por Submodelo (Optuna)** ⭐ MEDIA

**Problema**: V12 usa mismos hparams para todos los subsets.

**Solución V13**: Optuna tuning por cada (target, gender, pace_bucket)

**Evaluación**: ⚠️ **BUENA IDEA PERO COSTOSO**
- 12 submodelos × 40 trials × 120s timeout = **~16 horas de tuning**
- Con `--skip-tuning` flag se puede evitar
- **Recomendación**: hacer tuning solo una vez, guardar mejores params en config

---

### Mejora 9: **Módulos Separados** ⭐ MEDIA

**Problema**: `train_v12.py` tiene ~1000 líneas en un solo archivo.

**Solución V13**: Estructura modular (dataset.py, features.py, train_clf.py, etc.)

**Evaluación**: ✅ **EXCELENTE PARA MANTENIBILIDAD**
- Facilita testing, debugging, reutilización
- No mejora performance del modelo directamente
- **Hacerlo desde el inicio es más barato que refactorizar después**

---

## 📈 Mejoras FALTANTES (No Mencionadas en PLAN_V13.md)

### 1. **Walk-Forward Validation REAL**

**Falta en V12**: league stats con leakage  
**Falta en V13 PLAN**: no menciona explícitamente walk-forward para league_stats

**Propuesta**:
```python
# Para cada partido de test (ej: 2026-03-01)
# Calcular league_stats SOLO con partidos ANTERIORES a 2026-03-01
# Esto evita leakage de futuro
```

**Impacto**: 🔴 **CRÍTICO** - Sin esto, V13 tendrá el mismo leakage que V12

---

### 2. **Análisis de Distribución de Puntos por Liga**

**Falta**: Validar los umbrales de pace buckets (42/54 para Q3, 63/78 para Q4)

**Propuesta**:
```python
# Antes de entrenar, consultar:
SELECT league, 
       AVG(q1_home + q1_away + q2_home + q2_away) as q3_total_avg,
       PERCENTILE(q3_total, 0.33) as p33,
       PERCENTILE(q3_total, 0.66) as p66
FROM matches
GROUP BY league
HAVING COUNT(*) > 20
```

**Impacto**: 🟡 **IMPORTANTE** - Los umbrales genéricos pueden no funcionar para ligas específicas

---

### 3. **Feature de "Rest Days" (Descanso entre partidos)**

**Falta**: V12/V13 no consideran si un equipo juega back-to-back

**Propuesta**:
```python
# Para cada equipo, calcular días desde último partido
rest_days_home = (current_match_date - last_home_match_date).days
rest_days_away = (current_match_date - last_away_match_date).days

# Feature: is_back_to_back = rest_days < 2
# Impacto conocido: equipos en B2B rinden ~5-10% menos
```

**Disponibilidad en DB**: Se puede calcular con las fechas de partidos existentes

**Impacto**: 🟡 **POTENCIALMENTE ÚTIL** - Depende de si hay data suficiente

---

### 4. **Home/Away Splits en League Stats**

**Falta**: V12 calcula league_avg_total_points global, pero algunos equipos tienen splits enormes

**Propuesta**:
```python
# En vez de:
league_avg_total = AVG(total_points)  # todos los partidos

# Usar:
home_avg = AVG(home_total) WHERE venue = 'home'
away_avg = AVG(away_total) WHERE venue = 'away'
home_advantage = home_avg - away_avg
```

**Impacto**: 🟡 **MODERADO** - Mejora features de contexto de liga

---

### 5. **Live Simulation en Evaluación**

**Falta**: `eval_v13.py` debería simular timing live, no solo accuracy estática

**Propuesta**:
```python
# Para cada partido de test:
# 1. Simular llegada de datos en minutos {20, 22, 24} para Q3
# 2. Simular llegada de datos en minutos {30, 32, 34} para Q4
# 3. Medir:
#    - ¿A qué minuto la señal es estable?
#    - ¿Cuántos flips NO_BET→BET ocurren?
#    - ¿Cuánta ventana de apuesta queda?
```

**Impacto**: 🟢 **IMPORTANTE PARA VALIDACIÓN** - Sin esto, no sabemos si V13 funciona en live

---

### 6. **Mecanismo de "Model Not Found" Fallback**

**Falta**: Si un bucket de pace no tiene modelo (<100 muestras), ¿qué hacer?

**Propuesta**:
```python
# Jerarquía de fallback:
1. Modelo específico (q4_low_women)
2. Modelo medium del mismo target/gender (q4_medium_women)
3. Modelo gender-agnostic (q4_low_all)
4. Fallback MAE simple (promedio histórico del cuarto)
```

**Impacto**: 🟢 **NECESARIO PARA ROBUSTEZ**

---

### 7. **Detección de Anomalías en Datos Live**

**Falta**: V12/V13 asumen que si hay datos, son correctos. No detectan outliers.

**Propuesta**:
```python
# Validaciones antes de inferencia:
- score_q1 + score_q2 + score_q3 == score_current?
- graph_points minutos consecutivos? (sin gaps)
- pbp_events count coherente con tiempo jugado?

# Si alguna falla → data_quality = "poor" → NO_BET automático
```

**Impacto**: 🟡 **PREVENTIVO** - Evita predicciones con datos corruptos

---

### 8. **Métrica de "Time to Stable Signal"**

**Falta**: No hay métrica que mida cuántos ticks tarda la señal en estabilizarse

**Propuesta**:
```python
# En eval_v13.py con live simulation:
for match in test_set:
    signals = []
    for minute in [30, 31, 32, 33, 34]:
        signal = infer_at_minute(match, minute)
        signals.append(signal)
    
    # Medir:
    # - Primer minuto con señal BET
    # - Cuántos flips ocurrieron
    # - Ventana de apuesta restante
```

**Impacto**: 🟢 **CRÍTICO PARA DECIDIR SI V13 ES MEJOR QUE V12 EN LIVE**

---

## 🎯 Plan de Implementación Recomendado

### Fase 1: **Fundamentos** (1-2 días)
- [ ] **Walk-forward validation** para league_stats (EVITAR LEAKAGE)
- [ ] Centralizar config.py (ya existe)
- [ ] Estructura modular de código (dataset.py, features.py, etc.)
- [ ] Consultar DB para confirmar distribución de pace buckets

### Fase 2: **Core del Modelo** (3-5 días)
- [ ] Cutoff dinámico de features (snapshot_minute)
- [ ] Segmentación por ritmo (con fallback)
- [ ] Calibración Platt/Isotonic
- [ ] Umbrales de confianza separados Q3/Q4

### Fase 3: **Optimización** (2-3 días, opcional)
- [ ] Hiperparámetros por submodelo (Optuna)
- [ ] Gate de liga penalizado
- [ ] Features adicionales (rest_days, home/away splits)

### Fase 4: **Validación Live** (2-3 días)
- [ ] Live simulation en eval_v13.py
- [ ] Métrica "time to stable signal"
- [ ] Detección de anomalías en datos
- [ ] Testing manual con 5-10 partidos recientes

### Fase 5: **Integración** (1 día)
- [ ] Añadir "v13" a AVAILABLE_MODELS en telegram_bot.py
- [ ] Rama V13 en bet_monitor._run_inference_sync
- [ ] Testing end-to-end con monitor

---

## 📊 Viabilidad de V13: Conclusión Final

### ¿Es viable V13?

**SÍ**, pero con condiciones:

#### ✅ Datos Suficientes:
- 14,699 partidos completados → buen volumen
- 762 ligas → diversidad alta (pero muchas con pocas muestras)
- 80% hombres, 20% mujeres → desbalanceado pero manejable

#### ⚠️ Riesgos a Mitigar:
1. **Data leakage** (MISMO que V12 si no se implementa walk-forward)
2. **Buckets de women Q4 con <200 muestras** (necesita fallback)
3. **Tuning de hiperparámetros puede tardar 16+ horas** (usar --skip-tuning)
4. **762 ligas distintas** (filtrar a top 50-100 para empezar)

#### 🔴 Crítico Antes de Entrenar:
- **Implementar walk-forward para league_stats**
- **Consultar distribución real de puntos por liga**
- **Confirmar que hay ≥200 muestras por bucket mínimo**

---

## 📝 Recomendaciones Finales

### Lo que V13 **DEBE** tener (obligatorio):
1. ✅ Walk-forward validation (sin esto, no entrenar)
2. ✅ Cutoff dinámico (razón de ser de V13)
3. ✅ Config centralizado (simple, evita bugs)
4. ✅ Umbrales Q3/Q4 separados
5. ✅ NO_BET_CONFIRM_TICKS_Q4 = 0

### Lo que V13 **DEBERÍA** tener (recomendado):
6. ✅ Segmentación por ritmo (con fallback)
7. ✅ Calibración Platt
8. ✅ Live simulation en evaluación
9. ✅ Módulos separados

### Lo que V13 **PODRÍA** tener (nice-to-have):
10. ⏭️ Hiperparámetros por submodelo (Optuna)
11. ⏭️ Gate de liga penalizado
12. ⏭️ Features de rest_days, home/away splits

---

**Decisión final**: V13 es viable y resuelve problemas reales de V12, pero **SOLO SI se implementa walk-forward validation correctamente**. Sin eso, V13 repetirá los mismos errores de V12.

**Próximo paso**: Ejecutar análisis de distribución de puntos en DB para confirmar thresholds de pace buckets antes de escribir código de entrenamiento.
