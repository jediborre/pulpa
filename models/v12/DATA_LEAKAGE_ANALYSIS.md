# V12 - Data Leakage & Overfitting Analysis
## ⚠️ CRÍTICO: Resultados NO son realistas

---

## Hallazgo Principal: DATA LEAKAGE SEVERO

### Simulación con odds 1.41 (pesimista):
- Initial: $1,000
- Final: $9,532 
- ROI: **853%**
- Hit Rate: 86.2%
- Bets: 232

### Simulación con odds 1.83 (realista):
- Initial: $1,000
- Final: $**1,814,987** (sí, casi 2 millones)
- ROI: **181,398%**
- Hit Rate: 85.5%
- Bets: 275

**Esto es IMPOSIBLE en la realidad.** Ningún modelo de apuestas puede generar returns de 180,000%.

**Conclusión: El modelo tiene DATA LEAKAGE severo y los resultados de entrenamiento (74-89% accuracy) están INFLADOS.**

---

## Fuentes de Data Leakage Identificadas

### 1. 🔴 CRÍTICO: Team History Window usa resultados del MISMO match

**Problema:**
```python
# En build_samples()
samples.append(HybridSample(...))

# DESPUÉS de crear los samples de Q3 y Q4:
team_history[ht].append(1 if score["home"] > score["away"] else 0)
team_history[at].append(1 if score["away"] > score["home"] else 0)
```

Esto parece correcto, PERO el problema es que `home_prior_wr` y `away_prior_wr` se calculan con el historial ANTES de agregar el match actual. Sin embargo, el historial incluye TODOS los matches anteriores en el DB, y el split temporal es 80/20, lo que significa que matches "futuros" del test set SÍ influyen en el training.

**Impacto:** Bajo (el historial se construye secuencialmente)

### 2. 🔴 CRÍTICO: League Stats usa TODOS los matches del DB

**Problema:**
```python
def _compute_league_stats(conn) -> dict:
    rows = conn.execute("""
        SELECT m.league, m.match_id, m.home_score, m.away_score
        FROM matches m
        WHERE m.status_type = 'finished'
        ...
    """).fetchall()
```

Esto incluye TODOS los matches, incluyendo los que deberían estar en el test set.

**Impacto:** MODERATE-ALTO
- Features como `league_home_advantage`, `league_avg_total_points`, `league_std_total_points` están contaminadas con datos del futuro.
- Para una liga con 1000 matches, el modelo conoce el promedio real de TODOS ellos, no solo los "pasados".

### 3. 🟡 MODERATE: Monte Carlo Feature

`mc_home_win_prob` usa `pbp_home_pts_per_play * pbp_home_plays` que deriva de datos del match actual. Esto no es leakage per se, pero puede crear overfitting al crear una feature que es esencialmente una transformación compleja de otras features.

### 4. 🟡 MODERATE: Evaluación sobre datos de entrenamiento

En `eval_v12.py` y la simulación de bankroll, estamos evaluando sobre los MISMOS datos que se usaron para entrenar (o muy similares). No hay una separación estricta temporal.

---

## Por Qué los Números de Entrenamiento son Inflados

### Métricas reportadas en training:
- Classification Accuracy: 74.2%
- Classification F1: 0.763
- Regression R²: 0.553

### Métricas reportadas en evaluación (500 matches):
- Q3 Hit Rate: 81.76%
- Q4 Hit Rate: 89.20%
- ROI: +0.38 a +0.60 per bet

**Problema:** La evaluación de 500 matches usa el modelo entrenado con TODOS los datos y lo evalúa en un subconjunto que PUEDE estar contaminado por league stats.

---

## Qué es REALISTA y Qué NO

### ✅ REALISTA:
- MAE de regresión: 5.33 pts en total de quarter (~20-25 pts típicos)
  - Error del 20-25%, esto sí es realista
- R² de 0.553: Explica 55% de la varianza, razonable
- Volatilidad y swings como features importantes

### ❌ NO REALISTA:
- **81-89% hit rate en apuestas**: Debería ser 55-65% como máximo
- **ROI de +0.38 a +0.60 por apuesta**: Debería ser -0.05 a +0.05 si acaso
- **853% ROI con odds 1.41**: Imposible
- **181,000% ROI con odds 1.83**: Absurdo

### 🎯 LO QUE DEBERÍA ESPERARSE:
Con un modelo REAL sin leakage:
- Hit rate: 52-58% (ligeramente mejor que random)
- ROI: -0.02 a +0.05 por apuesta (difícil ser consistente positivo)
- Coverage: 20-40% (conservador)
- Max drawdown: 20-40% del bankroll

---

## Comparación con la Realidad del Betting

### Bookmakers son MUY buenos:
- Las líneas de apuestas son extremadamente eficientes
- El vigorish (juice) de 4.5-5% significa que necesitas 52.4%+ hit rate solo para BREAK EVEN
- Profesionales reales ganan 53-55% hit rate a largo plazo
- ROI anual de profesionales: 2-5% sobre bankroll

### V12 reporta:
- 85-89% hit rate → **30-35% MEJOR que profesionales reales**
- ROI de 853% → **170-400x mejor que profesionales reales**

**Esto confirma que los resultados están completamente inflados por data leakage.**

---

## Fix Requerido

Para hacer V12 realmente usable, se necesita:

### 1. League Stats con Rolling Window
```python
def _compute_league_stats_rolling(conn, cutoff_date: str) -> dict:
    """Solo usar matches ANTES de cutoff_date"""
    rows = conn.execute("""
        SELECT m.league, m.home_score, m.away_score
        FROM matches m
        WHERE m.status_type = 'finished'
          AND m.date < ?  -- SOLO matches pasados
    """, (cutoff_date,)).fetchall()
```

### 2. Team History Estrictamente Temporal
```python
# Para cada match, SOLO usar matches ANTES de este
for match in sorted_matches:
    # Predecir PRIMERO
    prediction = model.predict(features_built_from_past_only)
    # DESPUÉS agregar al historial
    team_history[match.home].append(result)
```

### 3. Walk-Forward Validation
```python
# En lugar de 80/20 split simple:
results = []
for cutoff_date in [monthly_dates]:
    train_data = all_data_before(cutoff_date)
    test_data = all_data_after(cutoff_date)
    model = train(train_data)
    results.append(evaluate(model, test_data))
```

### 4. Remover features sospechosas
- `league_home_advantage` (computed from all data)
- `league_avg_total_points` (computed from all data)
- `league_std_total_points` (computed from all data)

O recomputarlas con rolling windows.

---

## Documentación Honesta de los Números Actuales

### Lo que el modelo DICE que hace:
- 74-89% accuracy en ganador de quarter
- ROI de +38-60% por apuesta
- $1,000 → $9,532 (odds 1.41)
- $1,000 → $1,814,987 (odds 1.83)

### Lo que REALMENTE haría en producción:
- 50-55% accuracy (cerca de random)
- ROI de -5% a +5% por apuesta (difícil ser positivo)
- $1,000 → $500-1,200 (perder o ganar poco)
- Probable pérdida neta después de vigorish

---

## Conclusión

**V12 NO está listo para producción con dinero real.**

Los números de entrenamiento y evaluación están severamente inflados por data leakage. El modelo necesita:

1. ✅ Re-entrenamiento con features estrictamente temporales
2. ✅ Walk-forward validation en lugar de 80/20 split
3. ✅ Evaluación con holdout set verdaderamente "futuro"
4. ✅ Testing en producción real con paper trading antes de dinero real

**NO usar este modelo para apostar dinero real hasta fix completo.**

---

## Próximos Pasos

1. **URGENTE**: Fix league stats para usar solo datos pasados
2. **URGENTE**: Implementar walk-forward validation
3. **Importante**: Re-evaluar con datos verdaderamente holdout
4. **Importante**: Paper trading por 100+ bets antes de dinero real
5. **Nice-to-have**: Agregar más features sin leakage (rest days, injuries, etc.)
