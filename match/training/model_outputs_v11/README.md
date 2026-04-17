# V11 - Over/Under Model - Próxima Iteración

## Resumen Ejecutivo

V11 mejorará V10 implementando las recomendaciones consolidadas:

| Mejora | Descripción | Impacto Esperado |
|--------|-------------|------------------|
| Modelos separados por género | Hombres vs Mujeres | +5-10% accuracy |
| Modelos por liga | NBA, Euroleague, etc. | +3-5% accuracy |
| Auto-betting logic | Calcular edge y recomendar OVER | +20% ROI |
| Margen de confianza | Solo bet si edge > 3 pts | Mayor hit rate |

**Estado**: 🔴 Pendiente de implementación

---

## Problemas Identificados en V10

### 1. Modelo único para todos los géneros
- Hombres promedian ~40 pts/quarter
- Mujeres promedian ~32-35 pts/quarter
- Diferencia de ~8 puntos no captada

### 2. Modelo único para todas las ligas
- NBA: ~42-45 pts/quarter
- EuroLeague: ~38-40 pts/quarter
- LatAm: ~36-40 pts/quarter

### 3. Lógica de apuesta no automatizada
- Usuario debe calcular edge manualmente
- No hay recomendación automática de OVER/UNDER

### 4. Sin filtro de confianza
- Todas las predicciones se usan para betting
- No hay umbral para filtrar bets de baja confianza

---

## Objetivos de V11

### Objetivo Principal
Crear un **sistema de betting automatizado** que:
1. Detecte el género del partido
2. Detecte la liga
3. Seleccione el modelo apropiado
4. Calcule edge automáticamente
5. Recomiende SOLO OVER cuando edge > threshold

### Objetivos Secundarios
1. Mejorar accuracy con modelos especializados
2. Reducir bets fallidos con margen de confianza
3. Generar output listo para usar en vivo

---

## Plan de Implementación

### Fase 1: Modelos por Género

```
V11/
├── models/
│   ├── men/
│   │   ├── q3_home_stacking.joblib
│   │   ├── q3_away_stacking.joblib
│   │   ├── q3_total_stacking.joblib
│   │   ├── q4_home_stacking.joblib
│   │   ├── q4_away_stacking.joblib
│   │   └── q4_total_stacking.joblib
│   └── women/
│       ├── q3_home_stacking.joblib
│       ├── q3_away_stacking.joblib
│       ├── q3_total_stacking.joblib
│       ├── q4_home_stacking.joblib
│       ├── q4_away_stacking.joblib
│       └── q4_total_stacking.joblib
```

**Thresholds por género:**

| Género | Q3 Total | Q4 Total |
|--------|----------|----------|
| Hombres | 39-40 | 39-40 |
| Mujeres | 32-35 | 32-35 |

### Fase 2: Modelos por Liga (Opcional)

```
V11/models/
├── nba/
├── euroleague/
├── ncaa/
└── latam/
```

**Thresholds por liga:**

| Liga | Q3/Q4 Total |
|------|--------------|
| NBA | 42-45 |
| EuroLeague | 38-40 |
| NCAA | 35-38 |
| LatAm | 36-40 |

### Fase 3: Script de Predicción Automatizada

```python
def predict_and_bet(match_features, sportsbook_line):
    """
    Returns: {
        'prediction': float,
        'edge': float,
        'recommendation': 'OVER' / 'UNDER' / 'NO_BET',
        'confidence': 'HIGH' / 'MEDIUM' / 'LOW'
    }
    """
    
    # 1. Detectar género y liga
    gender = match_features['gender_bucket']
    league = match_features['league_bucket']
    
    # 2. Cargar modelo apropiado
    model = load_model(gender, league, target)
    
    # 3. Predecir
    prediction = model.predict(match_features)
    
    # 4. Calcular edge
    edge = prediction - sportsbook_line
    
    # 5. Recomendar
    if edge > 3:
        return {'recommendation': 'OVER', 'confidence': 'HIGH'}
    elif edge > 1:
        return {'recommendation': 'OVER', 'confidence': 'MEDIUM'}
    elif edge < -3:
        return {'recommendation': 'UNDER', 'confidence': 'HIGH'}
    elif edge < -1:
        return {'recommendation': 'UNDER', 'confidence': 'MEDIUM'}
    else:
        return {'recommendation': 'NO_BET', 'confidence': 'LOW'}
```

### Fase 4: Sistema de Live Betting

```
Timeline del partido:
├── Q1 ends → Update features
├── Q2 ends (Halftime) → Predecir Q3 y Q4 → Apostar
├── Q3 ends → Predecir Q4 → Apostar
└── Q4 ends → Final
```

---

## Features a Agregar

### Nuevas Features (Opcional)

| Feature | Descripción | Correlación Esperada |
|---------|-------------|---------------------|
| pace_home | Jugadas por minuto local | +0.3 |
| pace_away | Jugadas por minuto visitante | +0.3 |
| pace_diff | Diferencia de pace | +0.2 |
| pace_total | Pace total del partido | +0.35 |
| 3q_momentum | Diferencia Q3 - Q2 | +0.15 |

### Features Existentes (Mantener)

- league_bucket, gender_bucket
- home_team_bucket, away_team_bucket
- ht_home, ht_away, ht_total
- q1_diff, q2_diff, q3_diff
- score_3q_home, score_3q_away, score_3q_total
- gp_count, gp_last, gp_peak_*, gp_area_*, gp_*, gp_slope_*
- pbp_home_pts_per_play, pbp_away_pts_per_play
- pbp_home_plays, pbp_away_plays, pbp_plays_diff
- pbp_home_3pt, pbp_away_3pt

---

## Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────┐
│                    V11 SYSTEM                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │  Input       │     │  Gender      │                 │
│  │  Features   │────▶│  Detector    │                 │
│  └──────────────┘     └──────────────┘                 │
│                              │                          │
│                              ▼                          │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │  League     │────▶│  Model       │                 │
│  │  Detector   │     │  Selector    │                 │
│  └──────────────┘     └──────────────┘                 │
│                              │                          │
│                              ▼                          │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │  Prediction │────▶│  Betting     │                 │
│  │  Engine     │     │  Advisor     │                 │
│  └──────────────┘     └──────────────┘                 │
│                              │                          │
│                              ▼                          │
│  ┌──────────────┐                                   │
│  │  Output:     │                                   │
│  │  - OVER/UNDER/NO_BET                             │
│  │  - Edge     │                                   │
│  │  - Confidence                                   │
│  └──────────────┘                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Comparación V10 vs V11

| Aspecto | V10 | V11 |
|---------|-----|-----|
| Modelos por género | ❌ No | ✅ Sí |
| Modelos por liga | ❌ No | ✅ Opcional |
| Auto-betting logic | ❌ Manual | ✅ Automático |
| Filtro de confianza | ❌ No | ✅ Sí |
| Over recommended only | ❌ No | ✅ Sí |
| Live betting timing | ❌ No | ✅ Sí |

---

## Métricas Objetivo

| Métrica | V10 Actual | V11 Objetivo |
|---------|------------|--------------|
| OVER Hit Rate | 86-100% | 90-100% |
| OVER ROI | +65-91% | +80-100% |
| UNDER Hit Rate | 38-45% | N/A (no usar) |
| MAE (totales) | ~5 pts | ~4 pts |
| R² | ~0.5 | ~0.6 |

---

## Estructura de Archivos V11

```
model_outputs_v11/
├── README.md                          # Este documento
├── v11_train.py                       # Script de entrenamiento
├── v11_predict.py                     # Script de predicción
├── v11_live_betting.py                # Script para uso en vivo
├── models/
│   ├── men/
│   │   ├── q3_home_stacking.joblib
│   │   ├── q3_away_stacking.joblib
│   │   ├── q3_total_stacking.joblib
│   │   ├── q4_home_stacking.joblib
│   │   ├── q4_away_stacking.joblib
│   │   └── q4_total_stacking.joblib
│   └── women/
│       ├── q3_home_stacking.joblib
│       ├── q3_away_stacking.joblib
│       ├── q3_total_stacking.joblib
│       ├── q4_home_stacking.joblib
│       ├── q4_away_stacking.joblib
│       └── q4_total_stacking.joblib
├── metrics/
│   ├── men_metrics.csv
│   └── women_metrics.csv
└── betting_simulation/
    └── v11_betting_results.csv
```

---

## Tareas Pendientes

### 🔴 Alta Prioridad
- [ ] Crear script de entrenamiento V11 (separar por género)
- [ ] Entrenar modelos para hombres
- [ ] Entrenar modelos para mujeres
- [ ] Crear script de predicción con auto-betting

### 🟡 Media Prioridad
- [ ] Agregar pace features
- [ ] Crear modelos por liga (NBA, Euroleague, etc.)
- [ ] Implementar live betting timing

### 🟢 Baja Prioridad
- [ ] Backtest con datos reales
- [ ] Incluir juice/vig en ROI
- [ ] Historical validation

---

## Questions Pendientes

1. ¿Comenzamos con modelos por género o por liga?
2. ¿Cuántasligas específicas quieres incluir?
3. ¿Qué formato de output prefieres (CLI, API, CSV)?
4. ¿Cuál es el timeframe de implementación?

---

## Referencias

- V10 README: `model_outputs_v10/README.md`
- Betting results V10: `model_outputs_v10/betting_results.csv`
- Training script V10: `train_q3_q4_regression_v10.py`

---

*Documento creado: 2026-03-30*
*Para revisión: 2026-03-31*