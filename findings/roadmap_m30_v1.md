# Roadmap M30_V1

Roadmap operativo del modelo `m30_v1` para predecir Q4 winner desde snapshot minute 30.

Estado al 17 May 2026 — despues de momentum features, podas, separacion 10m/12m y analisis de brecha con m27_v1.

## Que es m30_v1

Modelo independiente de `m27_v1` entrenado con `snapshot_minute = 30`. Usa features extraidas de la ventana Q3 completo (para 10m) o mitad de Q3 (para 12m). Admite modo mixto y modo 10m-only via `--filter-10m`.

Trainer: `match/training/train_q4_m30_v1.py`

## Estado actual (17 May 2026)

| Version | Modelo | ROC AUC test | Accuracy |
|---|---|---|---|
| m30_v1 (pre-momentum) | Champion mixto | 0.584 | 0.562 |
| m30_v1 (pre-momentum) | Champion 10m-only | 0.585 | 0.564 |
| m30_v1 (con momentum) | **HistGB 10m-only** | **0.591** | **0.561** |
| m30_v1 (con momentum) | Champion mixto | 0.585 | 0.562 |
| m27_v1 (referencia) | Champion | 0.668 | 0.628 |

Mejora neta de momentum features: **+0.007 ROC AUC** (HistGB 10m: 0.585 → 0.591).

## Lo que ya se hizo

### 1. Podas y limpieza (sin impacto)

- ~12 features ruidosas/redundantes removidas (89 → 77)
- 10 constantes/colineales removidas para 10m (77 → 67)
- Resultado: identico rendimiento antes/despues de podar

### 2. Separacion 10m/12m (sin impacto)

- `--filter-10m` flag implementado
- Modelo 10m-only: Champion 0.585 vs mixto 0.584
- Separacion no mejora nada

### 3. Deep momentum features (CON IMPACTO)

**Origen**: usuario desafio la conclusion de "limite estructural" argumentando que humanos pueden predecir Q4 winner desde minuto 30 mediante patrones de momentum visibles en la trayectoria de `graph_points`.

**Exploracion de datos**: se confirmo que `graph_points.value` = score differential (home - away) por minuto. Se identificaron patrones concretos:
- **Momentum exhaustion**: equipo que remonta fuerte en Q3 pero "gasta" su recuperacion, pierde Q4
- **Lead erosion**: equipo cuya ventaja se reduce entrando a Q4, pierde Q4

**Features agregadas (24 total)**:
- Q3 graph momentum (8): lead erosion, late momentum, volatility, turning points, recovery spent, peak-to-final ratio, last 3min momentum, momentum acceleration
- PBP windows (16): recent_1m_* y recent_2m_* (home/away points, points_diff, home_event_share, max_run_pts, run_diff, last_scoring)

**Resultado**: HistGB mejoro de 0.585 a 0.591 ROC AUC (+0.007). Es la primera mejora real en toda la historia del proyecto.

### 4. Feature importance (XGB)

| Familia | Sum Importance | Features |
|---|---|---|
| PBP windows | **0.226** | 16 features |
| prior_strength | 0.160 | win rates, etc. |
| score_state | 0.118 | score_est, halftime, etc. |
| Q3 momentum | **0.101** | 8 features |
| transition_flags | 0.096 | comeback state, etc. |
| q3_partial_state | 0.092 | Q3 partial scores |

`recent_2m_last_scoring_away` es #9 global. Q3 momentum features estan en el rango #27-#65. Con un modelo mas profundo (HistGB depth=5) su contribucion real es mayor.

### 5. XGB depth experiment (depth 4→6)

Profundidad mayor hizo OVERFIT: XGB depth=6 bajo a 0.567 ROC AUC. Depth 4 es el optimo.

### 6. ROI gap con m27_v1

m27_v1 (0.668) es **no comparable** con m30_v1 (0.591) porque:
- m27_v1 usa `snapshot_minute = 27` que para NBA (12m) = **3 minutos adentro de Q4**
- m30_v1 usa `snapshot_minute = 30` que para 10m = fin de Q3 (sin datos de Q4)
- La ventaja de m27_v1 viene de tener datos reales de Q4 para NBA, no de mejor feature engineering

La comparacion justa es m30_v1 pre-momentum vs con-momentum: +0.007 ROC AUC.

## Archivos clave

- `match/training/train_q4_m30_v1.py` — trainer con momentum features y `--filter-10m`
- `match/training/M30_V1_FEATURE_PRUNING_NOTES.md` — historial completo de experimentos
- `match/training/model_outputs_m30_v1/q4_metrics.csv` — metricas modelo mixto con momentum
- `match/training/model_outputs_m30_v1/q4_10m_only/q4_metrics.csv` — metricas 10m-only con momentum
- `match/training/model_outputs_m30_v1/q4_10m_only/run_summary.json` — features finales (93)
- `temp_scripts/explore_db_schema.py` — explorador de esquema DB
- `temp_scripts/explore_game_patterns.py` — patrones de juego (momentum exhaustion, lead erosion)
- `temp_scripts/explore_comebacks.py` — analisis de remontadas

## Opciones para continuar

### 1. Agregar recent_3m windows (recomendado)

m27_v1 tenia `recent_3m_*` (8 features) que cubrian minutos 24-27 (inicio de Q3). En m30_v1 se removieron durante la poda. Al minuto 30, un window de 3 minutos cubriria minutos 27-30 (final de Q3), que es distinto del recent_2m. Podria capturar momentum sostenido que el 2m no ve.

### 2. Agregar trailing-deficit x recent-run interaction

m27_v1 tenia `trailing_now_recent_run_2m` y `trailing_now_recent_run_3m`. Estas features cruzan "quien va perdiendo" con "quien esta anotando ahora" — capturan exactamente el patron de "trailing team making a run". No existen en m30_v1.

### 3. Aumentar profundidad de HistGB

HistGB actualmente usa max_depth=5. Subir a 6-7 podria capturar mejor las interacciones entre Q3 momentum features y PBP windows. XGB no mejora con mas profundidad, pero HistGB si podria.

### 4. Abandonar y redirigir

Si +0.007 ROC AUC no justifica mas iteracion, redirigir esfuerzo a mejorar m27_v1 o explorar targets alternativos (Q4 totals, spreads).
