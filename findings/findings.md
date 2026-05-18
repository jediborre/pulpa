# Findings — Predicción Q4 (minuto 27/30)

## Resumen de modelos

| Modelo | Snapshot | ROC AUC (10m) | Yield (10m) | Apuestas | Estado |
|--------|----------|:-:|:-:|:-:|:-:|
| **m27_v1** | 27 | **0.668** | **+4.5%** | 643 | ✅ Mejor |
| **m27_v2** | 27 | 0.667 | — | — | ✅ Baseline sin H2H |
| **m27_v3** | 27 | **0.789** | **+13.0%** | 2,414 | ✅ Producción lista |
| m30_v1 | 30 | 0.576 | -3.6% | 90 | ❌ Ruido |
| v6.3 m30 | 30 | 0.581 | -19.8% | 2,561 | ❌ Quiebra |

## Hallazgos clave

### 1. Snapshot 27 > Snapshot 30 para 10m
m27_v1 (final de Q3) supera ampliamente a m30_v1 (3 min después). Los 3 minutos extra de Q3 añaden ruido, no señal. Con ~90% de Q3 capturado en m27 es suficiente.

### 2. Ningún modelo funciona en 12m a snap 27/30
- Para 12m: snap 27 = solo 3 min de Q3; snap 30 = solo 6 min de Q3
- Los tres modelos dan ROC AUC cerca de 0.5 o peor en 12m
- Se necesita snapshot ~33-36 para 12m (final de Q3 real)

### 3. m27_v1 es rentable pero marginal
- Yield +4.5% a momio 1.4 con stake fijo $25
- 643 apuestas en test (4,069 muestras totales)
- Efectividad 74.7% (break-even en 71.4%)

### 5. H2H Head-to-Head features: el mayor salto de calidad hasta ahora

Agregar 3 features de historial H2H (último resultado H2H, últimos 3, y Q1 avg diff) al m27_v2:

| Feature | Test AUC | Δ vs baseline |
|---------|:--------:|:-------------:|
| **m27_v2 baseline** | 0.667 | — |
| **m27_v2 + H2H features** | **0.789** | **+0.122** |

La feature `h2h_last_home_won` es la **#1 en importancia** del modelo (0.2282, 4x la siguiente). Las 3 H2H suman 0.2855 de importancia total (>28% del modelo).

**Cobertura**: 42% de las muestras tienen al menos 1 H2H previo (11,622 de 27,542). El resto reciben valor 0/default.

**Implicación**: para partidos con historial H2H, el modelo es mucho más preciso. Descargar más historial (cubrir el 58% restante) podría mejorar aún más.

#### 5.1 ¿Qué datos de H2H descargar?

Actualmente el modelo computa H2H desde la DB local (tablas `matches` + `quarter_scores`). La cobertura es 42% porque muchos pares de equipos no tienen encuentros previos en la DB.

Para mejorarla hay 3 vías:

| Vía | Qué descargar | Costo | Impacto estimado |
|-----|--------------|:-----:|:----------------:|
| **A. Backfill histórico** | Más fechas de partidos via `/sport/basketball/scheduled-events/{date}` | ~100 request por día de backfill | **Alto** — más matches en DB = más pares H2H cubiertos |
| **B. API H2H de SofaScore** | `GET /event/{id}/h2h` — devuelve match IDs de encuentros previos entre los 2 equipos | 1 request por match a evaluar | **Medio** — descubre match IDs que luego hay que scrapear |
| **C. API de equipo** | `GET /team/{id}/events/last/{N}` — últimos N partidos de cada equipo | ~1 request por equipo único | **Bajo** — indirecto, menos preciso |

**Recomendación**: empezar por (A) — backfill de más fechas históricas. Es lo que ya hace el scraper, solo hay que correrlo para más días. No requiere cambios de infraestructura y da cobertura automática para nuevos pares que entren por fecha.

El endpoint H2H (B) existe en la API de SofaScore pero no se usa actualmente. Podría implementarse como optimización: para un match nuevo, llama al H2H endpoint para conseguir match IDs de encuentros previos, y si no están en DB, los descarga bajo demanda.

### 6. Anatomía: el modelo no depende de "riqueza" de datos
Los partidos acertados vs fallados son casi idénticos en:
- PBP total (+0.4%)
- Puntos totales (+0.5%)
- Diferencia de marcador (+3.8%)
- Densidad de eventos (+0.4%)

La señal predictiva está en las *features específicas* (diferencias por cuarto, win rates, rachas), no en cuántos datos haya.

### 7. v6.3 m30 vs m30_v1
- v6.3 apuesta 28× más que m30_v1 (umbral laxo) -> quiebra el bankroll
- m30_v1 es más conservador (filtro confianza 0.58) -> yield menos negativo pero aún negativo
- Ambos son superados por m27_v1

### 8. Feature importance: m27_v2 vive de recent_2m_points_diff
- **`recent_2m_points_diff` domina con 18.4% de importancia** (3× el #2). El modelo es esencialmente un predictor de "quién anotó más en los últimos 2 minutos".
- Recent windows (2m + 3m) = **35.9% de toda la importancia** del árbol.
- Top 5 features: recent_2m_points_diff (0.184), prior_wr_diff (0.057), recent_2m_run_diff (0.036), score_halftime_diff_ratio (0.026), q3_partial_total (0.020).

### 9. Features nuevas (F8/G1/G4) no sumaron señal
- Scoring Runs (F8): 4.1% de importancia. `current_run_away` (#10, corr -0.18) es el mejor.
- Score Ratios (G1): 4.7% de importancia. `score_halftime_diff_ratio` (#4, imp 0.026) es el mejor.
- PBP Density (G4): 5.9% de importancia combinada. Solo `pbp_scoring_density` está muerta.
- AUC-weighted ensemble no mejoró nada (XGB y HistGB rinden casi igual).
- **Neto: +0.0009 AUC vs baseline v2** — ruido.

### 10. Poda de 7 features muertas
- 7 features con XGB imp = 0.0 fueron removidas: `halftime_leader`, `halftime_trailing_side`, `q3_partial_leader`, `trailing_now_is_home/away`, `gp_count`, `pbp_scoring_density`.
- AUC idéntico (0.6682). 93 → 86 features.

### 11. Mejores predictores individuales (por |correlación|)
| Rank | Feature | |corr| |
|------|---------|:----:|
| 1 | recent_2m_points_diff | 0.240 |
| 2 | recent_2m_run_diff | 0.239 |
| 3 | recent_3m_run_diff | 0.230 |
| 4 | recent_3m_points_diff | 0.224 |
| 5 | recent_2m_home_event_share | 0.207 |
| 6 | current_run_away | -0.180 |
| 7 | home_prior_wr | 0.115 |
| 8 | prior_wr_diff | 0.168 |
| 9 | score_est_diff | 0.142 |
| 10 | gp_last | 0.115 |

Los primeros 18 puestos son ocupados por features de ventanas recientes (2m/3m). El primer feature no-recent-window es `prior_wr_diff` (#15).

## Archivos en esta carpeta

| Archivo | Contenido |
|---------|-----------|
| `findings.md` | Este resumen |
| `roadmap_m30_v1.md` | Roadmap y experimentos de m30_v1 |
| `M27_V1_ROADMAP.md` | Roadmap de m27_v1 |
| `M27_V2_ROADMAP.md` | Roadmap de m27_v2 — experimentos, feature importance, próximas señales |
| `M27_V1_LEAGUE_TIERS.md` | Policy de ligas para m27_v1 (tiers/blacklist) |
| `M27_FEATURES_COMPARISON.md` | Comparación de features entre versiones |
| `M30_V1_FEATURE_PRUNING_NOTES.md` | Notas de poda de features en m30_v1 |
| `ligas_10min.md` | Lista de 1,185 ligas con cuartos de 10 min |
| `ligas_12min.md` | Lista de 16 ligas con cuartos de 12 min |
| `results_roi.txt` | Resultados históricos de ROI (varios modelos) |
| `m30_v1_por_qlen.csv` | Métricas de m30_v1, m27_v1 y v6.3 split por duración de cuarto |
| `anatomia_m27_v1.csv` | Promedios de variables por acierto/fallo en m27_v1 |
| `anatomia_raw.csv` | Datos crudos de anatomía (cada partido con predicción + métricas) |
| `m27_v2_feature_importance.csv` | Feature importance completa de m27_v2 (XGB, corr, 86 features) |

## Próximos pasos: buscar nuevas señales

El feature importance confirmó que el modelo depende casi enteramente de ventanas recientes (2m/3m). Las features low-effort de V9+ (F8 scoring runs, G1 score ratios, G4 PBP density) no sumaron señal. Para mejorar hay que buscar features con información *nueva*, no redundante.

### Ideas de nuevas features

1. **Modelo 12m separado** con snapshot 33-36 (Q3 casi completo real)
2. **Snapshots intermedios** (28, 29) para ver si hay punto óptimo
3. **Features de fouls / faltas** — ritmo de fouls por equipo, bonus situation
4. **Features de timeouts** — cuántos timeouts le quedan a cada equipo, momentum post-timeout
5. **Features de starting lineups** — quién juega (descansa estrella?), profundidad de banca
6. **Features de streak** — racha ganadora/perdedora del equipo, no solo del partido actual
7. **Features de and-ones / free throws** — ratio FTA/FGA, eficiencia en línea
8. **Features de pace real** — posesiones por minuto, no solo puntos
9. **External data** — Vegas lines, Elo ratings, power rankings
10. **Modelos separados por liga o por duración de cuarto** (10m vs 12m)
