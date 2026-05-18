# Findings — Predicción Q4 (minuto 27/30)

## Resumen de modelos

| Modelo | Snapshot | ROC AUC (10m) | Yield (10m) | Apuestas | Estado |
|--------|----------|:-:|:-:|:-:|:-:|
| **m27_v1** | 27 | **0.668** | **+4.5%** | 643 | ✅ Mejor |
| **m27_v2** | 27 | **0.667** / **0.671**\* | — | — | ✅ Pipeline limpio \*10m filter |
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

### 4. Anatomía: el modelo no depende de "riqueza" de datos
Los partidos acertados vs fallados son casi idénticos en:
- PBP total (+0.4%)
- Puntos totales (+0.5%)
- Diferencia de marcador (+3.8%)
- Densidad de eventos (+0.4%)

La señal predictiva está en las *features específicas* (diferencias por cuarto, win rates, rachas), no en cuántos datos haya.

### 5. v6.3 m30 vs m30_v1
- v6.3 apuesta 28× más que m30_v1 (umbral laxo) -> quiebra el bankroll
- m30_v1 es más conservador (filtro confianza 0.58) -> yield menos negativo pero aún negativo
- Ambos son superados por m27_v1

## Archivos en esta carpeta

| Archivo | Contenido |
|---------|-----------|
| `findings.md` | Este resumen |
| `roadmap_m30_v1.md` | Roadmap y experimentos de m30_v1 |
| `M27_V1_ROADMAP.md` | Roadmap de m27_v1 |
| `M27_V1_LEAGUE_TIERS.md` | Policy de ligas para m27_v1 (tiers/blacklist) |
| `M27_FEATURES_COMPARISON.md` | Comparación de features entre versiones |
| `M30_V1_FEATURE_PRUNING_NOTES.md` | Notas de poda de features en m30_v1 |
| `ligas_10min.md` | Lista de 1,185 ligas con cuartos de 10 min |
| `ligas_12min.md` | Lista de 16 ligas con cuartos de 12 min |
| `results_roi.txt` | Resultados históricos de ROI (varios modelos) |
| `m30_v1_por_qlen.csv` | Métricas de m30_v1, m27_v1 y v6.3 split por duración de cuarto |
| `anatomia_m27_v1.csv` | Promedios de variables por acierto/fallo en m27_v1 |
| `anatomia_raw.csv` | Datos crudos de anatomía (cada partido con predicción + métricas) |

## Próximos pasos sugeridos

1. Entrenar modelo específico para **12m** con snapshot 33-36
2. Probar **modelos separados por duración de cuarto** (10m vs 12m)
3. Probar snapshot intermedios (28, 29) para ver si hay punto óptimo
4. Investigar por qué m27_v1 funciona mejor exactamente en snap 27 vs 30 con análisis de importancia de features
