# 🔬 m27_v5_3 — qué arregla, por qué cada feature, datos que faltan y cómo se mide

> **Reentrenado el 2026-10-20** (v2: filtro de formatos de torneo + umbral 0.75 + calibración continua Platt). Entrenado originalmente el 2026-10-18.
> Código: [`models/m27_v5_3/`](../models/m27_v5_3) · Pipeline: [`tmp/m27_v5_3/`](../tmp/m27_v5_3) · Diagnóstico previo: [`findings_m27_2.md`](./findings_m27_2.md).
> Análisis live y decisiones operativas: [`ANALISIS_LIVE_M27_V5_3.md`](./ANALISIS_LIVE_M27_V5_3.md) · Conclusión ejecutiva: [`CONCLUSION_M27_V5_3.md`](./CONCLUSION_M27_V5_3.md) · Plan v5_4: [`PLAN_V5_4.md`](./PLAN_V5_4.md).
> Artefactos (ignorados por git, ver `.gitignore`): `models/m27_v5_3/model_outputs/` → pesos, `model_config.json`, `m27_v5_3_summary.json`, `league_calibration.json`, `feature_justification.md`, `m27_v5_3_diagnostic_dashboard.png`.

---

## 1. Resumen

| | m27_v5_2 | **m27_v5_3** |
|---|---|---|
| Extractor | 2 (uno para entrenar, otro para live) | **1 solo** (`features.snapshot_features`) |
| Snapshot | "27" entrenado con marcador de min 30 | **Minuto real t ∈ [25.5, 30]**, entrenado con snapshots en todo el rango + cierre de Q3 |
| Faltantes | 0 (confunde "dato ausente" con 0) | **NaN** (árboles) / imputación + indicadores (LR) |
| Priors | ELO final de hoy + fórmulas distintas en live | `PriorState` as-of, mismo código, refresco incremental cada 30 min |
| Cuotas | ML (a veces in-play) | ML **solo pre-partido** + spread/total con línea (cuando existe) |
| Simetría local/visitante | no | **espejo** (aumentación + promedio) |
| Calibración | isotónica in-sample | isotónica **fuera de muestra** (bloque B), simétrica, techo 0.95 | **Platt simétrica fuera de muestra** (bloque B): continua, sin meseta en el umbral |
| Formatos de torneo | todos en el train | **fuera**: amistosos, 2ª fase, promoción, preliminar, colocación |
| Umbral | P ≥ 0.68 elegido sobre test | **P ≥ 0.75** (decisión explícita, `OPERATING_CONF`) |

Resultados sobre el holdout C (último 18 % de **184.173 partidos elegibles** — tras el filtro de formatos; **91.842 filas / 33.152 partidos ancla**):

| Métrica | v5_2 (reportado) | **v5_3 (2026-10-20)** |
|---|---|---|
| AUC train / calib (B) / **test** | 0.6368 / – / 0.6376 | 0.6461 / 0.6493 / **0.6552** |
| Brecha train−test | −0.0007 | −0.0090 (sin sobreajuste) |
| Brier / LogLoss / Acc (test) | 0.2350 / 0.6623 / 59.6 % | **0.2304 / 0.6518 / 61.3 %** |
| MAE margen Q4 | – | 6.23 pts |

Tabla de umbrales en C, **solo ancla (t=30, un partido = una fila)**, comparable con v5_2:

| P ≥ | Apuestas | WR (IC95) | ROI @1.40 | ROI @1.83 |
|---|---|---|---|---|
| 0.65 | 9.764 | 71.7 % (70.8–72.6) | +0.4 % | +31.2 % |
| 0.68 | 6.988 | 74.0 % (73.0–75.0) | +3.6 % | +35.5 % |
| 0.70 | 5.457 | 76.3 % (75.1–77.4) | +6.8 % | +39.5 % |
| 0.72 | 4.206 | 78.0 % (76.7–79.2) | +9.1 % | +42.7 % |
| **0.75** *(operativo)* | **2.728** | **81.2 % (79.7–82.7)** | **+13.7 %** | **+48.7 %** |
| 0.80 | 1.312 | 86.3 % (84.3–88.0) | +20.8 % | +57.9 % |

> ⚠️ **La meseta desapareció.** Con la calibración isotónica los umbrales 0.75, 0.76, 0.78 y 0.80 elegían **las mismas 2.171 apuestas** (la isotónica es escalonada y la confianza quedaba cuantizada: 9 valores útiles en vivo). Con la **calibración Platt simétrica** (`models/common/calibration.py`) la confianza es continua (7.838 valores distintos), el LogLoss mejora (0.6522 → 0.6518) y **el umbral se puede afinar**. El óptimo de beneficio esperado está en **0.72 a cuota 1.40** y en **0.70 a cuota 1.83**; se mantiene 0.75 como posición prudente hasta disponer del EV real (ver [`PLAN_V5_4.md`](./PLAN_V5_4.md)).
>
> ⚠️ Honestidad: el equilibrio a cuota 1.40 es 71.4 % de WR. La cuota real de Q4 no se registra aún (ver §5), así que el ROI es teórico.

---

## 2. Qué arregla respecto a v5_2 (inconsistencias → corrección)

| # | Problema en v5_2 | Corrección en v5_3 | Dónde |
|---|---|---|---|
| 1 | `recent_2m_run_diff` siempre 0 en vivo (clave `run_diff` vs `recent_run_diff`); en train era `q3_diff*0.2` | `recent_pts_diff_2m/5m` reales desde `match_events`, mismo código | `features.py` |
| 2 | `d_strat_q4_net` siempre 0 en vivo (state sin `q_profile`) | `d_q4_net12` desde `PriorState` | `PriorState` |
| 3 | `h2h` y `team_strength` llegaban como *lista*, el extractor esperaba *dict* → 0 | H2H del historial propio (`PriorState`); `team_strength` ya no se usa | — |
| 4 | `rubber_band_gap`, `white_flag`, `d_wr`, `d_q4`, `d_lineup_*_ft`, `expected_ft_margin` con fórmulas distintas train/live | Una sola función. Auditoría de paridad: máx. diff 0.0 en features de estado/mercado | `audit_feature_parity.py` |
| 5 | "Snapshot 27" entrenado con marcador/pace/gp de min 30 | `minute` es feature; entrenado en t ∈ [25.5, 29.9] (2 por partido con eventos) + ancla t=30 | `build_dataset.py` |
| 6 | Si el partido ya estaba en Q4, el vector mezclaba puntos de Q4 | En Q4 / retraso se usa el cierre de Q3 (Q1+Q2+Q3), t=30 | `ctx_from_payload` |
| 7 | `odds_log_ratio` tomaba el primer mercado, que en vivo es **in-play** (KBL: pre 1.33/3.10 vs in-play 1.95/1.74) | `_parse_odds` descarta `isLive=True`; las cuotas in-play se resumen aparte (`odds_live`) | `match/scraper.py`, `mobile_client.py` |
| 8 | `team_strength` bajado el 12-18 oct 2026 para todo el histórico (look-ahead) | `d_season_wr` as-of calculado desde `matches.db` | `PriorState` |
| 9 | ELO "final" usado como feature de entrenamiento | `PriorState` cronológico (as-of), decaimiento por inactividad no mutante | `PriorState.prematch` |
| 10 | Roster ELO (8 features) con 47 % de cobertura y definición de "titular" distinta | Retirado: aportaba −0.0006 AUC y no es reproducible en vivo | §3 |
| 11 | Ceros como "dato ausente" (gp, odds, recent) | NaN + `has_ml` | `features.py` |
| 12 | `pbp_3pt_diff` constante 0 en train; crash si `play_by_play` vacío | Eliminada | — |
| 13 | Calibración isotónica in-sample + umbral elegido en test | Isotónica OOT (B) + umbral en B; test C intacto | pipeline |
| 14 | Sesgo local/visitante no controlado | Aumentación espejo + promedio simétrico (+ `is_home_view`) | `mirror()` |
| 15 | Ligas nuevas: lc_map ausente → fórmulas distintas | `fallback_info` compartida (género + 10m) | `features.py` |
| 16 | `league_calibration.json` nunca se usaba; ligas tóxicas solo podadas del train | `block_toxic_leagues` en inferencia (PASS) | `predict.py` |

Paridad (400 partidos del holdout, camino live vs fila del dataset): **PARIDAD OK**. Priors difieren solo por obsolescencia del estado (corr 0.89–0.99). Latencia: extracción 1.1 ms; `predict` completo ≈ 30 ms (payload en memoria, con el backfill corriendo en paralelo).

---

## 3. Features: para qué sirven, correlación y evidencia

Tabla completa con correlación, AUC univariado, cobertura, ganancia XGB y ablación individual: [`feature_justification.md`](../models/m27_v5_3/model_outputs/feature_justification.md). Resumen por grupo (caída de AUC al quitar el grupo, HGB, mismo split):

| Grupo | Features | ΔAUC al quitarlo | Qué aporta |
|---|---|---|---|
| **C mercado** | `has_ml`, `odds_log_ratio`, `mkt_p_home`, `expected_ft_margin`, `cur_vs_expected`, `mkt_margin_spread`*, `total_line`* | **0.0134** | Prior pre-partido más fuerte (corr +0.18, AUC uni 0.606). Todo lo demás es secundario |
| B fuerza | `d_elo_ft/q4`, `d_wr15`, `d_ft_margin15`, `d_q4_bayes`, `d_q4_net12`, `d_q3_net12`, `d_season_wr`, `h2h_*`, `elo_reliability` | 0.0026 | Calidad cuando no hay cuota (36-40 % de partidos). `d_ft_margin15` es la #2 en ganancia XGB (11.7 %), AUC uni 0.605 |
| D liga | `lg_avg_*`, `lg_q4_vol`, `lg_home_q4_rate`, `is_women` | 0.0011 | Escala de varianza y sesgo local por liga |
| A estado | marcador, momentum Q3, ventanas 2/5 min, ritmo, `gp_*`, `lead_z` | 0.0008 | Contexto; mejor AUC en blowouts |

\* Spread/total: **0 % de cobertura en train** (la línea no se guardaba) → hoy inertes; se activarán al terminar el backfill (§4).

**Lectura honesta:** las ablaciones individuales son ≈ 0 (alta redundancia); el valor real viene del bloque de mercado y de que los priors y el estado ahora se calculan igual en entrenamiento y en vivo. El AUC +0.015 sobre v5_2 sale de la suma de: más filas de entrenamiento (multi-snapshot), NaN en vez de 0, priors de margen (`d_ft_margin15`), calibración OOT y simetría — no de una sola feature milagrosa. Features nuevas por correlación univariada: `d_ft_margin15` (+0.192), `d_season_wr` (+0.169), `lead_z` (+0.090), `recent_pts_diff_5m` (+0.032).

Descartadas con evidencia (ver `findings_m27_2.md`): faltas/timeouts/pérdidas/triples acumulados de Q1-Q3 y calendario/descanso (AUC +0.0003 o negativo); roster ELO (−0.0006); `ts_delta_wp_bayes` (look-ahead).

---

## 4. Resiliencia al minuto (el punto débil de m27_v3)

Holdout C, partidos con eventos (umbral operativo 0.75):

| Minuto del snapshot | Filas | AUC | Apuestas | WR |
|---|---|---|---|---|
| 25.5–26.5 | 13,422 | 0.6606 | 1,220 | 81.4 % |
| 26.5–27.5 | 13,319 | 0.6544 | 1,199 | 79.9 % |
| 27.5–28.5 | 13,366 | 0.6566 | 1,211 | 79.9 % |
| 28.5–29.5 | 13,365 | 0.6504 | 1,144 | 80.1 % |
| 29.5–30 | 38,370 | 0.6547 | 3,213 | 81.4 % |

AUC entre 0.650 y 0.661: **no se desploma por el minuto**. Política de disparo (partidos con eventos, umbral 0.75): primer disparo 80.3 % WR (+12.4 % ROI@1.40), solo cierre de Q3 81.3 % (+13.8 %), tick aleatorio 80.5 % (+12.7 %) → da igual cuándo caiga el tick. Robustez probada (`audit_feature_parity.py`): sin eventos, sin gráfica, sin odds, equipo sin historial, Q1/Q2 ausentes (se derivan de eventos), ya en Q4, reloj en 27.4, liga 12m (no soportada), liga tóxica (PASS). **Fuera de [25.5, 30] devuelve `unavailable`**: antes de 25.5 por poco y **después de 30** (no solo antes de 25), porque más allá del cierre de Q3 el modelo predice el ganador de un Q4 ya empezado con datos de Q3.

Walk-forward (train ≤ año anterior, umbral 0.75):

| Corte | AUC | Apuestas | WR | ROI@1.40 |
|---|---|---|---|---|
| Train ≤ 2022 → Test 2023 | 0.6276 | 365 | 78.4 % | **+9.7 %** (v5_2: −2.8 %) |
| Train ≤ 2023 → Test 2024 | 0.6369 | 2,935 | 80.4 % | **+12.5 %** |
| Train ≤ 2024 → Test 2025-26 | 0.6483 | 13,681 | 79.9 % | **+11.8 %** |

Calibración (C): sesgo ≤ ±2.9 % en todo el rango, y **plano** en el tramo operativo (0.7,0.8] → **+1.2 %** (antes +2.4 % con la isotónica). Solo las colas de 0.9+ quedan subestimadas (−2.9 %, N pequeño).

### Comparativo por el camino LIVE (2,500 partidos del holdout, `compare_v5_2_vs_v5_3_live_path.py`)

| Min 27 | AUC | Brier | P ≥ 0.68 | P ≥ 0.72 |
|---|---|---|---|---|
| v5_2 | 0.6328 | 0.2353 | 257 apuestas, WR 75.5 % | 143, WR 83.9 % |
| **v5_3** | **0.6459** | **0.2333** | 338 (+32 %), WR 76.3 % | 236 (+65 %), WR 81.4 % |

Muestra pequeña (IC95 del WR ≈ ±5 pp): lo sólido es **AUC +0.013 y +30–65 % de volumen**; las diferencias de WR están dentro del ruido. El estado de v5_2 incluye los resultados del holdout (ventaja para v5_2); v5_3 usa el estado desde el inicio del holdout (desventaja).

---

## 5. Datos: qué falta, qué está en línea y cómo completarlo

Auditoría (`tools/backfill_missing_details.py audit/complete` + sondeo propio, 251,466 partidos):

| Dato | Falta | ¿Existe en línea? | ¿Lo usa v5_3? | Acción |
|---|---|---|---|---|
| Cuotas ML | 36 % (90,846) | **No: 0/40 en el sondeo (404)** | sí | Nada que descargar |
| **Línea de spread/total** | 17,178 partidos con precio pero **sin línea** | **Sí: 40/40** | sí (inerte hasta completar) | **Backfill (script abajo)** |
| Cuotas de partidos monitoreados (pudieron guardarse in-play) | 5,122 | 22/40 (55 %) | sí | Backfill `monitored` |
| `match_events` con `time_seconds` | solo existe 2022+ (≈ 88 k partidos válidos) | — | sí | Pre-2022: solo anclas |
| `graph_points` | 39 % | no sondeado | sí (NaN) | Opcional |
| `play_by_play` | 36 % | — | no | — |
| `lineups`/`player_stats` | 53 % | — | no | Sin valor medido |
| `team_statistics` | 27 % | — | no | Probado: sin señal |
| Cuota **Q4** / in-play | no existe histórico | solo en vivo | no (se registra) | `live_snapshots_v3` |

**Script nuevo:** [`tools/backfill_odds_lines.py`](../tools/backfill_odds_lines.py) — descarga solo `event/{id}/odds/1/all` (1 petición/partido, 1 req/s), guarda `line`, `home_initial`, `away_initial` solo con mercados pre-partido, es reanudable (tabla `odds_backfill_state`) y no re-descarga nada más.

```powershell
.venv\Scripts\python.exe tools\backfill_odds_lines.py audit --since 2023-01-01
.venv\Scripts\python.exe tools\backfill_odds_lines.py probe --mode missing --since 2024-01-01 --n 60   # mide cuántos existen en línea
.venv\Scripts\python.exe tools\backfill_odds_lines.py run --mode lines --spacing 1.0          # ~17k partidos, ~5-6 h
.venv\Scripts\python.exe tools\backfill_odds_lines.py run --mode monitored --spacing 1.0      # ~5k partidos, ~1.5 h
```

Estado al cerrar este documento: `--mode lines` en ejecución (≈2,050 de 17,178 hechos; si el proceso se interrumpe, relanzar el mismo comando: continúa donde quedó). **Al terminar:** reconstruir y reentrenar para activar `mkt_margin_spread` y `total_line`:

```powershell
.venv\Scripts\python.exe tmp\m27_v5_3\build_dataset.py
.venv\Scripts\python.exe tmp\m27_v5_3\train_m27_v5_3_pipeline.py        # ~20 min
.venv\Scripts\python.exe tmp\m27_v5_3\audit_feature_parity.py
```

Cobertura esperada de spread/total: ~8-18 % de los partidos 2024-26 (solo ligas con ese mercado). Mejora esperada: modesta; el valor es mayor en esas ligas.

### Captura en vivo (nuevo en monitor_v3)
- `_parse_odds` ignora mercados `isLive`; `fetch_full_match` agrega `odds_live` (ML en vivo, línea de spread/total implícita).
- El watcher guarda cuotas/líneas **pre-partido** al despertar (`_capture_pregame_odds`) y el evaluator las vuelve a inyectar al payload (en vivo SofaScore solo ofrece líneas in-play). `save_match_odds` ya no borra líneas pre-partido guardadas.
- Tabla **`live_snapshots_v3`** (1 fila por minuto de juego): marcador, `odds_live`, jugadores (minutos, puntos, faltas, +/−, rating, titular) y estadísticas de equipo. Permite en 4-8 semanas entrenar con cuotas in-play, fatiga y faltas de estrellas, que hoy no son backtesteables. El ML in-play con el marcador ya descontado es, además, el verdadero baseline para calcular EV real.
- `PriorState` se refresca cada ~30 min desde `matches.db` (`refresh_state`).

---

## 6. Puesta en producción y checklist (AGENTS.md regla 6)

1. ✅ Métricas completas (AUC train/calib/test, Brier, LogLoss, Acc, gap) — `m27_v5_3_summary.json`.
2. ✅ Gráficas: curvas de aprendizaje, calibración, importancia, walk-forward, resiliencia por minuto, ROI vs umbral — `m27_v5_3_diagnostic_dashboard.png`.
3. ✅ Walk-forward 3 cortes.
4. ✅ Matriz por liga (anchor, IC Wilson, sesgo, bandera de bloqueo) y ROI a 1.40 / 1.83 — `league_calibration.json`. Con el filtro de formatos aplicado, **ninguna liga** queda por debajo del equilibrio con la evidencia actual (0 bloqueadas por medición).
5. ✅ Integración: `models/registry.py` (`m27_v5_3`), `monitor_v3/config/constants.py` (`ACTIVE_MODELS`, primero), `evaluator.py` (precarga, ventana Q3, refresco de priors y **captura de inferencia por minuto**). La poda de formatos vive en `models/common/competition_stage.py` + `models/m27_v5_3/features.py` (`BLOCKED_STAGE_CATEGORIES`), **no** en `model_config.json`.
6. ✅ Paridad offline vs live: `audit_feature_parity.py` → PARIDAD OK.
7. ✅ Comparativo y siguientes pasos (§1, §4, §7).

**Configuración en producción (2026-10-20):**

| Parámetro | Valor | Dónde |
|---|---|---|
| `min_conf` | **0.75** (`OPERATING_CONF`, decisión explícita; el pipeline exige WR ≥ 78 % en B) | `model_outputs/model_config.json` |
| `min_margin` | 0.0 | ídem |
| Ventana de inferencia | **[25.5, 30.0]** — fuera, `unavailable` | `models/m27_v5_3/predict.py` |
| Formatos bloqueados | amistosos, 2ª fase, promoción, preliminar, colocación (con **motivo** en el log) | `models/common/competition_stage.py` |
| Ligas de 12 min | `unavailable` (NBA, G League, CBA, PBA…) | `features.py` |
| Modo sombra | `m27_v5_3` se registra pero **no alerta** en Telegram | `constants.SHADOW_MODELS` |
| Serie por minuto | `model_inference_v3`: una fila por modelo y minuto dentro de la ventana | `evaluator.py` |

> ℹ️ `m27_v5_3` está **primero** en `ACTIVE_MODELS` pero en **modo sombra** (`SHADOW_MODELS`): sus apuestas se registran en `bet_monitor_log_v3` y en `model_inference_v3`, pero no disparan alerta. Motivo: 40 apuestas resueltas con WR 72.5 % (IC95 56-85 %) no demuestran ventaja en vivo. Se reactiva quitándolo de `SHADOW_MODELS` cuando haya ≥ 200 apuestas resueltas.
>
> 📊 **Comparación live ya hecha (2026-10-18 → 19, modelo anterior):** 40 apuestas, WR 72.5 % frente al 77.8 % de backtest → dentro del ruido, y frente a `m27_v5_2` (11 apuestas, 82 %) la diferencia **no es significativa** (Fisher p = 0.706). En los snapshots compartidos `v5_3` era **mejor** que `v5_2` (Brier 0.1907 vs 0.1976). El gap lo explicaba la **regla de disparo** (`min_conf 0.72` sin filtro de margen), las **inferencias con minuto > 30** y la **lista de ligas tóxicas cuyos patrones no coincidían con los nombres reales**. Todo corregido; detalle y contrfactuales: [`ANALISIS_LIVE_M27_V5_3.md`](./ANALISIS_LIVE_M27_V5_3.md).

---

## 7. Siguientes pasos

1. **Registrar la cuota Q4/in-play y calcular EV real; mover el umbral de P a EV.** Es lo único que falta para dejar de decidir a ciegas: la isotónica tiene una meseta (0.75 ≈ 0.80) y el óptimo de beneficio se mueve de 0.72 a 0.65 según la cuota sea 1.40 o 1.83. `odds_live` ya se guarda en `live_snapshots_v3`; falta el join con `bet_monitor_log_v3`.
2. **Terminar el backfill de líneas** y reentrenar (§5). Medir ΔAUC de `mkt_margin_spread`/`total_line` en las ligas con mercado.
3. **4-8 semanas de `model_inference_v3`** → calibración y acierto **por minuto** dentro de la ventana válida, y residuo con fatiga/faltas/+− de jugadores y cuota in-play (meta ΔAUC ≥ +0.005 fuera de muestra). Es la serie que faltaba para analizar el modelo por minuto (hoy `bet_monitor_log_v3` guarda una sola fila por partido y modelo).
4. **Volver a auditar la lista negra con la evidencia nueva**: con el filtro de formatos aplicado, la calibración actual deja **0 ligas** por debajo del equilibrio; el bloqueo por medición queda vacío y se repoblará solo cuando haya evidencia.
5. Backfill de `graph_points` (39 % faltante, solo si la ablación de `gp_*` lo justifica; hoy ΔAUC +0.0005).
6. Extender a 12m (NBA/CBA/PBA) con su propio modelo (roadmap `m34_*`); `m27_v5_3` los rechaza explícitamente.
7. Revisar mensualmente `league_calibration.json` para actualizar los bloqueos con evidencia (IC95 del WR < 71.4 %).

---

> 🔎 **Auditoría en producción:** dónde pierde realmente el modelo, barrido de `min_conf` y guards → [`AUDITORIA_M27_V5_3_PERDIDAS.md`](./AUDITORIA_M27_V5_3_PERDIDAS.md).
> 📉 **Análisis live completo (v5_2 vs v5_3, lista negra, ablation por fase):** [`ANALISIS_LIVE_M27_V5_3.md`](./ANALISIS_LIVE_M27_V5_3.md).

---

> ?? **Auditor?a en producci?n:** d?nde pierde realmente el modelo, barrido de `min_conf` y guards ? [`AUDITORIA_M27_V5_3_PERDIDAS.md`](./AUDITORIA_M27_V5_3_PERDIDAS.md).
