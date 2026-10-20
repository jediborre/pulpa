# 🔬 m27_v5_3 — qué arregla, por qué cada feature, datos que faltan y cómo se mide

> Entrenado el 2026-10-18. Código: [`models/m27_v5_3/`](../models/m27_v5_3) · Pipeline: [`tmp/m27_v5_3/`](../tmp/m27_v5_3) · Diagnóstico previo: [`findings_m27_2.md`](./findings_m27_2.md).
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
| Calibración | isotónica in-sample | isotónica **fuera de muestra** (bloque B), simétrica, techo 0.95 |
| Umbral | P ≥ 0.68 elegido sobre test | elegido en **B**, test C intacto |

Resultados sobre el holdout C (último 18 % de 197,111 partidos elegibles, desde 2025-11-10; **98,476 filas / 35,480 partidos ancla**):

| Métrica | v5_2 (reportado) | **v5_3** |
|---|---|---|
| AUC train / calib (B) / **test** | 0.6368 / – / 0.6376 | 0.6454 / 0.6540 / **0.6530** |
| Brecha train−test | −0.0007 | −0.0076 (sin sobreajuste) |
| Brier / LogLoss / Acc (test) | 0.2350 / 0.6623 / 59.6 % | **0.2308 / 0.6529 / 61.1 %** |
| MAE margen Q4 | – | 6.20 pts |

Tabla de umbrales en C, **solo ancla (t=30, un partido = una fila)**, comparable con v5_2:

| P ≥ | Apuestas | % partidos | WR (IC95) | ROI @1.40 | ROI @1.83 |
|---|---|---|---|---|---|
| 0.65 | 11,126 | 31.4 % | 70.9 % (70.1–71.8) | −0.7 % | +29.8 % |
| **0.68** | 6,156 | 17.4 % | 75.5 % (74.4–76.5) | +5.6 % | +38.1 % |
| 0.72 *(operativo, elegido en B)* | 4,563 | 12.9 % | 77.8 % (76.5–78.9) | +8.9 % | +42.3 % |
| 0.75 | 3,627 | 10.2 % | 79.3 % (77.9–80.6) | +11.0 % | +45.1 % |
| 0.80 | 1,586 | 4.5 % | 84.9 % (83.1–86.6) | +18.9 % | +55.4 % |

v5_2 a P ≥ 0.68: 5,555 apuestas, WR 73.4 %, ROI@1.40 +2.7 %. La isotónica deja mesetas (0.68 ≈ 0.70), por eso los escalones no son lineales.

> ⚠️ Honestidad: el equilibrio a cuota 1.40 es 71.4 % de WR. A P ≥ 0.65 el ROI es negativo; el borde real solo aparece desde ~0.68. La cuota real de Q4 no se registra aún (ver §5), así que el ROI es teórico.

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

Holdout C, partidos con eventos (umbral operativo 0.72):

| Minuto del snapshot | Filas | AUC | Apuestas | WR |
|---|---|---|---|---|
| 25.5–26.5 | 14,390 | 0.6559 | 14.0 % | 78.8 % |
| 26.5–27.5 | 14,252 | 0.6510 | 13.8 % | 77.6 % |
| 27.5–28.5 | 14,338 | 0.6540 | 14.0 % | 76.8 % |
| 28.5–29.5 | 14,412 | 0.6490 | 13.6 % | 77.1 % |
| 29.5–30 | 41,084 | 0.6538 | 13.0 % | 77.9 % |

AUC entre 0.649 y 0.656: **no se desploma por el minuto**. Política de disparo (partidos con eventos): primer disparo 77.1 % WR (+7.9 % ROI@1.40), solo cierre de Q3 77.7 % (+8.8 %), tick aleatorio 77.5 % → da igual cuándo caiga el tick. Robustez probada (`audit_feature_parity.py`): sin eventos, sin gráfica, sin odds, equipo sin historial, Q1/Q2 ausentes (se derivan de eventos), ya en Q4, reloj en 27.4, liga 12m (no soportada), liga tóxica (PASS). Antes de min 25 devuelve `unavailable`.

Walk-forward (train ≤ año anterior, umbral 0.72):

| Corte | AUC | Apuestas | WR | ROI@1.40 |
|---|---|---|---|---|
| Train ≤ 2022 → Test 2023 | 0.6288 | 1,157 | 74.9 % | **+4.9 %** (v5_2: −2.8 %) |
| Train ≤ 2023 → Test 2024 | 0.6370 | 7,090 | 75.9 % | +6.3 % |
| Train ≤ 2024 → Test 2025-26 | 0.6484 | 22,942 | 77.3 % | +8.2 % |

Calibración (C): sesgo ≤ ±2.4 % entre 0.1–0.9; leve sobreestimación en (0.7,0.8] (+2.4 %) y subestimación en las colas (0–0.1: −6.3 %, filas con N pequeño).

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
4. ✅ Matriz por liga (anchor, IC Wilson, sesgo, bandera de bloqueo) y ROI a 1.40 / 1.83 — `league_calibration.json`.
5. ✅ Integración: `models/registry.py` (`m27_v5_3`), `monitor_v3/config/constants.py` (`ACTIVE_MODELS`, primero), `evaluator.py` (precarga, ventana Q3 y refresco de priors). `leagues.yaml` no cambia (la poda de tóxicas vive en `model_config.json`).
6. ✅ Paridad offline vs live: `audit_feature_parity.py` → PARIDAD OK.
7. ✅ Comparativo y siguientes pasos (§1, §4, §7).

Umbral en producción: `model_config.json` → `min_conf = 0.72`, `min_margin = 0.0` (elegido en B). Ligas tóxicas → `PASS`. Ligas de 12 min → `unavailable`.

> ℹ️ `m27_v5_3` quedó **primero** en `ACTIVE_MODELS` y, si emite `BET`, el monitor lo notificará como cualquier otro modelo. Para observarlo sin notificar, quitarlo de `ACTIVE_MODELS` o dejarlo registrando en `bet_monitor_log_v3` unas semanas (recomendado: comparar WR live contra el 77.8 % de backtest antes de operar con dinero).
>
> 📊 **Comparación live ya hecha (2026-10-18 → 19):** 40 apuestas, WR 72.5 % (IC95 ≈ 56-85 %) frente al 77.8 % de backtest → dentro del ruido, y frente a `m27_v5_2` (11 apuestas, 82 %) la diferencia **no es significativa** (Fisher p = 0.706). En los snapshots compartidos `v5_3` es **mejor** que `v5_2` (Brier 0.1907 vs 0.1976). El gap lo explica la **regla de disparo** (`min_conf 0.72` sin filtro de margen), las **inferencias con minuto > 30** (fuera del rango de entrenamiento) y la **lista de ligas tóxicas cuyos patrones no coinciden con los nombres reales**. Detalle y contrfactuales: [`ANALISIS_LIVE_M27_V5_3.md`](./ANALISIS_LIVE_M27_V5_3.md).

---

## 7. Siguientes pasos

1. **Terminar el backfill de líneas** y reentrenar (§5). Medir ΔAUC de `mkt_margin_spread`/`total_line` en las ligas con mercado.
2. **Registrar la cuota Q4/in-play** (ya se guarda `odds_live`) y calcular EV real; mover el umbral de P a EV.
3. **4-8 semanas de `live_snapshots_v3`** → residuo sobre m27_v5_3 con fatiga/faltas/+- de jugadores y cuota in-play (meta ΔAUC ≥ +0.005 fuera de muestra).
4. Backfill de `graph_points` (39 % faltante, solo si la ablación de `gp_*` lo justifica; hoy ΔAUC +0.0005).
5. Extender a 12m (NBA/CBA/PBA) con su propio modelo (roadmap `m34_*`); `m27_v5_3` los rechaza explícitamente.
6. Revisar mensualmente `league_calibration.json` para actualizar los bloqueos con evidencia (IC95 del WR < 71.4 %).

---

> ?? **Auditor?a en producci?n:** d?nde pierde realmente el modelo, barrido de `min_conf` y guards ? [`AUDITORIA_M27_V5_3_PERDIDAS.md`](./AUDITORIA_M27_V5_3_PERDIDAS.md).
