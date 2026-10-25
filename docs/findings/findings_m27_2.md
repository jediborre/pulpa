# 🔬 Findings m27_v5_2 → recomendaciones para m27_v3 (arquitectura, skew train/live, features)

> Auditoría del 2026-10-18 sobre `models/m27_v5_2`, `tmp/m27_v5/train_m27_v5_pipeline.py`, `tmp/m27_v5_2/train_m27_v5_2_pipeline.py`, `monitor_v3/models/evaluator.py` y `matches.db`.
> Scripts reproducibles en [`tmp/m27_v5_2_audit/`](../tmp/m27_v5_2_audit) (a1…a13). Todas las cifras salen de esos scripts (mismo split cronológico 82/18, 187,692 partidos limpios, HGB con los hiperparámetros de producción).

---

## 0. Resumen ejecutivo

1. **El techo del modelo no es de arquitectura, es de información.** Con los mismos datos, cambiar la arquitectura (liga, normalización, simetría espejo, cabeza gaussiana, LR vs. árboles) mueve el AUC entre 0.632 y 0.638. Todo converge a ~0.637. Para subir el win rate hay que **(a) arreglar el skew train→live** y **(b) agregar información nueva** (mercado por spread/total, odds iniciales, telemetría live acumulada), no re-podar features.
2. **El backtest no describe lo que corre en vivo.** El extractor live de `features.py` produce vectores distintos a los del entrenamiento en ≥10 de las 27 features (3 de ellas siempre en 0). Simulando lo que el extractor live realmente entrega, el **volumen cae −65 % (4,620 → ~1,500 apuestas) y el AUC −0.015** (§2).
3. **El “snapshot 27” es en realidad snapshot 30.** El dataset de entrenamiento usa Q1+Q2+Q3 completos (min 30) para `cur_score_diff`, `pace`, `gp_val`, y un proxy (`q3_diff*0.20`) para `recent_2m_run_diff`. En vivo a min 27 el modelo ve ~10 % menos marcador (§2.3).
4. **La mayor parte de las 27 features es redundante.** Quitar todo el bloque ELO/roster (8 features) baja el AUC **0.0006**; quitar todo el marcador/momentum (11) **0.0004**; quitar el bloque situacional **0.0011**. Solo mercado+h2h+standing (4 features) ya da 0.624 de 0.637. La “importancia 13.77 % de `d_lineup_starters_elo_ft`” en `modelos.md` es una ilusión de colinealidad (ablación = −0.0003 AUC).
5. **El win rate de ~74 % viene de dos priors apilados: favorito pre-partido + blowout.** El 55 % de las apuestas verdes están en |cur_diff| > 20. Reglas simples al mismo volumen: favorito pre-partido 67.8 %, líder tras Q3 63.1 %; el modelo combina ambos y llega a 74.4 %. Con cuota 1.40 el punto de equilibrio es 71.4 % → **el edge real es ~+3 pp y depende de la cuota real de Q4, que no se está registrando**.

---

## 1. Diagnóstico del “problema”: por qué v5 → v5_1 → v5_2 no suben

| Modelo | AUC (35.7k holdout) | Lectura |
|---|---|---|
| m27_v5 (37f) | 0.6381 | Techo real de la información disponible |
| m27_v5_1 (25f) | 0.6307 | Quitar `odds_log_ratio` y `ts_delta_wp` costó −0.007 |
| m27_v5_2 (27f) | 0.6361 | Los restauró y recuperó ~todo |

Ablaciones propias (HGB, mismo split; `a1_dataset_audit.py`):

| Conjunto | AUC | Brier |
|---|---|---|
| 27 features | 0.6367 | 0.2351 |
| solo marcador/momentum (11) | 0.5531 | 0.2461 |
| solo ELO/roster/estrategia (8) | 0.6065 | 0.2410 |
| solo mercado/h2h/standing (4) | **0.6241** | 0.2378 |
| solo `expected_ft_margin` (1) | 0.6141 | 0.2400 |
| `cur_diff` + `expected_ft_margin` (2) | 0.6225 | 0.2381 |
| 5 features top (`cur_diff`, `efm`, `odds`, `d_wr`, `starters_elo_ft`) | 0.6265 | 0.2372 |
| 27 − marcador / − ELO / − situacional | 0.6363 / 0.6361 / 0.6356 | — |
| 27 − `odds_log_ratio` / − `ts_delta_wp` | 0.6345 / 0.6347 | — |

**Conclusión:** el modelo = *prior de mercado* + *marcador*. Lo demás (roster ELO, pace, `gp_val`, `rubber_band`, `recent_2m`) no aporta medible. Agregar/podar más de lo mismo no mueve la aguja; por eso v5_1 y v5_2 oscilan ±0.007.

Otros hechos:
- **Techo teórico:** SD del margen Q4 = 7.7 pts. Para P ≥ 0.68 se necesita un margen esperado de +3.6 pts; para P ≥ 0.75, +5.2 pts. Solo los blowouts lo permiten.
- **Por régimen:** AUC del modelo = 0.61–0.63 para |cur_diff| < 20 y **0.695 para ≥ 20**. Base rate del local en Q4 con cur_diff ∈ (−20,20) ≈ 0.47–0.51 (sin reversión a la media); con > +20 sube a 0.60, con < −20 baja a 0.35.
- **Composición de apuestas verdes (raw P ≥ .68/≤ .32):** AWAY con cur_diff < −20 = 1,009 de 2,392; HOME con > +20 = 830 de 1,813. Fuera de blowouts, WR 64–87 % con N pequeño y mucha varianza.
- **Calibración isotónica in-sample NO es el problema** (Brier 0.2354 in-sample vs 0.2355 out-of-time; raw 0.2352). Descartado.
- **Cola alta sobreestimada:** deciles (0.8,0.9] y (0.9,1.0] con sesgo +6.5 % y +5.2 % → el `y_max=0.98` del isotónico es optimista; usar techo ~0.90.
- Ver §4: la ganancia de los modelos “con más info” en docs (`m27_v5` AUC 0.6708/Brier 0.2246) **no es reproducible** sobre el holdout actual (0.638/0.235); esos números del doc deben considerarse obsoletos.

---

## 2. Train-serving skew (lo más grave y lo más barato de arreglar)

Medido con `a5_parity.py` (250 partidos 2025+, extractor live vs fila del cache) y `a13_skew_impact.py` (impacto en métricas).

### 2.1 Bugs de código en `models/m27_v5_2/features.py`

| # | Feature | Qué pasa en vivo | Evidencia |
|---|---|---|---|
| 1 | `recent_2m_run_diff` | **Siempre 0.** Lee `recent_window_features(...).get("run_diff")` pero la clave devuelta es `recent_run_diff` (`prefix_run_diff`). | live std 0.0, 100 % ceros |
| 2 | `d_strat_q4_net` | **Siempre 0.** Busca `team_priors[t]["q_profile"]["q4_net_avg"]`, pero el state guardado tiene `q4_net_hist`/`q3_net_hist`. 0 de 7,645 equipos tienen `q_profile`. | `a8_state_and_time.py` |
| 3 | `h2h_bayes_ft_margin` | **Siempre 0.** El monitor entrega `parsed["h2h"]` como *lista* de partidos; el extractor espera un *dict* con `ft_margin_avg`. | `mobile_client.fetch_full_match` vs `features.py` |
| 4 | `ts_delta_wp_bayes` | **Siempre 0.** El evaluator inyecta `team_strength` como *lista* de filas; el extractor espera `{"home":…, "away":…}`. | `evaluator.py` L127 |
| 5 | `pbp_3pt_diff_q1_q3` | Si `play_by_play` está vacío, `pbp_events_upto([])` recibe una lista → `AttributeError` → predicción `unavailable`. Además la feature es **constante 0 en 100 % del train** (se puede borrar). | traceback a5 |
| 6 | `rubber_band_gap` | Fórmula distinta: train `cur − 0.75·efm` (std 13.4); live `−0.15·cur` si \|cur\|≥14 (std 1.9). corr −0.73. | a5 |
| 7 | `white_flag_risk_zone` | Train: 1 si \|cur\| ≥ 18 (hombres) / 24 (mujeres) / 14 (12m); live: ±1 si \|cur\| ≥ 22. | a5 |
| 8 | `d_lineup_starters_elo_ft` | Live = `starters_elo_q4 × 1.15` (ELO **Q4** reescalado); train = ELO **FT** de los jugadores. corr live/train = 0.69. | a5 |
| 9 | `d_wr_bayes`, `d_q4_bayes` | Train = win-rate / margen Q4 bayesiano de los últimos 15 juegos; live = `d_elo_ft/180`, `d_elo_q4/35`. corr 0.76 / 0.51. | a5 |
| 10 | `expected_ft_margin` | Train: `6.5·olr + 0.5·d_elo_ft/28`; live: `6.5·olr` (con odds) o `d_elo/25` (sin odds). corr 0.70. | a5 |
| 11 | `d_roster_drift_adv`, `d_lineup_top3/bench` | Train usa top-5 por ELO+rating+minutos **entre los que jugaron** (box score final); live usa `is_starter` de `lineups`. Definiciones distintas de “titular”. | `train_m27_v5_pipeline.py` vs `features.py` |
| 12 | `odds_log_ratio` | `_parse_odds` prefiere `fractionalValue` (cuota **actual**, en min 27 posiblemente in-play) sobre `initialFractionalValue`. Train son cuotas pre-partido (AUC vs ganador FT 0.763, estable 2023-26). Riesgo de odds que ya incorporan el marcador. | `match/scraper.py::_parse_odds` |

### 2.2 Impacto cuantificado (`a13_skew_impact.py`, mismo modelo, test reemplazado por lo que produce el extractor live)

| Paso | AUC | Apuestas P≥.68 / ≤.32 | WR |
|---|---|---|---|
| 0. test limpio (backtest) | 0.6367 | 4,620 | 74.4 % |
| 1–2. + `recent_2m`=0, `d_strat_q4_net`=0 | 0.6360 | 3,709 | 75.8 % |
| 3–4. + `h2h`=0, `ts`=0 | 0.6331 | 2,660 | 76.9 % |
| 5. + `rubber_band` fórmula live | 0.6217 | 1,655 | 76.0 % |
| 7. + escala ELO en `d_wr`/`d_q4` | 0.6212 | 1,533 | 76.0 % |
| 8. + sin odds | 0.6211 | 1,315 | 76.7 % |

**Efecto neto:** el win rate sube un poco (el modelo se vuelve más conservador) pero **el volumen se desploma −65 %**. Esto explica que “a P≥0.68 salgan muy pocas verdes” en vivo y que el equipo haya subido el umbral a 0.75 en `predict.py` (2,520 apuestas en backtest, mucho menos en vivo). `rubber_band_gap` es el bug que más daña (−0.011 AUC, −38 % de volumen).

### 2.3 Definición del snapshot
- `train_m27_v5_pipeline.py`: `cur_diff = q1+q2+q3` (min **30** completo), `pace = pts_3Q/30`, `gp_val` de minuto 30, `recent_2m_run_diff = q3_diff·0.20` (proxy, no real).
- `extract_features_v5_2(target_minute=27)`: `score_upto(27)` = ~97.6 pts vs 115.1 reales al min 30 (−15 %), `pace` con denominador 27, 19 % de `cur_score_diff` en 0 por eventos sin `time_seconds` vs 2 % en train.
- `pace_vs_league_prior`: live −0.76 vs train −0.03; `pace_pts_per_min` corr 0.37.
- El evaluator mantiene la señal en “monitoring_q3_window_for_green” hasta min 30, por lo que en la práctica la inferencia que dispara suele ser la de min ≥ 29-30; conviene **declarar y entrenar explícitamente el snapshot real** (30) o entrenar con features recalculadas a min 27 desde `match_events`.

### 2.4 Leakage / look-ahead en entrenamiento
- **`team_strength` no es as-of.** `fetched_at` de las 451,512 filas está entre 2026-10-12 y 2026-10-18 para partidos desde 2015: es el *standings del día del backfill*, no el previo al partido. 2019 = 0 % cobertura. `ts_delta_wp_bayes` tiene AUC 0.6275 contra el ganador FT: parte es persistencia de calidad (información del futuro). En vivo sí es as-of → distribución distinta.
- **Push excluido:** `q4_diff == 0` se descarta del train/test. En vivo existen `push` (~3 % en `bet_monitor_log_v3`). El ROI debe computarse con push = stake devuelto sobre todos los partidos.
- **Umbral y mix elegidos sobre el test** (0.65–0.80 se barrió en el holdout). Para reportar WR usar un tercer corte (validación para elegir umbral; test intacto).
- **Gate redundante:** `p_xgb > 0.5` y `exp_margin ≥ 1.8` casi nunca desempatan frente a `P_cal ≥ 0.68`.
- **Ruido de dataset:** brecha 2020-2022 sin datos y ELO con decaimiento por gaps >300 días; Corte 1 del walk-forward (train ≤2019 → test 2023) AUC 0.614, WR 69.4 %, ROI@1.40 −2.8 %: el modelo no es estable ante discontinuidades.

### 2.5 Hallazgos operativos
- En `bet_monitor_log_v3` (188 filas, 2026-10-04…05) **solo hay `m27_v3` y `v6_2`**: no hay un solo registro de `m27_v5_x` en producción. No existe aún evidencia live de v5_2. Antes de optimizar, correr `m27_v5_2` en modo sombra y registrar el vector de features.
- `git status` muestra `models/m27_v5_2/predict.py` modificado (umbral 0.75): `modelos.md` documenta 0.68.

---

## 3. Recomendaciones de arquitectura para m27_v3 (orden por impacto esperado)

### P0 — Arreglar el contrato train/live (sin esto, todo lo demás se mide mal)
1. **Una sola función de features** `build_features(payload, minute)` usada por entrenamiento y live. El pipeline de entrenamiento debe construir sus filas llamando al mismo extractor sobre los datos de `matches.db` (más lento, pero elimina el skew por construcción). Test de paridad automático: N partidos, diferencia máxima < 1e-4 por feature; falla el CI si no pasa.
2. Corregir los puntos 1-12 de §2.1. Orden de impacto por apuesta perdida: `rubber_band_gap` → `h2h`/`ts` (adaptadores lista→dict) → `d_strat_q4_net` (guardar `q_profile` en el state) → `recent_2m_run_diff` (`recent_run_diff`) → ELO FT/Q4 de titulares → escalas de `d_wr`/`d_q4`.
3. **Snapshot explícito:** entrenar a `minuto=27` real reconstruyendo marcador/pace/run/gp desde `match_events` (cobertura de eventos 100 %), o declarar m30 y evaluar solo cuando el reloj ≥ fin de Q3. No mezclar.
4. **As-of de `team_strength`:** reconstruirlo desde `matches` (W/L acumulado por equipo/competición hasta la fecha) en vez de usar la tabla actual.
5. **Odds iniciales:** parsear `initialFractionalValue` (apertura) y, si existe, la cuota de cierre pre-partido; guardar ambas. Nunca usar cuota in-play como `odds_log_ratio`.
6. Eliminar features muertas: `pbp_3pt_diff_q1_q3`, `white_flag_risk_zone` duplicado en `abs_cur_score_diff`, `lg_avg_q4_prior` (AUC 0.5006, solo útil como interacción).

### P1 — Cambios de modelado que sí ayudan (aunque poco), medidos
| Variante (a12) | AUC | WR @.68 | Nota |
|---|---|---|---|
| Base 27 | 0.6367 | 74.4 % | referencia |
| + identidad de liga (home Q4 rate, volatilidad, sesgo; target-encoding expanding) | 0.6375 | 74.2 % | +0.0008, aumenta volumen 3 % |
| + features normalizadas por volatilidad de liga | 0.6376 | 74.1 % | |
| Simetría espejo (aumentación + promedio) | 0.6372 | **74.5 %** | elimina sesgo local/visitante, gratis |
| Cabeza gaussiana μ/σ heterocedástica | 0.6382 | 72.9 % | mejor AUC, peor calibración |
| Promedio HGB + gaussiana | **0.6384** | 73.8 % | mayor AUC |
| LR lineal | 0.6322 | 73.1 % | −0.005 vs árboles |

Recomendación: **un solo HGB/XGB con restricciones monótonas** + simetría espejo + liga por target-encoding + cabeza de margen para `σ`; descartar el ensamble 0.4/0.35/0.25 de 3 modelos con las mismas 27 columnas (ganancia ≈ 0). Calibración isotónica fuera de muestra con techo 0.90.

### P2 — Reemplazos de ELO
- El ELO de equipo/jugador se actualiza con resultado binario y K fijo; no usa margen. Sustituir por **rating tipo Massey/ridge de margen por cuarto** (ratings ofensivo/defensivo por liga/temporada, shrinkage hacia la media de la liga) o **ELO con margen (MOV)**. El roster ELO de jugador es colineal con el ELO de equipo (ablación −0.0006 AUC): o se calcula con minutos reales (+/- por jugador, RAPM) o se retira.
- ELO “as-of” con liga como grupo (random effect) para ligas con pocos juegos; `elo_reliability` se podó pero debería interactuar con el prior.

### P3 — Selección de apuestas y evaluación
- Tres cortes: entrenamiento / validación (umbral, calibración) / test intacto. Reportar WR con intervalo de Wilson y ROI con **push** y cuota real.
- Umbral por **EV, no por P**: `EV = P·cuota_Q4 − 1 > margen`; requiere guardar la cuota real de Q4 (§4).
- Poda de ligas: reemplazar la lista manual de 16 “tóxicas” por un criterio estadístico (sesgo de calibración por liga con N ≥ 200, test binomial, expandiendo) revisado cada mes.

---

## 4. Features nuevas viables en vivo (y qué dijeron los experimentos)

### 4.1 Probadas en este análisis → **no mueven la aguja** (no invertir tiempo)
`a11_new_features_experiment.py` (78k partidos 2024+ con stats Q1-Q3, mismo HGB):

| Variante | AUC | WR |
|---|---|---|
| Base 27 | 0.6355 | 74.2 % |
| + calendario (descanso, back-to-back, partidos en 7 días) | 0.6350 | 74.4 % |
| + stats de equipo Q1-Q3 (diferencial de faltas, timeouts, pérdidas, triples, TL, 2PM, share de 3PT) | 0.6358 | 73.9 % |
| + ambas | 0.6356 | 74.9 % |

AUC univariado de esas features: 0.48–0.52; `fouls` es constante en `team_statistics` (no sirve). La idea “bonus/foul trouble/timeouts” de `modelos.md` §m27_v3 queda **descartada con evidencia** salvo con datos a nivel de jugador/posesión.

### 4.2 Con información genuinamente nueva (candidatas reales)
1. **Mercado completo pre-partido** (`match_odds`: `Point spread`, `Over/Under – Game total`, `1st half`, `1st quarter winner`): la línea de handicap y el total dan el margen esperado y el ritmo esperado directamente (`expected_margin = spread`, `expected_total`). Cobertura actual: spread/total 2024 = 6 %, 2025 = 17 %, 2026 = 18 %; ML 2025 = 63 %. Features: `spread_implied_margin`, `total_line`, `efm_per_min`, `cur_diff − 0.75·spread`, `pace_total_vs_line` (puntos reales Q1-Q3 vs total/4·3). Es la única fuente con AUC marginal esperado alto (el bloque de mercado ya aporta 0.624 solo).
2. **Movimiento de línea** (`initialFractionalValue` vs `fractionalValue`) = señal de dinero informado; requiere guardar ambas en `match_odds`.
3. **Cuota live del mercado de Q4/2ª mitad al minuto de disparo** (si SofaScore la expone): el verdadero baseline. Guardarla en el log aunque no se use como feature → permite calcular EV real y detectar cuándo el mercado ya descontó el blowout.
4. **Resultados del propio rival más cercanos y específicos:** `q4_net` y `q3_net` como *diferencial* del equipo en 3 horizontes (últ. 5 / 15 / temporada), contra el rival (`d_closer = q4_net_team − q4_net_allowed_rival`), con shrinkage por liga. El estado actual (`q4_net_hist`) ya existe; solo falta conectarlo (bug #2).
5. **Telemetría live de jugadores (flywheel de datos, no backtesteable hoy):** el endpoint `lineups` en vivo trae `secondsPlayed`, `points`, `personalFouls`, `plusMinus` por jugador a min 27. Features: minutos acumulados de los top-3 por equipo (fatiga), faltas de estrellas (4+), `plus_minus` de quienes están en cancha, puntos de banca vs. titulares en Q1-Q3. Hoy `player_stats` solo guarda el box score **final** (incluye Q4 → no sirve como feature histórica). **Acción:** persistir un snapshot de `lineups` en min 27/30 en cada partido monitoreado (tabla `live_snapshots_v3`) durante 4-8 semanas y entrenar un residuo sobre m27.
6. **Possessions/pace real** y eficiencia (pts por posesión Q3 vs Q1-Q2) desde `match_events` (tiros anotados + pérdidas + faltas ofensivas); y **racha final** con reloj real (últimos 2-3 min de Q3: `score_swing_last3`, `unanswered_points`, `time_since_last_score`). El `run_diff` actual está roto (bug #1) y el proxy de train es débil; arreglar y medir.
7. **Gráfica de presión de SofaScore:** `gp_val` hoy solo el valor puntual; agregar `gp_slope_3m/5m`, `gp_area_diff` últimos 6 min (ya existen en `graph_stats_upto`; `gp_slope_3m` se podó por entrenar con 0 en train — rehacerlo con valores reales).
8. **Contexto de competición:** fase (playoff/regular), juego decisivo, partidos restantes (`is_playoffs` de `leagues_classification` ya existe pero no entra al modelo).
9. **Interacciones explícitas para árboles poco profundos** (max_depth 4): `lead_z = cur_diff/√(pts_3Q)`, `cur − k·spread`, `efm × signo(cur)` (¿el líder también es favorito?), `blowout_leader_is_favorite`. Efecto medido pequeño (+0.0003) pero ayuda con features de mercado nuevas.

### 4.3 Qué **no** hacer
- No podar más features por importancia: es engañosa por colinealidad (ver §1).
- No re-entrenar con más ELO/roster: aporte cero medido.
- No subir el umbral para “mejorar WR” sin recontar el volumen live: con el skew actual cae a ~1,300-1,500 apuestas/test.

---

## 5. Plan sugerido para m27_v3 (siguiente iteración, con criterio de salida)

| Fase | Trabajo | Criterio de salida |
|---|---|---|
| F0 | Extractor único + test de paridad (§3-P0); correr m27_v5_2 en sombra y loguear vector + cuota Q4 | paridad < 1e-4; ≥ 300 partidos live logueados |
| F1 | Odds iniciales/spread/total + `team_strength` as-of + snapshot declarado | AUC ≥ 0.645 en holdout; volumen live/offline ≥ 85 % |
| F2 | HGB único monótono + espejo + liga + calibración OOT (techo 0.90) | Brier ≤ 0.2345; Walk-Forward estable (Corte 1 ROI@1.40 ≥ 0) |
| F3 | `live_snapshots_v3` (jugadores/minutos/faltas) 4-8 semanas → residuo | ΔAUC ≥ +0.005 OOT sobre F2 |
| F4 | Umbral por EV con cuota real, tres cortes, Protocolo AGENTS §6 completo | ROI@cuota real > 0 con IC95 % > 0 |

Expectativa realista: F0+F1 recuperan el volumen live y suben ~+0.005-0.010 AUC; con ello P≥0.70 podría sostener ~75-76 % de WR a volumen comparable a v5. Pasar de ~74 % a > 80 % solo es posible en el subconjunto ultra-selectivo (P ≥ 0.80, ~1,200 apuestas, ya observado) — no por arquitectura global.

---

## 6. Archivos de evidencia
- [`tmp/m27_v5_2_audit/a1_dataset_audit.py`](../tmp/m27_v5_2_audit/a1_dataset_audit.py): cobertura, ablaciones, calibración, régimen.
- [`a5_parity.py`](../tmp/m27_v5_2_audit/a5_parity.py): paridad de valores live vs train. [`a13_skew_impact.py`](../tmp/m27_v5_2_audit/a13_skew_impact.py): impacto del skew.
- [`a6_odds_semantics.py`](../tmp/m27_v5_2_audit/a6_odds_semantics.py), [`a7_team_strength_asof.py`](../tmp/m27_v5_2_audit/a7_team_strength_asof.py): semántica de odds y `team_strength`.
- [`a8_state_and_time.py`](../tmp/m27_v5_2_audit/a8_state_and_time.py): `q_profile` ausente y reconstrucción de marcador a min 27.
- [`a9_baselines_and_data.py`](../tmp/m27_v5_2_audit/a9_baselines_and_data.py): baselines simples, techo teórico, inventario de datos.
- [`a11_new_features_experiment.py`](../tmp/m27_v5_2_audit/a11_new_features_experiment.py), [`a12_arch_experiments.py`](../tmp/m27_v5_2_audit/a12_arch_experiments.py): features nuevas y variantes de arquitectura.
