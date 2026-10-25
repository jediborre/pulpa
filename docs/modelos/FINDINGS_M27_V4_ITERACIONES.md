# 🔬 Hallazgos de Entrenamiento, Auditoría de Fugas y Anti-Overfitting (`models/m27_v4`)

> **PROPÓSITO:** Este documento registra permanentemente todos los hallazgos cuantitativos, fugas de datos corregidas, causas raíz de overfitting/drift temporal y pruebas de estrés (cortes temporales Walk-Forward, cortes por minuto en vivo `m26..m32` y caída de endpoints) realizados durante el desarrollo del modelo de producción **`models/m27_v4`** sobre el universo histórico completo de **`227,028` partidos (`2015–2026`)**.
>
> **REGLA DE INGENIERÍA:** Antes de entrenar una futura versión (`m27_v5` o superior), **es obligatorio revisar las 6 trampas documentadas en las secciones 1 y 2** para no volver a incurrir en fugas de datos ni sobreajuste.

---

## 1. Tres Fugas de Datos (Data Leakage) Identificadas y Erradicadas

| # | Fuga Detectada | Archivo / Componente Afectado | Impacto Artificial (Antes) | Corrección Aplicada en Código (`m27_v4` y `m27_v3`) | Señal Limpia Real (Después) |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Fuga H2H del mismo día (`match_h2h`)** | `models/m27_v3/train.py` (`_compute_h2h_features`) | `match_h2h.match_date` se guardaba sin hora (`YYYY-MM-DD 00:00:00`) y `matches.match_date` con hora (`YYYY-MM-DD 19:30:00`). Al filtrar `g["dt"] < sample.dt`, el partido actual entraba en su propio historial H2H (`h2h_last_home_won` tenía $r = +0.401$ e inflaba el AUC de `0.668` a `0.789`). | Filtro estricto por día calendario `(g["dt"].date() < sample_date) and (g["match_id"] != sample_match_id)` + **Contracción Bayesiana** hacia `0.50` con peso $n / (n + 5)$. | $r = +0.1243$ con `q4_win` (100% pre-partido, cero leakage). |
| **2** | **Contaminación de reloj `12m` vs `10m` en FIBA** | `match/scraper.py` (`_period_from_game_seconds`) y `models/common/pbp_utils.py` | Se asumía que todo cuarto duraba `720s` (12 min). En partidos FIBA de 10 min (`600s`/cuarto), el minuto 27 nominal (`1620s` restantes) caía en el **minuto 1 del Q4 (`21:00` jugados)**, filtrando los primeros 60 segundos del Q4 dentro del entrenamiento "pre-Q4". | Parametrización dinámica `quarter_len_min` (`10` vs `12`) en `match/scraper.py` y `models/common/pbp_utils.py` consultando `leagues_classification`. | Corte exacto en el segundo `180` restante del Q3 (`m27` real en FIBA 10m) y `m33` en NBA 12m. |
| **3** | **Fuga Post-Partido en `player_stats`** | Propuesta previa (`PROPUESTA_NUEVO_MODELO_M27_V4.md`) | Usar `player_stats.sofascore_rating` o `minutes_played` del `match_id` actual leía las estadísticas **al finalizar el partido (FT)** ($r = +0.384$ espurio). | Prohibido leer `player_stats` del `match_id` en curso para features de entrada. En `models/m27_v4` se usa un **Rolling Pre-Match Tracker** de los últimos 10 partidos previos del equipo (`rolling_delta_plus_minus`, `rolling_top3_rating_diff`). | `rolling_delta_plus_minus` alcanza **$r = +0.2294$** con `q4_win` (la feature limpia #1 de todo el dataset). |

---

## 2. Tres Trampas de Overfitting y Drift Descubiertas en los Cortes Temporales (`V4.0` $\to$ `V4.4`)

Durante las pruebas de corte temporal (Walk-Forward) y auditoría de curvas de aprendizaje, descubrimos tres fenómenos estructurales en `matches.db` que hacían sobreajustar a los árboles GBDT tradicionales:

### 2.1 El Vacío FIBA `2019–2022` y la Necesidad de *Seasonal ELO Decay*
- **Diagnóstico:** En `matches.db`, los años `2019`, `2020`, `2021` y `2022` contienen únicamente partidos de `nba_12m` (`~2,200/año`) y **`0` partidos FIBA (`fiba_men_10m` / `fiba_women_10m`)**.
- **Efecto en `V4.0_raw`:** Cuando la historia FIBA se reanudó en `2023` (`38,167` partidos), los equipos europeos/latinoamericanos conservaban su ELO congelado de `2018` (5 años atrás). Al cortar el entrenamiento en `<= 2022` y probar en `2023`, el modelo sufría un desplome a `Test AUC = 0.5906` (`Gap = +0.1381`).
- **Solución implementada en `models/m27_v4`:** Implementamos **Decaimiento Estacional Automático (Seasonal ELO & Rolling Reset)** basado en los días de inactividad ($\Delta \text{days}$) de cada equipo:
  - Si $\Delta \text{days} > 60$ (cambio de temporada): el ELO regresa un **35% hacia `1500.0`** (`elo = 0.65 * elo + 0.35 * 1500`).
  - Si $\Delta \text{days} > 300$ (hueco multianual o ascenso/descenso): el ELO regresa un **75% hacia `1500.0`** y se limpia la ventana corta de plus/minus.
- **Resultado:** El `Corte 1` subió a **`Test AUC = 0.6268`** y **`WR@0.70 = 76.18%`** con una brecha mínima de `+0.0137`.

### 2.2 Drift de Cobertura en `team_statistics` vs Cobertura 100% en `match_events` (PBP)
- **Diagnóstico:** La tabla `team_statistics` (`period = 'Q3'`) tenía **`0.0%` de cobertura entre `2015–2021`**, `38.4%` en `2023` y `83.1%` en `2025–2026`. Además, auditamos su exactitud y solo en el `82.4%` de los partidos cuadraban los tiros anotados con `quarter_scores`.
- **Efecto en `V4.0_raw`:** El árbol aprendía un atajo falso (*covariate shift*): *"si `team_statistics` no es NaN, el partido es de 2024–2026"*, degradando la generalización entre épocas.
- **Solución implementada en `models/m27_v4`:** Eliminamos la dependencia de `team_statistics` y calculamos el perfil de tiro acumulado hasta el corte (`pbp_2pt_diff_q1_q3`, `pbp_3pt_diff_q1_q3`, `pbp_ft_diff_q1_q3`, `pbp_scoring_events_q3`) **directamente desde `match_events` (Play-by-Play)**, el cual tiene **100.0% de cobertura en los 12 años (`2015–2026`)** y **99.74% de exactitud matemática** contra `quarter_scores`.

### 2.3 Física Inversa del Q4 en `nba_12m` y el Ancla Lineal L2
- **Diagnóstico:** La dinámica del último cuarto cambia radicalmente según el arquetipo de la Tríada Canónica:
  - En **`fiba_women_10m`**: el equipo que lidera al final del Q3 **gana el Q4 el `58.9%` de las veces** ($r = +0.2016$). Domina la superioridad física/técnica.
  - En **`fiba_men_10m`**: el líder al Q3 **gana el Q4 el `51.8%` de las veces** ($r = +0.0566$). Equilibrio entre calidad del equipo y relajación por ventaja.
  - En **`nba_12m`**: el líder al Q3 **gana el Q4 solo el `44.3%` de las veces** ($r = -0.1190$). El equipo que va perdiendo al Q3 gana el Q4 el **`55.7%`** de las veces (*rubber-banding*, descanso de titulares en ventajas de 12+ puntos y ritmo de 48 minutos).
- **Efecto al apilar snapshots en `nba_12m`:** Con `28,250` partidos (5 veces menos muestra que `fiba_men_10m`), apilar 4 snapshots con árboles de profundidad 4 provocaba memorización de partidos (`Train AUC = 0.7892`, `Val AUC = 0.5943`, `Gap = +0.1949`).
- **Solución implementada en `models/m27_v4`:** Para `nba_12m` limitamos el stacking a 2 snapshots (`m33` y `m36`), restringimos los árboles a `max_depth = 3`, `max_leaf_nodes = 10`, `min_samples_leaf = 450`, `L2 = 60.0` e introdujimos un **Ancla Lineal L2 (`LogisticRegression C=0.08`) con 30% de peso en el ensemble**. La brecha Train-Val cayó de `+0.1949` a **`+0.0269`** y el Win Rate en test ($\ge 0.70$) subió de `60.16%` a **`73.86%`**.

---

## 3. Progresión Iterativa de Versiones (`V4.0_raw` $\to$ `V4.4_champion`)

Resultados auditados sobre `fiba_men_10m` ($N = 149,713$ partidos válidos con Q1–Q4 completos, split cronológico estricto 70% Train / 15% Val / 15% Test Out-of-Time):

| Versión | Descripción de Cambios | Train AUC | Val AUC | Test AUC | Brecha (`Train - Val`) | Test WR @ $P \ge 0.70$ ($N$ bets) | Yield @ $0.70$ (Cuota 1.40) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V4.0_raw`** | Hiperparámetros agresivos (`max_depth=8`, `max_leaf=63`, `min_leaf=10`, `L2=0.1`), sin ELO decay, H2H sin contracción | `0.8242` | `0.6246` | `0.6298` | **`+0.1996`** ❌ | `70.44%` (`4,360`) | `-1.39%` ❌ |
| **`V4.1_pruned`** | Poda de features con drift + PBP 2PT/3PT directo + Seasonal ELO Decay + Contracción Bayesiana H2H | `0.6761` | `0.6401` | `0.6411` | `+0.0360` ⚠️ | `74.82%` (`3,185`) | `+4.75%` ✅ |
| **`V4.2_tuned`** | Regularización óptima (`max_depth=4`, `max_leaf=15`, `min_leaf=400`, `L2=50`, `L1=15`, `lr=0.03`, `colsample=0.65`) | `0.6402` | `0.6396` | `0.6399` | **`+0.0006`** ✅ | `76.59%` (`2,132`) | `+7.23%` ✅ |
| **`V4.3_multisnap`** | Stacking Multi-Snapshot (`m27, m28, m29, m30`) + Modality Dropout (12%) + Calibración Isotónica en Val | `0.6443` | `0.6425` | `0.6405` | `+0.0018` ✅ | `76.95%` (`2,139`) | `+7.73%` ✅ |
| **`V4.4_champion`** ⭐ | **Ensemble Híbrido (85% GBDT + 15% Ancla Lineal L2) + Gate Dual con Cabeza Regresora de Margen Q4** | **`0.6417`** | **`0.6423`** | **`0.6406`** | **`-0.0006`** ⭐ | **`77.19%` (`2,345`)** | **`+8.07%`** ⭐ |

---

## 4. Pruebas Rigurosas Cortando la Información en Múltiples Puntos

### 4.1 Cuatro Cortes Históricos Walk-Forward (Expanding Window `2015–2026`)
Entrenamiento desde 2015 hasta cada fecha de corte $T_k$ y evaluación ciega en el bloque futuro inmediato $[T_k, T_{k+1}]$:

| Corte Temporal | Partidos Train $\to$ Test | Ventana de Test | Train AUC | Test AUC | Brecha (`Train - Test`) | Test WR @ $P \ge 0.70$ ($N$ Apuestas) | Yield @ Cuota 1.40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corte 1** | `62,879` $\to$ `19,463` | `2023-12` a `2024-04` | `0.6405` | `0.6268` | `+0.0137` ✅ | **`76.18%`** (`995` bets) | **`+6.65%`** |
| **Corte 2** | `82,342` $\to$ `22,457` | `2024-04` a `2025-01` | `0.6378` | `0.6349` | `+0.0029` ✅ | **`74.87%`** (`1,707` bets) | **`+4.82%`** |
| **Corte 3** | `104,799` $\to$ `22,457` | `2025-01` a `2025-11` | `0.6375` | `0.6397` | `-0.0022` ✅ | **`76.64%`** (`1,982` bets) | **`+7.30%`** |
| **Corte 4** | `127,256` $\to$ `22,457` | `2025-11` a `2026-10` | `0.6386` | `0.6409` | `-0.0023` ✅ | **`77.74%`** (`2,143` bets) | **`+8.84%`** |

### 4.2 Siete Cortes de Minuto en Vivo (`m26` a `m32` en `fiba_men_10m`)
Evaluación del mismo modelo campeón `V4.4` cortando el flujo de eventos PBP y gráfico minuto a minuto desde `m26` (4:00 restantes en Q3) hasta `m32` (2:00 jugados en Q4):

| Minuto de Corte | Estado del Partido (FIBA 10m) | ROC AUC (Test) | Accuracy General | WR @ $P \ge 0.65$ ($N$) | WR @ $P \ge 0.70$ ($N$) | WR @ $P \ge 0.75$ ($N$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`m26`** | Q3 — 04:00 restantes | `0.6407` | `60.29%` | `71.81%` (`5,353`) | **`77.12%`** (`2,115`) | `80.89%` (`811`) |
| **`m27`** ⭐ | **Q3 — 03:00 restantes (Óptimo Pre-Q4)** | **`0.6406`** | **`60.28%`** | **`71.82%`** (`5,333`) | **`77.31%`** (`2,102`) | **`81.14%`** (`790`) |
| **`m28`** | Q3 — 02:00 restantes | `0.6406` | `60.26%` | `71.65%` (`5,358`) | **`76.81%`** (`2,126`) | `80.78%` (`817`) |
| **`m29`** | Q3 — 01:00 restante | `0.6405` | `60.19%` | `71.56%` (`5,363`) | **`76.68%`** (`2,144`) | `80.93%` (`818`) |
| **`m30`** | Fin de Q3 — Descanso (00:00) | `0.6407` | `60.34%` | `71.67%` (`5,361`) | **`77.00%`** (`2,135`) | `80.83%` (`819`) |
| **`m31`** | Q4 — 01:00 jugado (Confirmación Live) | **`0.6659`** | `62.11%` | `73.76%` (`5,610`) | **`78.79%`** (`2,291`) | `82.78%` (`848`) |
| **`m32`** | Q4 — 02:00 jugados (Confirmación Live) | **`0.7069`** | `65.17%` | `76.59%` (`6,014`) | **`81.47%`** (`2,520`) | `85.17%` (`917`) |

### 4.3 Cinco Cortes por Caída de Endpoints en Vivo (Modality Stress Test)
Simulación de fallos parciales de la API de SofaScore en el conjunto de prueba (`N = 22,458`):

| Escenario de Corte de Modalidad | ROC AUC (Test) | Accuracy | WR @ $P \ge 0.70$ ($N$ Apuestas) | Yield @ $0.70$ (Cuota 1.40) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Todas las fuentes activas (Nominal)** | **`0.6406`** | `60.28%` | **`77.31%`** (`2,102` bets) | **`+8.23%`** |
| **2. Sin `graph_points` (`/graph` caído $\to$ `NaN`)** | `0.6408` | `60.29%` | **`77.36%`** (`2,107` bets) | **`+8.30%`** |
| **3. Sin `play_by_play` (`/incidents` caído $\to$ `NaN`)** | `0.6408` | `60.25%` | **`76.57%`** (`2,121` bets) | **`+7.20%`** |
| **4. Sin `player_stats` históricos (`NaN`)** | `0.6085` | `58.01%` | **`74.88%`** (`1,067` bets) | **`+4.83%`** |
| **5. Solo Marcador + ELO + Liga (Emergencia extrema)** | `0.6085` | `58.02%` | **`75.29%`** (`1,052` bets) | **`+5.41%`** |

---

## 5. Desempeño Final de la Tríada Canónica (`models/m27_v4`) en las 4 Cabezas

| Arquetipo (`models/m27_v4`) | Partidos ($N$) | Train AUC | Val AUC | Test AUC | Brecha (`Train-Val`) | Cabeza 1 & 2: Ganador Q4 + Gate Margen ($\ge 0.70$) | Cabeza 3: Over/Under Total Q4 ($\|\hat{y}-\text{Prior}\|\ge 2.5$) | Cabeza 4: Ganador Partido Completo (FT Live) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fiba_men_10m`** | `149,713` | `0.6417` | `0.6423` | **`0.6406`** | **`-0.0006`** | **`77.19%` WR** (`2,345` bets, **+8.07% Yield**) | **`60.9%` WR** (`r=+0.48`, MAE `6.5 pts`, **+12.7% Yield** @1.85) | **`AUC = 0.9340`** (`Acc = 84.8%`) |
| **`fiba_women_10m`** | `34,251` | `0.6841` | `0.6886` | **`0.6714`** | **`-0.0045`** | **`76.74%` WR** (`1,023` bets, **+7.44% Yield**) | **`58.0%` WR** (`MAE = 6.2 pts`, **+7.3% Yield** @1.85) | **`AUC = 0.9622`** (`Acc = 88.7%`) |
| **`nba_12m`** | `28,250` | `0.6584` | `0.6315` | **`0.6208`** | **`+0.0269`** | **`73.86%` WR** (`241` bets, **+3.40% Yield**) | **`61.4%` WR** (`MAE = 7.1 pts`, **+13.6% Yield** @1.85) | **`AUC = 0.9150`** (`Acc = 82.9%`) |

---

## 6. Ubicación de Artefactos, Scripts y Paquete de Producción

- **Paquete de Producción (`models/m27_v4/`):**
  - `models/m27_v4/features.py`: Constructor de las 41 features limpias (6 bloques A–F) con soporte tanto en vivo como histórico y lectura O(1) de `team_priors_state.pkl`.
  - `models/m27_v4/predict.py`: Motor de inferencia multi-cabeza (`predict_match`) con enrutamiento automático a la Tríada Canónica (`fiba_men_10m`, `fiba_women_10m`, `nba_12m`) y compuerta de consenso Dual-Head.
  - `models/m27_v4/train.py`: Punto de entrada oficial de re-entrenamiento y exportación de estados.
  - `models/m27_v4/model_outputs/`: Modelos serializados (`model_v4_fiba_men_10m.pkl`, `model_v4_fiba_women_10m.pkl`, `model_v4_nba_12m.pkl`, `team_priors_state.pkl` y `v4_training_report.json`).
- **Scripts de Auditoría y Gráficas (`tmp/eda_nuevo_modelo/`):**
  - `tmp/eda_nuevo_modelo/iterative_training_pipeline.py`: Pipeline completo de iteraciones `V4.0` $\to$ `V4.4`, cortes Walk-Forward, cortes por minuto y generación de las 5 gráficas `.png` en `tmp/eda_nuevo_modelo/v4_artifacts/`.
  - `tmp/eda_nuevo_modelo/export_v4_production_state.py`: Exportador O(1) de estados ELO/rolling (`9,452` equipos y `3,643` ligas).
