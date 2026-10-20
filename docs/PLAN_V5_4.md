# 🧪 v5_4 — qué se midió, qué se cayó y qué queda

> Fecha: 2026-10-20 · Base: holdout C (98.476 filas / 35.480 partidos ancla), artefactos desplegados de `m27_v5_3`, el mismo split y la misma calibración que producción.
> Reproducible: `tmp/m27_v5_3_audit/v54_*.py`.

---

## 0. Resumen en una tabla

| Idea | Veredicto | Cifra (dAUC vs V0) |
|---|---|---|
| F2 · patrón de ganadores de cuarto | ❌ **no aporta** | **−0.0002** (Brier +0.0000) |
| F3 · ventaja extrema por liga | ❌ **no aporta** | (incluida en V1: −0.0002) |
| F1 · prior contextual liga × estado | ❌ **empeora** | **−0.0164** (Acc −1.46 pp, LogLoss +0.0094) |
| Feature de mercado (1st quarter / partido) | ❌ **redundante** | mejor mezcla: peso 0.1 → **+0.0007** |
| P0 · recalibración condicional (Platt/isotónica por población) | ❌ **no mejora** | AUC plana; a 0.75 el WR **baja** (−1.6 pp) |
| **Calibración continua (Platt) en vez de isotónica** | ✅ **la que sí aporta** | LogLoss **0.6522 → 0.6518**, AUC 0.6550 → **0.6552**, meseta eliminada |
| **Población sin mercado** | ✅ **ahí está el edge** | WR **83.3 %** vs 74.1 % con mercado |
| **Calibración condicional al mercado** | ✅ candidata nueva | el modelo va **7 pp** sobreconfiado con mercado, 2 pp sin |
| **Arquitectura de distribución del margen** | ✅ la idea estructural que queda | la meseta 0.75 ≈ 0.80 delata la confianza cuantizada |
| EV real | ⏳ **prospectivo** | no hay mercado de "4th quarter" pre-partido |

### El hallazgo que sí cambió la calibración: la isotónica es escalonada

La isotónica es una función **escalonada**: dentro de un escalón la confianza no distingue nada. Efectos medidos:

- El holdout C tenía una **meseta**: los umbrales 0.75, 0.76, 0.78 y 0.80 elegían **las mismas 2.171 apuestas**. Por eso subir el umbral no hacía nada y no se podía afinar.
- Las confianzas en vivo de `m27_v5_3` tomaban solo **9 valores útiles** (0.727, 0.749, 0.751, 0.779, 0.799, 0.801, 0.826, 0.882, 0.915).

`PlattCalibrator` (`models/common/calibration.py`) es una calibración monótona **continua** y simétrica (`p_cal = sigmoid(a·logit(p_raw) + b)`, ajustada con `(z,y)` y `(-z,1-y)` para respetar la simetría local/visitante). Interfaz compatible con `IsotonicRegression` → `predict.py` no cambia.

| | isotónica | **Platt** |
|---|---|---|
| valores distintos de confianza (C) | 135 | **7.838** |
| AUC test | 0.6550 | **0.6552** |
| Brier | 0.2304 | 0.2304 |
| LogLoss | 0.6522 | **0.6518** |
| @0.75 · apuestas | 6.346 | **7.987** |
| @0.75 · WR | 82.11 % | 80.77 % |
| @0.75 · ROI@1.40 | +14.96 % | +13.08 % |
| @0.75 · **beneficio @1.40** | 949 u | **1.044 u** (+10 %) |
| @0.75 · **beneficio @1.83** | 3.190 u | **3.818 u** (+20 %) |

Con la calibración continua el umbral **sí se puede afinar** (curva suave, sin meseta) y el mismo umbral da más beneficio esperado.

### La frontera del umbral con la calibración continua (bloque C completo)

| umbral | apuestas | WR | ROI@1.40 | u@1.40 | ROI@1.83 | u@1.83 |
|---|---|---|---|---|---|---|
| 0.70 | 15.962 | 75.96 % | +6.34 % | 1.012 | +39.0 % | 6.225 |
| 0.72 | 12.330 | 77.96 % | +9.15 % | **1.128** | +42.7 % | 5.262 |
| 0.74 | 9.253 | 79.80 % | +11.72 % | 1.085 | +46.4 % | 4.260 |
| **0.75 (actual)** | 7.987 | 80.77 % | +13.08 % | 1.044 | +48.2 % | 3.818 |
| 0.78 | 5.182 | 83.94 % | +17.52 % | 908 | +53.6 % | 2.778 |
| 0.80 | 3.814 | 86.03 % | +20.44 % | 779 | +58.7 % | 2.190 |

El óptimo está en **0.72 a cuota 1.40** y en **0.70 a cuota 1.83**: el umbral depende de la cuota real, que es justo lo que falta por medir (P4). El 0.75 actual es una posición intermedia y prudente (WR más alto, menos volumen).

---

## 1. Por qué las features de estado no pueden funcionar

Medido sobre 202.577 partidos con 4 cuartos completos:

| Relación | Correlación con el ganador de Q4 |
|---|---|
| **Q1 real → Q4 real** | **−0.0079** (cero) |
| Q1 real → Q2 real → Q3 real (los tres cuartos) | se Lava en el ruido |
| **Mercado de "1st quarter winner" → Q4 real** | **+0.1733** |
| Mercado → Q1 real (control) | +0.2434 |

**Los cuartos que se juegan no dicen nada del cuarto siguiente.** El mercado sí. Es decir: el modelo no necesita saber *quién ganó el tercer cuarto* (el estado ya está en `cur_score_diff` y en los priors de ELO); necesitaría la **opinión del mercado sobre quién gana un cuarto**, y ese mercado solo existe para Q1 (en LMB) y en vivo.

El "efecto basura" que se midió (ventaja ≥ +22 al final de Q3 → el líder gana el Q4 el 60.7 %; ≤ −21 → el 38.7 %) es real, pero el modelo **ya lo tiene**: `cur_score_diff` + `lead_z` + `lead_vs_lg_vol` lo cubren, y por eso F3 no añade nada (dAUC −0.0002).

El prior contextual por liga (F1) **empeora** porque duplica información que ya está en `d_q4_bayes`/`lg_avg_q4`, y porque un prior por liga rompe el aprendizaje simétrico (el modelo se entrena con el espejo local/visitante) y desordena la calibración.

---

## 2. El mercado ya está dentro del modelo

| | AUC para el ganador de Q4 (LMB, n=13.979) |
|---|---|
| Modelo (p_cal) | **0.6310** |
| Mercado de "1st quarter winner" | 0.6079 |
| Mercado del partido (ML) | 0.6139 |
| Mejor mezcla modelo + w·logit(mercado) | w=0.1 → 0.6317 (+0.0007) |

Y `corr(p_model, p_mercado) = +0.79` (mercado de cuarto) y **+0.86** (mercado del partido): el modelo ya internalizó el precio. **Añadir el mercado como feature no aporta nada.**

Pero el mercado sí sirve para una cosa: **revelar un defecto de calibración.**

---

## 3. El hallazgo accionable: el modelo está sobreconfiado donde hay mercado

Población con mercado pre-partido vs sin mercado (anclas t=30, bloque C, umbral 0.75):

| población | apuestas | WR | ROI@1.40 | ROI@1.83 | confianza media del modelo |
|---|---|---|---|---|---|
| **con** mercado | 293 | **74.1 %** | **+3.7 %** | +35.5 % | 0.811 (el mercado dice 0.672) |
| **sin** mercado | 1.879 | **83.3 %** | **+16.6 %** | +52.4 % | 0.853 |

| | dice | acierta | error |
|---|---|---|---|
| con mercado | 81.1 % | 74.1 % | **−7.0 pp** |
| sin mercado | 85.3 % | 83.3 % | −2.0 pp |

Y el signo más fuerte: en la población **con** mercado, las apuestas más confiadas del modelo (umbral 0.85) son **las peores**: 9 apuestas, WR 55.6 %, ROI@1.40 **−22.2 %**. Es la firma clásica de la sobreconfianza: cuando el mercado existe y discrepa, el modelo va por delante y se equivoca.

### Contrfactuales de política (anclas, bloque C)

| política | apuestas | WR | ROI@1.40 | ROI@1.83 | beneficio esperado @1.40 |
|---|---|---|---|---|---|
| A · umbral 0.75 para todos (actual) | 2.172 | 82.0 % | +14.9 % | +50.1 % | **323 u** |
| B · umbral 0.75 **solo sin mercado** | 1.879 | 83.3 % | **+16.6 %** | **+52.4 %** | 312 u |
| D · 0.85 con mercado / 0.75 sin | 1.888 | 83.2 % | +16.4 % | +52.2 % | 310 u |

Quitar la población con mercado cuesta **3 % del beneficio esperado** y sube el ROI **+1.7 pp**. La decisión final la dará el **EV real** (que aún no se puede medir en retrospecto).

---

## 4. El EV real solo se puede medir hacia adelante

La API **no publica el mercado de "4th quarter" en pre-partido**: ofrece `1st half` y `1st quarter` (y solo en LMB: 15.549 partidos). El mercado de Q4 solo existe **in-play**, y ya se captura desde el 2026-10-20 (`q4_ml_home` / `q4_ml_away` en `live_snapshots_v3.odds_live_json`).

Consecuencia: **no hay backtest posible del EV**; hay que acumular datos. Herramienta lista: `tmp/m27_v5_3_audit/ev_calculator.py` (une apuestas ↔ cuota del Q4 en el minuto de la inferencia, calcula el EV real y el contrafactual de umbral por EV).

---

## 5. Plan de v5_4 (revisado con lo medido)

### P0 · Calibración condicional (sin reentrenar, bajo riesgo) — ❌ **RECHAZADA por la medición**

Recalibrar `p_cal` condicionando a la presencia de mercado (y a la categoría de fase), ajustando en el bloque **B** y midiendo en **C**:

| variante | AUC | thr 0.75 · apuestas | WR | ROI@1.40 |
|---|---|---|---|---|
| A isotonica global (actual) | 0.6536 | 2.172 | **82.04 %** | **+14.86 %** |
| B Platt por población | 0.6532 | 2.701 | 80.41 % | +12.58 % |
| C Isotónica por población | 0.6529 | 3.289 | 79.48 % | +11.27 % |
| D Platt + isotónica por fase | 0.6536 | 2.915 | 80.24 % | +12.34 % |

Ninguna mejora: el AUC queda plana y **el WR a 0.75 baja**. La recalibración condicional se descarta.

### P0' · Calibración continua (Platt) — ✅ **APLICADA**

Es el P0-bis: en vez de calibrar *por grupo*, calibrar de forma **continua** para eliminar la meseta de la isotónica. Detalle en §0. Implementado en `models/common/calibration.py` + `train_m27_v5_3_pipeline.py` (`CALIBRATION_KIND = "platt"`), con el backup de la isotónica en `model_outputs/calibrator_isotonic_backup.pkl`.

### P1 · Política por población
Umbral más alto (o no operar) en la población con mercado. La evidencia dice que ahí el modelo apenas rompe el equilibrio (74.1 % vs 71.4 %). La cifra exacta la fija el EV real.

### P2 · Arquitectura: distribución del margen
La calibración isotónica tenía una **meseta (0.75 ≈ 0.80)**: la confianza del modelo estaba cuantizada y no podía expresar diferencias finas. **La Platt lo resuelve en parte** (7.838 valores distintos), pero el modelo sigue calculando el pick con dos cabezas (clasificador y margen) que se contradicen en el 4.3 % de las filas, aunque en 0 de 6.346 apuestas. Un modelo de **distribución del margen de Q4** (quantile regression) daría de un solo objeto el pick y una confianza continua → más volumen con el mismo edge.

### P3 · Fine-tuning con foco en la banda de decisión
Hoy `w = 1/partidos` y se entrena en todo el espacio, incluido el ~50 % de filas que nunca se betting. Pesar ∝ probabilidad de que la fila se aposte (o focal loss centrado en p ∈ [0.6, 0.9]) concentra capacidad donde se decide. Una sola prueba con el harness existente (`phase_exclusion_ablation.py`).

### P4 · EV real (prospectivo)
Dejar correr el monitor y medir con `ev_calculator.py`. Es lo que decide el umbral final (0.70 a 1.83, 0.72 a 1.40) y si la población con mercado se opera o no.

### P2 · Arquitectura: distribución del margen — ❌ **RECHAZADA por la medición**

Se entrenó un modelo de **distribución** del margen de Q4 (cuantil 0.1/0.3/0.5/0.7/0.9 con `HistGradientBoostingRegressor(loss="quantile")` y aumento espejo) y se derivó `P(margen > 0)` de la distribución:

| variante | AUC | Brier | LogLoss | thr 0.75 · apuestas | WR | u@1.40 |
|---|---|---|---|---|---|---|
| V0 actual (clasificador) | **0.6537** | **0.2306** | **0.6524** | 2.728 | 81.2 % | **374** |
| V2 distribución del margen | 0.6064 | 0.2395 | 0.6715 | 1.638 | 76.0 % | 105 |
| V4 mezcla (peso 0.2) | 0.6534 | 0.2308 | 0.6527 | 2.505 | 81.0 % | 336 |

La distribución con 5 cuantiles es un clasificador claramente peor (AUC −0.047) y la mezcla no aporta nada (AUC −0.0003, y en el umbral operativo el beneficio baja de 374 a 336 u). Con el margen de un solo punto (el regresor actual) el modelo ya extrae lo que hay; la distribución no lo mejora.

### P3 · Fine-tuning con foco en la banda de decisión — ❌ **TRAMPA**

Reentrenar con pesos `exp(−((|p_raw−0.75|/0.12)²))` (solo el 15 % de filas conserva peso > 0.5):

| | AUC | Brier | thr 0.75 · apuestas | WR | u@1.40 |
|---|---|---|---|---|---|
| V0 actual | **0.6537** | **0.2306** | 2.728 | 81.2 % | **374** |
| V1 foco en la banda | 0.6462 | 0.2415 | **237** | **92.0 %** | **68** |

**El WR sube a 92 % pero el beneficio se desploma a 68 u (de 374)**: el modelo se especializa tanto que pierde el 91 % del volumen. Además el AUC y el Brier empeoran → no es solo una recalibración, el modelo se daña. Un WR alto con pocas apuestas **no** es una mejora si el umbral es lo que decide el tamaño de la cartera.

### P5 · Más datos: el backfill de líneas ya terminó — 🔄 **en curso**

El backfill de `match_odds` (modo `lines`) está **completo**: 17.176 partidos con `status='ok'`, 0 pendientes. Pero el dataset se construyó cuando el backfill iba por ~7.000:

| feature | cobertura en el dataset actual | con el backfill completo |
|---|---|---|
| `mkt_margin_spread` | 3.7 % (7.062 filas) | **~9.0 %** (17.192 partidos) |
| `total_line` | 3.6 % (6.914 filas) | **~9.0 %** (17.061 partidos) |

Reconstruir el dataset y reentrenar es el único vía de **más señal** que queda después de que las features nuevas midieran neutras: el total y el handicap del mercado son la escala del partido, que es justo lo que determina el margen de Q4.

### Descartado (con evidencia)
- F1 prior contextual por liga × estado → **−0.0164 AUC**.
- F2 patrón de quarters → **−0.0002 AUC** (y los cuartos reales no predicen el Q4: −0.008).
- F3 ventaja extrema → **−0.0002 AUC**.
- Feature de mercado → **redundante** (corr 0.79-0.86 con `p_cal`).
- **F5 backfill de "contexto de competencia"** (quién necesita ganar el Q4): es otra feature de estado, y las features de estado no funcionan. No gastar una semana en un backfill nuevo antes de tener el EV.
- **P0 recalibración condicional** (por población o por fase) → el AUC queda plana y el WR a 0.75 baja 1.6 pp.

---

## 6. Decisiones operativas tomadas (2026-10-20)

### 6.1 Qué formatos de torneo se descartan y cuáles se mantienen

| categoría | partidos fuera del train | decisión | motivo |
|---|---|---|---|
| AMISTOSOS | 3.486 | ❌ **descartado** | sin formato de liga (ritmo, localía y número de equipos distintos) |
| SEGUNDA FASE (*championship round*) | 3.653 | ❌ **descartado** | grupo con formato de segunda fase |
| PROMOCION | 2.583 | ❌ **descartado** | formato de supervivencia |
| PRELIMINAR | 2.290 | ❌ **descartado** | formato clasificatorio |
| PUESTO | 186 | ❌ **descartado** | partidos por puesto, no por título |
| ELIMINATORIA | 18.435 | ✅ **se mantiene** | el modelo acierta **84.3 %** ahí, por encima del 77.0 % de la temporada regular |
| COPA | 11.065 | ✅ **se mantiene** | **82.8 %** WR, ROI@1.40 +15.9 % |

> La decisión de mantener eliminatorias y copas es **opuesta** a la premisa original de la guía, y está respaldada por medición sobre el holdout C, no por criterio. Si en el futuro se construye el modelo propio para esas fases, el filtro por `stage_detail` es el sitio donde se activa.

### 6.2 Cambios en el monitoreo

| cambio | motivo |
|---|---|
| `ACTIVE_MODELS = ["m27_v5_3", "m27_v5_2"]` | `v6_2` apostó con **28 % de confianza** (min 27 del match 17230818) y su WR vivo es de los más bajos; `m27_v3` no expone `p_cal` (no se puede auditar ni calibrar) |
| `m27_v5_3` sale del **modo sombra** | el usuario lo sigue de cerca y pide verlo en las alertas. **Aviso:** sigue sin evidencia suficiente en vivo (40 apuestas, WR 72.5 %, IC95 56-85 %) |
| La alerta muestra **todos** los modelos con su motivo | antes un modelo en estado "vigilando la zona verde" **desaparecía** de la alerta y solo aparecía en la tarjeta de resultado (inconsistencia) |
| Estados en la tarjeta y el mensaje | `BET` (verde) · `PENDIENTE` (naranja, todavía puede apostar) · `NO BET (<umbral>)` · `NO BET — <motivo del descarte>` |
| Orden | `m27_v5_3` primero (`model_display_sort_key`) |

### 6.3 Corrección de la liquidación (bug del 2026-10-20)

La migración de `quarter_scores` a `quarter_scores_v2` (la tabla canónica) revelou que **la liquidación usaba un Q4 provisional**: 7 apuestas quedaron mal (ninguna de `m27_v5_2`/`m27_v5_3`). Se corrigió con dos guards:
1. No cerrar ni liquidar un partido que la API sigue marcando `inprogress`.
2. No liquidar si la suma de los cuartos no cuadra con el marcador final **y no hay prórroga** (el primer guard confundía los partidos AET con un Q4 parcial y los dejaba pendientes para siempre).

Bonus de la migración: `quarter_scores_v2` pasó de 2.916 a **250.143 partidos** con el dato definitivo de fin de partido, y el pipeline de entrenamiento ya lee esa tabla (mismos 184.173 elegibles, 0 diferencias).

