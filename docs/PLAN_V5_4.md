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
| **Población sin mercado** | ✅ **ahí está el edge** | WR **83.3 %** vs 74.1 % con mercado |
| **Calibración condicional al mercado** | ✅ candidata nueva | el modelo va **7 pp** sobreconfiado con mercado, 2 pp sin |
| **Arquitectura de distribución del margen** | ✅ la idea estructural que queda | la meseta 0.75 ≈ 0.80 delata la confianza cuantizada |
| EV real | ⏳ **prospectivo** | no hay mercado de "4th quarter" pre-partido |

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

### P0 · Calibración condicional (sin reentrenar, bajo riesgo)
Recalibrar `p_cal` **condicionando a la presencia de mercado** (y a la categoría de fase), ajustando en el bloque **B** y midiendo en **C**. Es la vía más barata y la que ataca el defecto medido (−7 pp de sobreconfianza). Se puede hacer como post-hoc sobre las probabilidades del modelo.

### P1 · Política por población
Umbral más alto (o no operar) en la población con mercado. La evidencia dice que ahí el modelo apenas rompe el equilibrio (74.1 % vs 71.4 %). La cifra exacta la fija el EV real.

### P2 · Arquitectura: distribución del margen
La calibración isotónica tiene una **meseta (0.75 ≈ 0.80)**: la confianza del modelo está cuantizada y no puede expresar diferencias finas. Un modelo de **distribución del margen de Q4** (quantile regression) daría de un solo objeto el pick y una confianza continua → más volumen con el mismo edge. Es el cambio estructural con más recorrido. Las dos cabezas actuales (clasificador y margen) se contradicen en el 4.3 % de las filas, aunque en **0 de 6.346 apuestas** (la regla de disparo ya filtra esas).

### P3 · Fine-tuning con foco en la banda de decisión
Hoy `w = 1/partidos` y se entrena en todo el espacio, incluido el ~50 % de filas que nunca se betting. Pesar ∝ probabilidad de que la fila se aposte (o focal loss centrado en p ∈ [0.6, 0.9]) concentra capacidad donde se decide. Una sola prueba con el harness existente (`phase_exclusion_ablation.py`).

### P4 · EV real (prospectivo)
Dejar correr el monitor y medir con `ev_calculator.py`. Es lo que decide el umbral final y si la población con mercado se opera o no.

### Descartado (con evidencia)
- F1 prior contextual por liga × estado → **−0.0164 AUC**.
- F2 patrón de quarters → **−0.0002 AUC** (y los cuartos reales no predicen el Q4: −0.008).
- F3 ventaja extrema → **−0.0002 AUC**.
- Feature de mercado → **redundante** (corr 0.79-0.86 con `p_cal`).
- **F5 backfill de "contexto de competencia"** (quién necesita ganar el Q4): es otra feature de estado, y las features de estado no funcionan. No gastar una semana en un backfill nuevo antes de tener el EV.
