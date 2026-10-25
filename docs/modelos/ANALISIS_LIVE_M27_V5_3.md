# 📉 m27_v5_3 en vivo vs m27_v5_2 — por qué "no le va tan bien" (y por qué en realidad sí)

> Análisis sobre `bet_monitor_log_v3`, ventana **2026-10-18 → 2026-10-19** (682 inferencias, 51 apuestas reales).
> Código: [`tmp/m27_v5_3_audit/`](../tmp/m27_v5_3_audit) · Contexto del modelo: [`findings_m27_v5_3.md`](modelos/findings_m27_v5_3.md).
> Reproducir: `.venv\Scripts\python.exe tmp\m27_v5_3_audit\diagnose_v5_3_vs_v5_2.py` (y `contrast_with_findings.py`, `confidence_parity.py`, `calibration_live.py`).

---

## 0. Conclusión en una línea

**El modelo `m27_v5_3` no es peor que `m27_v5_2`: es mejor.** Lo que está mal es la **regla de disparo con la que se opera** (`min_conf 0.72` sin filtro de margen, elegido maximizando beneficio a una cuota hipotética de 1.40) y una **lista de ligas tóxicas que no coincide con ningún nombre real de SofaScore**.

| | m27_v5_2 | **m27_v5_3** | Lectura |
|---|---|---|---|
| Apuestas live | 11 | **40** | 3.6× más volumen |
| Acierto live | 9/11 = **82 %** | 29/40 = **72.5 %** | −9.5 pp |
| ROI @1.40 | +14.55 % | **+1.50 %** | se Come casi todo el borde |
| Acierto en los **mismos** snapshots | 32/45 | **34/45** | v5_3 gana |
| Brier en los mismos snapshots | 0.1976 | **0.1907** | v5_3 mejor calibrado |
| Fisher exacto entre ambos | — | **p = 0.706** | **diferencia NO significativa** |

Con 40 apuestas el IC95% de 72.5 % es ≈ **[56 %, 85 %]**: tanto el 77.7 % de la tabla de ancla como el 81.4 % del camino live (`compare.log`, que es la referencia correcta porque el monitor dispara al primer tick) están dentro del ruido. **Ninguna de las dos versiones ha demostrado en vivo ser mejor que la otra.**

---

## 1. El gap NO está en el modelo, está en la regla

### 1.1 Las dos reglas de disparo (leídas del código)

```
m27_v5_2  (models/m27_v5_2/predict.py:117-128)
    p_cal >= 0.75   AND   p_xgb > 0.5   AND   exp_margin >= 1.8

m27_v5_3  (models/m27_v5_3/predict.py:113-120, model_config.json)
    p_cal >= 0.72   AND   exp_margin >= 0.0   AND   liga no tóxica
```

`v5_3` **relaja el umbral en 0.03 y anula por completo el filtro de margen** (`1.8 → 0.0`). Ese margen de 1.8 era un filtro de calidad real: exigía que el modelo previera un Q4 con ventaja de al menos 1.8 puntos.

### 1.2 Qué pasa en los snapshots que ambos modelos vieron

Sobre los **48 snapshots compartidos** (mismo `match_id` + mismo minuto de inferencia):

| | Valor |
|---|---|
| Dispara **ambos** | 9 |
| Dispara **solo v5_3** | **28** ← el gap de volumen |
| Dispara **solo v5_2** | **0** |
| Ninguno | 11 |

En esos mismos snapshots, `v5_3` se muestra **más confiado** que `v5_2` en 34 casos contra 6 al revés (confianza media `|p_cal−0.5|`: 0.238 vs 0.158, **+0.075**).

**Pero ese exceso de confianza NO se paga con más acierto:** en los 45 snapshots con Q4 resuelto, `v5_3` acierta **34/45** y `v5_2` **32/45**, con **mejor** Brier (0.1907 vs 0.1976) y mejor LogLoss (0.5713 vs 0.5777). El modelo más fino produce probabilidades más separadas, y la regla floja las convierte en 3× más apuestas.

### 1.3 La calidad de las apuestas que `v5_2` rechazaba

De las 28 apuestas exclusivas de `v5_3`:

- Con `p_cal ≥ 0.75` (lo que `v5_2` habría tomado): **64 %** de acierto
- Con `p_cal` entre 0.72 y 0.75 (la banda que `v5_2` descartaba): **62 %** de acierto

El equilibrio a cuota **1.40 es 71.4 %** de acierto. La banda marginal está **por debajo del equilibrio** y por encima solo funciona a 1.83 (equilibrio 54.6 %).

### 1.4 Contrfactuales sobre el histórico en vivo

Reconstruyendo la decisión con el `p_cal` y el margen **congelados en cada log** (333 filas con Q4 resuelto):

| Configuración | n | WR | ROI @1.40 | ROI @1.83 |
|---|---|---|---|---|
| **v5_3 como está** (0.72, margen 0, sin tope de minuto) | 40 | 72.5 % | **+1.50 %** | +32.67 % |
| v5_3 con `min_conf 0.75` | 27 | 77.8 % | **+8.89 %** | +42.33 % |
| v5_3 con `min_conf 0.75` + margen ≥ 1.8 (regla v5_2) | 27 | 77.8 % | +8.89 % | +42.33 % |
| v5_3 con `min_conf 0.72` + margen ≥ 1.8 | 40 | 72.5 % | +1.50 % | +32.67 % |
| v5_3 con tope de minuto 30 (rango de entrenamiento) | 38 | 76.3 % | +6.84 % | +39.66 % |

Dos lecturas importantes:

1. **El umbral es la restricción que aprieta.** Subirlo a 0.75 recorta 13 apuestas y sube el ROI de +1.50 % a +8.89 %. El filtro de margen **no aporta nada por sí solo** (0.72 + margen 1.8 = idéntico a 0.72 sin margen: el margen nunca fue < 1.8 cuando la confianza fue ≥ 0.72).
2. **El tope de minuto 30 vale ~5 pp de ROI** por sí solo (2 apuestas, 0 aciertos).

### 1.5 El modelo acierta lo que su propia calibración promete

Para las 18 apuestas de `v5_3` en ligas con calibración offline (`league_calibration.json`), sumando el WR offline que el modelo esperaba para cada liga:

- **Esperado: 13.8 aciertos**
- **Real: 13 aciertos** (desviación −0.8)

**El modelo en vivo rinde exactamente como su backtest.** No hay degradación live-vs-offline. Las 22 apuestas restantes (de 40) están en ligas **sin ninguna entrada** en la calibración offline — el modelo no tiene historial medido de ellas.

---

## 2. Defectos reales encontrados (independientes del rendimiento)

### 2.1 Se apuesta con el snapshot obsoleto (minutos 31-33)

`m27_v5_3` se entrenó con snapshots en **t ∈ [25.5, 30.0]** (`features.py:87`, `model_config.json:snapshot_range`) y **clampea internamente**: `ctx_from_payload` usa el **cierre de Q3 con t = 30** cuando el partido ya está en Q4. Verificado en los 336 logs de `v5_3`:

| `inference_minute` (log) | `snapshot_minute_used` (modelo) | filas |
|---|---|---|
| 27 | 27.0 – 27.8 | 30 |
| 28 – 29 | 28.0 – 30.0 | 13 |
| 30 | 30.0 | 241 |
| **31 – 38** | **30.0 (clampeado)** | **52** |

> ⚠️ **Corrección a la primera versión de este informe:** aquí no hay extrapolación fuera de distribución — la feature `minute` vale 30.0 y el vector es idéntico al del ancla entrenada. El problema es otro: **la apuesta se coloca sobre un Q4 que ya empezó y cuyo parcial el modelo ignora**.

El monitor permite apuestas ordinarias hasta el minuto **33** (`Q4_TOO_LATE_BET_MINUTE`, `constants.py:90`) y v5_3 disparó **2 apuestas en minutos 31 y 32, ambas perdidas**. Con n=2 no es evidencia estadística, pero el mecanismo es sólido: el modelo predice el ganador de Q4 con datos de Q3, cuando ya se han jugado 1-2 minutos del cuarto que decide la apuesta. Para `m27_v5_2` el problema es peor: su extractor (`extract_features_v5_2`) **no clampea** y sí extrapola con el minuto real.

**Consecuencia práctica:** el plazo válido de `m27_v5_3` es el **minuto 30**, no el 33 global del monitor.

### 2.2 La lista de ligas tóxicas no coincide con ningún nombre real

`is_toxic_league()` hace *substring match* sobre `TOXIC_LEAGUE_PATTERNS` (`features.py:89-99`). Contra las **3,761 ligas** del universo de entrenamiento y las **434** vistas en vivo:

| Patrón | ¿Poda del train? | ¿Bloquea en live? |
|---|---|---|
| `korean basketball league`, `tipos sbl`, `korisliiga, women`, `betsafe-lkl`, `nlb`, `1st dbbl`, `serie a2, women, group b`, `serie b, group b`, `israeli basketball super league` | ✅ | ✅ |
| `kbl`, `lmb, apertura` | ✅ (ligas históricas: `Kvindebasketligaen , KBL DM-Finaler`, `LMB, Apertura`) | ❌ nunca aparecen en vivo |
| **`cyprus division`, `georgia superli`, `kosovo superlig`, `albania super`, `armenia a-league`** | ❌ | ❌ **no casan con ninguna liga, ni del train ni live** |

`cyprus division` no casa con `Cyprus Basketball Division A` (la palabra "Basketball" va en medio) — y **`v5_3` perdió una apuesta en esa liga**.

**Y aquí está el hallazgo más grave:** `build_dataset.py:130` usa **la misma función** para podar el train (`not is_toxic_league(r[2])`). Es decir: los 5 patrones muertos **no podaron nada del entrenamiento tampoco**, y las 2 ligas que el propio modelo marcó como `block_candidate` **estaban en el set de entrenamiento**:

| Liga | WR offline | ROI @1.40 | ¿Estaba en el train? | ¿Bloqueada? |
|---|---|---|---|---|
| `1. Division B` | **42.1 %** (19 apuestas) | −41.0 % | **SÍ** | ❌ no |
| `Sparkasse Prva Liga Zeni` | **47.6 %** (21 apuestas) | −33.3 % | **SÍ** | ❌ no |

El modelo se entrenó con ligas que su propia calibración declara tóxicas. La mejora #16 del documento de findings ("`block_toxic_leagues` en inferencia") es **inerte**.

> 📌 La lista **no** está en `model_config.json` (como afirma AGENTS.md §4.5): `model_config.json` solo tiene el booleano `block_toxic_leagues: true`. Los patrones están hardcodeados en `models/m27_v5_3/features.py:89-94`. Quien quiera arreglarla debe editar ese archivo.

### 2.4 La premisa de la lista negra estaba escrita… pero nunca se aplicó

La premisa original está en [`docs/ligas/COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md`](ligas/COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md) (semáforo de arquetipos) y la infraestructura existía en `leagues_classification` (3.643 filas con `is_youth`, `is_college`, `competition_type`, `is_playoffs`, `is_final`, `quarter_duration_minutes`). **El filtro nunca se conectó al entrenamiento**: `build_dataset.py:130` solo aplicaba `not is_toxic_league(liga)`, una lista de 16 patrones escritos a mano (5 de los cuales no casaban con ningún nombre real).

| Regla del semáforo | Volumen en la base | **Lo que seguía dentro del train de v5_3** |
|---|---|---|
| 🔴 R1 Juveniles U16-U23 | 885 ligas / 7.820 part. | **851 ligas / 7.743 part.** |
| 🔴 R2 College NCAA | 64 ligas / 5.704 part. | **53 ligas / 2.914 part.** |
| 🔴 R3 Amistosos / torneos / pretemporada | 1.051 ligas / 16.672 part. | **1.011 ligas / 13.528 part.** |
| 🟡 A3 Playoffs / finales / eliminatorias | 851 ligas / 11.864 part. | **811 ligas / 11.441 part.** |
| 🟡 A1 NBA y ligas de 12 min | 44 ligas / 29.967 part. | 0 ✅ (ya implementado) |
| 🟡 A2 Femenino adulto | 617 ligas / 36.666 part. | con feature `is_women` ✅ |

**Tres bugs de la clasificación existente:**

1. **`leagues_classification.stage_detail` clasifica por liga base, no por variante.** `NBA, Playoffs`, `Argentina Liga Nacional, Playoffs`, `Brazil NBB, Playoffs`, `Mexico LNBP, Playoffs` o `Korisliiga, Playoffs` (18k partidos de eliminatorias) quedan marcados como **"Regular Season"**. Por eso el filtro tiene que leer el texto de la fase, no el bucket precalculado.
2. **"Championship Round" se reparte de forma inconsistente**: `NBL`, `NKL`, `Portugal Proliga`, `Korisliiga`, `Basketligaen` y `LNB` lo tratan como playoffs; `LNA`, `United League`, `Bulgaria NBL` y `Stoiximan GBL` como temporada regular. De 183 ligas con "championship", 13 quedan sin clasificar de forma coherente.
3. **Las ligas de 12 min están bien marcadas** (44 ligas / 29.967 partidos: NBA, G League, China CBA, PBA×5, NBA Summer League, WCBA, CBA Club Cup, MCBL…). Verificado contra la API (`period_length` 600/720) en 7 candidatas: ninguna resultó ser de 12 min (TBL, Chinese Taipei P League, NBL1 y CIBACOPA son de 10 min con anotación alta). Solo `Chinese Taipei T1 League` y `Super 20, Playoffs` no respondieron (datos antiguos).

**Solución implementada:** [`models/common/competition_stage.py`](../models/common/competition_stage.py) clasifica la fase con marcadores multilingües sobre el texto original (el, pt, it, fr, de, pl, cs, hr/sr, lt, ru, el, tr, hu, ro, bg, zh/ja/ko…) y devuelve la **categoría + el motivo legible**, que el modelo escribe en el `raw_json` de cada inferencia.

### 2.5 Lo que la medición dice de la lista negra

Sobre el bloque C (98.476 filas, artefactos desplegados, umbral 0.75) el modelo rinde **mejor** en las fases que la guía marcaba como ruido:

| categoría | apuestas | WR | ROI@1.40 | sesgo |
|---|---|---|---|---|
| REGULAR | 1.555 | 77.0 % | +7.9 % | +0.031 |
| **FORMATIVAS** (R1) | 596 | **84.9 %** | **+18.9 %** | −0.028 |
| **ELIMINATORIA** (A3) | 191 | **84.3 %** | **+18.0 %** | −0.040 |
| **COPA** (R3) | 128 | **82.8 %** | +15.9 % | −0.037 |
| **SEGUNDA FASE** | 45 | **73.3 %** | +2.7 % | +0.051 |
| PROMOCION | 20 | 85.0 % | +19.0 % | −0.064 |

La premisa de la guía se apoyaba en el **ritmo y los puntos** (playoffs: 144 pts vs 158.6 de media), no en el acierto del ganador de Q4, que es el objetivo del modelo. La única categoría floja es *Championship Round* (73.3 %, ROI +2.7 %).

> **Decisión aplicada (2026-10-20):** se descartan del train y se bloquean en inferencia **AMISTOSOS** y el bloque **segunda fase + promoción + preliminar + colocación**. **Se mantienen eliminatorias y copas** (rinden por encima de la temporada regular). Juveniles y NCAA quedan **pendientes de medición** por ablación.

### 2.6 Ablación: qué pasa si se excluyen esas fases del entrenamiento

Cuatro variantes reentrenadas desde el dataset ya construido (mismo split, misma semilla) y evaluadas **siempre sobre el mismo bloque C** (98.476 filas, 0.6530 AUC de base — reproduce exactamente el modelo desplegado). Umbral de lectura: **0.75**, el nuevo punto operativo.

| variante | filas train | AUC (C completo) | apuestas @0.75 | WR | ROI @1.40 | ROI @1.83 |
|---|---|---|---|---|---|---|
| **V0** real (lo de hoy) | 204.703 | 0.6530 | 10.588 | 79.18 % | +10.86 % | +44.91 % |
| **V1** sin amistosos | 201.862 | 0.6530 | 6.260 | **82.43 %** | **+15.40 %** | +50.84 % |
| **V2** = decisión tomada (V1 + 2ªFase+PROMO+PRELIM+PUESTO) | 191.636 | 0.6524 | 6.762 | **81.74 %** | **+14.43 %** | +49.58 % |
| **V3** = V2 + sin juveniles + sin NCAA | 188.795 | 0.6523 | 6.435 | 81.86 % | +14.61 % | +49.81 % |

**Lecturas:**

1. **La decisión tomada se confirma**: excluir amistosos + segunda fase + promoción + preliminar + colocación sube el acierto **+2.56 pp** y el ROI@1.40 **+3.57 pp** con un AUC prácticamente intacto (−0.0006).
2. **Juveniles y NCAA NO aportan nada**: V3 ≈ V2 en todas las métricas (+0.12 pp WR, +0.18 pp ROI). Peor aún, en el subconjunto que V3 conserva el **AUC baja −0.0054**: el modelo estaba extrayendo señal útil de esos partidos. **No se excluyen** (la premisa de la guía no se sostiene para este objetivo).
3. ⚠️ **El número de apuestas es muy sensible a la calibración**: quitar el 1.4 % de filas de entrenamiento mueve el conteo de 10.588 a 6.260 apuestas al mismo umbral. La calibración isotónica tiene tramos planos, así que un desplazamiento pequeño mueve muchas filas de un lado al otro del umbral. Consecuencia práctica: el umbral hay que elegirlo **junto** con la calibración, no de forma aislada.

### 2.7 Lo que el modelo **no** tiene (para contexto, no es la causa)

- **Cuota de mercado — es un efecto de selección, no una pérdida de datos:** `has_ml = True` en el **67.9 %** de las 336 inferencias en vivo, pero solo en el **37.5 %** de las 40 apuestas. La causa es que los 40 partidos apostados **no tienen ninguna cuota ML en `match_odds`** (0 de 40) → no es un fallo de extracción, es que el modelo **apuesta mucho más donde no tiene mercado**. La exactitud es idéntica en ambos subconjuntos (73 % con cuota vs 72 % sin), así que la confianza que produce sin datos de mercado **no aporta nada extra**: ese es el margen que el modelo se está dando a sí mismo.
- **Empates de Q4:** desde el commit `9121cb6` cuentan como pérdida, igual para ambas versiones → la comparación es justa.

---

## 3. Recomendaciones

### R1 — El umbral no se puede fijar sin la cuota real del Q4 (esto va primero de todo)

El pipeline elige `min_conf` **maximizando beneficio absoluto a cuota 1.40** en el bloque B (`train_m27_v5_3_pipeline.py:236`: `profit = w * 1.40 - n`). Esa función premia el volumen, no la calidad. La superficie real de beneficio (test C) muestra que el umbral óptimo **depende de la cuota**:

| `min_conf` | apuestas | WR | benef. esp. @1.40 | benef. esp. @1.83 |
|---|---|---|---|---|
| 0.65 | 32,318 | 70.67 % | −343 ❌ | **9,478** ← óptimo a 1.83 |
| 0.68 | 17,973 | 75.20 % | 949 | 6,761 |
| 0.70 | 17,708 | 75.28 % | 955 | 6,687 |
| **0.72 (actual)** | 13,277 | 77.68 % | **1,162** ← óptimo a 1.40 | 5,597 |
| 0.75 | 10,588 | 79.18 % | 1,149 | 4,754 |
| 0.80 | 4,589 | 84.79 % | 858 | 2,532 |

*(Cifras sobre todas las filas de C; el ancla t=30 da el mismo ranking.)*

Tres consecuencias:

1. **A 1.40, 0.72 y 0.75 están empatados** (−1.1 % de diferencia esperada). Eligiendo 0.72 no se gana nada: se agrega un 20 % más de volumen con 1.5 pp menos de acierto. La elección del pipeline es, en la práctica, arbitraria dentro del ruido.
2. **A 1.83 el óptimo se mueve a 0.65** (+69 % de beneficio esperado frente a 0.72). Si el mercado real paga ~1.83, el umbral actual está dejando dinero sobre la mesa; si paga ~1.40, está en el sitio correcto.
3. **Por eso el ROI es teórico**: la cuota real del cuarto nunca se registró. `odds_live` **ya se está guardando** en `live_snapshots_v3`; falta un join con `bet_monitor_log_v3` para pasar de "beneficio a cuota hipotética" a **EV real** y fijar el umbral por EV en vez de por intuición.

### R2 — Subir `min_conf` a 0.75 (barato, reversible, con evidencia)

| | |
|---|---|
| **Dónde** | `models/m27_v5_3/model_outputs/model_config.json` → `min_conf` |
| **Efecto live (contrafactual sobre los logs)** | n 40→27 · WR 72.5 %→**77.8 %** · ROI@1.40 +1.50 %→**+8.89 %** |
| **Efecto backtest (C, ancla t=30)** | WR 77.76 %→**79.29 %** · ROI@1.40 +8.86 %→**+11.01 %** |
| **Coste** | −32 % de volumen |
| **Por qué no rompe nada** | a 1.40 el beneficio esperado de 0.72 y 0.75 es el mismo (−1.1 %) |

> ⚠️ **El cambio no es durable si solo se edita el JSON.** `model_config.json` se regenera en cada reentrenamiento con `thr_star` (pipeline línea 445-447). Para que 0.75 sobreviva al próximo `train_m27_v5_3_pipeline.py` hay que tocar la **función objetivo** (línea 236): p. ej. maximizar beneficio con un techo de volumen, o exigir `WR ≥ 78 %` como restricción. Alternativa: dejar `thr_star` como está y aplicar un override de despliegue post-entrenamiento.

### R3 — Que `m27_v5_3` deje de apostar después del minuto 30

| | |
|---|---|
| **Dónde** | `monitor_v3/models/evaluator.py` (plazo por modelo) o `models/m27_v5_3/predict.py` validando el `snapshot_minute` **de entrada** (no `ctx["minute"]`, que ya viene clampeado a 30) |
| **Por qué** | el modelo predice el ganador de Q4 con el cierre de Q3; a partir del minuto 31 ya se han jugado 1-3 minutos del cuarto que decide la apuesta |
| **Efecto live (contrafactual)** | n 40→38 · WR 72.5 %→**76.3 %** · ROI@1.40 +1.50 %→**+6.84 %** |
| **Deuda** | el monitor permite ordinarias hasta el 33 (`Q4_TOO_LATE_BET_MINUTE`); el plazo válido de este modelo es 30 |
| **Extra** | `m27_v5_2` **no clampea** (`extract_features_v5_2` usa el minuto crudo) → también extrapola; merece el mismo tope |

### R4 — Arreglar el bloqueo de ligas tóxicas

- Reemplazar los **5 patrones muertos** (`cyprus division`, `georgia superli`, `kosovo superlig`, `albania super`, `armenia a-league`) por los nombres reales; lo ideal es casar por un id de liga estable en vez de por texto.
- Añadir las **2 ligas `block_candidate`** del propio modelo: `1. Division B` (WR 42.1 %) y `Sparkasse Prva Liga Zeni` (WR 47.6 %).
- Regla sugerida y ya disponible: bloquear cuando el **IC95% del WR offline de la liga** quede por debajo del equilibrio de la cuota objetivo (71.4 % @1.40) — está en `league_calibration.json`.
- ⚠️ La lista es **compartida con la poda del entrenamiento** (`build_dataset.py:130`): arreglarla cambia el set elegible → lo ideal es reentrenar después. Bloquear en inferencia, mientras tanto, es estrictamente mejor que no bloquear.
- Corregir la referencia en `AGENTS.md` §4.5: la lista **no** está en `model_config.json` (ahí solo está el booleano), está en `models/m27_v5_3/features.py:89-94`.

### R5 — No notificar como apuesta operable hasta tener evidencia suficiente

Con 40 apuestas resueltas y WR 72.5 % (IC95 ≈ 56-85 %) **no hay evidencia** de que el modelo tenga ventaja en vivo, y el propio documento de findings ya decía: *"comparar WR live contra el 77.8 % de backtest antes de operar con dinero"*. La comparación ya está hecha y es inconcluyente.

Concretamente: un flag por modelo en `monitor_v3/notifications/telegram_bot.py` (la señal llega por `_is_real_bet_signal`) que permita que `m27_v5_3` **registre en `bet_monitor_log_v3` sin disparar la alerta de apuesta**, hasta acumular **≥ 200 apuestas resueltas** por configuración. Se necesita un contador de apuestas resueltas por modelo y versión de configuración.

### R6 — Lo que NO hay que hacer

- **No reentrenar el modelo.** No hay degradación live-vs-offline: el modelo rinde 13 aciertos donde su propia calibración esperaba 13.8, y en los snapshots compartidos **gana** a `v5_2` (Brier 0.1907 vs 0.1976). El problema es de configuración y de higiene, no de modelo.
- **No cambiar `min_margin`.** Medido: a 0.72 el filtro de margen ≥ 1.8 **nunca llega a activarse** (40 apuestas idénticas con y sin él). El margen no es la restricción que aprieta; el umbral sí.
- **No emitir el bloqueo por `has_ml`.** El modelo acierta lo mismo con y sin cuota (73 % vs 72 %), así que filtrar por presencia de mercado no aporta. El tema de la cuota es un **sesgo de selección** (apuesta más donde no hay mercado: 37.5 % vs 67.9 % de las inferencias), no una pérdida de datos — y conviene medirlo **offline en el bloque C** antes de tratarlo como problema.

### R7 — Medir el gap de cobertura por liga

22 de las 40 apuestas (55 %) son en ligas **sin entrada** en `league_calibration.json`: el modelo opera con priors de liga genéricos (`fallback_info`) en más de la mitad de sus apuestas sin haber medido nunca su WR en Q4. Con 434 ligas vistas en vivo y 74 calibradas, el modelo está cubriendo un territorio que no conoce.

---

## 4. Contraste con `findings_m27_v5_3.md`

El documento de findings es sólido y en lo medible **se sostiene**. Esta tabla separa lo **confirmado**, lo **corregido** y lo que el documento **no cubría**.

| # | Afirmación del documento | Veredicto |
|---|---|---|
| 1 | v5_3 es superior a v5_2 (AUC 0.653 vs 0.638, Brier 0.231 vs 0.235) | ✅ **Confirmada y reforzada**: en los snapshots en vivo v5_3 también gana (Brier 0.1907 vs 0.1976, 34/45 vs 32/45) |
| 2 | Umbral operativo 0.72 → WR 77.8 %, ROI@1.40 +8.9 % (tabla §1, ancla t=30) | ⚠️ **Referencia equivocada para el disparo real**: el monitor dispara al **primer tick** (min 27 en 30 de 40 apuestas), así que la referencia correcta es §4 `compare.log` (camino live min 27): **WR 81.4 %, ROI +13.9 %** en 236 apuestas. Contra esa referencia, el 72.5 % live queda a −8.9 pp (dentro del ruido con n=40) |
| 3 | §4: "da igual cuándo caiga el tick" (primer disparo 77.1 % vs ancla 77.7 % vs aleatorio 77.5 %) | ⚠️ **Válido solo dentro de [25.5, 30]**: no se midió ningún tick > 30, que es donde el monitor sí disparó 2 veces (ambas perdidas) |
| 4 | §4: "Antes de min 25 devuelve `unavailable`" | ⚠️ **Incompleto, pero por otra razón**: el modelo **clampea** a t=30 (verificado en los 336 logs), así que no extrapolaba. Lo que faltaba es que la ventana de apuesta del monitor (hasta el min 33) **supera el rango válido del modelo (30)**: se colocaron 2 apuestas sobre un Q4 ya empezado, con datos de Q3 |
| 5 | §2 mejora #16: "`block_toxic_leagues` en inferencia (PASS)" | ❌ **Inerte**: 5 de 15 patrones no casan con ninguna liga (ni del train ni live); 2 más solo casan con ligas históricas; y las 2 ligas `block_candidate` del propio modelo **estaban en el train** |
| 6 | §3/§5: "36-40 % de partidos sin cuota" (el bloque "fuerza" cubre ese hueco) | ⚠️ **Sesgo de selección en vivo**: hay cuota en el 67.9 % de las inferencias pero solo en el **37.5 %** de las apuestas. No es pérdida de datos (0 de 40 partidos apostados tienen cuota ML en `match_odds`), es que **el modelo apuesta más donde no hay mercado** |
| 7 | §5: "Cuota ML no existe en línea (0/40 en el sondeo, 404)" | ✅ **Confirmado**: los 40 partidos apostados no tienen cuota ML en la base. La consecuencia real (el modelo prefiere apostar los que no la tienen) no estaba en el documento |
| 8 | §2/§6: "Paridad OK" (400 partidos, camino live vs dataset) | ✅ **Se sostiene** para el rango entrenado. No cubre el caso `t > 30` (la auditoría solo probó el snapshot 27) |
| 9 | §1: "comparar WR live contra el 77.8 % de backtest antes de operar con dinero" | ✅ **Hecho**: 72.5 % vs 77.8 % → dentro del ruido; la recomendación de esperar 200+ apuestas se confirma |
| 10 | §7 punto 6: "Revisar mensualmente `league_calibration.json` para actualizar los bloqueos" | ❌ **No se hizo**: la revisión nunca ocurrió, y mientras tanto el propio modelo identificó 2 ligas tóxicas que quedaron sin bloquear |

### Lo que el documento no cubría y este análisis añade

1. **Que el gap con v5_2 no es de modelo sino de regla** — y el contrafactual exacto (0.72 → 0.75: ROI@1.40 de +1.50 % a +8.89 %).
2. **Que el modelo en vivo rinde igual que su backtest** (13.8 aciertos esperados vs 13 reales según la calibración de liga): no hay degradación live-vs-offline que depurar.
3. **La apuesta posterior al minuto 30** como mecanismo de pérdida: el modelo clampea a t=30, así que a partir del minuto 31 está prediciendo el ganador de un Q4 **ya empezado** con datos de Q3 (2 apuestas, ambas perdidas).
4. **La lista de ligas tóxicas como mecanismo train/live compartido** (y por tanto doblemente inerte: no podó ni bloqueó).
5. **El sesgo de selección hacia partidos sin cuota de mercado.**
6. **La superficie de beneficio por umbral y cuota**: a 1.40, 0.72 y 0.75 están empatados; a 1.83 el óptimo baja a 0.65 (+69 % de beneficio). El umbral actual es, en el fondo, una apuesta sobre la cuota real del Q4 — que nunca se ha registrado.

---

## 5. Reproducción

```powershell
.venv\Scripts\python.exe tmp\m27_v5_3_audit\diagnose_v5_3_vs_v5_2.py   # gap, contrfactuales, ligas tóxicas
.venv\Scripts\python.exe tmp\m27_v5_3_audit\profit_surface.py          # beneficio esperado por umbral y cuota
.venv\Scripts\python.exe tmp\m27_v5_3_audit\check_minute_clamp.py      # ¿el modelo clampea el minuto? (sí, a 30)
.venv\Scripts\python.exe tmp\m27_v5_3_audit\contrast_with_findings.py   # contraste contra findings_m27_v5_3.md
.venv\Scripts\python.exe tmp\m27_v5_3_audit\confidence_parity.py        # confianza y discordancia en el mismo snapshot
.venv\Scripts\python.exe tmp\m27_v5_3_audit\calibration_live.py         # Brier/LogLoss/calibración por banda
.venv\Scripts\python.exe tmp\m27_v5_3_audit\losses_analysis.py          # desglose de pérdidas (previo)
.venv\Scripts\python.exe tmp\m27_v5_3_audit\threshold_sweep.py          # barrido de umbral (previo)
```

Todos leen `matches.db` de la raíz y solo escriben en `tmp/`. Ningún script modifica la base de datos.

---

## 6. Estado de aplicación (2026-10-20)

| # | Cambio | Dónde | Estado |
|---|---|---|---|
| R2 | `min_conf` 0.72 → **0.75** | `model_outputs/model_config.json` + `OPERATING_CONF` en el pipeline | ✅ aplicado y **durable** (el pipeline ya no puede volver a 0.72: exige WR ≥ 78 % en B) |
| R3 | Ventana de inferencia **[25.5, 30]** | `models/m27_v5_3/predict.py` (valida el `snapshot_minute` de **entrada**, no el clampeado) y `models/m27_v5_2/predict.py` | ✅ aplicado y probado (`test_window_guards.py`) |
| R4 | Bloqueo por **fase de torneo** con motivo | `models/common/competition_stage.py` + `models/m27_v5_3/features.py` (`block_reason`) | ✅ aplicado: se descartan del train y se bloquean en inferencia **amistosos, segunda fase, promoción, preliminar y colocación**; se **mantienen** eliminatorias y copas |
| R4b | Bloqueo por **medición propia** (IC95 < equilibrio) | `league_calibration.json` → `measured_toxic_leagues()` | ✅ aplicado: `1. Division B` (WR 42.1 %) y `Sparkasse Prva Liga Zeni` (WR 47.6 %) |
| R5 | **Modo sombra** para `m27_v5_3` | `monitor_v3/config/constants.py` (`SHADOW_MODELS`) + `monitor_v3/main.py` | ✅ aplicado: se registra pero **no alerta en Telegram** hasta reunir ≥ 200 apuestas resueltas |
| R6 | **Serie de inferencia por minuto** | nueva tabla `model_inference_v3` + `monitor_v3/models/evaluator.py` | ✅ aplicado: una fila por modelo y minuto dentro de la ventana válida, con `p_cal`, margen, señal, motivo y el vector completo de features |
| — | Juveniles y NCAA (R1/R2 de la lista negra) | — | ✅ **medido**: la ablación dice que **no** aporta nada (V3 ≈ V2) y que el AUC en el subconjunto conservado baja −0.0054 → **no se excluyen** |
| — | Reconstruir dataset + reentrenar con el filtro de fases | `build_dataset.py` + `train_m27_v5_3_pipeline.py` | ✅ **hecho** (2026-10-20 00:18). Respaldo en `tmp/m27_v5_3/backup_pre_fase_filter/` |

### Modelo reentrenado con el filtro de fases (2026-10-20)

| | antes (2026-10-18) | **ahora (2026-10-20)** | Δ |
|---|---|---|---|
| Partidos elegibles | 197.111 | **184.173** | −12.938 (formatos fuera) |
| AUC test (C) | 0.6530 | **0.6550** | **+0.0020** |
| Brier / LogLoss | 0.2308 / 0.6529 | **0.2304 / 0.6522** | mejor |
| Acc test | 61.10 % | **61.21 %** | +0.11 pp |
| WR @0.75 (ancla) | 79.18 % | **82.04 %** | **+2.86 pp** |
| ROI @1.40 @0.75 | +10.86 % | **+14.86 %** | **+4.00 pp** |
| Walk-forward (3 cortes) | WR 74.9 / 75.9 / 77.3 % | **75.0 / 81.5 / 80.3 %** | +0.1 / **+5.6** / **+3.0** pp |
| Sesgo de calibración (0.7, 0.8] | +2.4 % | **+0.3 %** | casi plano |
| Ligas bloqueadas por medición | 2 | **0** | la calibración nueva no tiene ninguna liga bajo el equilibrio |

Composición del set de entrenamiento tras el filtro: **REGULAR 163.475 · ELIMINATORIA 16.618 · COPA 9.982** filas. Cero filas de amistosos, 2ª fase, promoción, preliminar y colocación.

⚠️ **Meseta de la isotónica**: con el modelo nuevo, los umbrales 0.75 y 0.80 eligen prácticamente el mismo conjunto de apuestas (2.172 vs 2.171, mismo WR). Por eso subir de 0.72 a 0.75 no costó volumen y subió el acierto 4 pp.

**Auditoría de paridad** re-ejecutada con los artefactos nuevos: **PARIDAD OK** (features de estado y mercado con diff 0.0; los priors difieren solo por obsolescencia del estado, corr 0.92-0.99).

**Monitor reiniciado** tras los cambios (regla 7 de AGENTS.md): 2 procesos `python.exe` (daemon + bot), 0 ventanas `cmd`, arranque limpio sin errores.

