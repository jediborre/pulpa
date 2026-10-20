# 📉 m27_v5_3 en vivo vs m27_v5_2 — por qué "no le va tan bien" (y por qué en realidad sí)

> Análisis sobre `bet_monitor_log_v3`, ventana **2026-10-18 → 2026-10-19** (682 inferencias, 51 apuestas reales).
> Código: [`tmp/m27_v5_3_audit/`](../tmp/m27_v5_3_audit) · Contexto del modelo: [`findings_m27_v5_3.md`](./findings_m27_v5_3.md).
> Reproducir: `.venv\Scripts\python.exe tmp\m27_v5_3_audit\diagnose_v5_3_vs_v5_2.py` (y `confidence_parity.py`, `calibration_live.py`).

---

## 0. Conclusión en una línea

**El modelo `m27_v5_3` no es peor que `m27_v5_2`: es mejor.** Lo que está mal es la **regla de disparo con la que se opera** (`min_conf 0.72` sin filtro de margen, disparando también más allá del minuto 30 donde el modelo nunca fue entrenado) y una **lista de ligas tóxicas que no coincide con ningún nombre real de SofaScore**.

| | m27_v5_2 | **m27_v5_3** | Lectura |
|---|---|---|---|
| Apuestas live | 11 | **40** | 3.6× más volumen |
| Acierto live | 9/11 = **82 %** | 29/40 = **72.5 %** | −9.5 pp |
| ROI @1.40 | +14.55 % | **+1.50 %** | se Come casi todo el borde |
| Acierto en los **mismos** snapshots | 32/45 | **34/45** | v5_3 gana |
| Brier en los mismos snapshots | 0.1976 | **0.1907** | v5_3 mejor calibrado |
| Fisher exacto entre ambos | — | **p = 0.706** | **diferencia NO significativa** |

Con 40 apuestas el IC95% de 72.5 % es ≈ **[56 %, 85 %]**: el 77.7 % de backtest está dentro del ruido. **Ninguna de las dos versiones ha demostrado en vivo ser mejor que la otra.**

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

### 2.1 Se infiere fuera del rango de entrenamiento (minutos 31-38)

`m27_v5_3` se entrenó con snapshots en **t ∈ [25.5, 30.0]** (`features.py:87`, `model_config.json:snapshot_range`). Pero `predict.py:103` solo valida el **piso** (`MIN_VALID_MINUTE = 25.0`); **no hay techo**.

| | |
|---|---|
| Inferencias con minuto > 30 | **60 de 682** (minutos 31, 32, 33, 34, 35, 38) |
| De ellas, apuestas reales | 2 (minutos 31 y 32) → **0 aciertos** |

La feature `minute` es la primera del vector: fuera de rango, el modelo extrapola. Además el monitor sigue evaluando hasta el minuto 38.

### 2.2 La lista de ligas tóxicas no coincide con ningún nombre real

`is_toxic_league()` hace *substring match* sobre `TOXIC_LEAGUE_PATTERNS` (`features.py:89-99`). Contra las **434 ligas** realmente vistas en vivo:

| Patrón | ¿Coincide? |
|---|---|
| `korean basketball league`, `tipos sbl`, `korisliiga, women`, `betsafe-lkl`, `nlb`, `1st dbbl`, `serie a2, women, group b`, `serie b, group b`, `israeli basketball super league` | ✅ sí |
| `kbl`, `lmb, apertura`, `albania super`, `georgia superli`, `kosovo superlig`, `armenia a-league`, **`cyprus division`** | ❌ **nunca coincide con ninguna liga** |

`cyprus division` no casa con `Cyprus Basketball Division A` (la palabra "Basketball" va en medio) — y **`v5_3` perdió una apuesta en esa liga**.

Peor aún: la calibración offline del propio modelo marcó dos ligas como `block_candidate` y **ninguna de las dos está bloqueada**:

| Liga | WR offline | ROI @1.40 | ¿Bloqueada? |
|---|---|---|---|
| `1. Division B` | **42.1 %** (19 apuestas) | −41.1 % | ❌ no |
| `Sparkasse Prva Liga Zeni` | **47.6 %** (21 apuestas) | −33.3 % | ❌ no |

### 2.3 Lo que el modelo **no** tiene (para contexto, no es la causa)

- **Cuota de mercado:** `has_ml = False` en 25 de las 40 apuestas (62 %). El bloque "C mercado" es el que más aporta al AUC (0.0134). El modelo acierta igual con y sin cuota (73 % vs 72 %), pero en los partidos con cuota la handicap de mercado debería estar descontando ventaja y no se está descontando.
- **Empates de Q4:** desde el commit `9121cb6` cuentan como pérdida, igual para ambas versiones → la comparación es justa.

---

## 3. Recomendaciones

### Prioridad 1 — cambiar la configuración (impacto medido en vivo, sin reentrenar)

| Cambio | Dónde | Efecto live medido |
|---|---|---|
| `min_conf: 0.72 → 0.75` | `models/m27_v5_3/model_outputs/model_config.json` | n 40→27, WR 72.5 %→77.8 %, ROI@1.40 +1.50 %→**+8.89 %** |
| Techo de minuto: `SNAPSHOT_MAX = 30.0` | `models/m27_v5_3/predict.py` (junto a `MIN_VALID_MINUTE`) | descarta 2 inferencias OOD, WR →**76.3 %**, ROI →**+6.84 %** |

> Nota: `model_outputs/` está en `.gitignore`. El cambio de `min_conf` hay que aplicarlo también al artefacto desplegado y reiniciar el monitor (regla 7 de AGENTS.md).

### Prioridad 2 — corregir el bloqueo de ligas

- Reemplazar los 6 patrones muertos por los **nombres exactos** de las 434 ligas reales (o, mejor, casar por `league_id`/prefijo estable en vez de por texto).
- Añadir explícitamente las dos ligas que el propio modelo marcó `block_candidate` (`1. Division B`, `Sparkasse Prva Liga Zeni`).
- Regla sugerida: bloquear cuando el **IC95% del WR offline de la liga** caiga por debajo del equilibrio de la cuota objetivo (71.4 % @1.40) — ya está calculado en `league_calibration.json`.

### Prioridad 3 — no tomar decisiones con 40 apuestas

El backtest de `v5_3` (77.7 % WR) y el live (72.5 %) son compatibles. Con ~40 apuestas el IC95% es ±14 pp. **Se necesitan ≥ 200 apuestas resueltas por configuración** para poder discriminar (o usar un test secuencial). Mientras tanto, los cambios de la Prioridad 1 están justificados por el backtest, no por estos 40 casos.

### Prioridad 4 — revisar el gap entre `league_calibration.json` y las ligas en vivo

22 de las 40 apuestas (55 %) son en ligas sin entrada en la calibración. El modelo opera con priors de liga genéricos (`fallback_info`) en más de la mitad de sus apuestas sin haber medido nunca su WR en Q4.

---

## 4. Reproducción

```powershell
.venv\Scripts\python.exe tmp\m27_v5_3_audit\diagnose_v5_3_vs_v5_2.py   # gap, contrfactuales, ligas tóxicas
.venv\Scripts\python.exe tmp\m27_v5_3_audit\confidence_parity.py        # confianza y discordancia en el mismo snapshot
.venv\Scripts\python.exe tmp\m27_v5_3_audit\calibration_live.py         # Brier/LogLoss/calibración por banda
.venv\Scripts\python.exe tmp\m27_v5_3_audit\losses_analysis.py          # desglose de pérdidas (previo)
.venv\Scripts\python.exe tmp\m27_v5_3_audit\threshold_sweep.py          # barrido de umbral (previo)
```

Todos leen `matches.db` de la raíz y solo escriben en `tmp/`. Ningún script modifica la base de datos.
