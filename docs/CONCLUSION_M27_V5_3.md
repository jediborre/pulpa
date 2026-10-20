# ✅ Conclusión del análisis de m27_v5_3 — qué se descubrió, qué se cambió y qué queda

> Fecha: 2026-10-20 · Alcance: por qué `m27_v5_3` rendía por debajo de `m27_v5_2` en vivo, qué parte era problema del modelo y qué parte era de la configuración.
> Detalle completo y datos: [`ANALISIS_LIVE_M27_V5_3.md`](./ANALISIS_LIVE_M27_V5_3.md) · Ficha del modelo: [`findings_m27_v5_3.md`](./findings_m27_v5_3.md) · Auditoría de pérdidas: [`AUDITORIA_M27_V5_3_PERDIDAS.md`](./AUDITORIA_M27_V5_3_PERDIDAS.md).

---

## 1. La pregunta y la respuesta

> «¿Por qué a m27_v5_3 no le va tan bien como a m27_v5_2?»

**El modelo no era el problema.** En los snapshots que ambos modelos vieron en vivo, `m27_v5_3` acertaba **más** (34/45 vs 32/45) y estaba **mejor calibrado** (Brier 0.1907 vs 0.1976). El gap de 82 % vs 72.5 % venía de **tres cosas que no eran el modelo**:

| # | Causa real | Evidencia | Corrección |
|---|---|---|---|
| 1 | **La regla de disparo era más laxa**: `min_conf 0.72` sin filtro de margen, frente a `0.75` **y margen ≥ 1.8** de v5_2 | En 48 snapshots compartidos: ambos disparan 9, **solo v5_3 dispara 28**, solo v5_2 **0**. La banda marginal 0.72-0.75 rindió **62 %** (por debajo del equilibrio de 71.4 % a cuota 1.40) | `min_conf = 0.75`; el pipeline ya no puede volver a 0.72 (exige WR ≥ 78 % en B) |
| 2 | **Se apostaba con el snapshot obsoleto**: el monitor permite hasta el min 33 y el modelo se entrenó en [25.5, 30] | 2 apuestas en min 31 y 32 (ambas perdidas): el modelo predice el ganador de un Q4 ya empezado con datos de Q3 | Ventana **[25.5, 30]** con `unavailable` fuera, en `m27_v5_3` y `m27_v5_2` |
| 3 | **La lista de ligas tóxicas era decorativa**: 16 patrones escritos a mano, 5 sin coincidencia con ningún nombre real | `stage_detail` clasifica por liga base, no por variante → `NBA, Playoffs` figuraba como "Regular Season". 18k partidos de eliminatorias mal marcados | Clasificador multilingüe (`models/common/competition_stage.py`) + bloqueo por **medición propia** del modelo, con **motivo** en el log, el mensaje y la imagen |

**Y un cuarto hallazgo, el más importante de todos:** la premisa de la lista negra (descartar fases de torneo porque se planeaba un modelo propio para ellas) **nunca se aplicó**. `build_dataset.py` solo filtraba por los 16 patrones, así que **amistosos, copas, eliminatorias, championship round, formativas y NCAA** estaban **dentro** del entrenamiento.

---

## 2. La lista negra: qué se descartó y qué se kept

La premisa estaba escrita en [`COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md`](./COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md) (semáforo de arquetipos) y la infraestructura ya existía en `leagues_classification` (3.643 filas con `is_youth`, `is_college`, `competition_type`, `is_playoffs`, `is_final`, `quarter_duration_minutes`). Nunca se conectó.

**Se medió antes de decidir** (ablación: 4 variantes reentrenadas, mismo holdout C de 98.476 filas, umbral 0.75):

| variante | filas train | AUC | apuestas | WR | ROI @1.40 |
|---|---|---|---|---|---|
| V0 real | 204.703 | 0.6530 | 10.588 | 79.18 % | +10.86 % |
| V1 sin amistosos | 201.862 | 0.6530 | 6.260 | 82.43 % | +15.40 % |
| **V2 = decisión tomada** | 191.636 | 0.6524 | 6.762 | **81.74 %** | **+14.43 %** |
| V3 = V2 + sin juveniles y NCAA | 188.795 | 0.6523 | 6.435 | 81.86 % | +14.61 % |

**Conclusiones de la medición:**

1. **Se descartan** amistosos, 2ª fase (*championship round*), promoción/relegation, preliminares y colocación → **+2.56 pp de acierto y +3.57 pp de ROI@1.40** con AUC intacto (−0.0006).
2. **Se mantienen** eliminatorias y copas: el backtest medido da **84.3 %** y **82.8 %**, por encima del 77.0 % de la temporada regular. La premisa de la guía se apoyaba en el **ritmo y los puntos** (playoffs: 144 pts vs 158.6), no en el acierto del ganador de Q4, que es el objetivo del modelo. Para ese objetivo **no son ruido**.
3. **Juveniles y NCAA no se excluyen**: no aportan nada (V3 ≈ V2) y, en el subconjunto que V3 conserva, el **AUC baja −0.0054** → el modelo estaba extrayendo señal útil de ellos.
4. ⚠️ **El número de apuestas es muy sensible a la calibración**: quitar el 1.4 % de filas del train movió el conteo de 10.588 a 6.260 al mismo umbral (la isotónica tiene tramos planos). El umbral hay que elegirlo **junto** con la calibración.

---

## 3. Resultado del reentrenamiento (2026-10-20 00:18)

| | antes (18 oct) | **ahora** | Δ |
|---|---|---|---|
| Partidos elegibles | 197.111 | **184.173** | −12.938 |
| AUC test (C) | 0.6530 | **0.6550** | **+0.0020** |
| Brier / LogLoss | 0.2308 / 0.6529 | **0.2304 / 0.6522** | mejor |
| WR @0.75 (ancla) | 79.18 % | **82.04 %** | **+2.86 pp** |
| ROI @1.40 @0.75 | +10.86 % | **+14.86 %** | **+4.00 pp** |
| Walk-forward (3 cortes) | 74.9 / 75.9 / 77.3 % | **75.0 / 81.5 / 80.3 %** | +0.1 / **+5.6** / **+3.0** pp |
| Sesgo de calibración (0.7, 0.8] | +2.4 % | **+0.3 %** | casi plano |
| Paridad offline ↔ live | OK | **OK** (re-auditada) | — |

Composición del train tras el filtro: **REGULAR 163.475 · ELIMINATORIA 16.618 · COPA 9.982** filas. Cero de los formatos descartados.

---

## 4. Lo que ahora hace el sistema

| Cambio | Dónde | Efecto |
|---|---|---|
| `min_conf 0.75` | `model_outputs/model_config.json` + `OPERATING_CONF` en el pipeline | WR +2.9 pp, ROI +4.0 pp; **durable** |
| Ventana [25.5, 30] | `models/m27_v5_3/predict.py`, `models/m27_v5_2/predict.py` | fuera → `unavailable` (no se apuesta con snapshot obsoleto) |
| Bloqueo por formato de torneo | `models/common/competition_stage.py` + `features.py` | amistosos, 2ª fase, promoción, preliminar, colocación |
| Bloqueo por medición propia | `features.py::measured_toxic_leagues()` | liga con IC95% superior < equilibrio a 1.40 |
| **Motivo del descarte** | `features.py::block_reason()` → `raw_json` → log → **Telegram** → **imagen** | se ve **por qué** no se apostó |
| Modo sombra | `constants.SHADOW_MODELS` | `m27_v5_3` registra pero **no alerta** hasta 200 apuestas resueltas |
| Serie por minuto | tabla `model_inference_v3` | una fila por modelo y minuto dentro de la ventana válida |
| Cuota del mercado Q4 | `mobile_client.summarize_live_odds()` | `q4_ml_home` / `q4_ml_away`: **el dato que faltaba para calcular EV** |

### Cómo se ve el motivo

```
🟢 APUESTA Q4
m27_v5_2 83% → 🏠 Local FC
m27_v5_3 77% [NO BET — torneo amistoso o invitacion (categoria AMISTOSOS)] → 🏠 Local FC
```

En la tarjeta, la fila del modelo que no operó lleva el motivo en naranja:
`no operó: torneo amistoso o invitacion (categoria AMISTOSOS)`.

Motivos que puede mostrar el sistema:
- `torneo amistoso o invitacion (categoria AMISTOSOS)`
- `segunda fase / ronda de campeonato con grupos (categoria SEGUNDA_FASE)`
- `fase de promotion/relegation (formato de supervivencia) (categoria PROMOCION)`
- `fase preliminar o clasificatoria (categoria PRELIMINAR)`
- `fase de colocacion (partidos por puesto, no por titulo) (categoria PUESTO)`
- `liga medida como toxica por el modelo (WR 42.1 %, IC95 [23.1, 63.7] < equilibrio 71.4 % a cuota 1.40)`
- `liga en la lista negra de composicion`
- `Fuera de la ventana de entrenamiento (min 32.0 > 30.0); m27_v5_3 solo infiere en [25.5, 30.0]`
- `Ligas de cuartos de 12 min no soportadas por m27_v5_3`

---

## 5. Lo que queda

1. **El cálculo de EV real** — el paso que desbloquea el resto. Ya se captura la cuota del mercado Q4 (`q4_ml_home`/`q4_ml_away`); falta el join con `bet_monitor_log_v3` y el cálculo `EV = p_model × cuota_q4 − (1 − p_model)`. Es lo que permitirá fijar el umbral por EV en vez de por P: la isotónica tiene una meseta (0.75 ≈ 0.80) y el óptimo de beneficio se mueve de 0.72 a 0.65 según la cuota real sea 1.40 o 1.83.
2. **Verificar que `model_inference_v3` se llena** con partidos reales (requiere un partido que pase del min 27) → habilita el análisis por minuto.
3. **Decidir cuándo `m27_v5_3` sale del modo sombra** (≥ 200 apuestas resueltas).
4. **Terminar el backfill de líneas** de spread/total y reentrenar para activar `mkt_margin_spread` / `total_line`.
5. **Reauditar la lista negra** cuando la calibración acumule evidencia: con el filtro aplicado, hoy hay **0** ligas bloqueadas por medición; la lista se repoblará sola.

---

## 6. Bugs colaterales que aparecieron por el camino

- **La línea de handicap nunca se capturaba**: `re.match` exigía que el nombre empezara por `(`, y SofaScore devuelve `"Home (-3.5)"`. Corregido a `re.search`.
- **`stage_detail` clasifica por liga base, no por variante**: `NBA, Playoffs`, `Argentina Liga Nacional, Playoffs`, `Brazil NBB, Playoffs`, `Mexico LNBP, Playoffs` y `Korisliiga, Playoffs` (18k partidos de eliminatorias) estaban marcados como "Regular Season".
- **La tabla de cuartos tenía dos fuentes de verdad** (`quarter_scores` legacy con 252.956 partidos, que era la que leía el entrenamiento, y `quarter_scores_v2` "canónica" con 2.916, la que escribía el monitor) y **la liquidación usaba datos provisionales**: el monitor guardaba y liquidaba en el mismo instante, con un Q4 que podía estar todavía en curso. 7 apuestas quedaron mal liquidadas (ninguna de `m27_v5_2`/`m27_v5_3`). Corregido con una migración SQL (0,5 s, 250.098 partidos, 0 discrepancias) + dos guards: no cerrar un partido que la API sigue marcando `inprogress`, y no liquidar si la suma de los cuartos no cuadra con el marcador final.

> **No había nada que descargar**: los 2.876 partidos con menos de 4 filas en la tabla legacy tampoco tienen 4 periodos en la API (25/25 en la muestra: son partidos abandonados a los 2 cuartos). El dato ya estaba en la base.

---

## 7. Regla que queda aprendida

> **No decidir con 40 apuestas.** Con 40 apuestas resueltas, el IC95% del WR es de ±14 pp: el 72.5 % live y el 77.7 % de backtest son indistinguibles. La diferencia real se encontró mirando **la configuración, no los resultados**: el mismo modelo, con otra regla de disparo, rendía +4 pp de ROI. Y cuando hay que elegir un formato de torneo, **se mide antes de decidir** (la ablación cambió la lista negra: eliminatorias y copas se quedan, juveniles y NCAA también).
