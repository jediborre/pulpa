# Estudio preparatorio de `m27_v5_4`

> **Esto NO es el modelo.** Es el estudio que precede al modelo: qué falló en
> `m27_v5_3`, qué capas se perdieron del monitor viejo, y qué ideas quedan
> sobre la mesa con su evidencia medida. Fecha: 2026-10-20/21.

---

## 1. Qué le pasó a `m27_v5_3` (la evidencia, no la impresión)

| | n | WR | esperado | gap |
|---|---|---|---|---|
| apuestas vivas a 0.75+ (3 días) | 46 | **68.9 %** | 80.7 % | −12 pp (~2σ, p≈0.03) |

Con la cuota real de **~1.40** el punto de equilibrio es **71.4 %**: las 46
apuestas vivas suman **−2.6 unidades** (−5.7 % ROI). Es decir: **a 1.40 el
modelo está en el mejor caso en break-even y en el peor slightly negativa.**

Antes de culpar al modelo se descartó **todo lo que fuera código**:

| causa | veredicto |
|---|---|
| liquidación | ❌ las 5 de hoy cuadran con el Q4 real de `quarter_scores_v2` |
| features de marcador | ❌ vivo vs BD: diferencia media **−0.01**; solo el 8 % difiere ≥3 puntos |
| predicción | ❌ el backtest sobre los **mismos 37 partidos** da **70.3 %** y el vivo **70.3 %** |
| cuotas | ❌ la ruta viva usa las **pre-partido**, igual que el entrenamiento |
| minuto | ❌ acierto y calibración planos de 27 a 30 |
| efecto basura | ❌ en partidos decididos el modelo es **mejor** (83.4 %) |
| mezcla de ligas | ❌ el backtest en las 33 ligas vivas da 78.1 % (no 69 %) |
| cobertura de liga/equipo | ❌ en el backtest las ligas desconocidas son las **mejores** (82.3 %) |

Y la comprobación global: el acierto de **todas** las predicciones vivas es
**59.5 %** con confianza media **60.6 %** → bien calibrado. El mal día está en
la submuestra 0.75+ (n=6 hoy).

**El problema estructural es la muestra**: a 0.75 el modelo dispara **6 veces al
día** → 200 apuestas resueltas llevan un mes. Con la serie por minuto se puede
calcular la curva de WR vivo en **cualquier** umbral (`tools/live_threshold_curve.py`),
así que el veredicto llega antes sin tocar la apuesta.

---

## 2. Las premisas (definidas por el usuario, no negociables)

| # | premisa | consecuencia de diseño |
|---|---|---|
| **P1** | el modelo **debe depender de datos**; si no hay datos, **no apostar** | el modelo necesita un *gate* de cobertura: sin historia de entrenamiento para la liga/equipos, no hay betting |
| **P2** | si la liga es **débil**, no apostar | filtro de calidad de liga medido, no de criterio |
| **P3** | además, **post-filtros** | la pila de filtros vive FUERA del modelo, como en el monitor v1 |

---

## 3. Las capas que se perdieron (recuperadas de `monitor_v1`)

El `v6_2` rendía bien **con el monitor v1 encima**, no solo por el modelo. La
pila del v1 (`monitor_v1/bet_monitor.py`) tenía:

| capa | reglas | archivo:línea |
|---|---|---|
| **blacklist operativa de ligas** | ~45 patrones SQL (femenil, eliminatorias, juvenile, ligas descartadas a mano: NBA, ABA Liga, Basketligaen, Élite 2, Turkish Super League, Swedish Superettan, Portugal LBP, LNB, …) | `monitor_v1/FILTROS_LIGAS.md:52-102` |
| **filtro de edge (v6_2)** | `p_pick ≥ 0.58` y `edge = p_pick − 1/1.40 > 0` (solo apuesta si su probabilidad supera el equilibrio a 1.40) | `bet_monitor.py:2382-2390` |
| **stake por Kelly** | Kelly × 0.25 con tope del 5 %, mínimo de stake 25, y escalones 25/50/75/100 según `p_pick`, `edge` y `k_strength` | `bet_monitor.py:2392-2420` |
| **reglas por liga** | B1 League (rechaza la zona 50-70 %), B2 League, Primera FEB, LMB, Colombia LPB, Korean Basketball League | `bet_monitor.py:2601-2640` |
| **ligas volátiles** | umbral 0.55 (Brazil NBB, Puerto Rico BSN, Israeli National, …) y **rechazo si confianza ≥ 0.80** | `bet_monitor.py:2305-2321` |
| **penalización al local** | +5 % de umbral en Brazil NBB, Puerto Rico BSN, China CBA, New Zealand NBL | `bet_monitor.py:2310-2314` |
| **fase eliminatoria** | +5 % de umbral en playoffs/knockout/cup/final (excepción: EuroLeague play-in) | `bet_monitor.py:2586-2599` |
| **visitante en liga fuerte** | stake reducido (0.25) si el visitante va 30-45 % en EuroLeague/B1/B2/China CBA/Germany BBL/BNXT | `bet_monitor.py:2300-2303` |
| **sobreconfianza** | ≥ 0.80 → micro-stake (0.15-0.30) | `bet_monitor.py:2323-2325` |

### 3.1 ¿Sirven para `m27_v5_3`? **Medido, no supuesto**

Replay de la pila del v1 sobre el backtest (bloque C, 0.75) y sobre la serie viva:

| variante | backtest · apuestas | WR | u@1.40 | vivo · apuestas | WR |
|---|---|---|---|---|---|
| sin filtros (0.75) | 7.959 | 80.7 % | 1.036 | 7 | 42.9 % |
| pila v1 completa | 2.658 | 81.5 % | 376 | 4 | **25.0 %** |
| solo blacklist de ligas | 2.666 | 81.5 % | 375 | 4 | **25.0 %** |
| solo filtro de edge (v6_2) | 7.959 | 80.7 % | 1.036 | — | — |

**Conclusión: la pila del v1 NO es gratis para `m27_v5_3`.** La blacklist quita
el 67 % del volumen para +0.8 pp de WR, y **en vivo de hoy habría empeorado el
resultado** (25 % vs 42.9 %) porque descartó 2 aciertos y 1 fallo. El filtro de
edge no hace nada (0.75 ya supera el equilibrio de 1.40).

> **Por qué funcionaba con `v6_2` y no con `m27_v5_3`:** la blacklist del v1
> excluía las ligas femeniles y las eliminatorias porque **el `v6_2` no estaba
> entrenado en ellas**. `m27_v5_3` **sí** lo está, y el estudio de fases ya
> mostró que las eliminatorias rinden **más** (84.3 %) que la temporada regular.
> Los filtros del v1 eran **específicos del modelo que los usaba**; trasplantarlos
> sin medir es exactamente el error que warns este documento.

**Lo que sí se hereda como idea (a re-medir sobre `m27_v5_3`):** el *stake por
Kelly* (hoy el monitor usa una fija) y las reglas por liga/phase/volátil, pero
recalculadas sobre el modelo actual y con la cuota real.

---

## 4. IDEA PRINCIPAL · Despacho por condición (con detector tick a tick)

**Qué propone el usuario:** antes de lanzar la inferencia, un proceso
independiente que mire el partido tick a tick, detecte si ocurrió el evento que
el modelo necesita y, según eso, lance un modelo **especialista** u otro.

### 4.1 Qué eventos exige `m27_v3` (y por qué se desplomaba en vivo)

`m27_v3` (AUC teórico 0.789) se apoyaba en features que **exigen que el evento
ocurra**: `current_scoring_run`, `max_scoring_run`, `trailing_now_recent_run_2m/3m`,
`halftime_trailer_cutting_in_q3`, `pbp_density` (`models/m27_v3/features.py:100-124`).
En vivo, muchas veces ese evento no ocurre → la feature se degenera y la
efectividad real se hunde. **El modelo no puede saber que su condición no se
cumplió**, porque la feature "no hay racha" es indistinguible de "no hay info".

### 4.2 Lo medido: qué condiciones separan de verdad

Detectores calculados tick a tick desde `match_events` (hasta el min 27) sobre
el backtest (bloque C, 0.75) — **2.612 apuestas**:

| condición | con la condición | sin la condición | ¿sirve? |
|---|---|---|---|
| **C1 · racha activa ≥ 4 puntos** | **83.4 %** (n=1.063) | 79.0 % (n=1.549) | ✅ **+4.4 pp (p≈0.014)** |
| C2 · racha máxima ≥ 8 | 80.8 % (n=2.612) | 80.0 % (n=125) | ❌ se cumple en el 95 % |
| C3 · el que va perdiendo viene con racha 3 min | 80.8 % | 81.3 % (n=2.034) | ❌ al revés |
| C4 · densidad ≥ 30 eventos | 80.8 % | 80.0 % (n=15) | ❌ trivial |
| C5 · partido competitive (\|dif\| ≤ 8) | 80.8 % | 81.0 % (n=2.148) | ❌ al revés |
| C6 · el trailer de medio tiempo corta en Q3 | 80.8 % | 75.4 % (n=114) | ❌ al revés |

**C1 es el único detector que separa**, y de forma **monótona en toda la banda
de confianza**:

| banda | C1 sí | C1 no | diferencia |
|---|---|---|---|
| 0.75-0.80 | 77.8 % (n=523) | 74.6 % (n=838) | +3.2 pp |
| 0.80-0.85 | **87.4 %** (n=270) | 83.4 % (n=445) | +4.0 pp |
| 0.85+ | **90.4 %** (n=270) | 85.7 % (n=266) | +4.7 pp |

### 4.3 Y el modelo no lo sabe: la calibración depende de la población

| población | n | confianza media | WR real | gap |
|---|---|---|---|---|
| **C1 sí** (con racha) | 1.063 | 0.8152 | 83.4 % | **+1.9 pp** (el modelo se queda corto) |
| **C1 no** (sin racha) | 1.549 | 0.8059 | 79.0 % | **−1.6 pp** (el modelo se pasa) |

Esto es exactamente lo que un modelo único no puede arreglar: **la misma
confianza significa cosas distintas según haya racha viva o no**. Un despacho (o
un post-filtro) sí lo puede.

### 4.4 Diseño propuesto

```
monitor_v3 (actual)            monitor_v4 (independiente, NUEVO)
─────────────────────          ─────────────────────────────────────
polling tick a tick      ──▶   1. DETECTORES tick a tick sobre el PBP:
                                 · racha activa (C1)  ← el único validado
                                 · (futuro: faltas, tiros libres, rotaciones)
                              2. ¿C1 se cumple?
                                 NO → modelo A (o umbral más alto)
                                 SÍ → modelo B especialista en racha
                              3. la inferencia especialista solo se lanza
                                 si su precondición se cumple
```

**Costo de implementación: ~cero peticiones extra.** El detector necesita solo
los eventos del PBP, que **ya vienen en el payload** que el monitor descarga.

**Lo que falta decidir (con medición, no con criterio):**
1. ¿Umbral del detector en 4 puntos? Probar 3/4/5/6 sobre el backtest.
2. ¿Modelo especialista o solo un **ajuste de calibración por población**
   (más barato y menos riesgo de sobreajuste)?
3. ¿El dispatch sube el WR o solo reparte? Con la cuota de 1.40, C1 sí ya está
   en 83.4 % (muy por encima del equilibrio) y C1 no en 79.0 % (también por
   encima): **filtrar C1 no es lo que hay que hacer**; lo que hay que hacer es
   **bajar el umbral en C1 sí** (más volumen donde el edge es real) y
   **subirlo en C1 no**.

---

## 5. IDEA · La data en vivo como entrada principal, H2H y mercado como sustento

**Qué propone el usuario:** que el modelo se apoye en la data en vivo (cómo se
está jugando, momentum, PBP) y use H2H y mercado solo como sustento.

### 5.1 El mercado: hoy aporta AUC, pero **no** WR — y puede estar manipulado

El usuario tiene razón en que las cuotas pueden estar "fabricadas" en mercados
con poca acción. La pregunta medible es: **¿el edge del modelo viene del
mercado?**

| variante | AUC | WR@0.75 | apuestas | u@1.40 | WR@0.70 | apuestas |
|---|---|---|---|---|---|---|
| **con mercado** (desplegado) | 0.6552 | 80.7 % | 7.959 | 1.036 | 75.9 % | 15.912 |
| **sin mercado** (features a NaN) | 0.6422 | **81.5 %** | 6.867 | 969 | 76.3 % | 12.825 |

Y en la población **sin mercado** (37.511 filas, la que corresponde a la
premisa P1):

| | AUC | acierto |
|---|---|---|
| modelo con mercado | 0.6891 | 63.3 % |
| modelo sin mercado | **0.6901** | 63.0 % |

**El mercado aporta 0.013 de AUC, pero CERO gain de WR en el umbral operativo**, y
en la población sin mercado el modelo entrenado sin él es **igual o mejor**
(0.6901 vs 0.6891). O sea: **el edge no viene de las cuotas.** Quitarlas es una
decisión de **riesgo** (una señal manipulable no debería sostener el edge) con un
costo medido de ~1.3 pp de AUC y **cero** en WR.

### 5.2 Qué eventos hay disponibles (para buscar detectores nuevos)

| fuente | qué tiene | ¿sirve para detectores? |
|---|---|---|
| `match_events` / `play_by_play` (histórico) | **solo eventos de anotación**: cuarto, tiempo, jugador, puntos, marcador | ✅ runs, ventanas, densidad, ritmo |
| `live_snapshots_v3.players_json` (vivo) | por jugador: titular, minutos, puntos, **faltas**, plus-minus, rating | ⚠️ en vivo sí, pero **sin histórico** |
| `live_snapshots_v3.team_stats_json` (vivo) | stats de equipo por partido (tiros libres, tiros de 2, …) | ⚠️ el monitor solo captura el agregado `ALL` |

**Consecuencia clave:** cualquier detector nuevo que no sea de anotación
(faltas, tiempos muertos, sustituciones) **exige un backfill histórico** para
poder entrenarse. Con ~1 req/s y 250.000 partidos, un backfill de `player_stats`
son ~70 h. **Antes de proponer un detector de faltas, hay que medir si
`match_events` + `graph_points` no bastan** (ya dan ritmo, presión y rachas).

### 5.3 SÍ: hay forma de descargar las faltas, y es barato si se acota a las ligas rentables

**El endpoint es `GET /event/{id}/statistics`** → **1 request por partido**
(no 20), y trae las stats de equipo **por cuarto**:

```
periodos: ['1Q', '2Q', '3Q', '4Q', 'ALL']
3Q  Fouls 10-6 | Free throws 5-8 | Turnovers 4-5 | Time spent in lead 0-720
```

(verificado en vivo sobre el partido 16814806, con el parser canónico
`match/scraper.py::_parse_team_statistics`, que usa `homeValue`/`awayValue`).

**Lo que esto abre como detector** (todo disponible en vivo, sin pedir nada más):
- **faltas de Q3** (y su diferencia): la presión antes del cuarto → tiros libres
- **tiros libres de Q3**: el ataque al vidrio / el juego en la zona
- **pérdidas de Q3**
- **"time spent in lead" de Q3**: quién controló el cuarto → **el proxy de momentum
  más limpio que existe, y es independiente del mercado**

**Coste de un backfill selectivo** (la pregunta):

| universo | partidos | horas a 1 req/s | a 2 req/s |
|---|---|---|---|
| todo el dataset elegible | 184.206 | 51 h | 26 h |
| **las 13 ligas por encima del equilibrio** (WR ≥ 71.4 % con ≥ 30 bets) | **8.706** | **2.4 h** | **1.2 h** |
| las 13 ligas, solo los partidos con bets históricos | 3.581 | 1.0 h | 0.5 h |

Ritmo medido: **1.6 req/s sin 403** (el límite seguro documentado es 1 req/s).

**Cobertura medida** (muestra de 24 partidos de las ligas rentables, 2023-2025):
- **92 %** devuelven las estadísticas de equipo **con faltas**
- el desglose **por cuarto** está en una **minoría** (los partidos grandes: NBA,
  G League, NCAA; las ligas pequeñas traen solo el agregado `ALL`)

### 5.4 Lo que se midió de verdad (y cambia el plan)

| evento | histórico (10 min) | histórico (NBA/G League/WNBA/NCAA) | en vivo |
|---|---|---|---|
| puntos y tiros libres **por cuarto** | ✅ 100 % con valor | ✅ | ✅ |
| **faltas por cuarto** | ❌ **0 %** (el feed `incidents` no las trae: vienen a 0-0) | ✅ | ✅ |
| tiempos muertos / pérdidas por cuarto | ❌ 0 % | ✅ | ✅ |
| agregado `ALL` (faltas, tiros libres, tiempo en ventaja, mayor ventaja, cambios de ventaja) | ✅ 92 % | ✅ | ✅ (parcial hasta el momento) |

**El feed `incidents` en histórico solo trae la anotación**: el reconstructor por
cuarto (`match/scraper.py::_parse_period_stats`) sí genera las filas de faltas,
pero con **valor 0-0** en las ligas de 10 minutos. En las ligas grandes
(`/statistics` con 1Q-4Q) sí hay faltas por cuarto.

**Consecuencia directa:** una feature de **faltas por cuarto** no se puede
entrenar con historia en las ligas donde opera `m27_v5_3` (10 min). Quedaría como
señal **solo en vivo**, a partir de ahora. Lo que **sí** tiene paridad completa
(entrenamiento ↔ vivo) es el **perfil de anotación por cuarto** (puntos y tiros
libres), que además ya está parcialmente en el modelo (PBP, ritmo, presión).

> El agregado `ALL` en vivo es **acumulado hasta ese momento**, así que
> Q1+Q2+Q3 histórico == ALL parcial en vivo: esa es la vía de paridad para el
> agregado.

**Lo que sí se deja montado** (todo ya funcionando):
- `models/common/team_stats.py`: selección canónica compartida (la misma en
  vivo y en el backfill → paridad por construcción).
- `team_stats_v3` + `team_stats_backfill_state`: tablas nuevas.
- Captura viva ampliada: `live_snapshots_v3.team_stats_json` ahora guarda **Q1-Q4
  + ALL** (antes solo el agregado `ALL`).
- `tools/backfill_team_stats.py`: proceso independiente, reanudable, a 2 req/s
  (medido 1.24-1.6 req/s sin 403), con las mismas dos fuentes que el monitor
  (`incidents` + `statistics`).
- Opción **40** del `menu.bat` (bloque de ingesta) + alias de CLI
  `menu.bat backfill_team_stats`.

**Estado del backfill (2026-10-21): COMPLETO** — 7.358 partidos de las 15 ligas
rentables, **292.487 filas**, con desglose **Q1-Q4 en 5.800+ partidos** (250
partidos sin estadísticas en el API: `status='empty'`).

**Lo que decide el estudio de v5_4** (en orden):
1. ¿El perfil de anotación por cuarto (tiros libres sobre todo) añade algo sobre
   el Q4? → tiene paridad, se puede medir ya en el backtest.
2. Las faltas: primero se acumulan en vivo (la captura ya está) y se validan
   sobre la muestra viva, no sobre historia.
3. El agregado `ALL` (el "carácter del partido": muchas faltas, muchos tiempos
   muertos, muchas cambios de ventaja) como **filtro** de la población: ¿los
   partidos turbulentos son distintos para el Q4? Es un filtro, no una feature,
   y por lo tanto funciona con el agregado.

---

## 6. El resto de ideas (medidas y descartadas)

| idea | veredicto | evidencia |
|---|---|---|
| P0 · recalibración condicional por población/fase | ❌ | AUC plana; WR a 0.75 cae 1.6-2.6 pp |
| P0' · calibración continua (Platt) | ✅ **aplicada** | 7.838 confianzas (antes 135), LogLoss 0.6518, +10 % de beneficio |
| P1 (modelo) · política por población | ⏸ | la decide el EV (que no se puede medir: no hay mercado de Q4) |
| P2 · arquitectura de distribución del margen | ❌ | AUC 0.6064 vs 0.6537 |
| P3 · fine-tuning con foco en la banda | ❌ | WR 92 % pero 68 u de 374; AUC y Brier empeoran |
| P5 · más datos (líneas del mercado) | ❌ | la línea sola da AUC 0.595 < modelo 0.627; sumarla no mejora |
| F1 · prior contextual por liga × estado | ❌ | −0.0164 AUC |
| F2 · patrón de quarters | ❌ | −0.0002 AUC |
| F3 · ventaja extrema | ❌ | −0.0002 AUC |
| F4 · condición de racha activa (**C1**) | ✅ **viable** | +4.4 pp de WR, monótona en las 3 bandas de confianza |

---

## 7. Lo que hay que decidir antes de escribir una línea de `m27_v5_4`

1. **¿Umbral del detector C1?** (3/4/5/6 puntos) — se mide en el backtest.
2. **¿Especialista o ajuste de calibración por población?** El ajuste es más
   barato y con menos riesgo de sobreajuste; el especialista es más ceilings.
3. **¿Se mantiene la dependencia del mercado?** Medido: no aporta WR. Decisión de
   riesgo, no de rendimiento.
4. **¿P1 y P2 (gate de datos y calidad de liga) con qué reglas exactas?**
   - *Datos*: hoy la cobertura se midió y **no** discrimina (las ligas
     desconocidas rinden 82.3 %, las mejor). Habría que redefinir "datos" como
     lo que el modelo **realmente necesita**: el PBP denso, la gráfica, los
     cuartos.
   - *Liga débil*: hoy hay 0 ligas bloqueadas por medición
     (`league_calibration.json` → 0 `block_candidate`). Con el filtro de fase
     (amistosos/segunda fase/promoción/preliminar/puesto) ya se descartan ~12.000
     partidos de torneo. Falta medir "liga débil" con un criterio (WR histórico
     con n mínimo).
5. **¿Qué stake?** El v1 usaba Kelly; hoy es fija. Con la cuota real de 1.40 y
   el WR vivo actual (68.9 %), el Kelly **no** da stake positivo en las bandas
   altas: ese es el número que hay que mirar primero.
6. **¿Monitoreo independiente para los detectores?** Sí, es lo que propone el
   usuario y aísla la lógica de condiciones del ciclo del monitor principal.
