# 🔎 Auditoría de pérdidas en producción — `m27_v5_3`

> ⚠️ **Vigencia:** este documento analiza el modelo **anterior** (entrenado el 2026-10-18, umbral operativo 0.72). Desde el **2026-10-20** el modelo se reentrenó con el filtro de formatos de torneo y opera en **0.75** dentro de la ventana [25.5, 30] y en **modo sombra** (registra pero no alerta). Las conclusiones de §1–§4 describen lo que pasaba con la configuración antigua; el estado actual y sus contrfactuales están en [`ANALISIS_LIVE_M27_V5_3.md`](modelos/ANALISIS_LIVE_M27_V5_3.md) y los números del modelo nuevo en [`findings_m27_v5_3.md`](modelos/findings_m27_v5_3.md).
>
> **Ámbito:** apuestas reales del modelo `m27_v5_3` registradas en `bet_monitor_log_v3` (18-19 oct 2026).
> **Objetivo:** entender dónde pierde, descartar hipótesis con datos, medir el efecto de bajar `min_conf` y definir qué experimento hace falta para evaluar la señal "multi-momento".
> **Reproducible con:** `tmp/m27_v5_3_audit/losses_analysis.py` y `tmp/m27_v5_3_audit/threshold_sweep.py`.
> **Contexto de entrenamiento (holdout C):** [`findings_m27_v5_3.md`](modelos/findings_m27_v5_3.md).

---

## 1. Muestra

| Dato | Valor |
| --- | --- |
| Apuestas `BET` registradas | **40** (39 resueltas) |
| Resultado | **28 ✅ / 11 ❌ = 72 %** |
| Periodo | 18 → 19 oct 2026 |
| Ligas distintas | **28** (1,4 apuestas por liga) |
| Picks | HOME 25 · AWAY 14 |
| Evaluaciones con `p_cal`+`margin` y Q4 resuelto (para el barrido) | **314** |

> ⚠️ Con 39 apuestas, cualquier diferencia menor a ~15 pp es ruido. Las conclusiones de §2-§4 son direccionales; las de §5 se apoyan en el holdout de entrenamiento.

---

## 2. Dónde perdió

| Margen Q4 (pick − rival) | Conf | Liga |
| ---: | ---: | --- |
| −4 | 0.88 | Women's Basketball Division A (Chipre) |
| −4 | 0.83 | 1. ZLS, Women (Serbia) |
| −1 | 0.75 | Swedish Basketball Superettan |
| −1 | 0.75 | 1. Division (Finlandia) |
| **0** (18-18) | 0.83 | Poland 2nd Basketball League |
| −7 | 0.75 | Poland 2nd Basketball League |
| −11 | 0.75 | Poland 2nd Basketball League |
| −12 | 0.76 | A1 Women (Italia) |
| −12 | 0.75 | LNB, Women (Rumanía) |
| −8 | 0.73 | Cyprus Basketball Division A |
| −7 | 0.83 | FSHB Superliga (Albania) |

- **3 de 11** derrotas fueron por ≤4 puntos (una de ellas el empate, que desde la regla nueva cuenta como pérdida).
- **8 de 11** fueron palizas del rival en el cuarto (−7 a −12).
- 3 derrotas en **Poland 2nd** (2/5 en esa liga) es el único cluster repetido; con 5 casos no es concluyente.

---

## 3. Hipótesis descartadas

| Hipótesis | Dato | Veredicto |
| --- | --- | --- |
| El lado del pick | HOME 18/25 (72 %) vs AWAY 10/14 (71 %) | ❌ sin señal |
| Falta de cuotas de mercado | `has_ml=True` 71 % vs `False` 72 % | ❌ sin señal |
| Ligas femeninas | Mujeres 12/16 (**75 %**) vs hombres/mixto 16/23 (70 %) | ❌ al contrario |
| Pick yendo perdiendo al apostar | −5..−9 → 2/2 · +10 o más → 69 % | ❌ sin monotonía |
| El propio margen esperado (`expected_q4_margin`) | Todas las apuestas pasan su filtro (≥3) | ❌ sin varianza |
| Minuto 27 vs 28 | 27 → 76 % · 28 → 67 % | ⚠️ débil |

---

## 4. Hipótesis con evidencia

1. **Banda marginal de confianza (la más sólida en vivo).** 0.70-0.75 → **7/12 (58 %)** frente a **12/14 (86 %)** en 0.75-0.80. El umbral operativo (0.72) admite casos que apenas lo cruzan y **6 de las 11 pérdidas tienen conf ≤ 0.76**.
2. **Minuto fuera de la ventana entrenada.** `m27_v5_3` se entrenó en t ∈ [25.5, 30] y **no bloquea tardíos**: min ≥31 → **0/2** (Karkkila 19-20 en el 32; Teleorman 23-11 en el 31).
3. **Varianza irreducible.** 3/11 derrotas por ≤4 puntos: no es filtrable por modelo.
4. **Sin soporte estadístico por liga.** 28 ligas / 39 apuestas; 4 de las 11 derrotas son goleadas en ligas con 1-2 muestras. No hay base para "ligas malas".
5. **Empates.** 1 caso (Roben 18-18) que la regla nueva convierte en pérdida: efecto de negocio, no del modelo.

---

## 5. ¿Qué pasa si bajo `min_conf`? (barrido sobre 314 evaluaciones reales)

Simulación con la regla exacta del modelo (`BET` si `p_cal ≥ thr ∧ margin ≥ mm`, o espejo para visitante), usando `p_cal`/`margin` guardados y el Q4 ya liquidado:

| `min_conf` | Apuestas | Acierto | Yield @1.40 | Yield @1.83 | Nuevas vs 0.72 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.60 | 140 | 71 % | **−1.0 %** | +29.4 % | +99 (69 %) |
| 0.62 | 137 | 72 % | +1.2 % | +32.2 % | +96 (71 %) |
| 0.65 | 100 | 72 % | +0.8 % | +31.8 % | +59 (69 %) |
| **0.68** | 53 | 77 % | **+8.3 %** | +41.6 % | **+12 (83 %: 10✅/2❌)** |
| 0.70 | 53 | 77 % | +8.3 % | +41.6 % | +12 (83 %) |
| **0.72 (operativo)** | **41** | **76 %** | **+5.9 %** | +38.4 % | — |
| 0.75 | 27 | 81 % | +14.1 % | +49.1 % | — |
| 0.78 | 14 | 79 % | +10.0 % | +43.8 % | — |
| 0.80 | 12 | 75 % | +5.0 % | +37.3 % | — |
| 0.85 | 3 | 67 % | −6.7 % | +22.0 % | — |

**Acierto por banda de `p_cal`** (monótono → buena calibración):

| Banda | Acierto | vs break-even 52.4 % @1.91 |
| --- | ---: | --- |
| 0.50-0.55 | 46/80 (58 %) | rentable |
| 0.55-0.60 | 60/94 (64 %) | rentable |
| 0.60-0.65 | 27/40 (68 %) | rentable |
| 0.65-0.70 | 31/47 (66 %) | rentable |
| 0.70-0.75 | 19/26 (73 %) | rentable |
| 0.75-1.01 | 22/27 (81 %) | rentable |

### El dato que manda: el holdout de entrenamiento

La tabla del holdout C (98 476 filas / 35 480 partidos ancla) en [`findings_m27_v5_3.md`](modelos/findings_m27_v5_3.md) §1 dice lo contrario que la muestra viva:

| P ≥ | WR (holdout) | ROI @1.40 (holdout) | Muestra viva (53/41 apuestas) |
| ---: | ---: | ---: | --- |
| 0.68 | 75.5 % | **+5.6 %** | 77 % / +8.3 % |
| **0.72 (operativo)** | 77.8 % | **+8.9 %** | 76 % / +5.9 % |
| 0.75 | 79.3 % | +11.0 % | 81 % / +14.1 % |

**Conclusión:** en vivo, bajar a 0.68 suma +29 % de volumen y parece mejorar (+8.3 % vs +5.9 % @1.40), pero en el holdout (250 veces más datos) **0.72 rinde mejor que 0.68 (8.9 % vs 5.6 %)**. La señal viva es ruido de 12 apuestas → **no se debe bajar el umbral**; si acaso, subirlo a 0.75 mejora el ROI por apuesta en ambos conjuntos, a costa de −34 % de volumen.

---

## 6. ¿Evaluar en varios momentos del partido? (multi-momento)

**Hoy no es medible** con lo que guardamos:

- `ctx_from_payload()` deriva el minuto del **reloj en vivo** (`game_seconds_played`) e **ignora** `snapshot_minute`; el payload guardado está congelado en el instante de la inferencia. Re-inferir con otro minuto devuelve lo mismo.

Evidencia observacional disponible:

- Apuestas reales: min 27 → 22/29 (76 %) · min 28 → 4/6 (67 %) · min 29-30 → 2/2 · **min ≥31 → 0/2**.
- En el barrido, las entradas tardías (min 30-35) que se añadirían a 0.68 fueron **10/12 (83 %)** → **no hay evidencia de que apostar más tarde sea peor**.

**Experimento necesario para responderlo:**

1. **Instrumentar** cada tick: guardar por modelo y minuto `p_cal`, `margin` y `pick` (tabla nueva `model_ticks_v3` o columnas extra en `live_snapshots_v3`).
2. Tras ~2 semanas comparar tres políticas: **un tiro** (min 27, actual), **consenso** (la señal debe repetirse en 2 de 3 ticks) y **entrada en el primer tick válido con salida si se revierte**.
3. Métricas: acierto, volumen y yield @1.40/@1.83 por política.

---

## 7. Guards propuestos

| Guard | Efecto medido | Decisión |
| --- | --- | --- |
| **G1: bloquear `inference_minute > 30`** | 28/37 (76 %) quitando **0 ✅ / 2 ❌** | **Aplicar** (coste cero y fuera de la ventana entrenada) |
| G2: bajar `min_conf` a 0.68 | +12 apuestas, +8.3 % @1.40 en vivo… pero 5.6 % en holdout | **No aplicar** (contradicho por el holdout) |
| G3: subir `min_conf` a 0.75 | 81 % WR, +14.1 % @1.40, −34 % volumen | Opcional: solo si se prioriza ROI por apuesta |
| G4: excluir Poland 2nd (2/5) | 26/34 (76 %) | No aplicar (sobreajuste con 5 casos) |
| G5: vigilar picks +1..+9 al descanso | 5/8 (62 %) | Solo monitorizar |

---

## 8. Limitaciones y siguientes pasos

1. **n = 39** apuestas resueltas → cualquier guard por liga o por banda es sobreajuste.
2. **No se registra la cuota real** de Q4 en el log, así que el yield es teórico (@1.40 / @1.83). Instrumentar `decimal_odds` por apuesta habilitaría ROI real.
3. Acumular **≥150-200 apuestas** antes de tocar umbrales o listas de ligas.
4. Próximos análisis (cuando haya volumen): por **probabilidad calibrada** en vez de `confidence`, por **tier de liga** (`leagues_classification`) y correlación con `graph_points_count` / `h2h_available`.

---

## 9. Reproducibilidad

```powershell
# pérdidas, buckets y guards
.venv\Scripts\python.exe tmp\m27_v5_3_audit\losses_analysis.py

# barrido de min_conf con yields @1.40/@1.83 y bandas de p_cal
.venv\Scripts\python.exe tmp\m27_v5_3_audit\threshold_sweep.py
```
