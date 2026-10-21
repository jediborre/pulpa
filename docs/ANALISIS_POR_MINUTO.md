# Análisis por minuto: ¿hay un minuto ideal para apostar?

> Instrumentación: `model_inference_v3` (una fila por modelo y minuto válido).
> Análisis reproducible: `tmp/m27_v5_3_audit/analyze_by_minute.py` y
> `tmp/m27_v5_3_audit/inventory_minute_series.py`.

## 1. Qué datos hay

| | |
|---|---|
| filas | **1.164** (4 modelos, 95 partidos) |
| rango | minutos 26 a 33 |
| ventana | desde las 03:16 de hoy (arranque de la instrumentación) |
| cobertura | 91 de los 95 partidos tienen ya el Q4 en `quarter_scores_v2` → evaluables |

El modelo `m27_v5_3` aporta **301 filas evaluables** (84 partidos, minutos 27-30
respetando su gate `[25.5, 30]`).

## 2. Resultado: dentro de la ventana NO hay minuto mejor

`m27_v5_3`, con el resultado real del Q4:

| minuto | n | acierto del pick | p_cal medio | gap (p_cal − real) | Brier | ≥0.75 | WR a 0.75 |
|---|---|---|---|---|---|---|---|
| 27 | 71 | 62.0 % | 0.6081 | −1.2 pp | 0.2576 | 6 | 33.3 % |
| 28 | 70 | 61.4 % | 0.6061 | −0.8 pp | 0.2579 | 6 | 33.3 % |
| 29 | 73 | 58.9 % | 0.6057 | **+1.7 pp** | 0.2644 | 6 | 33.3 % |
| 30 | 61 | **62.3 %** | 0.6091 | −1.4 pp | **0.2549** | 4 | 50.0 % |

**Todo es plano.** Con n≈70 por celda, la desviación estándar del acierto es
~5.9 pp: el rango 58.9-62.3 % son 0.6 σ. Y la **calibración es excelente en
todos los minutos** (gap ≤ 1.7 pp): el acierto real (58.9-62.3 %) sigue a la
confianza media (0.606-0.609) casi exactamente.

> Dato relevante: la confianza media de estas filas es 0.606 y el acierto real
> ~60 %. **El modelo está bien calibrado en vivo**, que es lo que faltaba
> confirmar con datos en vivo.

## 3. Por qué el minuto no cambia nada

| | |
|---|---|
| partidos con más de un valor de `p_cal` | 84 de 85 |
| partidos donde el **pick** cambia entre minutos | **7 de 85** |
| recorrido típico de `p_cal` dentro de la ventana | **0.018** (mediana), máximo 0.095 |

Motivo mecánico (`models/m27_v5_3/features.py`):
- la feature `minute` se recorta a `SNAPSHOT_MAX = 30` (`t = min(max(minute, 20), 30)`);
- a partir del cierre de Q3 el marcador es el **definitivo** (`cur` sale de los cuartos);
- entre 27 y 30 solo cambian el marcador parcial de Q3 y las features de momentum, que
  decaen.

Es decir: dentro de la ventana la información **no cambia de verdad**, y el modelo
responde con un cambio medio de 0.018 en probabilidad.

## 4. El EV por minuto NO es medible con esta fuente

Para decidir el minuto ideal por **precio** haría falta la cuota del mercado que
se apuesta (ganador de Q4). SofaScore **no lo publica**:

| mercado | periodo | pre-partido | in-play |
|---|---|---|---|
| Ganador del partido | `Match` | 161.706 partidos | sí |
| Ganador de la 1ª mitad | `1st half` | 16.984 | no |
| Ganador del **1er cuarto** | `1st quarter` | 15.574 | puntual |
| Handicap / Total | `Match` | 17.200 / 17.069 | sí |
| **Ganador del 4º cuarto** | `4th quarter` | **nunca** | **nunca** |

Comprobado sobre la API en vivo (`event/{id}/odds/1/all`): los mercados que
devuelve son `Home/Away Match`, `Over/Under Match`, `Point spread Match` y a
veces `Home/Away 1st quarter`. **Nunca** `4th quarter`.

Consecuencia práctica: de los 4.334 snapshots de `live_snapshots_v3`, **0** tienen
`q4_ml_home`. El P4 (EV real) sigue bloqueado por la fuente, no por la
instrumentación.

> ⚠️ **Ojo con la cuota de la tarjeta**: la "Cuota 1.83" que sale en la imagen y en
> el mensaje es el **ML in-play del partido completo** (`ml_home`/`ml_away`), no el
> precio del mercado de Q4. Sirve como ilustración del payout, **no** como el
> precio de la apuesta: no se puede calcular EV con ella.

## 5. Conclusión y qué hacer

1. **El minuto no es una palanca de precisión.** Apostar en el 27 no cuesta
   acierto frente al 30, y la calibración es igual de buena en toda la ventana.
   La decisión es de operación (margen para timeouts, no quedar fuera por una
   interrupción), no de modelo.
2. **Seguir acumulando**: 95 partidos/día → en una semana hay ~600 por celda de
   minuto, suficiente para detectar una diferencia de 3-4 pp si existiera.
3. **Para el EV hace falta otra fuente**: la cuota del mercado de Q4 de la casa
   donde se apuesta. Con SofaScore es imposible, y por eso el umbral operativo
   sigue fijándose por el backtest (0.75) y no por el EV medido.
4. La instrumentación sirve además como **alarma de deriva**: si en unas semanas
   el gap de calibración por minuto se sale de ±3 pp, habría que revisar.
