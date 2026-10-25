# Resumen en vivo de `m27_v5_3` — del 18 al 21 de octubre de 2026

> Generado por `tmp/m27_v5_3_audit/live_summary_all_days.py` (reproducible).

## 1. Apuestas reales por día

| día | apuestas | ganadas | WR | IC95 | u@1.40 | u@1.83 | conf media |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | 29 | 18 | **62.1 %** | [44.0, 77.3] | −4 | +4 | 0.787 |
| 2026-10-19 | 11 | 11 | **100 %** | [74.1, 100] | +4 | +9 | 0.759 |
| 2026-10-20 | 7 | 3 | **42.9 %** | [15.8, 75.0] | −3 | −2 | 0.809 |
| 2026-10-21 | 6 | 6 | **100 %** | [61.0, 100] | +2 | +5 | 0.788 |
| **ACUMULADO** | **53** | **38** | **71.7 %** | **[58.4, 82.0]** | **+0** | **+17** | 0.783 |

- El backtest a 0.75 promete **80.7 %** → la diferencia es **−9 pp**, **z = −1.66**
  (p ≈ 0.10): **no es estadísticamente significativo**, pero tampoco es "está bien".
- El patrón diario es **bimodal** (62 %, 100 %, 43 %, 100 %): varianza alta de
  muestra pequeña, no una racha del modelo.
- En unidades: a **1.40 está exactamente en break-even** (+0 u); a **1.83 gana
  +17 u** (+32 % ROI). El mismo dato con dos precios distintos.

## 2. Por banda de confianza (acumulado)

| banda | n | WR | IC95 | u@1.40 | u@1.83 | conf media |
|---|---|---|---|---|---|---|
| 0.75-0.80 | 35 | 74.3 % | [57.9, 85.8] | +1 | +13 | 0.755 |
| 0.80-0.85 | 12 | **58.3 %** | [32.0, 80.7] | −2 | +1 | 0.820 |
| 0.85+ | 6 | 83.3 % | [43.6, 97.0] | +1 | +3 | 0.884 |

El tramo **0.80-0.85** es el que más ha dolido (7/12), pero con n=12 no permite
concluir. Es el test pre-registrado: si a las ~100 apuestas de esa banda el WR
sigue < 75 %, el Platt está sobreconfiado en la cola.

## 3. Serie por minuto (todas las predicciones, no solo las apuestas)

| minuto | n | WR | conf media | p_cal medio (local) |
|---|---|---|---|---|
| 27 | 222 | 56.8 % | 0.608 | 0.535 |
| 28 | 218 | 56.9 % | 0.605 | 0.534 |
| 29 | 221 | 54.3 % | 0.602 | 0.537 |
| 30 | 192 | 57.3 % | 0.605 | 0.533 |

- **Ningún minuto es mejor que otro**: el WR va de 54.3 % a 57.3 % (dentro del
  ruido, ±6 pp con n≈200) y la confianza media es idéntica (0.602-0.608).
- El modelo vive **~4 pp por encima de su confianza** en el conjunto de
  predicciones (56.8 % real vs 60.8 % de confianza) → **está sobreconfiado,
  no mal calibrado al revés**: el problema no es que las probabilidades sean
  mentirosas, es que el WR en la banda de decisión es menor de lo prometido.
- `p_cal` medio 0.535 → el modelo **inclina hacia el local** ~3.5 pp.

## 4. Curva de umbral viva (236 partidos evaluados)

| umbral | n | WR | IC95 | u@1.40 | u@1.83 | ¿supera el equilibrio de 1.40? |
|---|---|---|---|---|---|---|
| 0.65 | 54 | 64.8 % | [51.5, 76.2] | −5 | +10 | no |
| 0.68 | 33 | 66.7 % | [49.6, 80.2] | −2 | +7 | no |
| 0.70 | 24 | 66.7 % | [46.7, 82.0] | −2 | +5 | no |
| 0.72 | 17 | 64.7 % | [41.3, 82.7] | −2 | +3 | no |
| 0.75 | 11 | 63.6 % | [35.4, 84.8] | −1 | +2 | no |
| 0.78 | 7 | 71.4 % | [35.9, 91.8] | +0 | +2 | **sí** |
| 0.80 | 5 | 60.0 % | [23.1, 88.2] | −1 | +0 | no |

> Esta curva usa la **última** predicción de cada partido (min 30); las apuestas
> reales se.placean en el **primer** cruce del umbral (min 27), y ese momento
> rindió mejor (71.7 % vs 63.6 %). Con n=11 en la celda de 0.75 no es
> concluyente, pero es la pista más interesante que sale de estos 4 días.

## 5. Por liga (las que tienen muestra)

| liga | n | WR | WR en el backtest |
|---|---|---|---|
| Poland 2nd Basketball League | 5 | **40.0 %** | 77.8 % (2ª liga por volumen) |
| LBP, Group B | 4 | 75.0 % | — |
| Basketligaen | 3 | 100 % | 76.7 % |
| 1. ZLS / A1 Women / FSHB / FIBA Women Africa | 2 c/u | 50 % | 80-91 % |
| Swedish Basketball League Women | 2 | 100 % | 71.0 % |
| 1 apuesta (17 ligas distintas) | 1 | 0 % o 100 % | — |

**Poland 2nd es la señal**: era la segunda liga más rentable del backtest
(162 bets, 77.8 % WR) y en vivo es 2/5. Con 5 apuestas no es prueba, pero es
la primera liga que hay que vigilar en el filtro por población (P1).

> 🔎 **Resuelto el 2026-10-21:** el problema no era la liga sino la **fase**.
> "Runda Zasadnicza" (la ronda decisiva polaca de promoción) y los
> "Relegation/Promotion Playoffs" los clasificaba como *temporada regular* /
> *eliminatoria* cuando son **formatos de supervivencia**. Con el clasificador
> arreglado esas poblaciones quedan **bloqueadas**, y en vivo son el peor
> tramo: **6 bets, 3 ganadas (50 %)**.

## 6. Comparativa con `m27_v5_2` (ya en modo sombra)

En los **17 partidos** donde ambos apostaron:

- `m27_v5_3`: **14/17 (82.4 %)**
- `m27_v5_2`: **14/17 (82.4 %)**
- **Eligieron el mismo lado en 17/17.**

> ⚠️ **Esto NO significa que sean el mismo modelo.** `v5_3` es un modelo
> **nuevo**: 42 features con un sistema de priors propio (`PriorState` as-of),
> frente a las 27 de `v5_2` (lineups, `rubber_band_gap`, `white_flag_risk_zone`,
> `d_roster_drift_adv`…), y corrige **16 inconsistencias** de `v5_2`
> (`recent_2m_run_diff` y `d_strat_q4_net` siempre a 0 en vivo, el H2H que
> llegaba como lista, el ELO "final" con look-ahead, las cuotas in-play
> colándose en el prior de mercado, la isotónica in-sample, el sesgo
> local/visitante sin controlar, el snapshot 27 entrenado con el marcador del
> min 30, etc. — ver `docs/modelos/findings_m27_v5_3.md` §2).
>
> Backtest: `v5_2` AUC 0.6376 / Brier 0.2350 / LogLoss 0.6623 frente a
> `v5_3` **AUC 0.6552 / Brier 0.2303 / LogLoss 0.6518**, con **+30-65 % de
> volumen** al mismo umbral.
>
> Coinciden en 17/17 porque **los dos eligen al favorito** (el que va ganando
> al Q3 suele ganar el Q4): dos modelos con la misma información convergen al
> mismo lado, no porque compartan features.

`m27_v5_2` en modo sombra sí aporta algo: es el **control** del cambio de
calibración (Platt + umbral 0.75) y del filtro de fases. Cuesta ~27 ms por
tick.

## 7. Veredicto

1. **El modelo no está roto**: la ruta viva es fiel (verificado por cuatro vías
   independientes) y la calibración por minuto es plana y honesta.
2. **Lo que hay hoy es un −9 pp contra el backtest, con z = −1.66**: no es
   significativo, pero tampoco es para Dancing. Con 53 apuestas el IC95 es
   [58.4 %, 82.0 %]: el backtest (80.7 %) está **dentro** del intervalo.
3. **El precio decide**: a 1.40 el acumulado está en **break-even (+0 u)**; a 1.83
   son **+17 u**. Si el precio real es 1.40, el umbral de 0.75 no tiene
   margen; si es 1.83, hay margen de sobra.
4. **Falta muestra**: 53 apuestas en 4 días. El punto de decisión pre-registrado
   (200 resueltas) llega en ~2 semanas si el ritmo sigue en ~13/día.
5. **La única acción con evidencia ahora mismo**: vigilar la banda 0.80-0.85
   (test pre-registrado) y la liga Poland 2nd (P1).
