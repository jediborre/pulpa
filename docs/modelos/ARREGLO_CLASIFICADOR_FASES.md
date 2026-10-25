# Arreglo del clasificador de fases de torneo (2026-10-21)

> Documento de referencia del cambio en `models/common/competition_stage.py`.
> Reproducible: `tmp/m27_v5_3_audit/measure_classifier_fix.py` y
> `tmp/m27_v5_3_audit/check_blocked_populations.py`.

## 1. El problema

El clasificador de fases reconoce la mayoría de los formatos en los idiomas que
conoce, pero se le escapaban varios:

| fase no reconocida | deporte | clasificación que devolvía |
|---|---|---|
| `Runda Zasadnicza` (pl) | ronda decisiva con grupo (campeonato/promoción) | REGULAR ❌ |
| `Relegation Playoffs`, `Rel/Prom Playoff` (en) | supervivencia | ELIMINATORIA ❌ |
| `Promotion Playoffs`, `Placement Playoffs` (en) | supervivencia | ELIMINATORIA ❌ |
| `Turneu final de promovare` (ro) | final de promoción | REGULAR ❌ |
| `Final Round`, `Finals`, `Final` (en) | final de torneo | REGULAR o COPA ❌ |

El precedente de `v6_2` (`match/training/v6_2_league_name_exclusions.json`)
excluía explícitamente **`Championship Round`**, **`Promotion Round`** y
**`Relegation Round`**: el polaco "Runda Zasadnicza" es el mismo formato que el
inglés "Championship Round".

## 2. El arreglo

1. **Marcadores nuevos** (`models/common/competition_stage.py`):
   - `SEGUNDA_FASE`: `runda zasadnicza`, `runda mistrzowska`, `runda pierwsza`.
   - `PROMOCION`: `final de promovare`, `turneu final de promovare`, `faza de
     promovare`, `faza de retrogradare`, `aufsteiger`, `absteiger`,
     `fase de ascenso`, `fase de descenso`, `runda awansu`, `runda spadkowa`.
   - `ELIMINATORIA`: `final`, `finals` (las finales de torneo, que antes solo
     se detectaban en variantes como "final 16").
2. **Orden de comprobación**: `PROMOCION` se comprueba **antes** que
   `ELIMINATORIA`, porque una "final de promovare" lleva la palabra "final"
   pero es un formato de supervivencia, no una eliminatoria.

## 3. Impacto medido (sobre 250.000 partidos)

**1.842 partidos cambian de decisión** (bloqueado ↔ permitido):

| cambio | partidos | efecto |
|---|---|---|
| REGULAR → SEGUNDA_FASE | **1.370** | bloqueados (ronda polaca) |
| ELIMINATORIA → PROMOCION | **372** | bloqueados (relegation/promotion playoffs) |
| SEGUNDA_FASE → ELIMINATORIA | 86 | permitidos (finales de torneo) |
| PRELIMINAR → ELIMINATORIA | 7 | permitidos |
| REGULAR → PROMOCION | 6 | bloqueados |
| AMISTOSOS → ELIMINATORIA | 1 | permitido |

Además, **800 partidos** (695 REGULAR→ELIMINATORIA y 105 COPA→ELIMINATORIA)
cambian solo el **motivo** que se muestra en el log, no la decisión: son finales
de torneo, que están **permitidas** por diseño (miden 84.3 % de acierto).

## 4. ¿Bloquear es lo correcto? (medido, no supuesto)

| población | bets en C | WR a 0.75 | u@1.40 | veredicto |
|---|---|---|---|---|
| Ronda polaca (1.370 partidos) | 540 | 77.2 % | +44 | **por debajo de la media** (80.7 %) aunque ainda positivo |
| Relegation/promotion playoffs (372) | 6 | **50.0 %** | −2 | **por debajo del equilibrio (71.4 %)** |
| Resto | 7.959 | 80.7 % | +1.036 | referencia |

Y en la **serie viva** (53 apuestas de `m27_v5_3`), por categoría con el
clasificador ya arreglado:

| categoría | bets | WR |
|---|---|---|
| REGULAR | 44 | 75.0 % |
| **SEGUNDA_FASE** (la que ahora se bloquea) | 6 | **50.0 %** |
| COPA | 2 | 50.0 % |
| ELIMINATORIA | 1 | 100 % |

**Conclusión:** la población que el arreglo bloquea es, en vivo, **el peor
tramo** (3 de 6). Y en el backtest está por debajo de la media. Bloquear es
correcto por diseño (es el precedente de `v6_2`), por la evidencia viva y por
la del backtest.

**Consecuencia operativa:** con el arreglo, el registro vivo de `m27_v5_3`
pasa de 38/53 (71.7 %) a 35/47 (74.5 %) y el modelo deja de apostar en formatos
de supervivencia.

Se reconstruyó el dataset y se reentrenó para que **entrenamiento e inferencia
usen la misma regla** (si no, el train incluiría formatos que el vivo bloquea).
