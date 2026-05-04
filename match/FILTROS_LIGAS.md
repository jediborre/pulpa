# Filtros de Ligas — Monitor de Apuestas Q4

Documento de referencia que describe qué ligas son excluidas y por qué, tanto en el modelo de predicción v6.2 como en el monitor en tiempo real (`bet_monitor.py`).

---

## 1. Filtros del modelo v6.2

Archivo fuente: `match/training/v6_2_league_name_exclusions.json`  
Modo de coincidencia: **contiene** (case-insensitive)

Estas categorías se usan al entrenar para excluir muestras con baja señal o alta varianza:

### 1.1 Juveniles / Formativas
Patrones: `U14`, `U15`, `U16`, `U17`, `U18`, `U19`, `U20`, `U21`, `Youth`, `Kadetska`, `Espoirs`

> Alta varianza, pocas muestras por liga, comportamiento distinto a ligas de mayores.

### 1.2 Femenil — Alta varianza
Patrones: `Women`, `Women,`, `Women's`, `Femenina`, `Femenino`, `Liga Femenina`, `LFB`, `LF2`, `WNBL`, `WNBA`

> Distribución de puntajes diferente a ligas masculinas; modelos entrenados en datos mixtos pierden señal.

### 1.3 Torneos cortos / Eliminatorias
Patrones: `Playoffs`, `Playoff`, `Play-in`, `Play-In`, `Knockout`, `Knock-out`, `Cup`, `Cup,`, `Copa`, `Final`, `Finals`, `Semi-Finals`, `Semifinal`, `Quarterfinal`, `Qualification`, `Qualifiers`, `Superfinal`, `Challenge Cup`, `Placement`, `Relegation Round`, `Promotion Round`, `Championship Round`, `Ćwierćfinały`, `Финал`

> Partidos únicos con motivación atípica; historial insuficiente para calibrar edge.

### 1.4 Ligas con bajo edge histórico
Patrones: `Rwanda`, `Egypt`, `Indonesia`, `MPBL`, `Big V`, `Azerbaijan`, `Qatar`, `Emir Cup`, `MOS Cup`, `Africa League`

> Retroalimentación histórica insuficiente o spreads demasiado amplios.

### 1.5 Ligas inconsistentes
Patrones: `Superliga`, `Puerto Rico BSN`, `Germany BBL`, `Argentina Liga Nacional`

> Resultados inconsistentes entre temporadas; señal débil.

### 1.6 Ligas descartadas manualmente
Patrones: `Swedish Basketball Superettan`, `Superleague`, `Israeli Basketball Super League`, `1 Liga Kobiet`, `Saku I liiga`, `Champions League Asia-East`, `Golden Square`, `The League`

> Evaluadas individualmente y descartadas por falta de edge verificable.

---

## 2. Filtros del monitor en tiempo real (`bet_monitor.py`)

Función: `_get_pending_rows` — aplica al recuperar partidos pendientes de la base de datos.

Estos filtros son un **superconjunto operativo** de los del modelo: incluyen ligas que no deben monitorearse en producción, ya sea por el modelo o por razones operativas adicionales.

| Patrón SQL (`LIKE`) | Categoría |
|---|---|
| `%WNBA%` | Femenil |
| `%Women%` / `%women%` | Femenil |
| `%Feminina%` / `%Femenina%` | Femenil |
| `%Liga Femenina%` | Femenil |
| `%LF Challenge%` | Femenil |
| `%Playoff%` / `%PLAY OFF%` | Eliminatorias |
| `%Playout%` | Eliminatorias |
| `%U21 Espoirs Elite%` | Juvenil |
| `%Polish Basketball League%` | Liga inconsistente |
| `%SuperSport Premijer Liga%` | Liga descartada |
| `%Prvenstvo Hrvatske za d%` | Liga descartada |
| `%ABA Liga%` | Liga descartada |
| `%Argentina Liga Nacional%` | Liga inconsistente |
| `%Basketligaen%` | Liga descartada |
| `%lite 2%` | Liga descartada |
| `%EYBL%` | Juvenil / amateur |
| `%I B MCKL%` | Liga descartada |
| `%Liga 1 Masculin%` | Liga descartada |
| `%Liga Nationala%` | Liga descartada |
| `%NBL1%` | Liga bajo edge |
| `%PBA Commissioner%` | Torneo corto |
| `%Rapid League%` | Torneo corto |
| `%Stoiximan GBL%` | Liga descartada |
| `%Superleague%` | Liga descartada |
| `%Superliga%` | Liga inconsistente |
| `%Swedish Basketball Superettan%` | Liga descartada |
| `%Swiss Cup%` | Torneo corto |
| `%Финал%` | Eliminatoria |
| `%Turkish Basketball Super League%` | Liga descartada |
| `%NBA%` | Liga descartada (modelo no entrenado) |
| `%Big V%` | Liga bajo edge |
| `%Egyptian Basketball Super League%` | Liga bajo edge |
| `%Lega A Basket%` | Liga descartada |
| `%Liga e Par%` | Liga descartada |
| `%Liga Ouro%` | Liga descartada |
| `%Señal%` | Liga descartada |
| `%LNB%` | Liga descartada (incl. Chile LNB) |
| `%Meridianbet KLS%` | Liga descartada |
| `%MPBL%` | Liga bajo edge |
| `%Nationale 1%` | Liga descartada |
| `%Poland 2nd Basketball League%` | Liga descartada |
| `%Portugal LBP%` | Liga descartada |
| `%Portugal Proliga%` | Liga descartada |
| `%Saku I liiga%` | Liga descartada |
| `%Serie A2%` | Liga descartada |
| `%Slovenian Second Basketball%` | Liga descartada |
| `%Super League%` | Liga descartada |
| `%United Cup%` | Torneo corto |
| `%United League%` | Liga descartada |

*Última actualización: 2026-05-03*
