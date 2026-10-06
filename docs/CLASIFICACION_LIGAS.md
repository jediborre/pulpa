# 🏀 Clasificación Taxonómica y Tabla de Ligas (`leagues_classification`)

> **PROPÓSITO:** Este documento detalla la taxonomía multidimensional y el catálogo de metadatos de las **1,958 ligas y competiciones** registradas en [`matches.db`](file:///C:/Users/App/Desktop/pulpa/matches.db).
> Esta información está materializada en la tabla SQLite permanente `leagues_classification`, permitiendo a cualquier modelo de Machine Learning, script de backfill o análisis EDA obtener *features* de contexto (género, formato de tiempo, fase de playoffs, priors de anotación, localía y nivel competitivo) con un simple `JOIN`.

---

## 🏗️ 1. Definición y Ubicación de la Tabla

- **Ubicación Física:** Base de datos central [`matches.db`](file:///C:/Users/App/Desktop/pulpa/matches.db) en la raíz del proyecto.
- **Nombre de la Tabla:** `leagues_classification`.
- **Clave Primaria:** `league` (coincide exactamente con el valor del campo `matches.league`).
- **Población:** 1,958 registros únicos con cobertura sobre los 48,623 partidos.

### Esquema Relacional de `leagues_classification` (28 Columnas)

| Columna | Tipo SQLite | Nulo | Descripción Semántica / Rango de Valores |
| :--- | :--- | :---: | :--- |
| `league` | `TEXT` | No (PK) | Nombre exacto de la competición en `matches` (ej. `'Liga ACB, Playoffs'`). |
| `clean_name` | `TEXT` | No | Nombre base limpio sin fase o etapa (ej. `'Liga ACB'`). |
| `stage` | `TEXT` | No | Fase identificada (ej. `'Regular season'`, `'Playoffs'`, `'Finals'`, `'Group Stage'`). |
| `gender` | `TEXT` | No | Género: `'men'` (masculino) o `'women'` (femenino). |
| `is_women` | `INTEGER` | No | Flag binario: `1` si la liga es femenina, `0` si es masculina. |
| `is_youth` | `INTEGER` | No | Flag binario: `1` si es categoría formativa (menores de 23 años). |
| `age_category` | `TEXT` | No | Rango de edad: `'Senior'`, `'U23'`, `'U21'`, `'U20'`, `'U19'`, `'U18'`, `'U16'`, `'Youth'`. |
| `is_college` | `INTEGER` | No | Flag binario: `1` si es baloncesto universitario (NCAA Division I/II/III, NAIA, NJCAA). |
| `competition_type` | `TEXT` | No | Tipo: `'league'` (liga regular), `'playoffs'`, `'cup'` (copa/torneo corto), `'friendly'`, `'all_star'`. |
| `is_tournament` | `INTEGER` | No | Flag binario: `1` si es copa, torneo corto o playoffs; `0` si es liga regular. |
| `is_playoffs` | `INTEGER` | No | Flag binario: `1` si corresponde a eliminatorias postemporada o lucha por el título. |
| `is_international` | `INTEGER` | No | Flag binario: `1` si es torneo transnacional o de selecciones (Euroleague, FIBA, etc.). |
| `country_or_region`| `TEXT` | No | País o ámbito geográfico (ej. `'USA'`, `'Spain'`, `'Italy'`, `'Lithuania'`, `'International'`, etc.). |
| `tier_level` | `TEXT` | No | Nivel: `'top_pro'`, `'second_pro'`, `'lower_pro_amateur'`, `'college'`, `'youth'`, `'standard_pro'`. |
| `quarter_duration_minutes` | `INTEGER` | No | Duración reglamentaria del cuarto: `12` (NBA, CBA, PBA) o `10` (FIBA, NCAA). |
| `total_game_minutes` | `INTEGER` | No | Duración reglamentaria total del partido: `48` o `40` minutos. |
| `match_count` | `INTEGER` | No | Total de partidos de esa liga en la base de datos. |
| `finished_match_count` | `INTEGER` | No | Partidos finalizados con marcador válido. |
| `avg_home_score` | `REAL` | Sí | Promedio de puntos anotados por el equipo local. |
| `avg_away_score` | `REAL` | Sí | Promedio de puntos anotados por el equipo visitante. |
| `avg_total_points` | `REAL` | Sí | Promedio global de puntos por partido (prior base para apuestas Over/Under). |
| `home_win_pct` | `REAL` | Sí | Porcentaje de victorias locales (% ventaja de localía). |
| `ot_rate` | `REAL` | Sí | Porcentaje de partidos terminados en prórroga (paridad competitiva). |
| `avg_q4_total_points` | `REAL` | Sí | Promedio de puntos combinados en el 4º cuarto (prior esencial para modelos Q4). |
| `avg_q4_margin` | `REAL` | Sí | Diferencia absoluta promedio de puntos en el 4º cuarto. |
| `pbp_coverage_pct` | `REAL` | No | Porcentaje de partidos con jugadas anotadoras (`play_by_play`). |
| `graph_coverage_pct`| `REAL` | No | Porcentaje de partidos con curva de momentum (`graph_points`). |
| `updated_at` | `TEXT` | No | Timestamp ISO de la última consolidación. |

---

## 📊 2. Distribución y Balance Global de Ligas

A partir del análisis de las 1,958 ligas y sus 48,623 partidos, se obtienen las siguientes distribuciones consolidadas:

### Por Género
| Género | Ligas / Competiciones | Partidos Totales | Puntos Promedio | Ventaja Local (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Masculino (`men`)** | 1,426 (72.8%) | 38,737 (79.7%) | 153.8 pts | 56.8% |
| **Femenino (`women`)** | 532 (27.2%) | 9,886 (20.3%) | 135.5 pts | 55.5% |

> 📉 **Hallazgo ML:** El baloncesto femenino anota en promedio **18.3 puntos menos por partido** que el masculino, requiriendo umbrales y priors de totales claramente separados.

---

### Por Tipo de Competición y Fase
| Tipo (`competition_type`) | Ligas / Fases | Partidos Totales | % de Partidos | Descripción |
| :--- | :---: | :---: | :---: | :--- |
| **`league`** | 875 | 40,146 | 82.6% | Fases regulares de todos contra todos. |
| **`playoffs`** | 664 | 4,840 | 10.0% | Series eliminatorias, semifinales, cuartos y finales. |
| **`cup`** | 384 | 3,429 | 7.1% | Copas domésticas (Copa del Rey, Pokal, etc.) y torneos cortos. |
| **`friendly`** | 30 | 201 | 0.4% | Amistosos de preparación o pretemporada. |
| **`all_star`** | 5 | 7 | <0.1% | Partidos de estrellas / exhibición. |

---

### Por Categoría de Edad y Baloncesto Universitario
| Categoría | Ligas | Partidos | % Partidos | Observaciones |
| :--- | :---: | :---: | :---: | :--- |
| **Senior / Absoluta** | 1,425 | 44,510 | 91.5% | Competiciones profesionales o adultas estándar. |
| **Formativa / Juvenil (`is_youth`)**| 533 | 4,113 | 8.5% | U16 a U23, ligas de desarrollo y torneos junior. |
| **Universitaria (`is_college`)** | 58 | 5,622 | 11.6% | NCAA Division I, II, III, NAIA (alta varianza de tanteo). |

---

### Por Duración de Cuartos Reglamentarios
| Duración de Cuarto | Ligas | Partidos | Puntos Promedio | Competiciones Principales |
| :--- | :---: | :---: | :---: | :--- |
| **10 Minutos (FIBA / NCAA)** | 1,932 | 46,115 (94.8%) | 148.2 pts | Liga ACB, Euroleague, BBL, NCAA, etc. |
| **12 Minutos (NBA / CBA / PBA)**| 26 | 2,508 (5.2%) | 194.9 pts | NBA, NBA G League, CBA China, PBA Filipinas. |

> ⏱️ **Hallazgo ML:** Las ligas de 12 minutos presentan un tanteo promedio de **194.9 pts** vs **148.2 pts** en ligas de 10 minutos (+46.7 puntos de diferencia debidos a los 8 minutos extra de juego).

---

## 🏆 3. Muestra del Top 25 Ligas con Clasificación Completa

| Liga | Género | Tipo | Tier | Min/Q | Partidos | Pts Avg | Win% Loc | Q4 Pts Avg | PBP% | Graph% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NCAA Women's Division I, Reg.** | Women | League | College | 10 | 1,729 | 132.9 | 61.7% | 34.1 | 96.2% | 94.7% |
| **NCAA Men's Division I, Reg.** | Men | League | College | 10 | 1,324 | 145.4 | 64.1% | 40.2 | 97.4% | 96.1% |
| **NBA** | Men | League | Top Pro | 12 | 1,222 | 229.4 | 55.6% | 55.4 | 99.8% | 99.5% |
| **Poland 2nd Basketball League** | Men | League | Standard | 10 | 852 | 162.7 | 57.9% | 40.8 | 97.9% | 96.7% |
| **B League Premier (Japón)** | Men | League | Second Pro | 10 | 575 | 158.9 | 55.8 | 39.5 | 99.3% | 98.6% |
| **NCAA Men's Division II, Reg.** | Men | League | College | 10 | 557 | 152.1 | 58.7% | 39.2 | 93.9% | 91.2% |
| **Liga Argentina, Group Stage** | Men | League | Standard | 10 | 518 | 150.3 | 64.9% | 37.6 | 96.7% | 95.0% |
| **NBA G League** | Men | League | Second Pro | 12 | 516 | 225.1 | 52.7% | 53.8 | 98.8% | 98.4% |
| **NCAA (Genérica)** | Men | League | College | 10 | 385 | 144.9 | 60.5% | 39.8 | 95.8% | 94.0% |
| **China CBA** | Men | League | Top Pro | 12 | 372 | 204.6 | 57.8% | 50.1 | 97.6% | 97.0% |
| **NBL (Australia)** | Men | League | Top Pro | 10 | 367 | 179.3 | 56.4% | 44.2 | 98.9% | 98.4% |
| **Brazil NBB** | Men | League | Top Pro | 10 | 353 | 157.9 | 59.8% | 39.1 | 98.6% | 97.7% |
| **Euroleague** | Men | League | Top Pro | 10 | 352 | 165.2 | 64.8% | 41.5 | 99.4% | 99.4% |
| **Serie A2 (Italia)** | Men | League | Second Pro | 10 | 320 | 156.4 | 61.6% | 39.3 | 98.8% | 97.8% |
| **B League One (Japón)** | Men | League | Second Pro | 10 | 318 | 159.2 | 56.3% | 39.7 | 98.7% | 97.8% |
| **Élite 2 (Francia)** | Men | League | Standard | 10 | 314 | 163.5 | 59.2% | 41.2 | 98.4% | 98.1% |
| **WNBA (EE. UU.)** | Women | League | Top Pro | 10 | 304 | 163.8 | 57.2% | 40.8 | 99.7% | 99.7% |
| **Super League** | Men | League | Second Pro | 10 | 302 | 166.4 | 56.0% | 42.1 | 96.7% | 95.4% |
| **NCAA Men's Division III, Reg.** | Men | League | College | 10 | 300 | 148.2 | 58.3% | 38.6 | 92.0% | 89.7% |
| **Argentina Liga Nacional** | Men | League | Top Pro | 10 | 298 | 159.1 | 61.7% | 39.9 | 98.7% | 97.7% |
| **Liga ACB (España)** | Men | League | Top Pro | 10 | 289 | 166.7 | 58.8% | 41.9 | 99.7% | 99.7% |
| **Germany BBL** | Men | League | Top Pro | 10 | 282 | 171.1 | 56.7% | 43.1 | 99.6% | 99.6% |
| **Serie B, Group B (Italia)** | Men | League | Lower Pro | 10 | 278 | 153.2 | 55.4% | 38.5 | 97.1% | 95.7% |
| **MPBL (Filipinas)** | Men | League | Second Pro | 10 | 277 | 155.0 | 54.2% | 38.8 | 96.8% | 96.0% |
| **1 Liga Kobiet (Polonia Fem.)**| Women | League | Standard | 10 | 270 | 134.6 | 56.7% | 33.9 | 95.6% | 93.3% |

---

## 🤖 4. Guía de Uso para Modelos de Machine Learning (Feature Engineering)

La tabla `leagues_classification` permite enriquecer instantáneamente cualquier dataset de entrenamiento o inferencia en vivo:

### Ejemplo 1: Enriquecimiento Directo en SQL
```sql
SELECT 
    m.match_id,
    m.date,
    m.home_team,
    m.away_team,
    -- Variables contextuales de la liga para el modelo:
    lc.is_women,
    lc.is_youth,
    lc.is_college,
    lc.is_playoffs,
    lc.quarter_duration_minutes,
    lc.tier_level,
    -- Priors de anotación de la liga:
    lc.avg_total_points AS league_prior_total,
    lc.avg_q4_total_points AS league_prior_q4_points,
    lc.home_win_pct AS league_home_advantage
FROM matches m
JOIN leagues_classification lc ON m.league = lc.league
WHERE m.status_type = 'finished';
```

### Ejemplo 2: Carga en Pipeline de Python (Pandas / Polars)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("matches.db")

query = """
SELECT 
    m.match_id, m.league,
    lc.is_women, lc.is_playoffs, lc.quarter_duration_minutes,
    lc.avg_q4_total_points, lc.home_win_pct
FROM matches m
LEFT JOIN leagues_classification lc ON m.league = lc.league
"""
df = pd.read_sql_query(query, conn)
```

---

## 🔄 5. Mantenimiento y Actualización

Si se ingieren nuevos partidos históricos o nuevas ligas en `matches.db`, la tabla de clasificación puede regenerarse o actualizarse automáticamente ejecutando:
```powershell
.venv\Scripts\python.exe tmp\league_classification\build_and_populate_leagues.py
```
El script es idempotente, recalcula las métricas agregadas de cada liga y garantiza que los nuevos torneos queden catalogados inmediatamente.
