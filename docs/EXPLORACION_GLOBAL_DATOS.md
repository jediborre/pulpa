# 🌐 Exploración Global del Conjunto de Datos (`matches.db`)

> **PROPÓSITO:** Este documento proporciona un **inventario y balance panorámico exhaustivo** de los datos consolidados en la base de datos central [`matches.db`](file:///C:/Users/App/Desktop/pulpa/matches.db).
> Su meta es servir como mapa de referencia cuantitativo previo a la formulación de análisis exploratorios de datos (EDA), ingeniería de características (*feature engineering*) o entrenamiento de modelos de Machine Learning.

---

## 📌 1. Resumen Ejecutivo de Datos (KPIs Clave)

| Métrica | Valor Consolidado | Observaciones |
| :--- | :--- | :--- |
| **Partidos Totales Registrados** | **48,623** | Extraídos de SofaScore en un año de monitoreo y backfill |
| **Partidos Finalizados (`finished`)** | **48,514** (99.78%) | 47,020 terminados en tiempo regular (`Ended`) y 1,489 con prórroga (`AET`) |
| **Ventana Temporal** | **08-Oct-2025 al 05-Oct-2026** | Cobertura continua de 363 días de calendario |
| **Competiciones / Ligas Únicas** | **1,958** | Torneos de todo el mundo (primeras, segundas divisiones, juveniles y universitarias) |
| **Puntos Totales por Partido** | **158.48 pts** (promedio) | Mínimo: 0 (walkovers) / Máximo: 308 puntos |
| **Efecto Localía (*Home Advantage*)** | **57.20% victorias locales** | 27,748 victorias locales vs 20,745 victorias visitantes |
| **Eventos de Jugada a Jugada (`play_by_play`)**| **3,709,807 eventos** | Promedio de 85.6 jugadas anotadoras por partido |
| **Puntos de Momentum (`graph_points`)** | **1,719,271 registros** | Promedio de 40.6 mediciones por partido (minuto a minuto) |
| **Historiales Directos (`match_h2h`)** | **69,070 enfrentamientos** | 99.8% de partidos cuentan con datos H2H previos |

---

## 🧩 2. Matriz de Cobertura y Densidad por Componente

No todos los torneos transmiten la misma profundidad de datos. La siguiente tabla detalla qué porcentaje de los 48,623 partidos cuenta con cada nivel de enriquecimiento en la base de datos:

| Componente Relacional | Tabla Asociada | Partidos con Datos | Cobertura (%) | Filas Totales | Profundidad Promedio |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Parciales por Cuarto** | `quarter_scores` | **48,622** | **99.99%** | 188,689 | 3.88 cuartos / partido |
| **Historial Cara a Cara** | `match_h2h` | **48,543** | **99.84%** | 69,070 | 1.42 duelos previos / partido |
| **Rendimiento Individual** | `player_stats` / `lineups` | **44,122** | **90.74%** | 952,182 | 21.6 jugadores / partido |
| **Jugadas Anotadoras** | `play_by_play` | **43,312** | **89.08%** | 3,709,807 | 85.65 jugadas / partido |
| **Box Score por Equipo** | `team_statistics` | **43,266** | **88.98%** | 1,204,175 | 27.8 métricas / partido |
| **Curva de Momentum** | `graph_points` | **42,368** | **87.14%** | 1,719,271 | 40.58 minutos / partido |
| **Cuotas de Apuestas** | `match_odds` | **~25,000** | **~51.4%** | 64,034 | Cuotas 1X2, Over/Under y Hándicap |
| **Evaluaciones ML (Q3/Q4)**| `eval_match_results` | **11,125** | **22.88%** | 11,125 | Predicciones históricas V1-V7 |
| **Evaluaciones Modernas** | `eval_match_results_v2` | **474** | **0.97%** | 474 | Inferencia en vivo de `v6_2` y `m27_v3` |

> 💡 **Nota para EDAs:** Para modelos basados en la curva de momentum y snapshots al minuto 27 (como `m27_v3`), el universo analizable garantizado es de **42,368 partidos** (87.14% del total).

---

## 📅 3. Dimensión Temporal (Distribución de Partidos por Mes)

El volumen de partidos refleja con exactitud el ciclo natural del calendario internacional de baloncesto:

| Año-Mes | Partidos Registrados | % del Total | Hito del Calendario Deportivo |
| :---: | :---: | :---: | :--- |
| **2025-10** | 3,882 | 8.0% | Inicio de temporadas europeas y NBA |
| **2025-11** | 4,958 | 10.2% | Apertura de temporada NCAA y ligas americanas |
| **2025-12** | 4,990 | 10.3% | Jornadas regulares continuas |
| **2026-01** | 4,689 | 9.6% | Mitad de temporada y copas domésticas |
| **2026-02** | **8,069** | **16.6%** | Ventana FIBA + recta final de conferencias universitarias |
| **2026-03** | **8,515** | **17.5%** | **Pico Máximo:** *March Madness* (NCAA) + cierres de fase regular |
| **2026-04** | 3,953 | 8.1% | Playoffs NBA y fases eliminatorias europeas |
| **2026-05** | 2,159 | 4.4% | Finales de ligas nacionales |
| **2026-06** | 1,378 | 2.8% | Finales NBA y torneos de verano |
| **2026-07** | 1,860 | 3.8% | Torneos preolímpicos / continentales / ligas sudamericanas |
| **2026-08** | 1,554 | 3.2% | Torneos de selecciones y pretemporadas |
| **2026-09** | 1,809 | 3.7% | Arranque de copas y Supercopas |
| **2026-10** | 807 | 1.7% | Ingesta de partidos recientes en curso |

---

## 🏆 4. Dimensión de Ligas y Competiciones

De las **1,958 ligas** presentes en la base de datos, el volumen está concentrado según la siguiente jerarquía piramidal:

### Pirámide de Distribución por Volumen

| Tier de Liga | Rango de Partidos | Cantidad de Ligas | Partidos Acumulados | % de Datos |
| :--- | :---: | :---: | :---: | :---: |
| **Gigante** | $\ge 500$ | 8 | 7,293 | 15.0% |
| **Muy Grande**| $200 - 499$ | 27 | 7,789 | 16.0% |
| **Grande** | $100 - 199$ | 81 | 11,211 | 23.1% |
| **Mediana** | $50 - 99$ | 99 | 7,101 | 14.6% |
| **Pequeña** | $20 - 49$ | 228 | 6,712 | 13.8% |
| **Micro** | $< 20$ | 1,515 | 8,517 | 17.5% |

---

### Top 40 Ligas Más Representadas

La siguiente tabla lista las 40 competiciones con mayor número de partidos finalizados, junto con su promedio de anotación total y el porcentaje de victoria del equipo local:

| Liga / Competición | Partidos | Formato Tiempo | Pts Totales (Avg) | Win% Local |
| :--- | :---: | :---: | :---: | :---: |
| **NCAA Women's Division I, Regular season** | 1,729 | 10 min cuartos | 133.5 pts | 61.1% |
| **NCAA Men's Division I, Regular season** | 1,324 | 20 min mitades | 145.8 pts | 63.8% |
| **NBA** | 1,222 | 12 min cuartos | 229.4 pts | 55.6% |
| **Poland 2nd Basketball League** | 852 | 10 min cuartos | 162.7 pts | 57.9% |
| **B League Premier (Japón)** | 575 | 10 min cuartos | 158.9 pts | 55.8% |
| **NCAA Men's Division II, Regular season** | 557 | 20 min mitades | 152.1 pts | 58.7% |
| **Liga Argentina, Group Stage** | 518 | 10 min cuartos | 150.3 pts | 64.9% |
| **NBA G League** | 516 | 12 min cuartos | 225.1 pts | 52.7% |
| **NCAA (Genérica)** | 385 | 20 min mitades | 144.9 pts | 60.5% |
| **China CBA** | 372 | 12 min cuartos | 204.6 pts | 57.8% |
| **NBL (Australia)** | 367 | 10 min cuartos | 179.3 pts | 56.4% |
| **Brazil NBB** | 353 | 10 min cuartos | 157.9 pts | 59.8% |
| **Euroleague** | 352 | 10 min cuartos | 165.2 pts | 64.8% |
| **Serie A2 (Italia)** | 320 | 10 min cuartos | 156.4 pts | 61.6% |
| **B League One (Japón)** | 318 | 10 min cuartos | 159.2 pts | 56.3% |
| **Élite 2 (Francia)** | 314 | 10 min cuartos | 163.5 pts | 59.2% |
| **WNBA (EE. UU.)** | 304 | 10 min cuartos | 163.8 pts | 57.2% |
| **Super League (Turquía/Israel)** | 302 | 10 min cuartos | 166.4 pts | 56.0% |
| **NCAA Men's Division III, Regular season** | 300 | 20 min mitades | 148.2 pts | 58.3% |
| **Argentina Liga Nacional** | 298 | 10 min cuartos | 159.1 pts | 61.7% |
| **Liga ACB (España)** | 289 | 10 min cuartos | 166.7 pts | 58.8% |
| **Germany BBL** | 282 | 10 min cuartos | 171.1 pts | 56.7% |
| **Serie B, Group B (Italia)** | 278 | 10 min cuartos | 153.2 pts | 55.4% |
| **MPBL (Filipinas)** | 277 | 10 min cuartos | 155.0 pts | 54.2% |
| **Serie B, Group A (Italia)** | 276 | 10 min cuartos | 154.8 pts | 58.0% |
| **1 Liga Kobiet (Polonia Femenina)** | 270 | 10 min cuartos | 134.6 pts | 56.7% |
| **BNXT League (Bélgica/Holanda)** | 263 | 10 min cuartos | 162.9 pts | 56.7% |
| **RKL Division A (Lituania)** | 246 | 10 min cuartos | 161.4 pts | 57.7% |
| **Primera FEB (España LEB Oro)** | 246 | 10 min cuartos | 160.8 pts | 58.9% |
| **Korean Basketball League (KBL)** | 239 | 10 min cuartos | 159.9 pts | 54.0% |
| **Nostra.lt-RKL Division B** | 236 | 10 min cuartos | 158.7 pts | 58.1% |
| **Poland 1st Division Basketball** | 231 | 10 min cuartos | 168.3 pts | 58.4% |
| **NKL (Lituania)** | 226 | 10 min cuartos | 159.7 pts | 55.3% |
| **France Pro A (LNB)** | 223 | 10 min cuartos | 164.2 pts | 57.0% |
| **Israeli National League Basketball** | 222 | 10 min cuartos | 166.8 pts | 54.5% |
| **Liga LEB Plata (España)** | 218 | 10 min cuartos | 152.4 pts | 56.9% |
| **Greece A1** | 215 | 10 min cuartos | 158.1 pts | 59.5% |
| **Liga Femenina (España)** | 210 | 10 min cuartos | 132.8 pts | 56.2% |
| **Italy Serie A** | 208 | 10 min cuartos | 163.5 pts | 58.2% |
| **Adriatic League (ABA)** | 204 | 10 min cuartos | 164.0 pts | 60.3% |

---

## 📈 5. Marcadores y Comportamiento Estadístico

### Distribución de Tanteos
- **Promedio de Puntos Local:** `80.73` puntos.
- **Promedio de Puntos Visitante:** `77.75` puntos.
- **Promedio de Diferencia Local - Visitante:** `+2.97` puntos a favor de los locales.
- **Puntos Totales Promedio:** `158.48` puntos.

### Comparativa por Formato de Duración
1. **Ligas NBA / G-League (48 minutos):**
   - Tanteo promedio: **~227 puntos**.
   - Ritmo anotador: ~4.73 puntos por minuto de juego.
2. **Ligas FIBA Pro (40 minutos - ACB, Euroleague, BBL):**
   - Tanteo promedio: **~165 puntos**.
   - Ritmo anotador: ~4.12 puntos por minuto de juego.
3. **Ligas Universitarias NCAA (40 minutos):**
   - Tanteo promedio: **~145 puntos**.
   - Ritmo anotador: ~3.62 puntos por minuto de juego (posesiones más largas de 30s).
4. **Ligas Femeninas FIBA / NCAA Women:**
   - Tanteo promedio: **~133 puntos**.
   - Ritmo anotador: ~3.32 puntos por minuto de juego.

---

## 🧪 6. Consultas SQL Canónicas para Iniciar EDAs

A continuación se incluyen consultas SQL estándar listas para ejecutar sobre [`matches.db`](file:///C:/Users/App/Desktop/pulpa/matches.db):

### 1. Extracción de Muestra Limpia para Modelo Q4 (Sin Nulos ni Walkovers)
```sql
SELECT 
    m.match_id,
    m.league,
    m.date,
    m.home_team,
    m.away_team,
    m.home_score,
    m.away_score,
    q4.home AS q4_home,
    q4.away AS q4_away,
    CASE 
        WHEN q4.home > q4.away THEN 'HOME'
        WHEN q4.away > q4.home THEN 'AWAY'
        ELSE 'TIE'
    END AS q4_winner
FROM matches m
JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
WHERE m.status_type = 'finished'
  AND m.home_score > 0
  AND m.away_score > 0
  AND q4.home IS NOT NULL
  AND q4.away IS NOT NULL
ORDER BY m.date DESC;
```

### 2. Extracción de Momentum en el Minuto 27 (Para Snapshot M27)
```sql
SELECT 
    m.match_id,
    m.league,
    gp.minute,
    gp.value AS momentum_at_m27
FROM matches m
JOIN graph_points gp ON m.match_id = gp.match_id AND gp.minute = 27
WHERE m.status_type = 'finished';
```

### 3. Box Score Agregado de Equipos
```sql
SELECT 
    ts.match_id,
    ts.period,
    ts.stat_key,
    ts.home_value,
    ts.away_value
FROM team_statistics ts
WHERE ts.period = 'ALL'
  AND ts.stat_key IN ('fieldGoals', 'threePointFieldGoals', 'freeThrows', 'rebounds', 'turnovers');
```

---

## 🧭 7. Resumen de Conclusiones para la Fase de EDA

1. **Volumen Suficiente para Modelos Complejos:** Con 48,514 partidos finalizados y más de 42,000 partidos con series de tiempo minuto a minuto (`graph_points`) y jugadas anotadoras (`play_by_play`), el conjunto de datos es estadísticamente robusto para análisis de covarianza, árboles de decisión gradient boosted (LightGBM/XGBoost) y redes recurrentes.
2. **Segmentación Mandatoria por Liga:** Dado que el promedio de puntos varía drásticamente entre la NBA (229 pts), ligas FIBA (165 pts) y ligas femeninas/NCAA (133-145 pts), cualquier análisis exploratorio o modelo predictivo debe **estandarizar las variables de anotación por cuartos** o incluir el formato de duración como feature categórica / embedding.
3. **Calidad de Integridad Excelente:** Menos del 0.25% de partidos sufren de datos incompletos o cancelaciones. El 99.99% tiene registro exacto de cada uno de sus cuatro cuartos.
