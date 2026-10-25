# 📊 Impacto Dinámico de la Fuerza de Equipos y Roster en los Cuartos (Q1 a Q4)

> **PROPÓSITO:** Este documento analiza cómo medir la fuerza de los equipos de forma **dinámica y punto-en-el-tiempo** (eliminando el *data leakage* de promedios estáticos entre temporadas) y presenta los resultados empíricos medidos sobre **9,326 partidos** en [`matches.db`](file:///C:/Users/App/Desktop/pulpa/matches.db) evaluando cómo impactan el quinteto titular, la profundidad de banquillo y el estilo táctico nacional en el desenlace de cada cuarto (Q1, Q2, Q3 y Q4).

---

## ⚠️ 1. El Problema de la Fuerza Estática vs Dinámica

Calcular una métrica de "fuerza histórica global" para un equipo (como su porcentaje de victorias de toda la base de datos o su promedio de puntos del año) es un error conceptual crítico en Machine Learning:

1. **Variación entre Temporadas y Traspasos:** Las plantillas de baloncesto cambian radicalmente de un año a otro (fichajes de estrellas, salidas de jugadores clave, cambio de entrenador). El Golden State Warriors de 2017 no tiene nada que ver con el de 2024.
2. **Variación Intratemporada (Baches y Lesiones):** Un equipo que pierde a su base titular o a su pívot defensivo por lesión durante 3 semanas tiene una fuerza transitoria completamente diferente a la de su récord oficial en la tabla.
3. **Data Leakage:** Si promedias toda la temporada para asignarle una fuerza fija a un partido disputado en noviembre, estás introduciendo información del futuro (partidos jugados en marzo/abril).

### La Solución: Descomposición Dinámica de Fuerza
Para alimentar a los modelos de ML sin fugas de datos, la fuerza debe medirse en dos ejes:
- **Eje A — Roster Presente ese Día (Match Lineup Quality):** Quiénes juegan realmente ese partido según `lineups` y `player_stats` (rating de los 5 titulares, rating del banquillo, brecha titular-suplente y concentración en estrellas).
- **Eje B — Fuerza Rodante Previa (Rolling Pre-Match Form):** Balance de victorias (`wins`, `losses`), posición y racha previa (`form = ["W", "L", ...]`) registrados en `team_strength` justo antes de iniciar el encuentro.

---

## 📈 2. Resultados Empíricos: Impacto de las Fuerzas en Cada Cuarto (9,326 Partidos)

Se analizó la correlación lineal de Pearson ($r$) y la precisión direccional (% de acierto al pronosticar qué equipo gana el cuarto según su ventaja en esa métrica):

| Dimensión de Fuerza Relativa ($\Delta = \text{Local} - \text{Visitante}$) | Q1 (Cuarto 1) | Q2 (Cuarto 2) | Q3 (Cuarto 3) | Q4 (Cuarto 4) | Partido Final (FT) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Diferencial Titulares (Rating 5 Iniciales)** | **+0.545** (66.3%) | +0.433 (62.4%) | **+0.459** (62.2%) | +0.401 (61.4%) | **+0.799** (83.9%) |
| **Diferencial Banquillo (Rating Suplentes)** | +0.232 (55.0%) | **+0.379** (61.4%) | +0.295 (57.3%) | **+0.348** (59.2%) | +0.547 (70.0%) |
| **Diferencial % Victorias Previas (Tabla)** | +0.379 (55.6%) | +0.324 (54.3%) | +0.273 (50.4%) | **+0.296** (53.9%) | +0.570 (65.9%) |
| **Dependencia de 1 Estrella (Star Reliance)** | **-0.254** | **-0.254** | **-0.232** | **-0.214** | **-0.366** |

---

## 🔬 3. Hallazgos Críticos para el Modelado de Cuartos

### 1. El Primer Cuarto (Q1) Pertenece a los Titulares
- La correlación del quinteto titular con el resultado del primer cuarto es de **+0.545** (la más alta de todos los cuartos individuales), acertando el ganador de Q1 el **66.3% de las veces**.
- La influencia del banquillo en Q1 es marginal (+0.232), dado que en los primeros 10 minutos los entrenadores apenas ejecutan 1 o 2 cambios de descanso corto.

### 2. El Segundo Cuarto (Q2) es el Territorio del Banquillo
- En Q2, la correlación del banquillo salta bruscamente a **+0.379**, y su capacidad predictiva se iguala a la de los titulares (**61.4% de precisión**).
- Los equipos con suplentes de bajo nivel sufren parciales en contra demoledores en Q2 cuando dan descanso a sus estrellas.

### 3. El Cuarto 4 (Q4) y la Pérdida de Peso del Récord Previo
- La posición en la tabla o el balance histórico previo (`delta_standings_win_pct`) se diluye notablemente al llegar al último cuarto: su correlación cae a tan solo **+0.296**.
- En Q4, el **diferencial de banquillo (+0.348)** es casi tan importante como el de los titulares (+0.401), debido a que la acumulación de faltas personales (*foul trouble*) y la fatiga física obligan a tirar de suplentes.
- **Conclusión de Oro para el Monitor V2/V3:** Ningún modelo pre-partido puede batir a un modelo de live betting en Q4, porque en el minuto 27 el momentum real acumulado en pista supera al récord en el papel.

### 4. La Trampa de la "Monodependencia" (Star Reliance)
- Equipos donde un solo jugador anota más del 30-35% de los puntos presentan una correlación **sistemáticamente negativa** (-0.21 a -0.36) frente a equipos corales.
- Al llegar a la segunda mitad (Q3 y Q4), la defensa rival ajusta ayudas y dobles marcas sobre la estrella, provocando el colapso ofensivo del equipo dependiente.

---

## 🌍 4. Variaciones Tácticas y Estilos de Roster por País

El análisis cruzado entre `matches`, `lineups` y `player_stats` por ámbito nacional revela patrones estructurales evidentes:

| País / Región | Muestra Partidos | Rating Titulares | Rating Banquillo | % Puntos Banquillo | % Tiros Triples (3P Rate) | Asistencias / Pérdidas (AST/TO) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Japón (B-League)** | 901 | 6.97 | 6.44 | 34.0% | 42.3% | **1.73 (Máxima disciplina)** |
| **Alemania (BBL)** | 554 | 6.79 | 6.50 | **49.3%** | 41.5% | 1.19 |
| **Australia (NBL)** | 2,298 | 6.90 | 6.41 | 26.3% | **45.1% (Máximo volumen 3P)**| 1.29 |
| **España (ACB)** | 870 | 6.77 | 6.57 | **42.2%** | 42.4% | 1.27 |
| **Italia (Serie A/A2)** | 1,753 | 6.83 | 6.38 | 31.7% | 40.5% | 1.41 |
| **Lituania (LKL/NKL)**| 1,006 | 6.91 | 6.51 | 34.9% | 39.8% | 1.33 |
| **Polonia (PLK/1 Liga)**| 1,513 | 6.89 | 6.48 | 35.2% | 39.3% | 1.36 |
| **Grecia (GBL)** | 359 | 6.57 | 6.24 | 34.5% | 34.9% | **0.83 (Juego trabado/faltas)**|
| **USA (NBA/NCAA)** | 6,171 | **7.15** | **6.59** | 69.9% (NCAA/rot.)| 37.7% | 1.19 |

### Implicaciones para Features de Machine Learning:
1. **Factor de Profundidad (Alemania y España):** En la Liga ACB y la BBL alemana, el banquillo aporta más del **42% al 49% de los puntos**. Modelar la fatiga de titulares aquí es menos relevante; lo determinante es la calidad del banquillo.
2. **Volatilidad Perimetral (Australia):** Con un 45.1% de tiros de campo provenientes de triples y banquillos cortos (26.3% de puntos), los equipos australianos experimentan rachas y sequías anotadoras más pronunciadas en Q4.
3. **Control del Ritmo (Japón vs Grecia):** En Japón el ratio AST/TO de 1.73 denota posesiones muy seguras y pocos contraataques fáciles para el rival. En Grecia (0.83), el juego es físico y trabado, con muchas pérdidas y faltas personales que aumentan los tiros libres en el clutch.

---

## 🛠️ 5. Recomendación de Features para el Pipeline de Modelos

Para incorporar estas fuerzas dinámicas al entrenamiento de modelos:

```python
# Features calculadas por partido a partir del roster y forma previa:
features = [
    # 1. Fuerza relativa del quinteto inicial:
    "delta_starter_rating",       # (starter_rating_home - starter_rating_away)
    
    # 2. Fuerza relativa del banquillo:
    "delta_bench_rating",         # (bench_rating_home - bench_rating_away)
    
    # 3. Profundidad interna (caída titular vs reserva):
    "delta_roster_depth",         # ((starter - bench)_away - (starter - bench)_home)
    
    # 4. Concentración de estrella:
    "delta_star_dependency",      # (% pts top 1 anotador home - away)
    
    # 5. Estilo nacional / regional:
    "country_expected_3p_rate",   # Prior de triples del país
    "country_expected_bench_share",# Prior de banquillo del país
    
    # 6. Fuerza rodante de tabla:
    "delta_standings_win_pct"     # Diferencial de % victorias previas en la liga
]
```
