# 🌐 Plan Estratégico de Descargas Históricas y Backfilling Adaptativo (2015–2026)

> **PROPÓSITO:** Este documento establece el plan director de recolección de datos históricos (backfilling) para el sistema `pulpa`. Integra los hallazgos empíricos sobre la evolución de la telemetría en SofaScore (2015–2026) y presenta la herramienta especializada [`tools/smart_historical_backfill.py`](file:///C:/Users/App/Desktop/pulpa/tools/smart_historical_backfill.py) diseñada para ejecutar descargas segmentadas por clústeres y modos de completitud sin riesgo de bloqueos.

---

## 🔬 1. Hallazgo Empírico: ¿Qué Datos Existen Realmente en SofaScore según el Año?

Al auditar la API móvil de SofaScore realizando consultas reales a partidos de **2015, 2018, 2020 y 2024**, descubrimos la evolución histórica de su base de datos:

| Época / Rango de Años | Marcadores Cuartos (Q1–Q4) | Historial H2H y Marcador FT | Play-by-Play (PBP) | Gráfica de Momentum (`graph_points`) | Utilidad en el Sistema `pulpa` |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2015 – 2017** | ✅ **100% Disponible** | ✅ **100% Disponible** | ❌ *No existía en baloncesto* | ❌ *No existía en baloncesto* | **Siembra de ELO Dinámico y H2H Histórico** (10 años de fortaleza de franquicias). |
| **2018 – 2020** | ✅ **100% Disponible** | ✅ **100% Disponible** | 🟡 **Disponible en Ligas Top** (NBA, ABA, etc.) | 🟡 **Disponible en Ligas Top** (40-48 puntos) | **Entrenamiento ML en Ligas Élite** + ELO histórico global. |
| **2021 – 2026** | ✅ **100% Disponible** | ✅ **100% Disponible** | ✅ **Universal** (100-150 eventos) | ✅ **Universal** (45-55 puntos) | **Entrenamiento Pleno ML** para todos los modelos (`m27_v4`, `m34_nba_12m`). |

> 💡 **Lección Crucial de Ingeniería:**
> El script anterior de backfill (`match/cli.py backfill`) descartaba partidos si no tenían Play-by-Play o Graph Points (`_has_usable_data`). Esto provocaba que si intentabas descargar 2015, el 100% de los partidos eran rechazados como "incompletos".
> Para aprovechar los datos desde 2015, se creó el **Backfilling por Capas (Dual-Tier Architecture)**.

---

## 🏛️ 2. Arquitectura de Descarga por Capas (Dual-Tier)

La herramienta [`tools/smart_historical_backfill.py`](file:///C:/Users/App/Desktop/pulpa/tools/smart_historical_backfill.py) opera con dos modos de ingesta:

```mermaid
graph TD
    API["SofaScore API Móvil (Android JWT)"] --> Filter{"Evaluador de Capa (check_data_usability)"}

    Filter -->|Tiene Q1-Q4 + PBP + Graph Points| ML["🟢 CAPA 1: FULL ML (Entrenamiento Min 27/34)"]
    Filter -->|Tiene Q1-Q4 + FT (Sin PBP/GP)| ELO["🔵 CAPA 2: ELO / H2H (Fuerza Relativa y Cara a Cara)"]
    Filter -->|Incompleto o No Finalizado| Skip["⚪ Omitir / Descartar"]

    ML --> DB1["Tablas: matches, quarter_scores, play_by_play, graph_points"]
    ELO --> DB2["Tablas: matches, quarter_scores, match_h2h"]
```

1. **Capa 1: `FULL_ML` (2018–2025):** 
   - Exige que el partido contenga los 4 cuartos cerrados, eventos de anotación en Q1/Q2 y gráfica de presión.
   - Se destina directamente a nutrir los modelos de Gradient Boosting (`m27_v4`, `m34_nba_12m`).
2. **Capa 2: `ELO_H2H` (2015–2018):**
   - Exige únicamente los marcadores de los cuartos (`Q1`, `Q2`, `Q3`, `Q4`) y marcador final.
   - Guarda los registros en `matches` y `quarter_scores` para que el algoritmo de **Elo por cuartos y H2H** tenga 10 años de memoria y no empiece ciego en 1,500 puntos.

---

## 🛠️ 3. El Programa Especializado: `tools/smart_historical_backfill.py`

El programa ha sido implementado en la raíz bajo [`tools/smart_historical_backfill.py`](file:///C:/Users/App/Desktop/pulpa/tools/smart_historical_backfill.py).

### Opciones de Ejecución:
- `--start-date YYYY-MM-DD`: Fecha de inicio.
- `--end-date YYYY-MM-DD`: Fecha de finalización.
- `--cluster {all, fiba_men, nba_12m, fiba_women}`: Filtra automáticamente los partidos para descargar **únicamente las ligas que pertenecen al modelo deseado**.
- `--mode {auto, full_ml, elo_h2h}`:
  - `auto` *(Recomendado)*: Si el partido tiene PBP/GP lo guarda como `FULL_ML`; si solo tiene cuartos, lo guarda como `ELO_H2H`.
  - `full_ml`: Solo guarda si tiene telemetría completa de momentum.
  - `elo_h2h`: Acepta partidos con marcadores básicos de cuartos.
- `--include-blacklist`: Si no se pasa esta bandera, **omite automáticamente juveniles (U16-U23), College NCAA y partidos amistosos**, ahorrando miles de peticiones innecesarias.
- `--delay 0.9`: Segundos de espera entre partidos con perturbación aleatoria (*jitter*) para proteger la sesión del WAF Fastly.

---

## 📅 4. Plan de Ejecución en 3 Fases Cronológicas

Para maximizar el impacto en el rendimiento de los modelos sin sobrecargar la cuota de red, se establece el siguiente cronograma:

### 🥇 Fase 1: Descarga de Temporadas 2023–2025 (Prioridad Inmediata)
- **Objetivo:** Multiplicar por 3× el dataset de entrenamiento para lanzar [`m27_v4`](file:///C:/Users/App/Desktop/pulpa/docs/PROPUESTA_NUEVO_MODELO_M27_V4.md) con cero riesgo de suerte estacional.
- **Rango:** `2023-10-01` al `2025-10-07` (2 años completos).
- **Modo:** `--mode full_ml` o `--mode auto`.
- **Volumen Esperado:** ~90,000 partidos con telemetría completa (PBP + GP + Lineups).
- **Comando de Ejecución:**
  ```powershell
  .venv\Scripts\python.exe tools\smart_historical_backfill.py --start-date 2023-10-01 --end-date 2025-10-07 --cluster all --mode auto
  ```

---

### 🥈 Fase 2: Profundidad de Ligas Élite 2018–2023 (NBA y FIBA Top)
- **Objetivo:** Ampliar la muestra de la NBA (llegar a más de 8,000 partidos NBA históricos) y EuroLeague/ACB para robustecer el modelo `m34_nba_12m`.
- **Rango:** `2018-10-01` al `2023-09-30`.
- **Modo:** `--mode auto`.
- **Comando Especializado NBA:**
  ```powershell
  .venv\Scripts\python.exe tools\smart_historical_backfill.py --start-date 2018-10-01 --end-date 2023-09-30 --cluster nba_12m --mode auto
  ```

---

### 🥉 Fase 3: Génesis Histórico ELO y H2H 2015–2018
- **Objetivo:** Dotar a las variables $E0\text{–}E3$ (Elo de Q4) y $M8\text{–}M9$ (H2H) de una base histórica profunda de 10 años.
- **Rango:** `2015-01-01` al `2018-09-30`.
- **Modo:** `--mode elo_h2h`.
- **Volumen Esperado:** ~50,000 partidos con marcadores oficiales por cuarto.
- **Comando de Ejecución:**
  ```powershell
  .venv\Scripts\python.exe tools\smart_historical_backfill.py --start-date 2015-01-01 --end-date 2018-09-30 --cluster all --mode elo_h2h
  ```

---

## 🛡️ 5. Políticas de Operación Segura y Prevención de Bloqueos (WAF Fastly)

1. **Ritmo de Peticiones:**
   - La herramienta aplica por defecto un retraso de $0.9 \text{ s} \pm 0.3 \text{ s}$ por partido. Esto equivale a una velocidad de **~50 a 60 partidos por minuto**.
   - No genera ráfagas concurrentes para evitar la activación del reto anti-bot (`403 challenge`).
2. **Rotación Automática de JWT:**
   - Cada 10 partidos completados, el script invoca `token_pool.notify_match_done()`, rotando la firma y la sesión entre los tokens disponibles en `monitor_v3/config/tokens.json`.
3. **Reanudación Transparente:**
   - Si la descarga se interrumpe (por reinicio de PC o pérdida de conexión), al volver a ejecutar el comando se reanuda instantáneamente: cualquier partido que ya figure en `matches` es omitido en $0.1 \text{ ms}$ sin realizar peticiones HTTP a SofaScore.
