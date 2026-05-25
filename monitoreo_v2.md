# Especificación de Arquitectura de Producción: Monitor de Apuestas Modular — `bet_monitor_v2.py`

## Propósito General
`bet_monitor_v2.py` es un daemon asíncrono de alto rendimiento optimizado con `asyncio` que gestiona el ciclo de vida completo del monitoreo de baloncesto en SofaScore, la ejecución de inferencias predictivas en paralelo para la ventana del último cuarto (Q4) y la persistencia transaccional. El sistema opera bajo la zona horaria **UTC-6 (America/Mexico_City)**, está sujeto al cumplimiento estricto de la norma **Flake8** y se distribuye en una arquitectura modular desacoplada, con un sistema dinámico de control de ligas externa y un almacenamiento de datos relacional auditivo.

---

## 1. Estructura del Proyecto y Modularización
Queda estrictamente prohibido compilar el daemon en un único archivo. El sistema se estructurará obligatoriamente bajo el siguiente árbol jerárquico de paquetes:

```text
bet_monitor_v2/
│
├── config/
│   ├── __init__.py
│   ├── constants.py       # Constantes del juego, umbrales y variables de entorno.
│   └── leagues.yaml       # Capa de configuración externa y declarativa para filtros de ligas.
│
├── database/
│   ├── __init__.py
│   ├── connection.py      # Inicialización de SQLite, activación de modo WAL y timeouts.
│   └── repository.py      # Transacciones CRUD, carga de configuraciones dinámicas y queries indexados.
│
├── scrapers/
│   ├── __init__.py
│   ├── base_scraper.py    # Abstracción de clientes y manejo de rotación de sesiones/cookies.
│   ├── live_scraper.py    # Consultas rápidas (snapshots ligeros) empleando Obscura (h4ckf0r0day/obscura).
│   └── browser_client.py  # Navegación y descargas pesadas (FT) vía Playwright (Chrome Headless).
│
├── models/
│   ├── __init__.py
│   └── evaluator.py       # Orquestador de inferencia, caching de Motores (v12-v17) en hilos separados.
│
├── notifications/
│   ├── __init__.py
│   └── telegram_bot.py    # Despachador de alertas, apuestas y resultados con markup inline.
│
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Formateador de consola ANSI de colores y rotación diaria de archivos.
│   └── helpers.py         # Filtros de tiempo humano, control de delays adaptativos y EMA del ritmo.
│
└── main.py                # Punto de entrada, event loop asíncrono y control de señales (asyncio.Event).
```

---

## 2. Constantes Técnicas, Parámetros Globales y Variables de Entorno (`config/constants.py`)

### 2.1 Variables de Entorno Requeridas
El sistema debe consumir de forma estricta las siguientes variables desde el entorno global:
*   `TELEGRAM_BOT_TOKEN`: Token único de autenticación de la API del bot de Telegram.
*   `TELEGRAM_CHAT_ID`: Identificador de destino del canal o chat de Telegram donde se despachan alertas y resultados.
*   `SOFASCORE_SCRAPER_BACKEND`: Fijado mandatoriamente en `obscura` para delegar consultas rápidas al backend anti-detección basado en la biblioteca [Obscura (h4ckf0r0day/obscura)](https://github.com/h4ckf0r0day/obscura).

### 2.2 Constantes de Configuración

```python
# --- Rutas de Archivos de Configuración e infraestructura ---
DB_FILE_PATH = "/match/matches.db"          # Ubicación absoluta y estricta de la base de datos
LEAGUES_CONFIG_PATH = "config/leagues.yaml"  # Ruta del archivo dinámico de ligas

# --- Configuración de Red y Anti-Ban ---
MAX_CONCURRENT_FETCHES = 2         # Semáforo de control de concurrencia de red
GLOBAL_FETCH_MIN_SPACING_SECS = 1.5 # Espaciado mínimo absoluto entre peticiones consecutivas
GLOBAL_403_STREAK_TRIGGER = 6      # Número de errores 403 seguidos que disparan el cooldown
GLOBAL_403_COOLDOWN_SECS = 900     # Tiempo de enfriamiento (15 min) ante bloqueos
SESSION_ROTATE_EVERY = 40          # Rotación de sesión/cookies cada N llamadas exitosas
SESSION_ROTATE_PAUSE_SECS = 45     # Pausa obligatoria al rotar sesión
MONITORING_LOCK_DURATION_SECS = 300 # Lock de 5 min para pausar tareas secundarias al haber partidos Live

# --- Ritmo de Juego y Ventana de Monitoreo Q4 ---
SECS_PER_GAME_MIN = 170            # Estimación inicial de segundos reales por minuto de juego
Q4_ONLY_EARLY_WAKE_MINUTE = 27     # Minuto del partido para despertar el watcher completo
Q4_ONLY_WAKE_LEAD_MINUTES = 2      # Margen de seguridad previo al minuto de despertar
Q4_TOO_LATE_BET_MINUTE = 33        # Límite para alertas ordinarias (minutos superiores marcan TARDÍA)
Q4_TOO_LATE_HARD_MINUTE = 36       # Bloqueo total: No se procesan apuestas pasada esta marca de tiempo
Q4_LATE_BET_MAX_DISADVANTAGE = 5   # Máxima desventaja de puntos permitida para el pick en apuesta tardía
Q4_WAITING_MAX_TICKS = 8           # Ticks máximos de espera por sincronización de score de periodos previos antes de abortar
Q4_STALE_MAX_TICKS = 5             # Ticks máximos sin crecimiento en la gráfica antes de abortar

# --- Sondeos Prestart (Modo Sonda) ---
PRESTART_PROBE_MIN_SECS = 60       # Intervalo mínimo de sondeo para partidos no iniciados
PRESTART_PROBE_MAX_SECS = 300      # Intervalo máximo de sondeo (Cap del Backoff)
PRESTART_PROBE_BACKOFF = 1.25      # Factor multiplicador del temporizador exponencial
PROBE_GLOBAL_TIMEOUT_SECS = 21600  # Timeout global (6h) para evitar tareas colgadas en probe mode

# --- Polling y Descarga Final (FT) ---
POLL_INTERVAL_FF = 150             # Intervalo base para verificar partido finalizado
POLL_JITTER_FF = 45                # Jitter máximo para desincronizar polling final
FINAL_FETCH_EXTRA_SECS = 300       # Margen de seguridad tras tiempo estimado de fin
FINAL_FETCH_MIN_GP = 20            # Mínimo de puntos de gráfica requeridos para persistencia válida
FT_SCRAPE_SLOT_SPACING_BASE = 35.0 # Identificador base en segundos para encolar descargas FT

# --- Intervalos de Tareas Secundarias ---
SCHEDULE_REFRESH_HOURS = 8         # Frecuencia de actualización del itinerario
PENDING_RECHECK_SECS = 3600        # Recheck de resultados cada hora
POLL_NEAR_SECS = 45                # Polling corto dentro de la ventana de juego activa
IDLE_POLL_SECS = 180               # Polling largo en periodos de inactividad
NO_BET_CONFIRM_TICKS = 3           # Confirmaciones consecutivas necesarias para asentar NO_BET
UTC_OFFSET_HOURS = -6              # Desfase horario local
```

---

## 3. Modelo de Persistencia Transaccional (`database/`)
El almacenamiento de datos se centraliza estrictamente en la ruta `/match/matches.db` empleando **SQLite** con la activación obligatoria de `PRAGMA journal_mode=WAL;` y `PRAGMA busy_timeout=30000;`. Toda interacción debe gestionarse mediante transacciones atómicas. 

### Regla Estricta de Nomenclatura
**Todas las tablas del sistema deben incluir de forma mandatoria el sufijo `_v2` en su identificador.**

### Esquema Detallado de Tablas y Atributos

1.  **`bet_monitor_schedule_v2`**: Orquestación y estado vital del itinerario diario.
    *   `match_id` (TEXT, PRIMARY KEY), `home_team` (TEXT), `away_team` (TEXT), `league` (TEXT), `event_date` (TEXT).
    *   `scheduled_utc_ts` (INTEGER), `scheduled_utc` (TEXT), `status_type` (TEXT).
    *   `status` (TEXT): Control de flujo (`pending`, `in_progress`, `done`, `discarded`).
    *   `q4_checked` (INTEGER), `q4_signal` (TEXT), `q4_notified` (INTEGER), `q4_model` (TEXT).
    *   `final_fetched` (INTEGER), `final_fetch_at` (TEXT), `skip_reason` (TEXT), `updated_at` (TEXT).
    *   *Índices requeridos:* `idx_schedule_date_status_v2` sobre (`event_date`, `status`), `idx_schedule_utc_ts_v2` sobre (`scheduled_utc_ts`).

2.  **`bet_monitor_log_v2`**: Almacenamiento granular e histórico de auditoría por modelo e inferencia efectuada.
    *   `id` (INTEGER PRIMARY KEY AUTOINCREMENT).
    *   `match_id` (TEXT): ID único de SofaScore asociado.
    *   `model_version` (TEXT): Identificador único del modelo de ML (ej: `v12`, `v15`, `v17`).
    *   `target_quarter` (INTEGER): Cuarto objetivo (Fijado en 4).
    *   `inference_minute` (INTEGER): Minuto exacto del partido en el que se gatilló la ejecución de inferencia analítica.
    *   `graph_points_count` (INTEGER): Volumen total de puntos de la gráfica (`graph_points`) acumulados presentes al momento de calcular la predicción.
    *   `raw_json` (TEXT): Volcado íntegro, crudo y completo estructurado en formato JSON del snapshot analítico procesado (`fetch_match_by_id`) para auditoría offline posterior y re-entrenamiento.
    *   `signal_type` (TEXT): Dictamen del modelo (`BET`, `NO_BET`, `UNAVAILABLE`, `ERROR`).
    *   `picked_side` (TEXT): Lado seleccionado (`HOME`, `AWAY`, `NONE`).
    *   `confidence` (REAL): Score de confianza o probabilidad arrojado por el motor de inferencia.
    *   `actual_home_score` (INTEGER): Marcador del equipo local al momento de la inferencia.
    *   `actual_away_score` (INTEGER): Marcador del equipo visitante al momento de la inferencia.
    *   `result` (TEXT): Estado de liquidación del pick (`pending`, `win`, `loss`, `push`).
    *   `created_at` (TEXT): Timestamp con la fecha y hora de creación de la transacción.
    *   *Índices requeridos:* `idx_log_match_model_v2` sobre (`match_id`, `model_version`), `idx_log_result_v2` sobre (`result`).

3.  **`eval_match_results_v2`**: Matriz condensada de rendimiento predictivo por modelo.
    *   Contiene `match_id` (TEXT, PRIMARY KEY), `available` (INTEGER), y columnas dinámicas mapeadas en tiempo de ejecución por combinación de tag de versión (ej: `q4_signal__v12`, `q4_pick__v12`, `q4_confidence__v12`, `q4_outcome__v12`).

4.  **`quarter_scores_v2`**: Marcadores consolidados oficiales por periodo.
    *   `match_id` (TEXT, PRIMARY KEY), `q1_home`, `q1_away`, `q2_home`, `q2_away`, `q3_home`, `q3_away`, `q4_home`, `q4_away`, `ot_home`, `ot_away`.

5.  **`leagues_config_v2`**: Espejo transaccional de los filtros declarativos externos.
    *   `league_name_pattern` (TEXT, PRIMARY KEY): String o patrón SQL LIKE (ej: '%Women%', '%U21%').
    *   `filter_mode` (TEXT): Tipo de restricción (`FULL_MONITOR` / `FT_ONLY` / `EXCLUDE`).
    *   `updated_at` (TEXT).

---

## 4. Estrategia Multicapa Anti-Ban y Regulación de Concurrencia

### 4.1 Mecanismos de Control de Tráfico
1.  **Semáforo de Concurrencia Limitado:** Se define `_fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)` compartido globalmente. Toda petición HTTP saliente debe ejecutar un `async with _fetch_sem:`.
2.  **Garantía de Espaciado y Jitter Microscópico:** Tras liberar el semáforo, el cliente forzará un delay de `GLOBAL_FETCH_MIN_SPACING_SECS` sumado a un jitter estocástico aleatorio de `0.0` a `0.5` segundos.
3.  **Bloqueo de Monitoreo de Prioridad Live:** Cuando un partido entre en ventana activa (Live), el sistema fijará el timestamp global `_monitoring_lock_until = ahora + MONITORING_LOCK_DURATION_SECS`. Mientras este candado temporal esté vigente, el ciclo principal descartará automáticamente la ejecución de tareas secundarias pesadas (como `recheck_pending` o escaneos de mantenimiento).
4.  **Cola de Slots Serializada para Cierres (FT Scrape Slot Queue):** Para prevenir picos de peticiones si múltiples partidos concluyen a la misma hora, se implementará un calculador dinámico de slots en cola. El espaciado entre descargas se adaptará en función del volumen de partidos finalizados simultáneamente:
    $$\text{slot\_spacing} = \frac{\text{FT\_SCRAPE\_SLOT\_SPACING\_BASE}}{\max(1, \text{peticiones\_en\_vuelo})}$$
    Cada tarea de finalización deberá reclamar su slot en orden cronológico estricto antes de lanzar **Playwright con Chrome Headless**.
5.  **Manejo de Cooldown por Racha:** Si el sistema acumula `GLOBAL_403_STREAK_TRIGGER` errores concurrentes de tipo HTTP 403, suspenderá de inmediato todas las colas de peticiones y activará un estado de hibernación total durante `GLOBAL_403_COOLDOWN_SECS` (15 minutos).

### 4.2 Especificación de Funciones de Scrapping
El paquete `scrapers/` implementará dos operaciones core asíncronas con cargas de red asimétricas para optimizar la evasión antiban:

*   **`fetch_event_snapshot(match_id)` [Modo Sonda / Probe Mode]:**
    *   *Mecánica:* Abre exactamente un contexto de navegador único (`1 browser open`) y realiza una única petición (`1 request`) al endpoint directo de SofaScore de tipo `/event/{id}`.
    *   *Payload:* Extrae y devuelve exclusivamente el estado plano del partido (`status_type`, `status_description`) y el marcador global (`home_score`, `away_score`). 
    *   *Uso:* Ejecutado únicamente mientras el partido se encuentra programado y no ha comenzado (`notstarted`), previniendo el consumo innecesario de ráfagas HTTP concurrentes.
*   **`fetch_match_by_id(match_id)` [Loop Principal / Ventana Activa Q4]:**
    *   *Mecánica:* Abre un contexto de navegador único (`1 browser open`) y realiza una ráfaga secuencial controlada de **entre 7 y 10 peticiones HTTP internas** para compilar la totalidad de estructuras analíticas requeridas por los modelos de ML.
    *   *Payload:* Reúne y consolida los siguientes sub-endpoints en un único mapa de datos:
        1.  Objeto general del partido (`event`).
        2.  Incidentes cronológicos / Play-by-Play (`incidents`).
        3.  Puntos analíticos de la gráfica de rendimiento de juego (`graph`).
        4.  Historial mutuo de enfrentamientos (`H2H`).
        5.  Estadísticas avanzadas acumuladas del partido (`estadísticas`).
        6.  Alineaciones y rotaciones en duela (`alineaciones`).
        7.  Cuotas y momios de apuestas (`cuotas`).
        8.  *(Opcional / Condicional):* Eventos de equipo (`team events`) utilizados para el cálculo de H2H desagregado por cuartos y fuerza proyectada (`team strength`).
    *   *Uso:* Invocado exclusivamente cuando el watcher despierta para la evaluación crítica de la ventana (Minuto 27 de juego en adelante).

---

## 5. Capa de Configuración Dinámica de Ligas (`config/leagues.yaml`)
Para asegurar la flexibilidad operativa, se prohíbe el uso de cadenas SQL de tipo `LIKE` o nombres de ligas hardcodeados en el código. El sistema procesará las exclusiones y modos operativos mediante una capa declarativa.

### Estructura de Configuración del Archivo YAML (`config/leagues.yaml`)
El módulo `utils/helpers.py` o `config/constants.py` leerá este archivo en el arranque (y opcionalmente recargará en caliente si cambia). Soporta tanto coincidencia exacta por ID de SofaScore como patrones de texto (*sub-strings*):

```yaml
# =====================================================================
# CONFIGURACIÓN DE FILTROS Y RESTRICCIONES DE LIGAS (DIFERENCIACIÓN Q4)
# =====================================================================

# Ligas que se ignoran por completo (No generan ninguna acción)
excluded_leagues:
  patterns:
    - "Regional Cup"
    - "Development League"
    - "Friendly"
  ids:
    - 9999
    - 8888

# Ligas restringidas a Modo Degradado (Modo FT-Only: Solo descarga y guarda marcador final, NO Inferencia ML)
ft_only_leagues:
  patterns:
    - "WNBA"
    - "Women"
    - "Femenina"
    - "Feminina"
    - "U21"
    - "U19"
    - "ABA Liga"
    - "NBA"
    - "Lega A"
    - "Turkish Basketball Super League"
    - "Playoffs"
    - "Stoiximan GBL"
    - "Copas regionales"
  ids:
    - 1234
    - 5678

# Nota de Diseño: Cualquier liga del itinerario que NO coincida con estos patrones 
# se asume en modo FULL_MONITOR (Monitoreo adaptativo e inferencia de modelos ML en Q4).
```

El repositorio de base de datos (`database/repository.py`) puede sincronizar este archivo YAML con la tabla `leagues_config_v2` al inicio para mantener consistencia si se prefiere control mixto.

---

## 6. El Ciclo Principal (`main.py`) y Carga de Itinerarios
El orquestador principal controlará el flujo mediante un ciclo infinito interrumpible por señales del sistema (`asyncio.Event`):

1.  **Inicialización:** Ejecuta la creación y verificación de índices de base de datos sobre `/match/matches.db`, seguida de una rutina inicial de reconciliación de resultados colgados (`reconcile_pending_results`). Carga y parsea la configuración declarativa de `config/leagues.yaml`.
2.  **Descarga del Itinerario de Doble Fecha:** Utilizando Playwright u Obscura, consulta la API de SofaScore para la fecha local actual y la fecha del día siguiente (UTC+1, previniendo el desfase transnocturno). Filtra y alinea los partidos cuyos inicios correspondan a la zona horaria UTC-6 local.
3.  **Doble Filtro de Asignación Basado en Configuración Dinámica:**
    *   **Lista Blanca (`_get_pending_rows`):** Corresponde a los partidos cuyas ligas **NO** matchean con ninguna regla de `excluded_leagues` ni `ft_only_leagues` en el YAML. Se instancian en corrutinas `_watch_match` completas (Monitoreo adaptativo e Inferencia de modelos ML en ventana Q4).
    *   **Modo Degradado / FT-Only (`_get_all_pending_rows`):** Corresponde a los partidos que matchean con los patrones o IDs declarados bajo `ft_only_leagues`. Se instancian en corrutinas de ciclo de vida pasivo (`ft_only=True`) únicamente para calcular su hora de término y ejecutar la descarga final (FT) para enriquecer el almacenamiento histórico en `quarter_scores_v2`. Los partidos en `excluded_leagues` se omiten del event loop de inmediato.

---

## 7. Máquina de Estados del Watcher Co-rutina (`_watch_match`)

Cada partido descubierto es asignado a una corrutina independiente que maneja variables de estado locales bajo la siguiente lógica algorítmica:

### 7.1 Flujo Degradado (`ft_only=True`)
Espera pasivamente el timestamp de inicio estimado, reclama secuencialmente su slot en la cola de descargas (`_claim_ft_scrape_slot`) y deriva directamente a la rutina de captura final `_final_fetch_and_save`.

### 7.2 Modo Sonda Prestart (`q4_prestart_probe_mode`)
El watcher mitiga el desperdicio de ancho de banda suspendiendo las descargas pesadas de objetos HTML/JSON complejos antes del inicio del juego.
*   Realiza peticiones periódicas optimizadas exclusivamente al endpoint ligero de SofaScore vía `fetch_event_snapshot(match_id)`.
*   Aplica un temporizador con multiplicación exponencial continua (`PRESTART_PROBE_BACKOFF`), variando según el desfase (*drift*) temporal restante hacia el evento.
*   *Condiciones de Salida del Modo Sonda:*
    1.  Detección de estados en juego (`inprogress`, `live`).
    2.  Detección de término prematuro (`finished`).
    3.  Cálculo de seguridad por drift: Si el tiempo transcurrido supera el umbral estricto calculado por:
        $$\text{drift} \ge \text{secs\_per\_gmin} \times \text{Q4\_ONLY\_EARLY\_WAKE\_MINUTE}$$
        El watcher fuerza su salida del modo sonda para resolver el estado real mediante un fetch completo.
    4.  Activación del `PROBE_GLOBAL_TIMEOUT_SECS` (6 horas) para evitar fugas de memoria o procesos eternos en partidos suspendidos permanentemente inestables en estado `notstarted`.

### 7.3 Temporizador Adaptativo de Aproximación
Una vez el partido entra en estado activo y se encuentra fuera del modo sonda, el watcher evalúa continuamente el Live Score extrayendo **únicamente la variable del minutero**. El ciclo calcula su tiempo de suspensión óptimo (*tiered sleep*) reduciendo las consultas a medida que se acorta la distancia hacia la ventana de evaluación de Q4 (Minuto 27 objetivo):

```
[Inicio Partido] 
       │
       ▼
(Despertar Inicial: Start + 5 min)
       │
       ├─► [Si está Cancelado/Atrasado] ──► [Actualizar BD] ──► [Finalizar Watch]
       │
       ├─► [Si está en Blacklist de Configuración] ──► [Calcular FT + 1h] ──► [Programar Descarga FT con Playwright]
       │
       └─► [Si está En Curso (Normal)]
                 │
                 ▼
     ┌──► [Bucle Temporizador Adaptativo (Obscura)]
     │           │
     │           ├─► Minutos restantes > 10? ──► Dormir lapso adaptativo largo (Cap 3 min) y reevaluar
     │           ├─► Minutos restantes ≤ 5? ───► Dormir 90s y reevaluar
     │           ├─► Minutos restantes ≤ 2? ───► Dormir 60s y reevaluar
     │           └─► Minutos restantes ≤ 1? ───► Dormir 45s y reevaluar
     │
     └─── [¿Se alcanzó el Minuto 27 objetivo?]
                 │
                 ▼
       [Llamar fetch_match_by_id (Descarga Completa)]
                 │
                 ▼
       [Fase de Inferencia Ventana Q4]
```

---

## 8. Pipeline de Inferencia de Modelos ML y Reglas de Validación
Al consolidar la descarga completa del partido mediante `fetch_match_by_id`, el payload se procesa en un hilo de ejecución independiente (`asyncio.to_thread`) contra el catálogo de modelos activos configurados en `ACTIVE_MODELS` (ej: `v6_2` y `v12`), implementando un patrón Singleton para el almacenamiento en caché de los motores analíticos en la variable `_ENGINE_CACHE`.

### Criterios de Aborto por Calidad de Datos
*   **Falta de Crecimiento (Stale Graph):** Si los puntos de la gráfica (`graph_points`) se mantienen estáticos durante `Q4_STALE_MAX_TICKS` consecutivos, la tarea aborta registrando la incidencia como `graph_stale_timeout`.
*   **Ausencia de Datos Históricos:** Si tras transcurrir un lapso equivalente en tiempo real a 55 minutos (`NO_GRAPH_REAL_SECS`) el partido carece de estructuras gráficas de rendimiento, se descarta el procesamiento predictivo de inmediato.

### Clasificación y Gestión de Señales de Entrada/Salida
Las respuestas analíticas se supeditan a una regla de guardias temporales basada en el minuto de juego exacto verificado en SofaScore dentro de la ventana de Q4 (Minuto 27 a 38 máximo):
*   **Señal Operable (`BET_HOME` / `BET_AWAY`):**
    *   Si el minuto actual es $\le 33$ (`Q4_TOO_LATE_BET_MINUTE`): Se cataloga y despacha como **APUESTA ORDINARIA Q4**.
    *   Si el minuto actual se comprende entre el rango $34$ y $36$ (Rango de Apuesta Tardía):
        *   **Filtro Temprano por Diferencia de Marcador:** El sistema debe evaluar obligatoriamente el marcador en vivo al momento de la inferencia. Si la desventaja del equipo seleccionado (`picked_side`) supera el límite de puntos definido en `Q4_LATE_BET_MAX_DISADVANTAGE`, la apuesta **se descarta por completo** (ej: Marcador 10 - 17, desventaja de 7 puntos, con pick `HOME` $
ightarrow$ SE DESCURTA / NO SE LANZA). Si el marcador está cerrado o dentro del umbral tolerado (ej: Marcador 10 - 11, desventaja de solo 1 punto, con pick `HOME` $
ightarrow$ SE MANTIENE / "SE QUEDA"), el flujo continúa y se notifica formalmente en Telegram como `⚪️ APUESTA TARDIA Q4`.
    *   Si el minuto actual es $\ge 36$ (`Q4_TOO_LATE_HARD_MINUTE`): Se anula por completo la señal y se fuerza el estado de la tarea a omitido por fuera de tiempo.
*   **Persistencia y Registro en Log Transaccional:** Cada inferencia calculada (independientemente de si arroja `BET` o `NO_BET`) debe registrar obligatoriamente una nueva fila en `bet_monitor_log_v2` inyectando el total de puntos de la gráfica en `graph_points_count`, el minuto del partido en `inference_minute` y respaldando todo el árbol devuelto por la API en el campo BLOB/TEXT `raw_json`.
*   **Señal Neutra (`NO_BET`):** Requiere de una persistencia de validación de `NO_BET_CONFIRM_TICKS` confirmaciones consecutivas antes de registrarse formalmente en base de datos como no bettable para mitigar fluctuaciones momentáneas de la API.
*   **Señal Incierta o No Disponible (`UNAVAILABLE`):** Incrementa contadores de reintento (`q4_unavailable_ticks`) y reprograma una reevaluación inmediata bajo el intervalo de contingencia corto `POLL_NEAR_SECS`.

---

## 9. Calibración Dinámica de Ritmo y Jitter Adaptativo (`utils/helpers.py`)

### Filtro de Media Móvil Exponencial (EMA) para `secs_per_gmin`
Para contrarrestar las distorsiones de tiempo real provocadas por interrupciones, faltas y revisiones de video, el sistema no asume una constante fija de duración. Cada vez que el minutero de SofaScore registra un avance real medible, recalculación el ritmo de juego aplicando la siguiente ecuación matemática:

$$\text{rate} = \frac{\text{wall\_time\_elapsed}}{\text{minute\_delta}}$$

$$\text{secs\_per\_gmin} = (0.7 \times \text{secs\_per\_gmin}) + (0.3 \times \text{rate})$$

El resultado se restringe dentro de los límites operativos estrictos de un intervalo de fijación empleando un método de truncamiento:

$$\text{secs\_per\_gmin} = \max(60.0, \min(\text{secs\_per\_gmin}, 360.0))$$

### Algoritmo de Jitter Estocástico por Fase
Para evitar comportamientos cíclicos fácilmente detectables, la función `_jitter_sleep_secs` altera proporcionalmente los tiempos de suspensión calculados en función de la criticidad de la phase actual de la corrutina:

| Fase del Partido | Porcentaje de Jitter | Techo Máximo Absoluto | Suelo Mínimo Absoluto |
|---|---|---|---|
| Inicial / Reposo / Alejado (`q4_far`) | 20% | 45s | 20s |
| Ventana Crítica Activa (`q4_window`) | 12% | 18s | 12s |
| Reintento por Error de Red (`error_retry`) | 22% | 55s | 15s |

*Nota:* Si el parámetro de criticidad `urgent=True` se encuentra activo dentro de la ventana de evaluación, el rango de distribución de jitter se reduce de forma mandatoria a un máximo estricto de 5% o 6 segundos.

---

## 10. Rutina de Conciliación y Auditoría Asíncrona
El módulo `database/repository.py` debe implementar mecanismos cíclicos independientes para auditar y cerrar discrepancies en los registros de la base de datos `/match/matches.db`:

*   **`reconcile_pending_results`**: Proceso síncronizado ejecutado en el arranque que inspecciona las entradas de la tabla `bet_monitor_log_v2` con estatus `pending`. Cruza el identificador único contra los marcadores de `quarter_scores_v2` consolidados y actualiza el casillero final computando el dividendo a valores booleanos estandarizados (`win`, `loss`, `push`). Adicionalmente, mapea mediante introspección estructural (`PRAGMA table_info`) las columnas de la tabla matricial `eval_match_results_v2` para liquidar los outcomes rezagados de todos los tags de modelos en Q4.
*   **`_recheck_pending_finished_schedule_once`**: Tarea asíncrona ejecutada con una cadencia de 2 horas. Su objetivo es barrer aquellos registros en `bet_monitor_schedule_v2` que posean un estatus colgado en `pending` o `final_fetched=0` pero cuyo timestamp teórico de ejecución ya haya expirado con creces. Invoca de forma controlada a Playwright para forzar su captura de datos, evitando la acumulación de basura transaccional histórica.

---

## 11. Formateo de Interfaces de Logs y Alertas de Telegram

### Arquitectura Estricta de Logging Diario y Consola
Los mensajes de log se escriben en archivos rotativos dentro del directorio `/logs/monitor_YYYY-MM-DD.log`. La salida estándar de consola debe usar el formateador ANSI de acuerdo a los siguientes lineamientos tipográficos:

*   **Estructura Base Obligatoria:** `{Fecha Hora UTC-6 sin texto explícito de zona} [NIVEL] [COMPONENTE] Mensaje descriptivo`
*   **Códigos de Color por Nivel:** `DEBUG` (Gris), `INFO` (Azul), `WARNING` (Amarillo), `ERROR` (Rojo).
*   **Identificadores de Componente Permitidos:** `[PROGRAMACION]`, `[MONITOREO]`, `[DESCARGA]`, `[EVALUACION]`.
*   **Visualización Estándar de Matches:** `{horario_match} {match_id} {home} vs {away}` *(Donde `{horario_match}` se imprime en **Amarillo** y el `{match_id}` en **Azul**).*
*   **Señalizadores Visuales de Fase Final:** Siempre que un log registre procesos activos de lectura dentro del último cuarto, se antepondrá obligatoriamente el tag con emoji: `Q4 🟠`.
*   **Inyección de Excepciones Críticas:** Cualquier error de red derivado de denegaciones de acceso o bloqueos del servidor perimetral debe imprimir de forma explícitamente e inequívoca la cadena de texto: `HTTP 403/404` bajo el color **Rojo brillante**.

### Formato de Mensajería en el Canal de Telegram
Todos los despachos dirigidos al bot deben usar tipografías limpias y apegarse estrictamente a la matriz de emojis prefijados en función del dictamen predictivo y el estado del cierre, consumiendo obligatoriamente las variables de entorno `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`:

*   **Alertas de Apuesta Inicial (Fase Live Ventana Q4):**
    *   *Buena Apuesta (Bettable):* `🟢 APUESTA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   *No Operable (Not Bettable):* `🟡 APUESTA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   *Apuesta Fuera de Tiempo (Rango Minuto 34-36 y que cumpla el filtro de marcador):* `⚪️ APUESTA TARDIA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
*   **Alertas de Confirmación de Resultado Final (Fase Cierre FT):**
    Al consolidar la verificación final tras el término del partido, se recupera el mensaje original y se antepone de forma mandatoria el emoji de resolución conservando la estructura interna intacta:
    *   `✅🟢 APUESTA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   `❌🟢 APUESTA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   `✅🟡 APUESTA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   `❌🟡 APUESTA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   `✅⚪️ APUESTA TARDIA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`
    *   `❌⚪️ APUESTA TARDIA Q4 [{modelo}] → 🏠/✈️ {nombre_equipo_pick}`

---

## 12. Comentarios de Reglas de Formateo Interno (Cabecera Obligatoria)
*(Nota de desarrollo: Todos los archivos de código fuente creados bajo esta especificación técnica modular deben incluir textualmente el siguiente bloque en su sección superior de comentarios para garantizar la consistencia en futuras iteraciones realizadas por Inteligencia Artificial).*

```python
# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
#    Cualquier aproximación monolítica de archivo único viola esta especificación.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    /match/matches.db y todas las tablas sin excepción finalizan con el sufijo '_v2'.
# 3. TABLA DE LOGS: 'bet_monitor_log_v2' se particiona por modelo y contiene 
#    obligatoriamente los campos 'raw_json' (TEXT), 'inference_minute' (INT), 
#    y 'graph_points_count' (INT) junto con marcadores reales del juego.
# 4. CONFIGURACIÓN DE LIGAS: Prohibido hardcodear filtros o patrones de texto 
#    en las consultas SQL o lógica directa. Debe consumirse declarativamente 
#    desde config/leagues.yaml o cargarse dinámicamente desde la BD SQLite.
# 5. Formato Log: {Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]
#    - Colores ANSI: INFO=Azul, WARNING=Amarillo, ERROR=Rojo.
# 6. Formato Matches en Log: {horario_match} {match_id} {home} vs {away}
#    - horario_match en Amarillo, match_id en Azul (sin texto UTC-6).
# 7. Errores de red críticos: Imprimir explícitamente "HTTP 403/404" en ROJO.
# 8. Monitoreo avanzado en progreso de cuarto final: Usar obligatoriamente "Q4 🟠".
# 9. Telegram Prefijos de Apuestas: 🟢 (Bettable), 🟡 (No Bettable), ⚪ (Tardía).
# 10. Telegram Resultados FT: Prefijar con ✅ (Ganada) o ❌ (Perdida) manteniendo emoji base.
# 11. Conversión de tiempos siempre legibles en formato humano (ej. 1 dia 2h 15min / 45s).
# =====================================================================
```
