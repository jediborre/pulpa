# Arquitectura del Monitor de Apuestas — `bet_monitor.py`

> Documento técnico para rediseño. Describe el comportamiento actual completo del daemon de monitoreo.

---

## 1. Propósito General

`bet_monitor.py` es un daemon asyncio que:

1. **Descarga el itinerario diario** de partidos de baloncesto desde SofaScore.
2. **Lanza una corutina `_watch_match`** por cada partido pendiente.
3. Cada watcher **monitorea el partido en tiempo real**, esperando las ventanas de cuartos Q3 y Q4.
4. En la ventana correcta, **ejecuta inferencia ML** (modelos v12–v17) y envía **alertas por Telegram** si el modelo dice BET.
5. Al terminar el partido, **descarga y persiste** el resultado final en la base de datos.

Se ejecuta como parte de la API Flask (`api.py`) o directamente, y se detiene con un `asyncio.Event`.

---

## 2. Constantes Clave

| Constante | Valor | Significado |
|---|---|---|
| `SECS_PER_GAME_MIN` | 170 | Segundos reales por minuto de juego (inicial) |
| `MAX_CONCURRENT_FETCHES` | 2 | Semáforo: máximo de fetches simultáneos |
| `SKIP_Q3` | `True` | Q3 deshabilitado; solo monitorea Q4 |
| `Q4_ONLY_EARLY_WAKE_MINUTE` | 27 | Minuto de juego al que despierta el watcher |
| `Q4_ONLY_WAKE_LEAD_MINUTES` | 2 | Margen antes del wake minute |
| `PRESTART_PROBE_MIN_SECS` | 60 | Sonda prestart: intervalo mínimo |
| `PRESTART_PROBE_MAX_SECS` | 300 | Sonda prestart: intervalo máximo |
| `PRESTART_PROBE_BACKOFF` | 1.25 | Factor de crecimiento del intervalo de sonda |
| `POLL_INTERVAL_FF` | 150 | Intervalo base del loop final_fetch (antes: 90s) |
| `POLL_JITTER_FF` | 45 | Jitter adicional en final_fetch: `random(0, 45s)` |
| `FINAL_FETCH_EXTRA_SECS` | 300 | Segundos extra tras tiempo estimado de fin |
| `FINAL_FETCH_MIN_GP` | ~20 | Mínimo de graph_points para guardar resultado |
| `FT_SCRAPE_SLOT_SPACING` | 35.0 | Separación mínima entre descargas FT concurrentes |
| `POLL_NEAR_SECS` | ~45 | Intervalo en ventana Q3/Q4 activa |
| `IDLE_POLL_SECS` | ~180 | Intervalo máximo en reposo |
| `NO_BET_CONFIRM_TICKS` | ~3 | Ticks de confirmación antes de aceptar NO BET |
| `Q4_TOO_LATE_BET_MINUTE` | 33 | A partir de aquí solo alerta tardía |
| `Q4_TOO_LATE_HARD_MINUTE` | 36 | A partir de aquí no apuesta en ningún caso |
| `Q4_WAITING_MAX_TICKS` | ~8 | Ticks máx esperando score Q3 antes de abortar |
| `Q4_STALE_MAX_TICKS` | ~5 | Ticks con graph_points sin crecer → abortar Q4 |
| `NO_GRAPH_REAL_SECS` | ~3300 (55 min) | Si no hay gráfica tras este tiempo → descartar |
| `MAX_FETCH_ERRORS` | ~8 | Errores consecutivos antes de descartar partido |
| `SCHEDULE_REFRESH_HOURS` | 8 | Frecuencia de refresco del itinerario |
| `PENDING_RECHECK_SECS` | ~3600 (1h) | Frecuencia del recheck de resultados pendientes |
| `UTC_OFFSET_HOURS` | -6 | Zona horaria local para fechas y logs |

---

## 2b. Qué Usa el Monitor del Scraper

El monitor llama a **dos funciones** de `scraper.py`:

### `fetch_event_snapshot(match_id)` — sonda ligera

Abre UN contexto de browser → hace **1 request** a `/api/v1/event/{id}`.

Devuelve:
```python
{
  "status_type": "notstarted" | "inprogress" | "finished" | ...,
  "status_description": "1st Quarter" | ...,
  "status_code": int,
  "home_score": int,
  "away_score": int,
}
```

**Se usa en:** probe mode (mientras el partido no ha empezado). Es barato — 1 request por tick, sin datos pesados.

**Problema conocido:** algunas ligas devuelven `status_type=""` permanentemente.

---

### `fetch_match_by_id(match_id)` — scrape completo

Abre UN contexto de browser → navega a la página del partido para establecer sesión → hace **hasta 7 requests** dentro de ese contexto:

| # | Endpoint | Datos | Requerido |
|---|---|---|---|
| 1 | `/event/{id}` | Metadata, scores por cuarto, equipos, liga | ✅ siempre |
| 2 | `/event/{id}/incidents` | Play-by-play: canastas, faltas, timeouts, sustituciones | ✅ siempre |
| 3 | `/event/{id}/graph` | Curva de presión/momento (graph_points) | ✅ siempre |
| 4 | `/event/{id}/h2h` | Historial H2H (agregado) | opcional |
| 5 | `/event/{id}/statistics` | Stats globales por equipo (FG%, rebotes, etc.) | opcional |
| 6 | `/event/{id}/lineups` | Alineaciones + rating SofaScore por jugador | opcional |
| 7 | `/event/{id}/odds/1/all` | Cuotas pre-partido (1x2, spread, over/under) | opcional |

Adicionalmente, si se resuelve el H2H via teams (preferido sobre el endpoint /h2h):
- `/team/{home_id}/events/last/N` → partidos recientes con scores por cuarto
- `/team/{away_id}/events/last/N` → idem

Y para team strength:
- Endpoints de `pregameForm` y performance por equipo

**Costo total real:** 1 browser open + ~7–10 HTTP requests dentro del mismo contexto de sesión. El browser se abre y cierra por cada llamada a `fetch_match_by_id`.

---

### Cómo Funciona el Backend Obscura

```
_browser_context(warmup_url, backend="obscura")
    │
    ├── Conecta al Chrome ya corriendo via CDP en localhost:9222
    │   (NO lanza browser nuevo — usa el proceso Obscura existente)
    │
    ├── new_context() → nuevo perfil de cookies/localStorage
    │
    ├── page.goto(warmup_url) → navega al partido para establecer cookies SofaScore
    │
    ├── ctx.request.get(endpoints...) → requests con cookies del browser real
    │
    └── ctx.close() → descarta el contexto (no cierra el browser Obscura)
```

El truco anti-ban: SofaScore bloquea requests HTTP directas (403) pero permite las que vienen desde un browser real con cookies válidas. Obscura mantiene un Chrome persistente que parece un usuario real.

---

### Impacto en Requests por Partido

Cada ejecución del loop principal del watcher hace:
- **1 `fetch_match_by_id`** → ~7 requests + 1 browser context open/close
- Durante probe mode: **1 `fetch_event_snapshot`** por tick → 1 request + 1 browser open/close

Con 20 partidos activos y semáforo de 2:
- Máximo 2 `fetch_match_by_id` simultáneos = máximo ~14 requests en vuelo
- El resto de watchers espera en la cola del semáforo

---

## 3. Módulos y Dependencias

```
bet_monitor.py
├── scraper.py          → fetch_match_by_id(), fetch_event_snapshot(), _browser_context()
├── anti_block.py       → espaciado global de requests, cooldown 403, rotación de sesión
├── db.py               → save_match(), get_match(), get_conn(), init_db()
├── training/
│   ├── v12/infer_match_v12.py  → run_inference(match_id, target)
│   ├── v13/infer_match_v13.py  → run_inference(match_id, target)
│   ├── v15/inference.py        → V15Engine.load() → engine.predict(...)
│   ├── v16/inference.py        → V15Engine.load() (mismo nombre de clase)
│   ├── v17/inference.py        → V15Engine.load()
│   └── infer_match.py          → run_inference(...) para versiones legacy
└── telegram_bot.py     → consume status_text(), schedule_text()
```

**Variables de entorno relevantes:**
- `SOFASCORE_SCRAPER_BACKEND=obscura` — backend Playwright anti-detección
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

---

## 4. Base de Datos SQLite (WAL)

Tablas principales:

| Tabla | Propósito |
|---|---|
| `bet_monitor_schedule` | Un row por partido: estado, señales Q3/Q4, timestamps |
| `bet_monitor_log` | Registro de cada apuesta enviada o simulada |
| `eval_match_results` | Resultados por modelo/tag para análisis de rendimiento |
| `quarter_scores` | Marcadores reales por cuarto (Q1–Q4) de cada partido |
| `settings` | Configuración dinámica (no ampliamente usada) |

**Configuración:** `PRAGMA journal_mode=WAL`, `busy_timeout=30000ms`

### Schema resumido de `bet_monitor_schedule`

```
match_id, home_team, away_team, league, event_date
scheduled_utc_ts, scheduled_utc, status_type
status: pending → done / discarded
q3_checked, q3_signal, q3_notified, q3_model
q4_checked, q4_signal, q4_notified, q4_model
final_fetched, final_fetch_at
skip_reason, updated_at
```

---

## 5. Arquitectura Anti-Ban

El sistema tiene múltiples capas para evitar bloqueos HTTP 403 de SofaScore:

### 5.1 Módulo `anti_block.py` (global)

| Mecanismo | Parámetro | Valor |
|---|---|---|
| Espaciado mínimo entre requests | `GLOBAL_FETCH_MIN_SPACING_SECS` | 1.5s |
| Jitter adicional | | 0–0.5s aleatorio |
| Streak 403 trigger | `GLOBAL_403_STREAK_TRIGGER` | 6 errores consecutivos |
| Cooldown tras streak | `GLOBAL_403_COOLDOWN_SECS` | 900s (15 min) |
| Rotación de sesión cada N fetches | `SESSION_ROTATE_EVERY` | 40 |
| Pausa por rotación | `SESSION_ROTATE_PAUSE_SECS` | 45s |

### 5.2 Semáforo concurrencia (`_fetch_sem`)

`asyncio.Semaphore(MAX_CONCURRENT_FETCHES=2)` — máximo 2 fetches en paralelo en todo el proceso.

### 5.3 Monitoring Priority Lock (`_monitoring_lock_until`)

Cuando un partido entra en estado live, se activa un lock de 5 min (`_MONITORING_LOCK_SECS`). Durante ese tiempo, los rechecks periódicos (`recheck_pending`, `recheck_schedule_ft`) son omitidos para no interferir con fetches críticos de Q4.

### 5.4 FT Scrape Slot Queue (`_claim_ft_scrape_slot`)

Serializa las descargas de resultado final. Si 20 partidos terminan simultáneamente, se les asigna slots en cola:
- Partido 1: slot = ahora
- Partido 2: slot = ahora + 35s
- Partido 3: slot = ahora + 70s
- ...

Implementado con `threading.Lock` sobre `_ft_scrape_next_slot`.

### 5.5 Jitter en `_final_fetch_and_save`

El loop de polling del resultado usa: `POLL_INTERVAL_FF + random.uniform(0, POLL_JITTER_FF)` = **150s + random(0–45s)** para desincronizar múltiples loops concurrentes.

---

## 6. Flujo de Itinerario

### 6.1 Fetch del itinerario (`_fetch_all_events_for_date_sync`)

1. Abre un contexto de browser Playwright en `sofascore.com/basketball`
2. Consulta `api.sofascore.com/api/v1/sport/basketball/scheduled-events/{fecha}` para la fecha local Y la fecha UTC+1 (overnight)
3. Filtra eventos cuya `startTimestamp` convertida a UTC-6 coincide con `local_date`
4. Devuelve lista ordenada por `scheduled_utc_ts`

### 6.2 Inserción en DB (`_upsert_schedule_row`)

- Si el `match_id` ya existe → actualiza solo campos no nulos
- Si es nuevo → inserta con `status=pending`

### 6.3 Refresco periódico

`run_monitor` refresca el itinerario cada `SCHEDULE_REFRESH_HOURS=8h` o cuando cambia la fecha local.

---

## 7. Filtrado de Ligas

Hay **dos niveles** de pendientes:

### `_get_pending_rows` (inferencia completa)

Excluye ~50 ligas mediante `NOT league LIKE '%...%'`. Ejemplos de excluidas:
- WNBA, Women, Feminina, Femenina
- NBA, Lega A, Turkish Basketball Super League
- Playoffs, U21, ABA Liga, Stoiximan GBL
- Ligas de desarrollo y copas regionales

### `_get_all_pending_rows` (solo descarga FT)

Sin ningún filtro de liga. Devuelve todos los pendientes. Usado para lanzar watchers en modo `ft_only=True`.

En `run_monitor`, el conjunto diferencia `all_pending - pending` determina los partidos que van en modo FT-only.

---

## 8. Ciclo Principal `run_monitor`

```
run_monitor()
│
├── init_tables() + reconcile_pending_results()
├── _refresh_schedule_dates(hoy, mañana)
├── _log_daily_summary()
├── Resume: lanzar watchers para pendientes ya en DB
│
└── loop cada 60s:
    ├── Refresco itinerario (si >8h o nuevo día)
    ├── pending_today + pending_tomorrow → lanzar _watch_match() si no activo
    ├── all_pending − pending → lanzar _watch_match(ft_only=True)
    ├── Limpiar tareas terminadas
    ├── Cada ~1h: _recheck_pending_outcomes_once() (si no hay cooldown ni carga)
    └── Cada ~2h: _recheck_pending_finished_schedule_once()
```

**Condiciones de skip del recheck:**
- Cooldown 403 activo
- Más de `PENDING_RECHECK_MAX_FETCHES_IN_FLIGHT` fetches en vuelo (backpressure)

---

## 9. Máquina de Estados del Watcher (`_watch_match`)

Cada partido tiene su propia corutina. El estado se mantiene en variables locales.

```
[INICIO]
    │
    ├── ft_only=True → ir a MODO FT-ONLY (sección 9.1)
    │
    ▼
[ESPERAR INICIO] (scheduled_ts - 120s)
    │
    ▼
[LOOP PRINCIPAL]
    │
    ├── q4_prefetch_armed? → dormir hasta minuto (Q4_ONLY_EARLY_WAKE_MINUTE - lead)
    │                         ↓ agotado → q4_prefetch_armed=False
    │
    ├── q4_prestart_probe_mode? → MODO SONDA (sección 9.2)
    │
    ├── q3_done AND q4_done → FINAL FETCH (sección 9.5) → break
    │
    ├── fetch_match_by_id() [semáforo, anti-ban]
    │   ├── error N veces → discard y break
    │   └── datos vacíos → retry
    │
    ├── _is_two_half()? → discard
    │
    ├── calibrar secs_per_gmin (EMA 0.7/0.3)
    │
    ├── status_type == "finished" → cerrar Q3/Q4 + guardar + break
    │
    ├── Q3 logic (si !q3_done && !SKIP_Q3) → sección 9.3
    │
    └── Q4 logic (si !q4_done) → sección 9.4
```

### 9.1 Modo FT-Only

```
[FT-ONLY]
├── Esperar scheduled_ts si aún no ha llegado
├── _claim_ft_scrape_slot() → esperar slot
├── _final_fetch_and_save()
└── _update_row(status=done)
```

### 9.2 Modo Sonda Prestart (`q4_prestart_probe_mode`)

Se activa con `SKIP_Q3=True` para evitar fetches completos mientras el partido no ha comenzado.

Usa `fetch_event_snapshot(match_id)` → respuesta ligera `{status_type, status_description, home_score, away_score}`.

```
snap_status?
├── "finished"    → salir probe mode → _claim_ft_scrape_slot() → stagger → loop normal
├── "inprogress"  → salir probe mode → _set_monitoring_lock() → stagger → loop normal
│   (o cualquier status live)
├── "notstarted"  → incrementar probe_ticks, aumentar delay con backoff
│                   - drift < 20min: ×1.25
│                   - drift 20–45min: ×1.22
│                   - drift > 45min: ×1.30
│                   max=300s
├── ""  (vacío)   → si drift >= secs_per_gmin * Q4_ONLY_EARLY_WAKE_MINUTE
│                   → salir probe mode (fetch completo resolverá el estado)
└── otro          → loggear como status inesperado, continuar con backoff
```

**Bug conocido resuelto:** algunas ligas (U17 Brazil, etc.) devuelven `status_type=""` permanentemente → el watcher quedaba atrapado en probe mode. Fix: condición de salida por drift.

### 9.3 Lógica Q3 (deshabilitada con `SKIP_Q3=True`)

```
minute > q3_cut + 6       → window_missed
minute >= q3_cut - wake   → ventana activa:
    ├── datos insuficientes → retry POLL_NEAR_SECS
    ├── BET/NO_BET con confirmación NO_BET_CONFIRM_TICKS
    ├── UNAVAILABLE → q3_unavailable_ticks++, retry
    └── notified/last_tick/ERROR → q3_done=True
minute < q3_cut - wake    → far sleep (proporcional a minutos restantes)
```

### 9.4 Lógica Q4

Ventana activa: `minute >= Q4_EARLIEST_MINUTE (27)` hasta `Q4_LAST_MINUTE (38)`.

```
minute > q4_cut+6 OR >= 38  → window_missed
minute >= 27:
    ├── score_ok AND gp4 >= q4_mgp:
    │   ├── BET → notificar + q4_done=True
    │   ├── NO_BET confirmado (ticks >= NO_BET_CONFIRM_TICKS) → registrar + q4_done=True
    │   ├── NO_BET incierto → q4_no_bet_ticks++, retry
    │   ├── UNAVAILABLE → q4_unavailable_ticks++, retry
    │   └── ERROR → q4_done=True
    └── score no disponible:
        ├── q4_waiting_score_ticks++
        ├── si stale_ticks >= Q4_STALE_MAX_TICKS → abortar "graph_stale_timeout"
        └── si waiting_ticks >= Q4_WAITING_MAX_TICKS → abortar "q3_placeholder_timeout"

minute < 27:
    └── tiered sleep proporcional a game-minutes restantes:
        ≤1 min: sleep = min(45, pace*0.35)
        ≤2 min: sleep = min(60, pace*0.45)
        ≤5 min: sleep = min(90, pace*0.55)
        >10 min: sleep = min(180, pace*0.5)   ← cap 3 min para recalibrar ritmo
        else:   sleep = min(120, pace*0.6)
```

**Poll adaptativo por urgencia:**
- `mins_left <= 2`: poll = 20s
- `mins_left <= 4`: poll = 30s
- else: `min(POLL_NEAR_SECS, Q4_REEVAL_FAST_SECS)`

### 9.5 Final Fetch (`_final_fetch_and_save`)

```
1. Estimar minutos restantes hasta minuto 48
2. Esperar: mins_remaining * secs_per_gmin + FINAL_FETCH_EXTRA_SECS (300s)
3. Loop hasta status=finished (max 40 min):
   ├── anti-ban: cooldown check + spacing delay
   ├── fetch_match_by_id()
   │   ├── error → log + poll 150s+jitter + continuar
   │   └── ok:
   │       ├── status=finished → BREAK
   │       └── status!=finished → log + poll 150s+jitter
   └── poll_sleep = 150 + random(0, 45)  ← desincroniza múltiples loops
4. Guardar con db.save_match() + marcar final_fetched=1
```

---

## 10. Pipeline de Inferencia

### 10.1 Configuración de Modelos (`_model_config`)

Diccionario `{target: version}` configurable. Ejemplo:
```python
{"q3": "v16", "q4": "v17"}
```

### 10.2 Flujo de inferencia (`_run_inference_sync`)

Llamado vía `asyncio.to_thread`.

| Versión | Módulo | Input |
|---|---|---|
| v12 | `training.v12.infer_match_v12` | match_id, target |
| v13 | `training.v13.infer_match_v13` | match_id, target |
| v15 | `training.v15.inference.V15Engine` | league, quarter_scores, graph_points, pbp_events |
| v16 | `training.v16.inference.V15Engine` | idem |
| v17 | `training.v17.inference.V15Engine` | idem |
| legacy | `training.infer_match` | match_id, version, target |

Los engines v15/v16/v17 se cachean en `_ENGINE_CACHE` (singleton por versión).

### 10.3 Señales de salida

| Señal | Acción |
|---|---|
| `BET_HOME` / `BET_AWAY` | Notificación Telegram + registro en bet_monitor_log |
| `NO_BET` | Silencioso (o notificación si es el último tick de confirmación) |
| `UNAVAILABLE` | Retry con backoff |
| `ERROR` | Registrar y marcar q3/q4_done |

### 10.4 Guardia de apuesta tardía

- `Q4_TOO_LATE_BET_MINUTE = 33`: aviso "apuesta tardía" pero aun notifica
- `Q4_TOO_LATE_HARD_MINUTE = 36`: no apuesta en ningún caso

### 10.5 Registro en `eval_match_results`

Cada ejecución de inferencia guarda en `eval_match_results` columnas dinámicas por tag de modelo:
- `q4_signal__TAG`, `q4_pick__TAG`, `q4_confidence__TAG`, `q4_outcome__TAG`

`outcome` empieza en `"pending"` y se reconcilia después con `quarter_scores`.

---

## 11. Reconciliación de Resultados

### `reconcile_pending_results(db_path)`

Corre síncrona al inicio y periódicamente.

**Fase 1 — `bet_monitor_log`:**
- Busca rows con `result='pending'` y señal BET
- Cruza con `quarter_scores` (Q3/Q4)
- Asigna `win`, `loss`, o `push`

**Fase 2 — `eval_match_results`:**
- Descubre tags de modelos con `PRAGMA table_info`
- Para cada tag y cuarto, si `available=1` y `outcome='pending'`:
  - Cruza con `quarter_scores`
  - Asigna `hit`, `miss`, o `push`

### `_recheck_pending_outcomes_once`

Versión async que además vuelve a scrape los partidos que todavía no tienen `quarter_scores`.

### `_recheck_pending_finished_schedule_once`

Busca rows de `bet_monitor_schedule` con `status=pending` y `final_fetched=0` cuyo `scheduled_utc_ts` haya pasado hace más de `PENDING_SCHEDULE_MIN_AGE_SECS`. Los scrape y, si están finished, guarda el resultado.

---

## 12. Notificaciones Telegram

Enviadas por `_notify(msg, reply_markup, notify_type, quarter)`.

### Tipos de notificación:
- **BET**: Apuesta encontrada. Incluye equipo pick, confianza, modelo, liga.
- **result**: Resultado de apuesta (win/loss/push). Incluye marcador del cuarto.
- **alert**: Avisos del sistema (errores graves, cooldowns largos).

Los mensajes de resultado incluyen botones inline:
- `Ver Match` → callback interno
- `Sofascore` → link directo al partido

---

## 13. Helpers de Calibración de Ritmo

### `secs_per_gmin` (EMA adaptativa)

Cada vez que `minute` avanza:
```python
rate = (wall_time_elapsed) / (minute_delta)
secs_per_gmin = 0.7 * secs_per_gmin + 0.3 * rate
secs_per_gmin = clamp(60.0, 360.0)
```

Inicializa en `SECS_PER_GAME_MIN=170`. Se actualiza continuamente durante el partido para ajustar tiempos de sleep a la velocidad real del juego.

### `_jitter_sleep_secs(base, phase, urgent)`

Agrega jitter proporcional al base sleep para evitar patrones fijos de requests:

| Phase | Jitter % | Max abs | Floor |
|---|---|---|---|
| startup / q3_far / q4_far | 20% | 45s | 20s |
| q3_window / q4_window / no_graph | 12% | 18s | 12s |
| error_retry | 22% | 55s | 15s |

Con `urgent=True`: jitter reducido a 5%/6s.

---

## 14. Estado Global (`MONITOR_STATUS`)

Diccionario mutable consultado por `status_text()` (Telegram bot):

```python
{
  "running": bool,
  "started_at": str (ISO UTC),
  "stop_requested": bool,
  "checked_q3": int,
  "checked_q4": int,
  "bets_sent": int,
  "no_bet": int,
  "discarded": int,
  "active_matches": [match_id, ...],
  "today_total": int,
  "tomorrow_total": int,
  "last_event": str,
}
```

---

## 15. Logging

### Consola (`_ensure_monitor_console_logging`)

Handler StreamHandler con formatter `MonitorConsoleFormatter` que añade colores ANSI por nivel:
- DEBUG: gris
- INFO: blanco
- WARNING: amarillo
- ERROR: rojo

### Archivo diario (`_ensure_daily_file_logging`)

Logs rotados en `MONITOR_LOG_DIR` (por defecto `match/logs/`), formato: `monitor_YYYY-MM-DD.log`.

### `_log(msg)` vs `logger.warning()`

- `_log`: nivel INFO, prefijo `[MONITOR]`, va a consola + archivo diario
- `logger.warning/error`: nivel WARNING/ERROR, va al logger estándar

### Prefijos notables en logs

| Prefijo | Significado |
|---|---|
| `[DESCARGA]` | Operaciones de descarga de resultado final |
| `[FT]` | Watcher en modo ft_only (liga filtrada) |
| `[FINISHED]` | Partido detectado como terminado |
| `[LOCK MON 5 MIN]` | Partido live → lock anti-interferencia activo |

---

## 16. Limitaciones Conocidas / Áreas de Mejora

### Problemas de diseño actuales

1. **`fetch_event_snapshot` devuelve status vacío** para ciertas ligas (e.g. U17 Brazil). El watcher ahora sale del probe mode por drift, pero no sabe el estado real. Un endpoint más confiable mejoraría esto.

2. **`FT_SCRAPE_SLOT_SPACING` fijo** (35s). No se adapta a la cantidad de partidos simultáneos. Con 50 partidos la cola tarda ~30 min en vaciarse.

3. **Filtro de ligas hardcodeado** como strings SQL LIKE. Difícil de mantener. Candidato a tabla de configuración en DB o YAML externo.

4. **`SECS_PER_GAME_MIN` compartida entre partidos** en el mismo watcher pero no entre watchers. Cada watcher calibra su propio ritmo, lo cual es correcto, pero la inicialización de 170s puede ser muy distinta de la realidad para ligas de 10 min/cuarto (NBA-style).

5. **Modelos cargados con import dinámico** (`importlib.import_module`). Los engines v15/v16/v17 se cachean en `_ENGINE_CACHE` pero si hay error de carga, el primer partido lo detecta y los siguientes reciben error inmediato hasta que el proceso se reinicia.

6. **Sin backpressure dinámico en watchers activos**. Si hay 40 partidos simultáneos, 40 corutinas compiten por el semáforo de 2 slots. El tiempo de espera por partido crece linealmente con la carga pero no hay un mecanismo de throttle de alto nivel.

7. **`bet_monitor_schedule` nunca limpia partidos viejos**. Con el tiempo acumula meses de datos. Sin índice por `event_date` los queries se vuelven lentos.

8. **`reconcile_pending_results` no tiene índice óptimo**. Recorre toda la tabla `bet_monitor_log` con `result='pending'` sin índice compuesto.

9. **Probe mode no tiene timeout global**. Un partido que siempre devuelve `notstarted` durante 6+ horas sigue en probe mode (aunque el backoff llega a 300s, sigue en el event loop para siempre).

10. **El itinerario de mañana se carga igual que hoy**. Si se cancela un partido de mañana, no se detecta hasta el siguiente refresco de 8h.

### Ideas para la próxima versión

- Separar itinerario, watchers y inferencia en procesos/workers independientes con cola de mensajes.
- Persistir el estado del watcher en DB para sobrevivir reinicios sin re-detectar inicio de partido.
- Configuración de ligas en tabla `settings` con un script de gestión.
- Rate limiter global con token bucket en lugar del semáforo simple.
- Endpoint livescore websocket si SofaScore lo expone (evitaría polling completamente).
- Circuit breaker por liga: si una liga falla consistentemente con 403, desactivarla temporalmente.
- Métricas de latencia de inferencia y tasa de BET por modelo en tiempo real.
