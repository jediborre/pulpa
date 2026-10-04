# 🚀 Plan Maestro: Arquitectura y Hoja de Ruta de Monitor V3

> **Estado:** 🟡 EN DEFINICIÓN Y POC INICIAL  
> **Objetivo:** Diseñar e implementar `monitor_v3`, un sistema de monitoreo e inferencia en vivo de básquetbol de alta velocidad, ultra bajo consumo de memoria y resistente a bloqueos anti-bot (Cloudflare Turnstile) mediante una arquitectura desacoplada sin costo de proxies.

---

## 1. Auditoría: ¿Qué recaba actualmente `monitor_v2`?

Para que `monitor_v3` reemplace o supere a `monitor_v2`, debe recopilar con total fidelidad el 100% de la información requerida por los modelos de ML (`v6_2`, `m27_v3`) y la auditoría operativa.

### A. Metadatos y Calendario del Día
* **Endpoint origen:** `/api/v1/sport/basketball/scheduled-events/{YYYY-MM-DD}`
* **Campos clave:**
  * `match_id` (Identificador único).
  * `home_team`, `away_team` (Nombres oficiales de los equipos).
  * `league` (Torneo / Liga, e.g., NBA, EuroLeague, ACB).
  * `scheduled_utc_ts` y `scheduled_utc` (Horario de inicio programado).
  * Estado inicial (`pending`).

### B. Sonda Pre-Partido y Seguimiento en Vivo (Live Watcher)
* **Endpoint origen:** `/api/v1/event/{match_id}` (Snapshot ligero)
* **Frecuencia:**
  * Pre-partido: Cada 1 a 3 minutos cuando el partido está próximo a iniciar.
  * Q1 a Q3: Sondas periódicas con cálculo de tiempo estimado para Q4.
  * Entrada a Q4 (minuto 27 o inicio de cuarto): Sondeo rápido de alta frecuencia con jitter aleatorio (ej. 2s a 5s).
* **Campos extraídos en tiempo real:**
  * Estado del partido (`status_description`, e.g., '1st quarter', '3rd quarter', '4th quarter', 'ended').
  * Marcador acumulado global (`home_score`, `away_score`).
  * Marcador por cuartos (`quarter_scores` en vivo: Q1, Q2, Q3, Q4).
  * Minuto inferido de juego (a través de eventos PBP / tiempo de reloj).
  * `graph_points`: Puntos de la curva de presión y momentum histórico del partido (utilizados como feature vital por los modelos).

### C. Evaluación de Modelos ML e Inferencia de Apuestas
* **Modelos activos:** `v6_2` (Q4 general) y `m27_v3` (minuto 27 específico).
* **Campos registrados en logs (`bet_monitor_log_v2` $\rightarrow$ `_v3`):**
  * `match_id`, `model_version`, `target_quarter`.
  * `inference_minute` (Minuto exacto en que se corrió la inferencia).
  * `graph_points_count` (Cantidad de puntos de la gráfica al momento de inferir).
  * `raw_json` (Instantánea completa del JSON analizado para auditoría).
  * `signal_type` (`BETTABLE`, `NO_BET`, `TOO_LATE`).
  * `picked_side` (`HOME` o `AWAY`).
  * `confidence` (Probabilidad calibrada del modelo).
  * `actual_home_score`, `actual_away_score` (Marcador en el momento de la señal).
  * `inference_json` (Detalle de probabilidades, umbrales y features extraídas).

### D. Cierre Final de Partido (Full Time - FT)
* **Endpoint origen:** Ráfaga completa de endpoints vía `fetch_match_by_id`:
  * `/api/v1/event/{id}`
  * `/api/v1/event/{id}/incidents` (Play-by-play completo)
  * `/api/v1/event/{id}/graph` (Curva final de momentum)
  * `/api/v1/event/{id}/h2h/events` (Historial cara a cara)
  * `/api/v1/event/{id}/statistics` (Tiros de 2, triples, rebotes, faltas)
  * `/api/v1/event/{id}/lineups` (Jugadores y quintetos)
* **Persistencia FT:**
  * Marcadores por cuartos en `quarter_scores_v3` (Q1, Q2, Q3, Q4, OT).
  * Almacenamiento relacional histórico en `matches`, `play_by_play`, `graph_points`, `match_h2h`.
  * Liquidación de apuestas (`reconcile_pending_results()`): marca `result` como `WIN` o `LOSS`.
  * Despacho de confirmación final por Telegram con emoji ✅ o ❌.

---

## 2. Nueva Arquitectura Propuesta para Monitor V3

### El Cambio de Paradigma: Harvester + Fast HTTP Worker
En V2, cada petición o sondeo ejecuta Playwright / Chrome Headless, consumiendo entre 200 y 400 MB de RAM por instancia y generando fugas de CDP detectables por Cloudflare.

En V3 separamos radicalmente la **cosecha de credenciales** de la **extracción de datos**:

```
┌────────────────────────────────────────────────────────┐
│ Capa 1: Harvester de Sesión (Bajo Demanda / Cada 3h)   │
│ - Camoufox o Chrome con perfil persistente             │
│ - Resuelve reto de Cloudflare 1 sola vez               │
│ - Exporta CookieJar en memoria y CIERRA el navegador   │
└──────────────────────────┬─────────────────────────────┘
                           │ (Inyecta cookies en caliente)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Capa 2: Fast HTTP Engine (Asíncrono Permanente)        │
│ - curl_cffi / httpx con TLS JA4 spoofing               │
│ - Peticiones HTTP/2 directas en 40-80 ms               │
│ - Menos de 30 MB de RAM en ejecución continua          │
└──────────────────────────┬─────────────────────────────┘
                           │ (JSONs limpios)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Capa 3: Orquestador y ML (Asyncio Event Loop)          │
│ - bet_monitor_schedule_v3                              │
│ - Inferencia desacoplada (models/registry.py)          │
│ - bet_monitor_log_v3 + Alertas Telegram                │
│ - Inserción atómica en base de datos matches.db        │
└────────────────────────────────────────────────────────┘
```

### Ventajas Técnicas de V3:
1. **Consumo de Memoria:** Pasa de ~350 MB continuos a **< 35 MB de RAM**.
2. **Latencia por Petición:** Pasa de 2.5s - 5.0s (abrir página en Chrome) a **40 - 90 ms** (HTTP/2 directo).
3. **Cero Costo en Proxies:** Utiliza tu propia IP residencial con perfil persistente o cabeceras móviles.
4. **Resistencia a Cloudflare:** Suplanta handshakes TLS reales (JA3/JA4) evitando la detección de bots en capa de red y eliminando la fuga CDP `Runtime.enable`.

---

## 3. Estructura de Módulos Propuesta (`monitor_v3/`)

```
monitor_v3/
├── config/
│   ├── __init__.py
│   ├── constants.py            # Constantes de tiempo, timeouts, modelos activos
│   └── leagues.yaml            # Configuración declarativa de ligas (excluidas, ft_only)
├── core/
│   ├── __init__.py
│   ├── session_manager.py      # Gestor de cookies, rotación y refresco
│   ├── harvester.py            # Navegador ligero (Camoufox/Perfil) para cookies
│   └── http_client.py          # Cliente asíncrono ultra-rápido (curl_cffi / httpx)
├── database/
│   ├── __init__.py
│   ├── connection.py           # Conexión canónica a matches.db en raíz
│   ├── repository.py           # Operaciones CRUD para tablas _v3
│   └── schemas.py              # Definición de DDL y migraciones
├── models/
│   ├── __init__.py
│   └── evaluator.py            # Integración con models/registry.py (v6_2, m27_v3)
├── scrapers/
│   ├── __init__.py
│   ├── schedule_scraper.py     # Descarga de partidos del día
│   ├── live_scraper.py         # Sondeo rápido de Q4 / PBP
│   └── match_detail_scraper.py # Descarga atómica de ráfaga FT
├── notifications/
│   ├── __init__.py
│   └── telegram_bot.py         # Formateo de apuestas 🟢🟡⚪ y resultados ✅❌
├── utils/
│   ├── __init__.py
│   ├── logger.py               # Logger ANSI coloreado reglamentario
│   └── helpers.py              # Jitter gaussiano, formateo de tiempos
└── main.py                     # Bucle principal de eventos asyncio
```

---

## 4. Matriz de Seguimiento del Estado del Proyecto (Status Tracker)

> Esta tabla permite rastrear el progreso tarea por tarea y retomar el trabajo en sesiones posteriores sin perder el contexto.

| ID | Fase / Tarea | Estado | Prioridad | Entregable / Archivo | Notas |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **F0.1** | Auditoría de requerimientos de datos de V2 | 🟢 Completada | Alta | `docs/PLAN_MONITOR_V3.md` | Lista completa de endpoints y campos validada. |
| **F0.2** | Documentación de investigación anti-bot | 🟢 Completada | Alta | `docs/ANALISIS_BROWSERS_ANTI_BOT.md` | Commit `db2a862` sincronizado en main. |
| **F1.1** | PoC de consulta HTTP directa con headers móviles | 🟡 En Curso | Alta | `tmp/monitor_v3_poc/test_mobile_api.py` | Validar si los endpoints responden sin reto 403. |
| **F1.2** | PoC de Harvester de Sesión (Camoufox / Chrome Perfil) | ⚪ Pendiente | Alta | `tmp/monitor_v3_poc/test_session_harvester.py` | Extraer CookieJar y pasarlo a `curl_cffi`. |
| **F1.3** | PoC de descarga FT completa sin navegador | ⚪ Pendiente | Alta | `tmp/monitor_v3_poc/test_fast_ft_download.py` | Validar descarga de los 6 JSONs (h2h, pbp, graph). |
| **F2.1** | Creación del paquete `monitor_v3/` y scaffolding | ⚪ Pendiente | Media | Directorio `monitor_v3/` | Estructura de carpetas modular. |
| **F2.2** | Implementación de `core/session_manager.py` | ⚪ Pendiente | Alta | `monitor_v3/core/session_manager.py` | Cacheo y renovación automática de cookies. |
| **F2.3** | Implementación de `core/http_client.py` | ⚪ Pendiente | Alta | `monitor_v3/core/http_client.py` | Peticiones HTTP/2 con TLS impersonation. |
| **F3.1** | Tablas `_v3` en `matches.db` canónica | ⚪ Pendiente | Alta | `monitor_v3/database/repository.py` | `schedule_v3`, `log_v3`, `quarter_scores_v3`. |
| **F4.1** | Scrapers especializados (Schedule, Live, Detail) | ⚪ Pendiente | Alta | `monitor_v3/scrapers/*.py` | Reemplazo de los métodos síncronos de Playwright. |
| **F4.2** | Game Watcher Asíncrono para Q4 | ⚪ Pendiente | Alta | `monitor_v3/main.py` | Bucle de monitoreo liviano con jitter. |
| **F5.1** | Integración del evaluador ML de modelos | ⚪ Pendiente | Alta | `monitor_v3/models/evaluator.py` | Conexión con `models/registry.py`. |
| **F5.2** | Notificaciones Telegram V3 | ⚪ Pendiente | Media | `monitor_v3/notifications/telegram_bot.py` | Mensajes con prefijos 🟢🟡⚪ y ✅❌. |
| **F6.1** | Pruebas de estrés y benchmarking V2 vs V3 | ⚪ Pendiente | Media | `tmp/monitor_v3_poc/benchmark_v2_vs_v3.py` | Comparativa de RAM, CPU y tasa de éxito. |
| **F7.1** | Integración en `menu.bat` (Opción Monitor V3) | ⚪ Pendiente | Baja | `menu.bat` | Lanzador interactivo y modo CLI. |

---

## 5. Próximo Paso Inmediato

Ejecutar la tarea **F1.1**: Desarrollar la Prueba de Concepto (PoC) en `tmp/monitor_v3_poc/test_mobile_api.py` para probar la respuesta de las APIs de SofaScore con emulación de cabeceras móviles y cliente TLS (`httpx` / `curl_cffi`).
