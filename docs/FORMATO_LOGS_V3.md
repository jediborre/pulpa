# 🎨 Formato de Logs del Monitor V3 (`monitor_v3`)

> **Ámbito:** daemon de producción `monitor_v3` (API móvil, zero-browser), sus scrapers, el pool de tokens, el bot de Telegram y los scripts auxiliares que importen el logger (por ejemplo el backfill de `tmp/probe_404_audit/cleanup_stale_schedule.py`).
> **Implementación única:** `monitor_v3/utils/logger.py`. Nadie debe formatear logs por su cuenta ni usar `print()` para eventos de producción.

---

## 1. Anatomía de una línea

Cada evento se escribe **dos veces**: coloreado en consola y en texto plano en el archivo diario.

**Consola:**

```text
{Fecha Hora} [NIVEL] [COMPONENTE] {prefijo Q4 🟠} [TAG] Mensaje transformado
```

**Archivo `logs/monitor_v3_<YYYY-MM-DD>.log`:**

```text
{Fecha Hora} [NIVEL] [COMPONENTE] {prefijo Q4 🟠} [TAG] Mensaje sin ANSI
```

Ejemplo real anotado (consola):

```text
2026-10-18 15:30:44 [WARNING] [MONITOREO] [PROBE] Partido obsoleto sin arranque | 12:00 17277719 Enosi Armenion Athliton vs GAOP
└──── fecha UTC-6 ────┘ └ nivel ┘ └ componente ┘ └ tag ┘ └──── bloque de partido (horario + id + equipos) ────┘
```

| Parte | Regla |
| --- | --- |
| **Fecha Hora** | `%Y-%m-%d %H:%M:%S` en **UTC-6** (`get_utc6_now()`), sin sufijo de zona. |
| **NIVEL** | Uno de `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| **COMPONENTE** | Subsistema que emite el evento (ver §3). |
| **Prefijo `Q4 🟠 `** | Solo cuando el emisor pasa `q4_orange=True` (ventana de cuarto final). |
| **TAG `[...]`** | Categoría temática opcional dentro del mensaje (ver §6). |
| **Mensaje** | Texto descriptivo, con bloque de partido si aplica (ver §5). |

---

## 2. Niveles y colores

### 2.1 Niveles (tabla oficial)

| Nivel | Función | Color consola | Código ANSI | Cuándo usarlo |
| --- | --- | --- | --- | --- |
| `DEBUG` | `log_debug(component, msg)` | Gris | `\033[90m` | Trazas de desarrollo. En producción apenas se usa. |
| `INFO` | `log_info(component, msg)` | Azul | `\033[94m` | Flujo operativo normal y seguimiento. |
| `WARNING` | `log_warning(component, msg)` | Amarillo | `\033[93m` | Desvíos temporales, obsolescencia, reintentos seguros, token degradado. |
| `ERROR` | `log_error(component, msg)` | Rojo | `\033[91m` | Fallos críticos o de red (`HTTP 403/404` se resaltan en rojo brillante). |

### 2.2 Paleta completa (`monitor_v3/utils/logger.py`)

| Constante | Código | Uso en el sistema |
| --- | --- | --- |
| `COLOR_RESET` | `\033[0m` | Cierre de cualquier color inline. |
| `COLOR_GREEN` | `\033[92m` | `ETA`, `SOLO FT`, `[EVAL COMPLETADA]`, `[BROADCAST]`, `WIN`, `[FT]`, ganador del marcador. |
| `COLOR_BRIGHT_RED` | `\033[91;1m` | Etiqueta `[LIVE]`, `HTTP 403`/`HTTP 404`, perdedor del marcador. |
| `COLOR_ORANGE` | `\033[38;5;208m` | Tags `[PROBE]`/`[WATCHER]`, `Q4` y `MIN` dentro del mensaje. |
| `COLOR_CYAN` | `\033[96m` | Componentes `MONITOREO` y `TELEGRAM`. |
| `COLOR_MAGENTA` | `\033[95m` | Componentes `SYSTEM` y `TOKEN_POOL`. |
| `COLOR_WARNING` | `\033[93m` | Nivel `WARNING` y componente **`[DATABASE]` / `[DB]`**. |
| `COLOR_ERROR` | `\033[91m` | Nivel `ERROR`, `LOSS` y la flecha `→`. |
| `COLOR_DEBUG/INFO` | `\033[90m` `\033[94m` | Color del nivel. |
| `COLOR_LATE_APUESTA` | `\033[37m` | Definida y reservada para apuestas tardías (en la práctica el aviso va por emoji en Telegram). |

> Los colores son **solo consola**: al escribir el archivo se eliminan con `re.sub(r"\033\[[0-9;]*m", "", msg)`.

---

## 3. Componentes

| Componente | Color | Responsabilidad | Archivos típicos |
| --- | --- | --- | --- |
| `SYSTEM` | Magenta | Arranque/parada del daemon, itinerario, configuración global. | `monitor_v3/main.py` |
| `MONITOREO` | Cyan | Ciclo de vida de cada partido: watcher, sonda, vivo, obsolescencia. | `monitor_v3/main.py` |
| `EVALUACION` | Verde | Resultado de la inferencia de los modelos activos. | `monitor_v3/main.py`, `monitor_v3/models/evaluator.py` |
| `DESCARGA` | Verde | Extracción de datos (API móvil) y persistencia final FT. | `monitor_v3/scrapers/*`, `monitor_v3/core/mobile_client.py` |
| `TELEGRAM` | Cyan | Envíos a Telegram y bot interactivo. | `monitor_v3/notifications/*` |
| `TOKEN_POOL` | Magenta | Rotación, fallos y regeneración de tokens JWT. | `monitor_v3/core/token_manager.py` |
| `DATABASE` | Sin color | Errores de la capa SQLite `_v3` (config, liquidación). | `monitor_v3/database/repository.py` |
| `DB` | Sin color | Lecturas puntuales de BD (`live_snapshots_v3`, suscriptores). | `repository.py`, `telegram_bot.py` |

---

## 4. Transformaciones automáticas del mensaje

Aplicadas por `_log_base()` **antes** de imprimir. El desarrollador escribe texto plano; el logger decora:

1. **Separador de pipe:** cada `" | "` se pinta en **verde**.
2. **Tags naranja:** `[PROBE]` y `[WATCHER]` → naranja.
3. **Errores de red en rojo brillante:** `HTTP 403` y `HTTP 404` (y los `403`/`404` sueltos, con guardas de regex para **no** corromper IDs de partido ni otras cifras) → `\033[91;1m`.
4. **Prefijo de cuarto final:** si se pasa `q4_orange=True`, la línea se antepone con `Q4 🟠 `.
5. **Colores inline del emisor:** se permite incrustar `COLOR_GREEN`, `COLOR_BRIGHT_RED`, etc. dentro del mensaje (patrón usado para `[LIVE]`, `ETA`, `SOLO FT`, `[EVAL COMPLETADA]`, `[BROADCAST]`).
6. **Palabras clave** (`_colorize_keywords`): `WIN` → verde, `LOSS` → rojo, `Q4` y `MIN` → naranja, `[FT]` → verde y la flecha `→` → rojo.
7. **Limpieza en archivo:** todo ANSI se elimina al escribir el `.log`.

---

## 5. Formato estándar de partido

Todo mensaje referido a un partido usa el bloque que devuelve `format_match_log(horario, match_id, home, away)`:

```text
18:30 15935079 Cleveland Cavaliers vs New York Knicks
```

| Campo | Color |
| --- | --- |
| `horario` (`HH:MM` en UTC-6) | **Amarillo** |
| `match_id` | **Azul** |
| `home vs away` | Sin color |

Regla: los mensajes globales (itinerario, tokens, resumen) **no** llevan este bloque; los de partido, **siempre**.

Marcador en vivo (`colorize_scores()`): **ganador en verde**, **perdedor en rojo brillante**, empate sin color.

---

## 6. Catálogo de tags

| Tag | Significado | Nivel habitual | Ejemplo |
| --- | --- | --- | --- |
| `[WATCHER]` | Creación/cierre de la corrutina dedicada a un partido. | INFO / WARNING | `[WATCHER] Start: 15:00 17221477 Azul y Rojo vs Colegio Los Leones Quilpe` |
| `[PROBE]` | Sonda ligera pre-partido y espera a la ventana Q4. | INFO / WARNING / ERROR | `[PROBE] En Vivo: ...` |
| `[FT-ONLY]` | Espera pasiva para ligas de la categoría `ft_only`. | INFO | `[FT-ONLY] Espera pasiva: 1h 12min \| ...` |
| `[LIVE]` | Monitoreo activo en vivo (se pinta en rojo brillante). | INFO / WARNING / ERROR | `[LIVE] Iniciando monitoreo móvil en vivo \| ...` |
| `[EVAL]` | Resultado de inferencia (en v3 se emite para el NO_BET definitivo). | INFO | `[EVAL] NO_BET definitivo \| Model: m27_v3 \| ...` |
| `[EVAL COMPLETADA]` | Todos los modelos se asentaron; el watcher espera el FT. | INFO | `[EVAL COMPLETADA] Todos los modelos asentados → esperando FT \| ...` |
| `[FT]` | Persistencia final: cuartos, histórico y liquidación. | INFO / WARNING / ERROR | `[FT] Final y liquidación de apuestas persistidos \| ... (gp=40)` |
| `[BROADCAST]` | Envío de un mensaje a varios suscriptores. | INFO | `[BROADCAST] Enviando 1390836289, 1922076265` |
| `[SKIP]` | Acción omitida deliberadamente (p. ej. sin token o sin suscriptores). | WARNING | `[SKIP] Sin suscriptores en monitor_subscribers (DB settings)` |
| `[FAIL]` | Fallo de entrega a un `chat_id` concreto. | ERROR | `[FAIL] chat_id=... \| Bad Request: ...` |
| `[ERROR]` | Error devuelto por la Bot API de Telegram. | ERROR | `[ERROR] chat_id=...: chat not found` |
| `[DB]` | Incidencia leyendo la base de datos (suscriptores, stats, snapshots). | ERROR | `[DB] Error leyendo subscribers: ...` |
| `[RESUMEN]` | Resumen diario de Telegram despachado. | INFO | `[RESUMEN] Resumen diario enviado (2026-10-18)` |
| `[BOT_V3]` | Bot interactivo (polling, comandos, resiliencia de red). | INFO / WARNING / ERROR | `[BOT_V3] Bot interactivo V3 iniciado (polling).` |
| `[BACKFILL]` | Scripts auxiliares de ingesta (p. ej. saneamiento de `_schedule_v3`). | INFO / ERROR | `[BACKFILL] Saneamiento finalizado: backfilled=4 no_graph=1` |

---

## 7. Catálogo de mensajes por fase

### 7.1 Arranque y parada (`SYSTEM`)

```text
[INFO] [SYSTEM] ==================================================
[INFO] [SYSTEM] 🏀 INICIANDO MONITOR V3 (NATIVE MOBILE API + MULTI-JWT)
[INFO] [SYSTEM] Modelos Activos: <lista de ACTIVE_MODELS separada por comas>
[INFO] [SYSTEM] Daemon V3 inicializado y operando. Monitoreando partidos en tiempo real...
[WARNING] [SYSTEM] El proxy local HTTP Toolkit (127.0.0.1:8000) no parece responder. Intentando conexión...
[INFO] [SYSTEM] Señal de interrupción recibida. Deteniendo daemon...
[INFO] [SYSTEM] Monitor V3 detenido de forma limpia y segura.
```

### 7.2 Itinerario (`SYSTEM`)

```text
[INFO] [SYSTEM] Actualizando itinerario: 2026-10-18 y 2026-10-19
[INFO] [SYSTEM] Itinerario 2026-10-18 guardado: 1324 partidos
[ERROR] [SYSTEM] Error en tarea de itinerario: <excepción>
```

### 7.3 Sonda pre-partido (`MONITOREO [PROBE]`)

```text
[INFO]    [MONITOREO] [PROBE] 16:45 17326049 CD Reducto vs CA Capurro | ETA 1h 20min
[INFO]    [MONITOREO] [PROBE] Drift Exit: 12:00 17277719 Enosi Armenion Athliton vs GAOP
[INFO]    [MONITOREO] [PROBE] En Vivo: 15:00 17221476 Luis Matte Larrain vs CD Sergio Ceppi
[INFO]    [MONITOREO] [PROBE] Finalizado detectado: <match_display>
[WARNING] [MONITOREO] [PROBE] Partido obsoleto | <match_display>
[WARNING] [MONITOREO] [PROBE] Partido obsoleto sin arranque | 12:00 17277719 Enosi Armenion Athliton vs GAOP
[WARNING] [MONITOREO] [PROBE] Evento eliminado en el API (HTTP 404) → se abandona | <match_display>
[WARNING] [MONITOREO] [PROBE] 8 fallos consecutivos → se abandona | <match_display>
[ERROR]   [MONITOREO] [PROBE] Error sondeo <match_display>: HTTP 404
```

### 7.4 Monitoreo en vivo (`MONITOREO [LIVE]`)

```text
[INFO]    [MONITOREO] [LIVE] Iniciando monitoreo móvil en vivo | <match_display>
[INFO]    [MONITOREO] Q4 🟠 15:00 17221476 Luis Matte Larrain vs CD Sergio Ceppi [Q4 | MIN~30] Score: 78 - 71
[INFO]    [MONITOREO] [LIVE] Fin detectado → Derivando a FT | <match_display>
[INFO]    [MONITOREO] Q4 🟠 [EVAL COMPLETADA] Todos los modelos asentados → esperando FT | <match_display>
[WARNING] [MONITOREO] [LIVE] Partido obsoleto | <match_display>
[ERROR]   [MONITOREO] [LIVE] Error en ciclo <match_display>: <excepción>
```

- La línea de marcador es el **log en vivo principal**: `[{cuarto} | MIN~{minuto}] Score: {home} - {away}`.
- Cuando el cuarto es `Q4`, la línea lleva el prefijo `Q4 🟠 ` y los colores de ganador/perdedor.

### 7.5 Evaluación (`EVALUACION [EVAL]`)

```text
[INFO] [EVALUACION] Q4 🟠 [EVAL] NO_BET definitivo | Model: m27_v5_3 | Motivo: descartada: torneo amistoso o invitacion (categoria AMISTOSOS) | <match_display>
```

- El campo `Motivo:` aparece solo cuando el modelo fue **descartado** (liga o formato de torneo, ventana de entrenamiento, liga de 12 min). Si simplemente no alcanzó el umbral, no hay motivo: lo muestra el umbral.

- Una **señal operable** no tiene línea de consola propia: su rastro es la fila insertada en `bet_monitor_log_v3`, el marcado `BET_SENT` de `bet_monitor_schedule_v3` y el `[BROADCAST]` que emite `[TELEGRAM]` al despachar la alerta.

### 7.6 Persistencia final y liquidación (`DESCARGA [FT]`)

```text
[INFO]    [DESCARGA] [FT] Final y liquidación de apuestas persistidos | CD Universitario Concepcion vs CD Gimnástico (gp=40)
[INFO]    [DESCARGA] [FT] Ya finalizado/notificado previamente, omitiendo | <home> vs <away>
[INFO]    [DESCARGA] [FT] Descartado (sin gráfica) | <home> vs <away>
[WARNING] [DESCARGA] [FT] Cierre con gráfica corta (gp=12) | <home> vs <away>
[ERROR]   [DESCARGA] [FT] Error procesando final <match_id>: <excepción>
```

Cada liquidación individual se registra en la capa de datos:

```text
[INFO] [DATABASE] Apuesta liquidada 17221472 (m27_v3): LOSS
```

### 7.7 Pool de tokens (`TOKEN_POOL`)

```text
[INFO]    [TOKEN_POOL] Pool activo listo con 3 tokens en rotación.
[INFO]    [TOKEN_POOL] ➕ Token regenerado automáticamente (...a1b2c3d4e5f6). Pool: 4 tokens.
[INFO]    [TOKEN_POOL] #2 de 4 [...a1b2c3d4e5f6]
[WARNING] [TOKEN_POOL] Fallo registrado en token ...a1b2c3d4e5f6 (Fallos: 1/3 | Código: HTTP 403)
[WARNING] [TOKEN_POOL] ❌ Token ...a1b2c3d4e5f6 desafiado o caducado (HTTP 403). Eliminado permanentemente de la lista de tokens válidos.
[INFO]    [TOKEN_POOL] ➡️ Switcheando inmediatamente al siguiente token disponible: ...0f9e8d7c6b5a (3 restantes en pool)
[ERROR]   [TOKEN_POOL] ⚠️ Se han agotado todos los tokens válidos del pool. Por favor captura o sincroniza nuevos tokens desde la App en menu.bat.
[ERROR]   [TOKEN_POOL] Error en /token/init: HTTP 403 - {"error": {"code": 403, "reason": "challenge"}}
```

### 7.8 Telegram y bot interactivo (`TELEGRAM`)

```text
[INFO]    [TELEGRAM] [BROADCAST] Enviando 1390836289, 1922076265
[WARNING] [TELEGRAM] [SKIP] TELEGRAM_BOT_TOKEN no configurado
[ERROR]   [TELEGRAM] [FAIL] chat_id=1922076265 | Bad Request: chat not found
[INFO]    [TELEGRAM] [RESUMEN] Resumen diario enviado (2026-10-18)
[INFO]    [TELEGRAM] [BOT_V3] Bot interactivo V3 iniciado (polling).
[WARNING] [TELEGRAM] [BOT_V3] Corte de red en polling (NetworkError: httpx.ReadError: ). Reintentando... (fallos acumulados: 7)
[WARNING] [TELEGRAM] [BOT_V3] Fallo de red en /status (intento 2/3): httpx.ReadError. Reintento en 2.0s
[ERROR]   [TELEGRAM] [BOT_V3] No se pudo completar mensaje largo (3/4) tras 3 intentos: httpx.ReadError
[ERROR]   [TELEGRAM] [BOT_V3] Excepción no controlada en handler: KeyError: 'x'
```

> Los errores del bucle de polling de PTB llegan al `add_error_handler` con `update=None`; se registran con **throttle de 60 s** para no inundar la consola con tracebacks completos.

---

## 8. Archivo de log rotativo

| Aspecto | Regla |
| --- | --- |
| Ruta | `logs/monitor_v3_<YYYY-MM-DD>.log` (relativa a la raíz del repo). |
| Rotación | Diaria, por fecha UTC-6. Un archivo `append` por día. |
| Codificación | UTF-8. |
| Contenido | `{fecha} [NIVEL] [COMPONENTE] {prefijo}{mensaje}` **sin** códigos ANSI. |
| Robustez | Cualquier fallo de escritura se ignora en silencio (`except: pass`): el logging **nunca** tumba al daemon. |
| Consola | Se fuerza `sys.stdout`/`sys.stderr` a UTF-8 para evitar `UnicodeEncodeError` con emojis en Windows. |

---

## 9. Formato de los mensajes de Telegram

### 9.1 Títulos de alerta de apuesta

| Situación | Título |
| --- | --- |
| Apuesta operable | `🟢 APUESTA Q4` |
| Apuesta tardía | `⚪️ APUESTA TARDIA Q4` |
| Sin valor | `⚪️ NO BET Q4` |

### 9.2 Prefijos por modelo (umbral de baja confianza: `< 0.30`)

| Prefijo | Significado |
| --- | --- |
| `🟢` | Apuesta bettable (confianza ≥ 30 %). |
| `🟡` | Apuesta de baja confianza (< 30 %). |
| `⚪️` | NO BET o apuesta tardía. |
| `🟡⚪️` | Tardía **y** de baja confianza. |

Ejemplo de línea por modelo: `🟢 APUESTA Q4 [m27_v5_3 78%] → 🏠 CD Universitario Concepcion`

Desde el 2026-10-20 **todos** los modelos VISIBLES del panel aparecen en la alerta, no solo los que apostaron, y cada uno con su motivo:

```text
🟢 APUESTA Q4
m27_v5_3 67% [PENDIENTE — aun puede apostar hasta el cierre de Q3] → 🏠 Falco KC-Szombathely
m27_v5_2 59% [NO BET (<75%)] → 🏠 Falco KC-Szombathely
```

- `PENDIENTE`: el modelo todavía puede apostar (vigila la zona verde hasta el cierre de Q3).
- `NO BET (<X%>)`: no superó el umbral `<X>` (el umbral real del modelo).
- `NO BET — <motivo>`: fue descartado; el motivo explica por qué.

> 🔕 **Política de umbral válido (2026-10-22):** la alerta **solo se envía si al menos un modelo VISIBLE tiene apuesta real** (`BET` por encima de su umbral). Si el único que apostó es un modelo en **modo sombra** (`SHADOW_MODELS`, hoy `m27_v5_2`), el log queda registrado en `bet_monitor_log_v3` / `model_inference_v3` pero **no se manda nada a Telegram**. Cuando sí hay apuesta real, las filas informativas del resto de modelos visibles se mantienen.

### 9.3 Resultados y confirmaciones FT

- Título: `🟢 RESULTADO Q4` / `⚪️ RESULTADO TARDIO Q4` (mismo criterio de prefijos que las alertas).
- Por modelo: `✅` ganada, `❌` perdida.
- Agregados: `✅ win`, `❌ loss`, `➖ push`, `⏳ pendiente`.
- 🔕 **Solo apuestas con umbral válido (2026-10-22):** la tarjeta de resultado se envía únicamente si hay un `BET` real de un modelo visible. Los `NO_BET`, los pendientes y los modelos en modo sombra **no generan mensaje** (antes un `NO_BET` de `m27_v5_2` al 56 % se colaba como `❌ RESULTADO Q4` porque `"BET" in "NO_BET"` era `True` y había fallback al modelo sombra). El filtro vive en `select_notifiable_logs()` / `select_notifiable_predictions()` de `telegram_bot.py`, aplicado también como red de seguridad dentro de los propios senders.
- 📉 **Cierres sin gráfica (`skip_reason="no_graph"`):** si el payload FT no trae `graph_points` (ligas menores que no publican curva de momentum), se guardan igual los marcadores por cuarto, se liquidan las apuestas y **sí se envía la tarjeta de resultado**; sale sin la curva de momentum, pero con el panel, marcadores y ✅/❌. Antes el `return` temprano se comía el aviso (caso `17330673` Vasto Verde Blumenau, 2026-10-22).

### 9.4 Reportes y bot interactivo

- Formato de mensajes: `parse_mode="HTML"`; mensajes largos se trocean a **3 800 caracteres** respetando saltos de línea.
- Bot interactivo (`bot_runner.py`): teclado persistente `🎯 Apuestas` / `📊 Status` y comandos `/start`, `/signals`, `/status`.
- Los envíos salientes se reintentan hasta 3 veces con backoff (1 s → 2 s → 4 s) ante `NetworkError`/`TimedOut`, y respetan `RetryAfter`.
- Los avisos de apuesta y de resultado FT se envían además como **imagen** (tarjeta y, en FT, tarjeta + gráfica de momentum): ver [`IMAGEN_APUESTA_Q4.md`](IMAGEN_APUESTA_Q4.md). Si la imagen falla, el aviso cae a texto plano.
- Cada partido lleva un **`#N`** secuencial dentro del día (columna `daily_seq`, reinicia en 1 cada fecha) para facilitar el seguimiento en el chat y en el listado `🎯 Apuestas`.
- `/status` responde con **imagen de dos secciones (`HOY` y `MES`)** más el texto equivalente; el resumen diario de las 23:55 usa el mismo motor con el desglose por categoría.
- `/status` (texto e imagen) y el resumen del día de `/signals` solo cuentan los modelos de `STATUS_MODELS` (`m27_v5_3`): los retirados (`m27_v4`, `m27_v3`, `v6_2`) y los de modo sombra no aparecen ni suman en los totales (2026-10-22). Sus logs siguen en `bet_monitor_log_v3` para auditoría.

---

## 10. Reglas de oro (checklist de cambios)

1. **Estructura modular**: el logger vive solo en `monitor_v3/utils/logger.py`.
2. **BD**: `matches.db` en la raíz; el sufijo de tablas v3 es `_v3` (no afecta al log).
3. **Tabla de logs**: `bet_monitor_log_v3` guarda `raw_json`, `inference_minute` y `graph_points_count`.
4. **Ligas**: configuración declarativa en `config/leagues.yaml`.
5. **Formato**: `{Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]` con INFO azul, WARNING amarillo, ERROR rojo.
6. **Partidos**: siempre `{horario} {match_id} {home} vs {away}`.
7. **Errores de red**: imprimir explícitamente `HTTP 403/404` en rojo brillante.
8. **Cuarto final**: usar obligatoriamente el prefijo `Q4 🟠`.
9. **Apuestas en Telegram**: prefijos `🟢` bettable, `🟡` no bettable, `⚪️` tardía.
10. **Resultados FT en Telegram**: prefijar `✅` ganada o `❌` perdida.
11. **Tiempos**: siempre en formato humano (`format_human_time`: `45s`, `12min 30s`, `1h 4min`, `4 dia 3h 15min`).
12. **Obsolescencia**: un partido sin datos 3.5 h después de su hora de inicio se abandona (`WATCH_OBSOLETE_SECS`).
13. **404 de SofaScore**: es un fallo permanente (evento eliminado o re-creado con otro id) → el watcher abandona y lo marca, nunca reintenta en bucle.

---

## 11. Comandos útiles de inspección

```powershell
# Todos los errores del día
Select-String -Path logs\monitor_v3_2026-10-18.log -Pattern '\[ERROR\]'

# Seguimiento de un partido concreto
Select-String -Path logs\monitor_v3_2026-10-18.log -Pattern '17353606'

# Sonda pre-partido
Select-String -Path logs\monitor_v3_2026-10-18.log -Pattern '\[PROBE\]'

# Ventana de cuarto final
Select-String -Path logs\monitor_v3_2026-10-18.log -Pattern 'Q4 '

# Apuestas liquidadas
Select-String -Path logs\monitor_v3_2026-10-18.log -Pattern 'Apuesta liquidada'

# Últimas 20 líneas
Get-Content logs\monitor_v3_2026-10-18.log -Tail 20
```

---

## 12. API para desarrolladores

```python
from monitor_v3.utils.logger import log_info, log_warning, log_error, format_match_log

match_display = format_match_log("18:30", match_id, home, away)

log_info("MONITOREO", f"[WATCHER] Start: {match_display}")
log_warning("MONITOREO", f"[PROBE] Partido obsoleto | {match_display}")
log_error("DESCARGA", f"[FT] Error procesando final {match_id}: {e}")

# Prefijo Q4 🟠 para eventos dentro de la ventana de cuarto final
log_info("EVALUACION", f"[EVAL] NO_BET definitivo | Model: {model} | {match_display}", q4_orange=True)
```

- Componentes válidos: `SYSTEM`, `MONITOREO`, `EVALUACION`, `DESCARGA`, `TELEGRAM`, `TOKEN_POOL`, `DATABASE`, `DB`.
- Tags recomendados: los de §6. Si añades uno nuevo, actualiza este documento.
- No usar `print()` para eventos de producción: no queda en el archivo diario ni se colorea.

- **Liquidaci?n sin pendientes**: `reconcile_pending_results()` liquida comparando el **parcial de Q4** (si el cuarto empata, la apuesta cuenta como **PERDIDA**). Si un partido no publica cuartos, un barrido cada 2 h reintenta su FT (`PENDING_SWEEP_*`) y, tras 24 h sin datos, marca la apuesta como `void` (anulada): deja de contar en apuestas y acierto, pero queda auditada. As? nunca quedan `pending` permanentes. El resumen diario ejecuta un barrido final antes de enviarse.
