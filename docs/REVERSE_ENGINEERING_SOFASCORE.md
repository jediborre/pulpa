# 🧪 Reverse Engineering de la App SofaScore (para parchear/actualizar el APK)

> **Fecha:** 2026-10-04
> **Propósito:** Dejar el procedimiento exacto, los datos de servidor, la arquitectura
> de red de la app y las herramientas usadas para volver a derivar el contrato móvil
> cuando se actualice el APK (y por tanto cambien versión, salt o huella TLS).
> **Resultado:** El WAF de Fastly exige **huella TLS/HTTP2 OkHttp Android** + un
> **`User-Agent` firmado con MD5 y ventana temporal**. Con eso, la PC consume la API
> directamente sin proxy. Ver `docs/HALLAZGOS_403_SOFASCORE.md`.

---

## 1. Datos del servidor de SofaScore

| Dato | Valor |
|---|---|
| Host API móvil | `https://api.sofascore.com/api/v1` |
| Infraestructura | **Fastly + Varnish** (NO Cloudflare) |
| Cabeceras típicas | `server: Varnish`, `retry-after: 0`, `strict-transport-security: max-age=300` |
| DNS | `A 140.248.179.52`, `AAAA 2a04:4e42:9c::820` (rangos Fastly) |
| Cuerpo de error | `{"error": {"code": 403, "reason": "..."}}` |
| `reason` observado | `"Forbidden"` (`POST /token/init`), `"challenge"` (`GET /sport/...`, `/event/...`) |

**Consecuencia:** no sirve intentar "pasar Cloudflare"; es un WAF propio sobre Fastly
que valida huella de cliente + firma temporal.

---

## 2. Arquitectura de red de la app (contrato móvil)

La app usa **OkHttp** con una cadena de interceptores que firma las peticiones:

1. **Interceptor `User-Agent` + `X-Timestamp`** (clase ofuscada `Lmmg;`)
2. **Interceptor `Authorization` + `app-version` + `X-Token-Refresh`** (`Lnmg;`)
3. **Interceptor `X-Premium-Token` + `Cache-Control`** (`Llmg;`)

### 2.1 User-Agent firmado (el "truco")

Pseudocódigo exacto reconstruido del bytecode:

```java
long t = System.currentTimeMillis() / 1000 / 100;   // unix_seconds / 100
String raw = t + "sofa2012";                         // "sofa2012" = salt fijo
String hash = md5(raw.getBytes());                   // MD5 en hex minúsculas
String version = "260921003".substring(0, 6);        // -> "260921"
String ua = "com.sofascore.results/" + Integer.parseInt(version) + "/" + hash.substring(0, 6);
// => com.sofascore.results/260921/<6 chars del MD5>
headers.add("User-Agent", ua);
headers.add("X-Timestamp", String.valueOf(Instant.now().toEpochMilli())); // ms epoch
```

> ⚠️ La firma cambia cada **100 segundos**. Un `User-Agent` estático (p. ej.
> `com.sofascore.results/260921/022538`) se vuelve inválido y el WAF responde 403.

### 2.2 Cabeceras que envía la app

| Cabecera | Valor / origen |
|---|---|
| `User-Agent` | firmado (ver 2.1) |
| `X-Timestamp` | `Instant.now().toEpochMilli()` |
| `Authorization` | `Bearer <AUTH_TOKEN>` (solo en `GET`/`HEAD`) |
| `app-version` | `"260921"` (primeros 6 dígitos de `260921003`) |
| `X-Token-Refresh` | refresh de token (worker `RegistrationWorker-...`) |
| `X-Premium-Token` | solo si hay premium (se consulta en prefs) |
| `Cache-Control` | `max-age=0` (o `max-stale=604800` según conectividad) |
| `Accept-Language` | p. ej. `en-US,en;q=0.9` |
| `Accept` | `application/json` |
| `Accept-Encoding` | `gzip` |

### 2.3 Mapa de clases ofuscadas (APK versión `260921003`)

| Clase | Rol |
|---|---|
| `Lmmg;` | Interceptor User-Agent + X-Timestamp |
| `Lnmg;` | Interceptor Authorization + app-version + X-Token-Refresh |
| `Llmg;` | Interceptor X-Premium-Token + Cache-Control |
| `Lue0;->g([B)String` | **MD5** hex (lowercase, zero-padded) |
| `Ll6j;->J(I,String)` | `substring(0, n)` |
| `Lw7i;->h(I,String)` | concatenación `StringBuilder(String).append(int)` |
| `Ljdd;->i()` | `Instant.now().toEpochMilli()` |
| `Lx1h;` | Referencia a `SCSWebviewCookieJar` (cookies del WebView, para ads/Equativ) |
| `Lp1i;->f` | Flujo `https://www.sofascore.com/android-auth` (login Google) |

### 2.4 Endpoints que usa el monitor

| Endpoint | Uso |
|---|---|
| `sport/basketball/{date}/{tz_offset_secs}/categories` | Descubrir categorías con partidos |
| `category/{id}/scheduled-events/{date}` | Partidos de una categoría |
| `event/{id}` | Metadatos + marcadores |
| `event/{id}/incidents` | Play-by-play |
| `event/{id}/graph` | Curva de momentum |
| `event/{id}/statistics` | Stats de equipo |
| `event/{id}/lineups` | Alineaciones |
| `event/{id}/h2h` | Cara a cara (agregado) |
| `event/{id}/odds/1/all` | Cuotas |
| `POST token/init` | Emisión de JWT (6 meses) |

> `sport/basketball/scheduled-events/{date}` **NO existe** (devuelve 404). Usar el
> flujo de `categories`.

### 2.5 Reloj de juego (minuto real, fuente fiable)

`GET event/{id}` expone `event.time`:

```json
{"played": 1932, "periodLength": 600, "overtimeLength": 300,
 "totalPeriodCount": 4, "clockRunning": false,
 "currentPeriodStartTimestamp": 1791131780}
```

- **`played`**: segundos acumulados del reloj de juego → **minuto real = `played // 60`**.
  Ej.: `played=1932` → min 32 (4º cuarto), `played=1442` → min 24 (3er cuarto).
- **`periodLength`**: segundos por periodo (600 = 10 min FIBA; 720 = 12 min NBA).
- `overtimeLength`, `totalPeriodCount`, `clockRunning`, `currentPeriodStartTimestamp`.

⚠️ **NO inferir el minuto desde PBP ni desde la gráfica.** Los `incidents` incluyen
marcadores de periodo cuyo `timeSeconds` es el **fin** del periodo (1200→20, 1800→30,
2400→40), por lo que el minuto inferido salta al final del cuarto y dispara el
monitoreo/FT antes de tiempo. La gráfica (`graphPoints`) sí trae `minute` fiable, pero
`time.played` es la fuente directa.

Implementado en `match/scraper.py` (`_parse` expone `match.game_seconds_played`) y usado
en `monitor_v3/main.py` y `monitor_v2/main.py` con fallback a PBP/gráfica.

### 2.6 URL canónica de un partido (botón "📱 Sofascore" de Telegram)

Estructura real:

```
https://www.sofascore.com/{deporte}/match/{slug}/{custom_id}#id:{match_id}
```

- **`{deporte}`**: `basketball`.
- **`{slug}`**: el campo `event.slug` del API. **NO es simplemente `home-away`**;
  SofaScore lo devuelve en su propio orden (a veces `away-home`). Ej.:
  `ldlc-asvel-lyon-villeurbanne-bcm-gravelines-dunkerque` para
  "BCM Gravelines-Dunkerque vs LDLC ASVEL". Por eso **hay que usar el `event.slug`
  tal cual**, no reconstruirlo desde `home_slug`/`away_slug`.
- **`{custom_id}`**: `event.customId` corto (ej. `HvbsLvb`, `fEicsGRpi`). **Es
  obligatorio**: sin él la ruta resuelve a **404**.
- **`#id:{match_id}`**: fragmento legacy/informativo (no se envía al servidor).

Ejemplo válido:
`https://www.sofascore.com/basketball/match/al-ula-al-kuwait/fEicsGRpi#id:17249511`

**Implementación:** `_sofascore_match_url()` en
`monitor_v3/notifications/telegram_bot.py` (y `monitor_v2`). Requiere que se le pase
`match_data` (el payload del partido con `match.event_slug` y `match.custom_id`).
Si no se pasa `match_data`, cae a un fallback `{home_slug}-{away_slug}` sin
`custom_id`, que **puede dar 404**. Por eso `_final_fetch_and_save` (resultados) debe
pasar `match_data=data` (arreglado 2026-10-04: antes no lo hacía y los botones de
RESULTADO daban 404).

---

## 3. Combinación que pasa el WAF (probada 200 OK)

- **Cliente TLS:** `tls_client.Session(client_identifier="okhttp4_android_13")`
  (también `okhttp4_android_12`), `random_tls_extension_order=False`.
- **User-Agent:** firmado (sección 2.1).
- **Cabeceras:** las de la sección 2.2 (`app-version`, `Accept-Language`, etc.).
- **Token:** JWT de `tokens.json` (o emitido con `POST token/init`).

Combinaciones que **NO** funcionan (dan 403): `httpx`, `requests`, `curl_cffi`
(`chrome`, `chrome_android`), `tls_client` con `chrome_131`, o cualquier cliente
**sin** el UA firmado.

---

## 4. Herramientas usadas

| Herramienta | Para qué |
|---|---|
| `adb` (Android SDK platform-tools) | `run-as`, extraer `shared_prefs`, `pm clear`, monkey, `settings`, forward de sockets |
| `adb shell run-as com.sofascore.results` | Leer prefs (`AUTH_TOKEN`), listar datos, `cat` de DBs |
| **androguard** (parser DEX crudo) | Localizar clases/métodos y constantes string en el APK |
| **tls_client** (bogdanfinn) | Huella TLS/HTTP2 OkHttp Android |
| **curl_cffi** | Huellas Chrome (descartadas para este WAF) |
| **HTTP Toolkit** | Intercepción MITM histórica (hoy rompe la app: el WAF detecta MITM) |
| WebView CDP (`webview_devtools_remote_<pid>` + `adb forward tcp:9222`) | Leer cookies del WebView |
| `apk-mitm` + `uber-apk-signer` | Parchear el APK (quitar SSL pinning, `debuggable=true`) |
| `Sofascore.apk` / `Sofascore-patched.apk` | Binarios en `tmp/monitor_v3_poc/` |

---

## 5. Procedimiento para re-derivar si se actualiza el APK

1. **Extraer el token** (opción 28 de `menu.bat` o `tools/sync_token_from_adb.py`):
   `adb shell run-as com.sofascore.results cat shared_prefs/com.sofascore.results_preferences.xml`.
2. **Confirmar que la PC da 403** con `tmp/debug_connection/test_direct.py`.
3. **Localizar los interceptores** con androguard (script `tmp/debug_connection/find_header_methods.py`):
   buscar `const-string` con `X-Timestamp`, `User-Agent`, `Authorization`,
   `X-Premium-Token`, `X-Token-Refresh`.
4. **Volcar el bytecode** del interceptor de User-Agent
   (`tmp/debug_connection/dump_intercept_bytecode.py`) y de la función de hash
   (`tmp/debug_connection/dump_hash_fn.py`). Reconstruir:
   - salt (p. ej. `sofa2012`),
   - ventana temporal (p. ej. `currentTimeMillis/1000/100`),
   - algoritmo de hash (p. ej. MD5) y longitud del prefijo (p. ej. 6),
   - versión base (p. ej. `260921003`).
5. **Probar** la combinación `tls_client okhttp4_android_*` + UA firmado con
   `tmp/debug_connection/test_okhttp_signed.py`.
6. Actualizar `monitor_v3/config/constants.py` (`SOFASCORE_APP_VERSION`,
   `SOFASCORE_UA_SALT`) y `monitor_v3/utils/helpers.py` (`build_signed_ua`).

---

## 6. Comandos útiles de ADB

```powershell
# Token actual
adb shell run-as com.sofascore.results cat shared_prefs/com.sofascore.results_preferences.xml

# Reset de sesión de la app
adb shell pm clear com.sofascore.results
adb shell monkey -p com.sofascore.results -c android.intent.category.LAUNCHER 1

# WebView devtools (si está habilitado por ser debuggable)
adb shell cat /proc/net/unix | findstr webview_devtools
adb forward tcp:9222 localabstract:webview_devtools_remote_<pid>
```

---

## 7. Notas

- La app **no persiste** cookies de SofaScore en `app_webview/Default/Cookies`
  (solo cookies publicitarias). El bypass **no** depende de cookies.
- La app **no** usa SDK de Cloudflare/Turnstile ni librerías nativas de firma
  específicas (solo SDKs de ads/APM).
- El login se hace vía `https://www.sofascore.com/android-auth` en un Custom Tab.
- El `AUTH_TOKEN` vive en `shared_prefs/com.sofascore.results_preferences.xml` y
  dura ~6 meses.

---

## 8. PoC sin wrapper pesado: `curl_cffi` (libcurl) con JA3 OkHttp

`tls_client` (blob Go) es solo un medio para enviar el **ClientHello correcto**. Se
demostró que basta un cliente HTTP ligero con el JA3 exacto. `curl_cffi` acepta
`ja3=` y `akamai=` directamente.

**JA3 OkHttp Android (capturado):**
```
771,4865-4866-4867-49195-49196-52393-49199-49200-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-51-45-43-21,29-23-24,0
```

**Uso (probado 200 OK en `token/init` y todos los endpoints):**
```python
from curl_cffi import requests as cffi

JA3 = "771,4865-4866-4867-49195-49196-52393-49199-49200-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-51-45-43-21,29-23-24,0"
s = cffi.Session(ja3=JA3, impersonate="chrome")
r = s.post("https://api.sofascore.com/api/v1/token/init", headers=sign_headers(), json=payload)
```

- Scripts: `tmp/debug_connection/capture_ja3.py` (captura el JA3 vía proxy local) y
  `tmp/debug_connection/test_curl_cffi_ja3.py` / `test_curl_cffi_get.py` (PoC).
- **Cómo capturar el JA3:** levantar un listener TCP local que haga de proxy HTTP
  (`CONNECT`), apuntar `tls_client`/OkHttp a él y parsear el ClientHello (debe incluir
  la extensión `0` = SNI; si se conecta a una IP, el JA3 sale sin SNI y falla).
- La PoC confirma que **el WAF evalúa el ClientHello TLS + el UA firmado**, no la
  librería. `httpx`/`requests` fallan porque su ClientHello no coincide.

### 8.1 Benchmark `tls_client` vs `curl_cffi` y decisión

Benchmark riguroso (IDs de partido **disjuntos** por cliente, peticiones alternadas,
partido completo = 7 endpoints en paralelo, mediana de 15 partidos, 2 corridas).
Script: `tmp/debug_connection/bench_clients_opt.py`.

| Cliente | mediana por partido |
|---|---|
| `tls_client` (`okhttp4_android_13`) | **~12 ms** |
| `curl_cffi` `ja3=` solo (sesión compartida) | ~16 ms |
| `curl_cffi` `AsyncSession` | ~18 ms |
| `curl_cffi` shared con `impersonate` | ~19 ms |

Notas del benchmark:
- Un benchmark ingenuo (sesión por thread + pool recreado por partido) daba `curl_cffi`
  ~72 ms; al **reutilizar pool de threads y sesión compartida** (curl_cffi usa handle
  curl thread-local interno) bajó a ~16 ms. La optimización movió la aguja.
- La clave es **reutilizar conexiones** (keep-alive); recrear sesiones/threads por
  partido paga handshakes TLS en cada descarga.

**Decisión: se mantiene `tls_client` en producción.**
- Es marginalmente más rápido (~12 ms vs ~16 ms por partido) y ya está validado con
  ~2400 partidos y 0 fallos.
- La diferencia real es despreciable: el descargador espacia los partidos 0.7–1.3 s
  (jitter anti-bot), así que ~4 ms/partido es <1% del tiempo total.
- `curl_cffi` queda documentado como alternativa ligera válida (sin el blob Go) por si
  se quiere eliminar esa dependencia: bastaría `curl_cffi.Session(ja3=JA3)` en
  `mobile_client.py`.
