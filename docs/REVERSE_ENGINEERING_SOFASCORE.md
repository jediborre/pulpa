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
