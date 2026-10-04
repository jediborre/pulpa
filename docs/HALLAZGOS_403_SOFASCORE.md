# 🔍 Hallazgos: HTTP 403 en `api.sofascore.com` (WAF Fastly/Varnish)

> **Fecha:** 2026-10-04
> **Contexto:** `fetch-range` / `fetch-date --backend mobile` dejan de funcionar con
> `All connection attempts failed` y/o `403`. Se investiga si el problema son los JWT
> o el canal de red. Scripts de esta línea en `tmp/debug_connection/`.

## 0. ✅ RESUELTO (2026-10-04)

**Causa raíz:** el WAF de Fastly exige **huella TLS/HTTP2 OkHttp Android** *y* un
**`User-Agent` firmado con MD5 y ventana temporal** que la app calcula en cada request:

```
UA = "com.sofascore.results/260921/" + md5(str(unix_segundos // 100) + "sofa2012")[:6]
```

**Solución implementada:** `tls_client` con `client_identifier="okhttp4_android_13"`
+ `User-Agent` firmado + cabeceras `X-Timestamp`, `app-version`, `Accept-Language`,
`Cache-Control` y `Authorization: Bearer`. **Sin proxy.** Detalles completos y
procedimiento de re-derivación en `docs/REVERSE_ENGINEERING_SOFASCORE.md`.

**Resultado:** `token/init` y todos los endpoints de producción devuelven `200`.
Descarga real del rango `2026-09-01` → `2026-10-03` (33 fechas): **~2406 partidos
descargados, 0 fallos, 0 bans**, con rotación de tokens cada 10 partidos funcionando
(los 9 tokens rotaron repetidamente). No hubo ban de IP ni de token en esta corrida.

**Manejo de bans (rate limiting):** se observaron `403` a nivel de **token** tras
descargas intensas (los tokens se "quemaban"). Implementado:
- `TokenPool.ensure_min_tokens()` → emite JWT nuevos vía `POST /token/init` sin teléfono.
- En `MobileClient.request`, ante `401/403`: se purga el token, se emite uno nuevo y se
  reintenta automáticamente.
- Rotación de sesión cada 10 partidos (`notify_match_done`), ahora sí conectada a
  `fetch-date`/`fetch-range`.
- **No se observó ban por IP** (86 partidos seguidos con la misma IP pública). Si en el
  futuro apareciera, documentarlo aquí.

> ⚠️ **Emoji/encoding:** los logs de `TOKEN_POOL` usan emojis; en consola Windows
> (cp1252) rompían la ingesta con `UnicodeEncodeError`. `monitor_v3/utils/logger.py`
> ahora fuerza UTF-8.

---

## 1. Resumen ejecutivo (histórico)

- Los tokens JWT eran **válidos** (la app oficial seguía mostrando datos con ellos).
- La PC **no podía** consumir la API móvil directamente: recibía `403`.
- La infraestructura real es **Fastly + Varnish** (WAF propio de SofaScore), **no Cloudflare**.
- Cambiar solo la huella TLS, el token, la versión de HTTP, el stack Python o forzar
  IPv4/IPv6 **no** evitaba el 403: faltaba el **UA firmado**.
- **HTTP Toolkit (proxy `127.0.0.1:8000`) ya no ayuda**: al activar la intercepción MITM
  por ADB, la propia app recibe 403; al desactivarla, la app vuelve a funcionar.

## 2. Evidencia de infraestructura (Fastly/Varnish)

```
$ curl -sI https://api.sofascore.com/...
server: Varnish
retry-after: 0
strict-transport-security: max-age=300
content-type: application/json
```
- DNS: `A 140.248.179.52`, `AAAA 2a04:4e42:9c::820` → rangos de **Fastly**.
- Cuerpo de error: `{"error": {"code": 403, "reason": "..."}}`.
  - `reason: "Forbidden"` en `POST /api/v1/token/init`.
  - `reason: "challenge"` en `GET /sport/...`, `/event/...`.

> ⚠️ AGENTS.md y docs antiguos decían "Cloudflare". Es incorrecto: es el WAF de
> SofaScore sobre Fastly/Varnish.

## 3. Qué se probó y qué NO funcionó

| Prueba | Herramienta | Resultado |
|---|---|---|
| Petición directa HTTP/1.1 | `httpx` | `403 Forbidden` |
| Petición directa con headers móviles | `httpx` | `403 Forbidden` |
| Impersonación TLS Chrome/Android | `curl_cffi` (`chrome`, `chrome_android`, `chrome131_android`, `chrome99_android`) | `403 challenge` |
| Impersonación TLS OkHttp | `tls_client` (`okhttp4_android_10/11/12/13`) | `403 challenge` |
| Token fresco recién emitido por la app | `httpx`/`curl_cffi` | `403` (mismo resultado) |
| Fuerza de IPv6 | `curl -6` | `403 Forbidden` |
| `POST /token/init` desde la PC | `httpx` | `403 Forbidden` (la app sí obtiene token) |
| Proxy HTTP Toolkit con la app interceptada | HTTP Toolkit | la app también recibe 403 |

**Conclusión:** el 403 **no** depende de token, IP, TLS fingerprint ni del stack HTTP.
Depende de algo propio del cliente Android legítimo.

## 4. Pistas encontradas en la app (APK)

- Clases de red relevantes: `SCSWebviewCookieJar`, `convertCookieManager`,
  `ApiLevelUtil.getCookieManager`, `WEBVIEW_COOKIE`.
  → La app **inyecta cookies del WebView (`android.webkit.CookieManager`) en OkHttp**.
  Es probable que el WAF exija una cookie/estado de sesión obtenido por un WebView real
  (p. ej. resultado de un `challenge` de Varnish) que la app guarda **en memoria**.
- `app_webview/Default/Cookies` (extraído por `run-as`): **solo** cookies publicitarias
  (`adnxs.com`, `doubleclick.net`). **No** aparece `cf_clearance` ni cookie de SofaScore.
  Las cookies de sesión del WebView no se persisten a disco.
- Cabeceras detectadas en el DEX: `X-Premium-Token`, `X-Token-Refresh`, `X-Android-Version`.
  No se halló una firma `X-So-...` ni SDK de Cloudflare/Turnstile.
- No hay librerías nativas de firma específicas de SofaScore (solo SDKs de ads/APM).

## 5. Próximas líneas de investigación

1. **WebView remote debugging:** buscar socket `webview_devtools_remote_<pid>` en
   `/proc/net/unix`, hacer `adb forward` y leer cookies vía CDP (`Network.getCookies`).
2. **Decompilar la capa de red** (`SCSWebviewCookieJar`, interceptores OkHttp) para
   identificar cabeceras/cookies/firmas obligatorias. Herramienta sugerida: jadx.
3. **Capturar el `ClientHello` real** de la app con `tmp/debug_connection/passthrough_proxy.py`
   (requiere que la app honre el proxy; Android puede necesitar VPN).
4. **Usar el teléfono como egress real** (VPN/reverse) para saber si el WAF evalúa
   reputación de red más allá del fingerprint.
5. **Revisar si el WAF exige un `x-*` calculado por JNI** o un header de dispositivo.

## 6. Estado del canal de descarga

- `monitor_v3/config/constants.py` fuerza `SOFASCORE_PROXY_URL = http://127.0.0.1:8000`
  cuando el `.env` contiene `smartproxy`. Con HTTP Toolkit apagado, todas las peticiones
  fallan con `All connection attempts failed`.
- La rotación cada 10 partidos (`TokenPool.notify_match_done`) **no se invoca** en
  `match/cli.py` (`fetch-date`/`fetch-range`); solo existe en un test de `tmp/`.
- El proxy residencial Smartproxy del `.env` devuelve `407 Proxy Authentication Required`
  (credenciales expiradas o inválidas) en `tmp/debug_connection/test_smartproxy.py`.
