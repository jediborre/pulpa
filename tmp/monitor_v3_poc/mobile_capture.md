# 📱 Guía de Laboratorio: Captura de Tráfico de la App Android de SofaScore

> **Ubicación:** `tmp/monitor_v3_poc/mobile_capture.md`  
> **Propósito:** Documentar el procedimiento paso a paso para interceptar el tráfico HTTPS de la app oficial de SofaScore en Android usando **HTTP Toolkit** (o Mitmproxy) en un entorno de pruebas controlado.  
> **Objetivo:** Extraer las cabeceras exactas (`User-Agent`, `X-So-...`, tokens, cookies nativas) que utiliza la app en partidos en vivo para replicarlas directamente en Python sin requerir emuladores en producción.

---

## 1. ¿Por qué HTTP Toolkit es la herramienta recomendada?

A diferencia de Fiddler o Charles Proxy tradicionales, **[HTTP Toolkit](https://httptoolkit.com/)** es una herramienta de código abierto especializada en Android que:
* Detecta automáticamente emuladores en ejecución (Android Studio AVD, Genymotion, LDPlayer, Nox).
* **Inyecta automáticamente certificados CA en el almacén del sistema (`/system/etc/security/cacerts`)** sin que tengas que luchar con permisos de Android 14+.
* **Bypassea automáticamente el SSL Pinning** de librerías comunes (como `OkHttp`, `TrustManager` y certificados empaquetados) mediante Frida en caliente con un solo click.

---

## 2. Preparación del Entorno (1 sola vez, ~10 minutos)

Tienes dos alternativas para realizar este laboratorio:

### Alternativa 1: Emulador de Android en PC (Recomendada si tienes Android Studio o emulador)
1. **Descargar e Instalar HTTP Toolkit en Windows:**
   * Descarga el instalador gratuito desde [httptoolkit.com](https://httptoolkit.com/).
2. **Iniciar un Emulador:**
   * **Opción A (Android Studio):** Abre un AVD (preferentemente con imagen de sistema **Google APIs** y arquitectura x86_64, sin "Google Play Store" bloqueado para permitir privilegios root adb fáciles).
   * **Opción B (Emulador ligero como Genymotion o LDPlayer):** Inicia el emulador con la opción de root habilitada.
3. **Instalar SofaScore en el Emulador:**
   * Descarga el APK oficial de SofaScore desde una fuente confiable (ej. APKMirror o Google Play) e instálalo arrastrando el archivo a la ventana del emulador.

---

### Alternativa 2: Teléfono Android Físico de Prueba
Si prefieres usar un teléfono real:
1. Conecta el teléfono por USB con **Depuración USB** activada.
2. En HTTP Toolkit, haz click en **"Android Device via ADB"**.
3. La aplicación de HTTP Toolkit se instalará automáticamente en tu teléfono y configurará la VPN local de intercepción.

---

## 3. Procedimiento de Captura Paso a Paso

1. **Vincular HTTP Toolkit con Android:**
   * En HTTP Toolkit en tu PC, haz click en el botón verde **"Android Device via ADB"** o **"Android Emulator"**.
   * Verás que HTTP Toolkit se conecta, inyecta su proxy y abre una sesión de captura en vivo.
2. **Filtrar el Tráfico:**
   * En la barra de búsqueda superior de HTTP Toolkit, escribe el filtro:
     ```
     sofascore
     ```
     para ignorar el tráfico del sistema operativo o de Google Play.
3. **Generar los Eventos en la App:**
   * En el emulador/teléfono, abre la aplicación **SofaScore**.
   * Ve a la sección **Básquetbol**.
   * Abre un partido que esté **en vivo** o que haya finalizado recientemente (para ver la gráfica de momentum y las estadísticas por cuarto).
   * Toca en las pestañas: **Resumen**, **Incidentes (Jugada a jugada)** y **Estadísticas**.
4. **Inspeccionar las Peticiones en HTTP Toolkit:**
   * Regresa a la ventana de HTTP Toolkit en la PC. Verás una lista de peticiones en color verde (`GET 200`).
   * Busca peticiones dirigidas a `api.sofascore.com` o subdominios similares.

---

## 4. ¿Qué datos específicos debemos anotar?

Al hacer click sobre una petición exitosa de un partido (ej. `/api/v1/event/...` o `/api/v1/event/.../incidents`), haz click en la pestaña **"Headers"** de la petición y copia los siguientes campos clave:

```http
GET /api/v1/event/XXXXXX/incidents HTTP/2
Host: api.sofascore.com
User-Agent: [Copiar valor exacto]
Accept: [Copiar valor exacto]
X-So-...: [Copiar cualquier cabecera que empiece con X- o similar]
Authorization / Token: [Si existe algún header de autenticación]
Cookie: [Si envía alguna cookie en la petición]
```

### Exportación Directa a cURL / Python:
En HTTP Toolkit, puedes hacer **click derecho sobre la petición $\rightarrow$ "Copy as..." $\rightarrow$ "Copy as cURL"** o **"Copy as Python Request"**.

---

## 5. Validación Inmediata en `pulpa`

Una vez copiado el comando cURL o los headers reales, los pegaremos en el script de prueba:
`tmp/monitor_v3_poc/replay_captured_request.py`

Si al ejecutar el script en Python con esas cabeceras recibimos **`200 OK` con el JSON de los incidentes**:
* **Habremos derrotado a Cloudflare de forma definitiva.**
* Podrás desinstalar o apagar el emulador para siempre.
* Tendremos la fórmula exacta para implementar el cliente nativo de `monitor_v3`.
