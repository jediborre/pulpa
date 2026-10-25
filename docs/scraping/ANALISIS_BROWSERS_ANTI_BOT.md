# Guía Exhaustiva: Motores de Navegación, Automatización Stealth y Evasión Anti-Bot en GitHub

> **Propósito:** Documentar los hallazgos técnicos derivados de las pruebas con Obscura, el análisis arquitectural de bloqueos anti-bot (Cloudflare Turnstile, DataDome, Akamai) y el catálogo completo de las alternativas más avanzadas y viables disponibles en GitHub para el scraping y monitoreo en vivo en `pulpa`.

---

## 1. Contexto y Lecciones Aprendidas de Obscura

Durante las pruebas con **Obscura** ([`h4ckf0r0day/obscura`](https://github.com/h4ckf0r0day/obscura)), un motor headless ligero (~30 MB de RAM) programado en Rust con interfaz CDP (`obscura serve`), se determinó que es **inviable para sitios web interactivos modernos** como SofaScore.

### Causas Fundamentales del Fallo (HTTP 403 Forbidden):
1. **Ausencia de un Motor DOM Completo:**
   - Obscura utiliza `html5ever` para construir un árbol DOM básico y un runtime V8 mínimo (`deno_core`).
   - No implementa la especificación completa del Living Standard DOM (APIs como `MutationObserver`, `IntersectionObserver`, `Storage APIs`, `ServiceWorkers`, `AbortController`, eventos sintéticos de frameworks).
   - **Consecuencia:** Los scripts de empaquetado de React y Next.js nunca se ejecutan. La página se queda en su estado de carga inicial (`"Loading..."`), impidiendo la generación de tokens de cliente y cookies de navegación requeridas por las APIs.
2. **Desconexión entre el CookieJar interno y Playwright CDP:**
   - Cuando Playwright se conecta vía CDP (`connect_over_cdp`), sus llamadas HTTP de contexto (`ctx.request.get()`) usan la pila de red nativa de Playwright, no el cliente HTTP de Obscura.
   - Como Obscura no sincroniza cookies bidireccionalmente a través del protocolo CDP Fetch/Network, cualquier llamada a endpoints API se dispara sin cookies de sesión, provocando el bloqueo inmediato.
3. **Limitaciones de Evaluación Asíncrona:**
   - `--eval` y `op_fetch_url` sufrieron problemas de resolución de promesas (timeout de 100ms) y bloqueo de cookies en peticiones cross-origin (`www.sofascore.com` hacia `api.sofascore.com`).
   - A pesar de corregir los bugs en el código Rust (`ops.rs` y `runtime.rs`), el problema raíz persiste: **un motor que no ejecuta los scripts del sitio jamás podrá superar los retos de Cloudflare ni autenticar sesiones dinámicas.**

> **Principio Clave:** Crear un motor de navegación completo desde cero en Rust requiere años de esfuerzo para emular lo que Chromium y Gecko ya hacen. Las soluciones que realmente triunfan en producción toman motores reales (Chromium o Firefox) y los modifican a bajo nivel para eliminar las huellas de automatización.

---

## 2. Anatomía de la Detección Anti-Bot Moderna (2025 - 2026)

Los sistemas de protección modernos (Cloudflare Turnstile/Bot Fight Mode, DataDome, Akamai, Kasada) analizan peticiones en cuatro capas concurrentes:

```
┌────────────────────────────────────────────────────────┐
│ 1. Capa de Red / TLS Handshake (JA3, JA4, HTTP/2 Frames)│
├────────────────────────────────────────────────────────┤
│ 2. Detección de Fugas de Protocolo (CDP, WebDriver)    │
├────────────────────────────────────────────────────────┤
│ 3. Huella Digital del Navegador (Canvas, WebGL, Fonts) │
├────────────────────────────────────────────────────────┤
│ 4. Retos Interactivos y de Comportamiento (Turnstile)  │
└────────────────────────────────────────────────────────┘
```

1. **Capa TLS / JA3 / JA4:** Inspecciona el orden de los ciphers suites, extensiones TLS, curvas elípticas y parámetros HTTP/2. `requests`, `urllib` o `httpx` estándar se delatan aquí al instante.
2. **Capa CDP / WebDriver:** Cloudflare detecta `navigator.webdriver = true`, pero más recientemente detecta **fugas del Chrome DevTools Protocol**, principalmente la llamada a `Runtime.enable`, que introduce micro-desfases en V8 que el código JS de Cloudflare mide con alta precisión.
3. **Capa de Fingerprinting:** Comprueba coherencia entre GPU (WebGL), Canvas rendering, AudioContext, lista de fuentes instaladas, WebRTC y resolución de pantalla.
4. **Capa de Comportamiento:** Análisis de movimiento de ratón, foco de ventana y velocidad de interacción.

---

## 3. Catálogo Completo de Alternativas en GitHub

A continuación se presentan las mejores opciones activas y probadas por la comunidad de scraping, organizadas por arquitectura:

---

### Categoría A: Motores Parcheados en C++ a Nivel de Engine (Evasión Máxima)

#### 1. [Camoufox](https://github.com/daijro/camoufox) ⭐ (Recomendación Principal)
* **Lenguaje / Base:** C++ / Firefox Gecko (con cliente Python).
* **Cómo funciona:** Es un fork de Firefox donde toda la ofuscación de huellas digitales (Canvas, WebGL, fuentes, WebRTC, OS fingerprint, Audio) se realiza directamente en el código fuente de C++ del motor, en lugar de inyectar scripts en JavaScript (los cuales Cloudflare detecta revisando `Function.prototype.toString`).
* **Ventajas:**
  * Al ser Firefox real, ejecuta 100% de Next.js, React, Webpack y Cloudflare Turnstile.
  * Diseñado como sustituto directo (drop-in) de Playwright (`from camoufox.sync_api import Camoufox`).
  * Sin fugas de CDP de Chrome (funciona bajo el protocolo de Firefox).
  * Incluye emulación de geolocalización, zona horaria y WebRTC emparejada automáticamente con la IP del proxy.
* **Desventajas:** La descarga inicial del binario customizado de Firefox pesa ~150 MB.

#### 2. [Scrapling](https://github.com/D4Vinci/Scrapling)
* **Lenguaje:** Python.
* **Cómo funciona:** Framework moderno de scraping y parsing adaptativo. Incorpora internamente a Camoufox como motor bajo su clase `StealthyFetcher`.
* **Ventajas:** Combina la evasión indetectable de Camoufox con un motor de parsing CSS/XPath hasta 8 veces más rápido que BeautifulSoup.
* **Ideal para:** Pipelines de extracción rápida de datos donde se requiere saltar Cloudflare sin configurar Playwright manualmente.

---

### Categoría B: Automatización Chromium Hardened (Sin WebDriver / Sin Fugas CDP)

#### 3. [Patchright](https://github.com/Kaliiiiiiiiii/Patchright) & [Rebrowser-Patches](https://github.com/rebrowser/rebrowser-patches) ⭐
* **Lenguaje:** TypeScript / Python (parches sobre Playwright).
* **Cómo funciona:** Reemplazo directo de Playwright que soluciona la fuga fundamental descubierta en 2024-2025: la fuga de `Runtime.enable` en el protocolo CDP de Chrome.
* **Ventajas:**
  * **Cero cambios de código:** Se reemplaza `from playwright.sync_api import sync_playwright` por `from patchright.sync_api import sync_playwright`.
  * Utiliza Google Chrome o Chromium real del sistema.
  * Ejecuta React/Next.js a la perfección y mantiene cookies y sesiones intactas.
  * Resuelve la detección de Cloudflare en modo headless.

#### 4. [SeleniumBase (UC Mode)](https://github.com/seleniumbase/SeleniumBase)
* **Lenguaje:** Python.
* **Cómo funciona:** Framework maduro con el modo **UC Mode** (Undetected-Chromedriver Mode). Inicializa Chrome de forma independiente y se acopla después para que el navegador jamás exponga flags de automatización.
* **Ventajas:**
  * Soporte específico para resolver Cloudflare Turnstile automáticamente con funciones auxiliares (`sb.uc_gui_click_captcha()`).
  * Gran comunidad y mantenimiento continuo frente a cambios de Cloudflare.
* **Desventajas:** Basado en Selenium; suele requerir modo con interfaz visible ("headed") o pantalla virtual (`xvfb`) para máxima efectividad.

#### 5. [Nodriver](https://github.com/ultrafunkamsterdam/nodriver)
* **Lenguaje:** Python (`asyncio`).
* **Cómo funciona:** Del mismo creador de `undetected-chromedriver`. Es una reimplementación moderna que **elimina WebDriver por completo**. Se comunica con Chrome directamente mediante WebSockets asíncronos nativos.
* **Ventajas:**
  * Extremadamente rápido y ligero al no tener la sobrecarga de un driver intermedio.
  * Diseñado desde cero para ser 100% asíncrono, lo que encaja de forma natural con arquitecturas como `monitor_v2`.
* **Desventajas:** Su API difiere de Playwright/Selenium; requiere adaptar los métodos de interacción.

#### 6. [DrissionPage](https://github.com/g1879/DrissionPage)
* **Lenguaje:** Python.
* **Cómo funciona:** Proyecto con más de 12k estrellas en GitHub que fusiona el control directo de Chromium (vía CDP) con un cliente de peticiones HTTP en una sola interfaz fluida.
* **Ventajas:**
  * Permite cambiar sin problemas entre navegación visual (para saltar la protección) y modo de requests puro (para extraer los JSONs de la API a alta velocidad compartiendo cookies).
  * No utiliza Selenium ni WebDriver.

---

### Categoría C: Ecosistema Rust (Rendimiento, Concurrencia y Red)

#### 7. [rquest](https://github.com/0x676e67/rquest) ⭐ (El estándar actual en Rust)
* **Lenguaje:** Rust.
* **Cómo funciona:** Cliente HTTP y WebSocket construido sobre BoringSSL diseñado específicamente para **suplantar huellas TLS y HTTP/2**.
* **Ventajas:**
  * Emula fielmente handshakes de Chrome 133, Firefox y Safari (JA3/JA4, ciphers, ALPN, curvas y cabeceras H2).
  * Consumo mínimo de memoria (<5 MB) y velocidad de nivel nativo.
  * **Casos de uso:** Si una sesión (cookies) se obtiene una vez mediante un navegador, `rquest` puede consultar las APIs protegidas de SofaScore millones de veces sin ser bloqueado por Cloudflare WAF, eliminando por completo la necesidad de mantener un navegador abierto.

#### 8. [Chromiumoxide Stealth](https://crates.io/crates/chromiumoxide_stealth) / [Chaser-Oxide](https://github.com/0xchasercat/chaser-oxide)
* **Lenguaje:** Rust.
* **Cómo funciona:** Integraciones y forks sobre `chromiumoxide` (el cliente CDP de Rust para Chromium) que inyectan evasión en el transporte de red y eliminan banderas de automatización en `DocumentCreated`.
* **Desventajas:** Requiere tener Chrome/Chromium instalado; no es un motor autónomo.

#### 9. [BrowserOxide](https://github.com/yfedoseev/browser_oxide) & [Eoka](https://docs.rs/eoka)
* **Lenguaje:** Rust.
* **Estado:** Proyectos experimentales/investigativos que intentan crear un motor nativo o una capa mínima sobre V8 (`deno_core`) y BoringSSL.
* **Limitación:** Sufren de la misma barrera arquitectural que Obscura al carecer de un motor DOM completo para frameworks SPA modernos.

---

### Categoría D: Clientes HTTP con Suplantación TLS en Python (Sin Navegador)

#### 10. [curl_cffi](https://github.com/lexiforest/curl_cffi) ⭐
* **Lenguaje:** Python (enlaces CFFI sobre `curl-impersonate`).
* **Cómo funciona:** Extensión de `libcurl` que altera la negociación TLS para suplantar los handshakes de Chrome, Safari y Firefox.
* **Ventajas:**
  * Sintaxis idéntica a `requests` (`requests.get("...", impersonate="chrome124")`).
  * Soporta sesiones asíncronas (`AsyncSession`).
  * Velocidad 10 a 20 veces superior a cualquier navegador.

#### 11. [tls-client](https://github.com/bogdanfinn/tls-client)
* **Lenguaje:** Go (con wrappers para Python, NodeJS, etc.).
* **Cómo funciona:** Cliente HTTP especializado en evasión de TLS/JA3/JA4. Muy popular en bots de monitorización de zapatillas y compras automáticas donde la velocidad es crítica.

---

### Categoría E: Navegadores para Agentes Autónomos e Infraestructura

#### 12. [Steel Browser](https://github.com/steel-dev/steel-browser)
* **Lenguaje:** TypeScript / Docker / Rust.
* **Cómo funciona:** Navegador open-source diseñado específicamente para agentes de IA y scraping avanzado con gestión de sesiones, proxies integrados y stealth incorporado.

#### 13. [Crawlee (Python)](https://github.com/apify/crawlee-python)
* **Lenguaje:** Python (creado por Apify).
* **Cómo funciona:** Suite integral de scraping que maneja colas, proxies, reintentos y soporte nativo para `Camoufox`, `Playwright` y `BeautifulSoup`.

---

## 4. Matriz Comparativa General

| Herramienta | Base Tecnológica | Nivel de Evasión (Cloudflare/Turnstile) | Soporte SPA (Next.js / React) | Compartición de Cookies Navegador $\leftrightarrow$ API | Impacto de Integración en `pulpa` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Obscura** | Rust (`html5ever`+`v8`) | ❌ Falla en APIs (403) | ❌ Incompatible | ❌ Roto | Ya integrado pero inviable |
| **Camoufox** | Firefox Gecko (C++) | 🟢 Excelente (Parche C++) | ✅ 100% Nativo | ✅ Nativo | **Mínimo** (Drop-in Playwright) |
| **Patchright / Rebrowser** | Chromium (Playwright Patched) | 🟢 Muy Alto (Fix CDP Leaks) | ✅ 100% Nativo | ✅ Nativo | **Cero** (Cambio de import) |
| **Nodriver** | Chromium (Pure Async CDP) | 🟢 Alto (Sin WebDriver) | ✅ 100% Nativo | ✅ Nativo | Medio (Requiere refactor a su API) |
| **SeleniumBase UC** | Chromium (Selenium Driverless) | 🟢 Alto (Auto-Turnstile) | ✅ 100% Nativo | ✅ Vía Driver | Medio (API de SeleniumBase) |
| **DrissionPage** | Chromium (CDP + Requests) | 🟡 Bueno | ✅ 100% Nativo | ✅ Integrado | Medio |
| **curl_cffi / rquest** | C/BoringSSL (Sin Browser) | 🟢 Excelente en WAF (Red) | ❌ No renderiza DOM | Manual (Pasa cabeceras Cookie) | Bajo (Excelente para endpoints directos) |

---

## 5. Estrategia y Recomendaciones para `pulpa`

Para resolver el scraping de SofaScore sin consumir recursos excesivos ni sufrir bloqueos por Cloudflare, se recomienda adoptar un **enfoque por etapas**:

### Recomendación Inmediata (Drop-in Replacement sin romper nada)
1. **Adoptar [Camoufox](https://github.com/daijro/camoufox) o [Patchright](https://github.com/Kaliiiiiiiiii/Patchright):**
   - Ambos respetan la interfaz de Playwright que ya está programada en [`match/scraper.py`](file:///C:/Users/App/Desktop/pulpa/match/scraper.py).
   - Camoufox es la opción más resistente a largo plazo al evitar los detectores específicos de Chromium.
   - Patchright permite seguir usando el Chrome existente pero silenciando la fuga `Runtime.enable`.

### Recomendación de Alto Rendimiento (Híbrido Browser + TLS Client)
2. **Arquitectura Híbrida (Navegador Ligero para Calentamiento + `curl_cffi` / `rquest` para APIs):**
   - Usar **Camoufox** únicamente para la sonda inicial (`warmup`), resolviendo el challenge y extrayendo las cookies de sesión una vez.
   - Pasar ese CookieJar a **`curl_cffi`** (o **`rquest`** en Rust) para descargar los JSONs de partidos e incidentes (`api.sofascore.com/api/v1/event/...`).
   - **Beneficio:** Velocidad de milisegundos por partido, cero consumo de memoria continuo de navegadores abiertos y bypass total de Cloudflare WAF.
