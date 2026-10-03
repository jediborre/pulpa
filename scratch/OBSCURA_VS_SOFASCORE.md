# Obscura vs SofaScore — Análisis y Limitaciones

## Binario Fixeado

**Ubicación:** `tools/obscura/v0.1.5/obscura-fixed.exe`

**Fixes aplicados sobre v0.1.5:**
1. **Cross-origin cookies** (`ops.rs`): Se eliminó `if !is_cross_origin` para que `fetch()` JS envíe cookies incluso a subdominios como `api.sofascore.com`
2. **Timeout de promesas** (`runtime.rs`): Se aumentó `resolve_promises` de 100ms → 30s para dar tiempo a que requests HTTP pesados se completen

---

## Qué partes SÍ funcionarían con Obscura

### 1. `obscura fetch` CLI (modo fetch)
```bash
obscura fetch URL --stealth --wait N --timeout M --dump text|html|links
```
✅ Funciona para páginas **estáticas** o con JS mínimo
✅ Verificado: obtiene el HTML de `sofascore.com/basketball`
✅ Verificado: extrae `document.title`, `__NEXT_DATA__`, links

### 2. Casos de uso donde Obscura es viable
- Scraping de APIs **sin autenticación** o con API key en URL
- Páginas con contenido en HTML estático (no renderizado por JS)
- Sitios que usan cookies via `Set-Cookie` HTTP (no via JS)
- Extracción de datos con `--eval` síncrono (`document.title`, `document.querySelector`, etc.)

---

## Qué partes NO funcionan y por qué

### 1. Navegación vía CDP (`obscura serve` + Playwright)

| Componente | Funciona? | Razón |
|---|---|---|
| `page.goto()` | ⚠️ Parcial | Navega pero no ejecuta JS complejo |
| `page.title()` | ✅ Sí | Viene en el HTML inicial |
| `page.evaluate(fetch())` | ❌ No | V8 no ejecuta Next.js/React |
| `ctx.request.get()` | ❌ No | Playwright usa su propio stack HTTP (sin cookies de Obscura) |
| `page.wait_for_function()` | ❌ No | El JS que espera nunca se ejecuta |

### 2. Flujo completo de descarga

| Función | Backend Chrome | Backend Obscura | Por qué falla Obscura |
|---|---|---|---|
| `fetch_finished_match_ids_for_date()` | ✅ OK | ❌ | Usa `ctx.request.get()` → no comparte cookies |
| `fetch_match_by_id()` | ✅ OK | ❌ | Usa `page.evaluate(fetch())` → JS no ejecuta |
| `fetch_event_snapshot()` | ✅ OK | ❌ | Ídem |
| `fetch_live_match_ids()` | ✅ OK | ❌ | Usa `ctx.request.get()` |

---

## Qué le falta a Obscura para funcionar con SofaScore

### Nivel 1: Bugs (fixeables en horas)

| Bug | Archivo | Impacto | Status |
|---|---|---|---|
| Cross-origin cookies bloqueadas | `obscura-js/src/ops.rs` | API returns 403 | ✅ FIXED |
| Timeout de promesas 100ms | `obscura-js/src/runtime.rs` | Promesas nunca resuelven | ✅ FIXED |
| `ctx.request.get()` no comparte cookies | Arquitectura CDP | APIs via Playwright fallan | ❌ Sin fix (arquitectural) |

### Nivel 2: Features faltantes (días-semanas)

| Feature | Necesaria para | Descripción |
|---|---|---|
| **Full Fetch API polyfill** | `page.evaluate(fetch())` | Obscura implementa `fetch()` via `op_fetch_url` pero le faltan features como AbortController, streaming, etc. |
| **XMLHttpRequest síncrono** | `page.evaluate()` síncrono | Permitiría extraer datos sin async/await |
| **Storage API** (localStorage, sessionStorage) | Sitios que usan auth tokens en storage | SofaScore podría usar localStorage para sesión |
| **Service Workers** | Sitios que cachean via SW | No implementado en Obscura |

### Nivel 3: Limitaciones arquitecturales (meses)

| Limitación | Descripción |
|---|---|
| **V8 mínimo sin DOM completo** | Obscura usa un parser HTML (html5ever) para construir el DOM, pero no ejecuta scripts de framework. React/Next.js requieren un DOM completo con todas las APIs del navegador (events, timers, fetch, MutationObserver, IntersectionObserver, etc.) |
| **Sin ejecución de JavaScript de página real** | Obscura ejecuta su propio JS (el que evaluás explícitamente), pero NO ejecuta los `<script>` tags de la página. Next.js funciona inyectando chunks de webpack que deben ejecutarse en orden. |
| **Cookie jar no sincronizado con Playwright** | Playwright tiene su propio stack HTTP que no usa el `CookieJar` de Obscura. Para sincronizarlos, habría que interceptar TODAS las requests de Playwright vía CDP Fetch domain y redirigirlas por el cliente HTTP de Obscura. |
| **Sin soporte para ES modules nativos** | Next.js usa dynamic imports (`import()`) que requieren un module loader. Obscura implementa uno básico pero puede fallar con bundles complejos. |

---

## Diagrama de flujo: qué pasa hoy vs qué debería pasar

### Flujo actual con Chrome (funciona)
```
Playwright → connect_over_cdp → chrome.exe
  │
  ├── page.goto("sofascore.com")
  │   └── Chrome ejecuta React/Next.js completo
  │       ├── Establece cookies (JS)
  │       ├── Renderiza DOM
  │       └── Carga datos de API
  │
  ├── ctx.request.get("api.sofascore.com/...")
  │   └── Chrome incluye cookies automáticamente ✅
  │
  └── page.evaluate(fetch("api.sofascore.com/..."))
      └── Chrome ejecuta fetch con cookies ✅
```

### Flujo con Obscura (hoy, falla)
```
Playwright → connect_over_cdp → obscura.exe (serve)
  │
  ├── page.goto("sofascore.com")
  │   └── Obscura navega pero NO ejecuta React/Next.js
  │       ├── Cookies NO se establecen (JS no corre)
  │       ├── DOM muestra "Loading..."
  │       └── Datos de API no se cargan
  │
  ├── ctx.request.get("api.sofascore.com/...")
  │   └── Playwright usa su propio HTTP (sin cookies) ❌
  │
  └── page.evaluate(fetch("api.sofascore.com/..."))
      └── op_fetch_url: sin cookies → 403 ❌
```

### Flujo con Obscura (ideal, requeriría los fixes arquitecturales)
```
Playwright → connect_over_cdp → obscura.exe (serve mejorado)
  │
  ├── page.goto("sofascore.com")
  │   └── Obscura ejecuta scripts de página (React/Next.js)
  │       ├── Cookies establecidas via op_set_cookie ✅
  │       ├── DOM renderizado completo ✅
  │       └── Datos cargados en JS global ✅
  │
  ├── ctx.request.get("api.sofascore.com/...")
  │   └── CDP Fetch domain intercepta → usa ObscuraHttpClient con CookieJar ✅
  │
  └── page.evaluate(fetch("api.sofascore.com/..."))
      └── op_fetch_url: cookies del CookieJar (ahora cross-origin) ✅
```

---

## Resumen

| Aspecto | Chrome | Obscura (hoy) | Obscura (con fixes mayores) |
|---|---|---|---|
| Ejecuta React/Next.js | ✅ | ❌ | ❌ (requeriría reescribir V8 → Chromium) |
| Cookies de sesión | ✅ | ❌ | ✅ (con interceptación CDP Fetch) |
| APIs con sesión | ✅ | ❌ (403) | ✅ (con fixes de cookies) |
| Páginas estáticas | ✅ | ✅ | ✅ |
| Consumo de RAM | ~200 MB | ~30 MB | ~30 MB |
| Velocidad de inicio | ~2s | ~0.1s | ~0.1s |
| Anti-detección | ❌ | ✅ (stealth) | ✅ |

**Conclusión:** Obscura es ideal para sitios simples sin JS pesado. Para SofaScore (Next.js/React), Chrome headless sigue siendo la única opción viable. Los fixes que aplicamos (cross-origin cookies + timeout de promesas) son correcciones correctas pero insuficientes ante la limitación fundamental de que Obscura no ejecuta los scripts de la página.
