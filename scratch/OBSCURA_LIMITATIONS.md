# Obscura - Limitaciones para Descarga de Días Anteriores

## Resumen

Después de múltiples pruebas, **Obscura no es compatible** con el flujo de descarga de partidos de días anteriores (opción 5 del menu.bat).

## Problemas Encontrados

### 1. Modo `obscura fetch` directo
- **Problema**: Devuelve 403 Forbidden en las APIs de SofaScore
- **Causa**: No mantiene sesión/cookies entre diferentes llamadas
- **Ejemplo**:
  ```bash
  obscura fetch "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-05-28" --stealth
  # Resultado: {"error": {"code": 403, "reason": "Forbidden"}}
  ```

### 2. Modo `obscura serve` + Playwright
- **Problema**: Las cookies no se comparten correctamente con el contexto de Playwright
- **Causa**: Obscura no expone las cookies de sesión al CDP (Chrome DevTools Protocol)
- **Ejemplo**:
  ```python
  browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222')
  page.goto('https://www.sofascore.com/basketball/2026-05-28')
  resp = page.request.get('https://api.sofascore.com/api/v1/...')
  # Resultado: Status 403
  ```

### 3. `obscura fetch` con `--eval`
- **Problema**: Solo ejecuta código JavaScript síncrono
- **Causa**: No soporta promesas/async, y las llamadas a APIs son asíncronas
- **Intentos fallidos**:
  - `async/await`: Devuelve `None` o `{}`
  - `XMLHttpRequest` síncrono: Bloqueado por el navegador moderno
  - Variables globales + `wait_for_function`: Se cuelga indefinidamente

## Qué SÍ Funciona con Obscura

✅ Navegar a páginas públicas de SofaScore:
```bash
obscura fetch "https://www.sofascore.com/basketball" --stealth --wait 10
```

✅ Extraer datos del DOM con `--eval` síncrono:
```bash
obscura fetch "https://www.sofascore.com/basketball" --dump text -e "document.title"
# Resultado: "Basketball live results & schedule | Sofascore"
```

❌ Acceder a APIs que requieren sesión/cookies

## Solución Actual

El script `scraper.py` usa **Chrome headless** directamente, que:
- Mantiene sesión correctamente
- Comparte cookies entre navegación y API calls
- Soporta código asíncrono

**No se recomienda reemplazar Chrome con Obscura** para este flujo.

## Alternativas Exploradas

1. **Usar `obscura fetch` con `--dump html` y parsear**: Los datos de partidos se cargan dinámicamente vía JavaScript, no están en el HTML inicial
2. **Aumentar `--wait` tiempo**: No ayuda, los datos siguen sin cargarse en el DOM
3. **Usar `__NEXT_DATA__`**: Solo contiene metadata, no los eventos/partidos

## Conclusión

Obscura es útil para:
- Scraping de páginas estáticas
- Verificar que una URL existe
- Extraer datos del DOM renderizado (si están disponibles síncronamente)

Obscura **NO** es viable para:
- APIs que requieren autenticación/cookies de sesión
- Flujos que requieren múltiples requests con sesión compartida
- Datos cargados dinámicamente vía JavaScript asíncrono

**Recomendación**: Continuar usando Chrome headless para la descarga de días anteriores.
