"""
PoC: Cosechador de Sesión (Session Harvester) con Playwright Chrome -> Paso a HTTP Client
Ubicación: tmp/monitor_v3_poc/test_chrome_harvester.py
Propósito:
    1. Abrir una sesión de Chrome con Playwright para visitar una página de evento.
    2. Interceptar y capturar las cabeceras exactas y las cookies de sesión generadas.
    3. Realizar una petición in-page con fetch() para verificar que da 200 OK.
    4. Extraer el CookieJar y User-Agent de Chrome.
    5. Cerrar Chrome por completo (liberando RAM).
    6. Intentar consultar la API directamente desde Python (httpx) usando esas cookies cosechadas.

Hipótesis:
    El bloqueo 403 (Varnish) se debe a la ausencia de cookies de sesión / tokens que
    establece el navegador al cargar la página principal. Si un Harvester ligero obtiene
    las cookies una vez, un cliente HTTP puro puede hacer las peticiones subsiguientes
    a velocidad luz sin necesidad de mantener el navegador abierto.
"""

import sys
import time
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import httpx
from playwright.sync_api import sync_playwright

MATCH_ID = "15935071" # Knicks vs Spurs
EVENT_URL = f"https://www.sofascore.com/event/{MATCH_ID}"
API_URL = f"https://api.sofascore.com/api/v1/event/{MATCH_ID}"

def run_poc():
    print("=" * 75)
    print("   PoC MONITOR V3: SESSION HARVESTER -> HTTP CLIENT")
    print(f"   Match ID: {MATCH_ID}")
    print(f"   Event URL: {EVENT_URL}")
    print("=" * 75)

    captured_cookies = {}
    user_agent = ""
    in_page_data = None
    
    print("\n[PASO 1] Iniciando Chrome Headless para cosechar credenciales...")
    start_browser_time = time.perf_counter()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        print(f"  -> Navegando a {EVENT_URL}...")
        nav_start = time.perf_counter()
        page.goto(EVENT_URL, timeout=30000, wait_until="domcontentloaded")
        print(f"  -> Página cargada en {(time.perf_counter() - nav_start):.2f}s")
        
        # Breve espera para que los scripts establezcan cookies
        time.sleep(2)

        # Probar fetch in-page (como lo hace monitor_v2)
        print("  -> Ejecutando fetch() in-page dentro de Chrome...")
        js_code = f"""
        async () => {{
            try {{
                const res = await fetch('{API_URL}');
                const json = await res.json();
                return {{ status: res.status, ok: res.ok, data: json }};
            }} catch (err) {{
                return {{ status: 0, ok: false, error: err.toString() }};
            }}
        }}
        """
        in_page_data = page.evaluate(js_code)
        print(f"  -> Resultado in-page: Status {in_page_data.get('status')}, OK={in_page_data.get('ok')}")

        # Extraer cookies del contexto
        raw_cookies = context.cookies()
        user_agent = page.evaluate("navigator.userAgent")
        for c in raw_cookies:
            captured_cookies[c["name"]] = c["value"]

        print(f"  -> Cookies cosechadas: {len(captured_cookies)} cookies")
        for k, v in captured_cookies.items():
            print(f"     * {k}: {v[:30]}..." if len(v) > 30 else f"     * {k}: {v}")

        # Cerrar navegador completamente
        context.close()
        browser.close()
        
    browser_elapsed = time.perf_counter() - start_browser_time
    print(f"[PASO 1 COMPLETO] Navegador cerrado en {browser_elapsed:.2f}s. RAM liberada.")

    print("\n" + "=" * 75)
    print("[PASO 2] Probando Petición Directa con httpx (SIN NAVEGADOR) usando cookies")
    print("=" * 75)
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Referer": EVENT_URL,
        "Origin": "https://www.sofascore.com",
        "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    # Probar con cliente HTTP directo
    with httpx.Client(cookies=captured_cookies, headers=headers) as client:
        test_urls = [
            ("Snapshot Evento", API_URL),
            ("Incidents PBP", f"https://api.sofascore.com/api/v1/event/{MATCH_ID}/incidents"),
            ("Graph Momentum", f"https://api.sofascore.com/api/v1/event/{MATCH_ID}/graph"),
        ]

        for label, url in test_urls:
            t0 = time.perf_counter()
            resp = client.get(url, timeout=10.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            if resp.status_code == 200:
                print(f"  [OK 200] {label:<16} | {elapsed_ms:>6.1f}ms | {len(resp.content)} bytes (JSON)")
            else:
                print(f"  [{resp.status_code}]   {label:<16} | {elapsed_ms:>6.1f}ms | {resp.text[:100]}")

    print("\n" + "=" * 75)
    print("   Fin de la Prueba de Cosechador de Sesión.")
    print("=" * 75)

if __name__ == "__main__":
    run_poc()
