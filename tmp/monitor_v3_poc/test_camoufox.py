"""
PoC: Test de Navegación y Evasión con Camoufox (Firefox Gecko C++ Anti-detect)
Ubicación: tmp/monitor_v3_poc/test_camoufox.py
Propósito:
    1. Lanzar Camoufox en modo headless.
    2. Navegar a SofaScore (evento y home).
    3. Verificar si salta el reto de captcha.html o si carga la página real.
    4. Probar fetch in-page a la API de SofaScore.
    5. Cosechar cookies y probarlas afuera con curl_cffi.
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from camoufox.sync_api import Camoufox

MATCH_ID = "15935071"
EVENT_URL = f"https://www.sofascore.com/event/{MATCH_ID}"
API_URL = f"https://api.sofascore.com/api/v1/event/{MATCH_ID}"

print("=" * 75)
print("   PoC MONITOR V3: CAMOUFOX ANTI-DETECT BROWSER TEST")
print(f"   Target URL: {EVENT_URL}")
print("=" * 75)

start_time = time.perf_counter()

from camoufox.pkgman import launch_path

real_exe = str(Path(launch_path()).resolve())
print(f"Ruta ejecutable Camoufox: {real_exe}")

with Camoufox(executable_path=real_exe, headless=True) as browser:
    print(f"Camoufox iniciado en {(time.perf_counter() - start_time):.2f}s")
    page = browser.new_page()
    
    print(f"\n[1] Navegando a {EVENT_URL}...")
    t0 = time.perf_counter()
    page.goto(EVENT_URL, timeout=30000)
    print(f"    Cargado en {(time.perf_counter() - t0):.2f}s")
    print(f"    URL Final: {page.url}")
    print(f"    Page Title: {page.title()}")
    
    # Verificar si fue redirigido a captcha.html
    if "captcha.html" in page.url:
        print("    [RESULTADO] Redirigido a captcha.html")
    else:
        print("    [RESULTADO] ¡NO FUE REDIRIGIDO A CAPTCHA! Página real alcanzada.")

    # Ejecutar fetch in-page
    print("\n[2] Ejecutando fetch() in-page...")
    js_code = f"""
    async () => {{
        try {{
            const r = await fetch('{API_URL}');
            const txt = await r.text();
            return {{ status: r.status, ok: r.ok, length: txt.length, preview: txt.slice(0, 150) }};
        }} catch(e) {{
            return {{ status: -1, ok: false, error: e.toString() }};
        }}
    }}
    """
    res = page.evaluate(js_code)
    print(f"    Fetch Status: {res.get('status')}, OK={res.get('ok')}, Len={res.get('length')}")
    print(f"    Preview: {res.get('preview')}")
    
    # Extraer cookies
    cookies = page.context.cookies()
    print(f"\n[3] Cookies en Camoufox ({len(cookies)}):")
    cookie_dict = {}
    for c in cookies:
        print(f"    * {c['name']} ({c['domain']}) = {c['value'][:30]}...")
        cookie_dict[c['name']] = c['value']

print("\n" + "=" * 75)
print(f"   Prueba finalizada en {(time.perf_counter() - start_time):.2f}s total.")
print("=" * 75)
