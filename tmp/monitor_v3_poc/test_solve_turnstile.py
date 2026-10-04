"""
PoC: Auto-resolución y Click en Widget de Cloudflare Turnstile
Ubicación: tmp/monitor_v3_poc/test_solve_turnstile.py
Propósito:
    1. Navegar a captcha.html usando Camoufox.
    2. Localizar el iframe de challenges.cloudflare.com.
    3. Hacer click en la casilla de verificación de Turnstile.
    4. Esperar a que Cloudflare valide y emita la cookie cf_clearance.
    5. Extraer cf_clearance y probar la llamada a la API.
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from camoufox.sync_api import Camoufox
from camoufox.pkgman import launch_path

real_exe = str(Path(launch_path()).resolve())

MATCH_ID = "15935071"
CAPTCHA_URL = f"https://www.sofascore.com/captcha.html?redirectUrl=https%3A%2F%2Fwww.sofascore.com%2Fevent%2F{MATCH_ID}"

print("=" * 75)
print("   PoC MONITOR V3: RESOLUCIÓN INTERACTIVA TURNSTILE CON CAMOUFOX")
print(f"   URL: {CAPTCHA_URL}")
print("=" * 75)

with Camoufox(executable_path=real_exe, headless=True) as browser:
    page = browser.new_page()
    print("Navegando a captcha.html...")
    page.goto(CAPTCHA_URL, wait_until="load", timeout=30000)
    
    print(f"URL actual: {page.url} | Título: {page.title()}")
    
    # Esperar a que el iframe de Turnstile aparezca
    print("Esperando iframe de Cloudflare Turnstile...")
    time.sleep(3)
    
    turnstile_frame = None
    for frame in page.frames:
        if "challenges.cloudflare.com" in frame.url:
            turnstile_frame = frame
            break
            
    if turnstile_frame:
        print(f"¡Iframe Turnstile encontrado!: {turnstile_frame.url[:80]}...")
        try:
            # Buscar el checkbox o contenedor dentro del iframe
            print("Buscando selector del checkbox...")
            checkbox = turnstile_frame.wait_for_selector("input[type=checkbox], #challenge-stage, .ctp-checkbox-label", timeout=5000)
            if checkbox:
                print("Haciendo click en la casilla de Turnstile...")
                checkbox.click()
            else:
                print("No se encontró selector de checkbox directo, intentando click en centro del iframe...")
                # Click en el centro del iframe
                box = turnstile_frame.frame_element().bounding_box()
                if box:
                    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        except Exception as e:
            print(f"Error al interactuar con Turnstile: {e}")
    else:
        print("No se encontró iframe de Turnstile.")
        
    print("\nEsperando 6s para procesamiento de Cloudflare...")
    time.sleep(6)
    
    print(f"URL tras resolución: {page.url}")
    print(f"Título tras resolución: {page.title()}")
    
    cookies = page.context.cookies()
    print(f"\nCookies obtenidas ({len(cookies)}):")
    cf_clearance_found = False
    for c in cookies:
        print(f"  * {c['name']} = {c['value'][:30]}... (domain: {c['domain']})")
        if c['name'] == 'cf_clearance':
            cf_clearance_found = True
            print("  >>> ¡ENCONTRADA COOKIE CF_CLEARANCE! <<<")
            
    if cf_clearance_found or len(cookies) > 0:
        # Intentar fetch in-page
        print("\nProbando fetch in-page tras resolución:")
        js = f"fetch('https://api.sofascore.com/api/v1/event/{MATCH_ID}').then(r => r.json()).then(d => ({{ ok: true, data: d }})).catch(e => ({{ ok: false, error: e.toString() }}))"
        res = page.evaluate(js)
        print(f"Resultado fetch in-page: {res.get('ok')}")
        if res.get("ok"):
            print("¡ÉXITO TOTAL! Datos de evento obtenidos correctamente.")
