"""
PoC: Test de Warmup en Home Page de SofaScore y Ejecución de Fetch
Ubicación: tmp/monitor_v3_poc/test_home_fetch.py
Propósito:
    Verificar si navegando a la página principal (https://www.sofascore.com) se establece
    la sesión limpia sin reto captcha, permitiendo consultar endpoints de la API (eventos, calendario).
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

MATCH_ID = "15935071"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    )
    page = ctx.new_page()
    
    print("Navegando a https://www.sofascore.com/ ...")
    t0 = time.perf_counter()
    page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=15000)
    print(f"Home cargada en {(time.perf_counter() - t0):.2f}s | URL: {page.url} | Title: {page.title()}")
    
    # Evaluar fetch in-page
    print("\nEjecutando fetch a la API de evento...")
    js_code = f"""
    async () => {{
        try {{
            const r = await fetch('https://api.sofascore.com/api/v1/event/{MATCH_ID}');
            const txt = await r.text();
            return {{ status: r.status, ok: r.ok, length: txt.length, preview: txt.slice(0, 150) }};
        }} catch(e) {{
            return {{ status: -1, ok: false, error: e.toString() }};
        }}
    }}
    """
    res = page.evaluate(js_code)
    print(f"Resultado Evento: Status {res.get('status')}, OK={res.get('ok')}, Len={res.get('length')}")
    print(f"Preview: {res.get('preview')}")
    
    # Evaluar fetch al calendario
    print("\nEjecutando fetch al calendario de hoy...")
    today_str = time.strftime("%Y-%m-%d")
    js_cal = f"""
    async () => {{
        try {{
            const r = await fetch('https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{today_str}');
            const txt = await r.text();
            return {{ status: r.status, ok: r.ok, length: txt.length, preview: txt.slice(0, 150) }};
        }} catch(e) {{
            return {{ status: -1, ok: false, error: e.toString() }};
        }}
    }}
    """
    res_cal = page.evaluate(js_cal)
    print(f"Resultado Calendario ({today_str}): Status {res_cal.get('status')}, OK={res_cal.get('ok')}, Len={res_cal.get('length')}")
    print(f"Preview: {res_cal.get('preview')}")
    
    # Extraer cookies
    cookies = ctx.cookies()
    print(f"\nCookies obtenidas ({len(cookies)}):")
    cookie_dict = {}
    for c in cookies:
        print(f"  * {c['name']} ({c['domain']}) = {c['value'][:30]}...")
        cookie_dict[c['name']] = c['value']
        
    ctx.close()
    browser.close()

# Ahora probar esas cookies con curl_cffi afuera del navegador
print("\n" + "=" * 60)
print("PROBANDO CON CURL_CFFI USANDO LAS COOKIES COSECHADAS:")
print("=" * 60)

try:
    from curl_cffi import requests as c_req
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }
    
    resp_curl = c_req.get(
        f"https://api.sofascore.com/api/v1/event/{MATCH_ID}",
        cookies=cookie_dict,
        headers=headers,
        impersonate="chrome124",
        timeout=10
    )
    print(f"curl_cffi Status: {resp_curl.status_code}, Len: {len(resp_curl.content)}")
    print(f"curl_cffi Body preview: {resp_curl.text[:150]}")
except Exception as exc:
    print(f"curl_cffi error: {exc}")
