"""
PoC: Inspección de Página y Cookies con Playwright Chrome
Ubicación: tmp/monitor_v3_poc/inspect_page_content.py
Propósito:
    Comprobar qué ve realmente Chrome al navegar a SofaScore:
    - Título de la página
    - Contenido del DOM (¿pasa Cloudflare o se queda en reto?)
    - Cookies en context y cookies en page
    - Ejecutar fetch_event_snapshot directamente desde match.scraper para comparar
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from match.scraper import fetch_event_snapshot, STANDARD_UA
from playwright.sync_api import sync_playwright

MATCH_ID = "15935071"

print(f"Probando fetch_event_snapshot({MATCH_ID}) desde match.scraper:")
try:
    snap = fetch_event_snapshot(MATCH_ID, backend="chrome")
    print(f"[OK DIRECTO DE SCRAPER.PY] Snapshot: {snap}")
except Exception as e:
    print(f"[ERROR DIRECTO DE SCRAPER.PY]: {e}")

print("\nInspeccionando DOM y cookies detalladas:")
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    
    url = f"https://www.sofascore.com/event/{MATCH_ID}"
    print(f"Navegando a {url}...")
    page.goto(url, wait_until="domcontentloaded", timeout=15000)
    
    print(f"Page Title: {page.title()}")
    print(f"Current URL: {page.url}")
    
    cookies = ctx.cookies()
    print(f"Total Cookies en Context: {len(cookies)}")
    for c in cookies:
        print(f"  Cookie: {c['name']} (domain: {c['domain']}) = {c['value'][:25]}...")
        
    # Verificar si Cloudflare está en el DOM
    content = page.content()
    if "Just a moment" in content or "Cloudflare" in content or "Turnstile" in content:
        print("[ALERTA] Reto de Cloudflare detectado en la página!")
    else:
        print("[INFO] No se detectó texto evidente de Cloudflare en el DOM inicial.")
        
    ctx.close()
    browser.close()
