"""
PoC: Inspección del HTML y Widgets en captcha.html
Ubicación: tmp/monitor_v3_poc/inspect_captcha_page.py
Propósito:
    Analizar qué tipo de reto implementa SofaScore en captcha.html:
    - ¿Es Cloudflare Turnstile?
    - ¿Tiene iframe o script de challenges.cloudflare.com?
    - ¿Se resuelve automáticamente esperando unos segundos o requiere interacción?
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from camoufox.sync_api import Camoufox
from camoufox.pkgman import launch_path

real_exe = str(Path(launch_path()).resolve())

with Camoufox(executable_path=real_exe, headless=True) as browser:
    page = browser.new_page()
    url = "https://www.sofascore.com/captcha.html?redirectUrl=https%3A%2F%2Fwww.sofascore.com%2F"
    print(f"Navegando a {url}...")
    page.goto(url, wait_until="networkidle", timeout=30000)
    
    print(f"Título: {page.title()}")
    print(f"URL actual: {page.url}")
    
    # Esperar 5 segundos para ver si el reto se auto-resuelve
    print("Esperando 5s para observar auto-resolución del reto...")
    time.sleep(5)
    print(f"URL tras 5s: {page.url}")
    
    content = page.content()
    print(f"Longitud de contenido: {len(content)} caracteres")
    
    # Buscar patrones conocidos
    has_turnstile = "challenges.cloudflare.com" in content or "turnstile" in content.lower()
    has_recaptcha = "google.com/recaptcha" in content
    has_hcaptcha = "hcaptcha.com" in content
    
    print(f"¿Tiene Turnstile?: {has_turnstile}")
    print(f"¿Tiene reCAPTCHA?: {has_recaptcha}")
    print(f"¿Tiene hCaptcha?: {has_hcaptcha}")
    
    # Listar iframes en la página
    frames = page.frames
    print(f"Total iframes: {len(frames)}")
    for f in frames:
        print(f"  Frame URL: {f.url}")
        
    cookies = page.context.cookies()
    print(f"Cookies generadas tras espera ({len(cookies)}):")
    for c in cookies:
        print(f"  * {c['name']} = {c['value'][:30]}...")
