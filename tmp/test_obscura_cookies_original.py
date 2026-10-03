"""
Ubicación original: scratch/test_obscura_cookies_original.py
Propósito / Qué hacía:
Prueba de cookies con el binario original de Obscura antes del parche.
"""

"""Test cookies con binario ORIGINAL"""
import subprocess, time, os, sys, json, socket
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
os.environ["SSL_CERT_FILE"] = str(CERT_PATH)

subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

proc = subprocess.Popen(
    [str(ROOT / "tools" / "obscura" / "v0.1.5" / "obscura.exe"), "serve", "--port", "9222", "--stealth"],
    cwd=str(ROOT / "tools" / "obscura" / "v0.1.5"), env=os.environ,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(4)

from playwright.sync_api import sync_playwright
p = sync_playwright().start()
browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=8000)
page = browser.contexts[0].new_page()

page.goto('https://www.sofascore.com/basketball', timeout=15000, wait_until='commit')
time.sleep(3)
print(f"Title: {page.title()}", flush=True)

cookies = browser.contexts[0].cookies()
print(f"Cookies (Playwright): {len(cookies)}", flush=True)
for c in cookies:
    print(f"  {c['name']}={c['value']} domain={c['domain']} path={c['path']} httponly={c.get('httpOnly')}", flush=True)

dc = page.evaluate("document.cookie")
print(f"document.cookie: '{dc}'", flush=True)

page.close()
browser.close()
p.stop()
proc.terminate()
try: proc.wait(timeout=3)
except: proc.kill()
print("DONE", flush=True)
