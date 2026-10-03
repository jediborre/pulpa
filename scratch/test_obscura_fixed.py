"""Test rápido del fix de Obscura - cookies cross-origin"""
import subprocess
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
OBSCURA_EXE = ROOT / "tools" / "obscura" / "v0.1.5" / "obscura-fixed.exe"
OBSCURA_DIR = ROOT / "tools" / "obscura" / "v0.1.5"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"

os.environ["SSL_CERT_FILE"] = str(CERT_PATH)
sys.stdout.reconfigure(line_buffering=True)

# Kill existing obscura
subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["taskkill", "/IM", "obscura-fixed.exe", "/F"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

print(f"[1] Iniciando Obscura FIXED desde {OBSCURA_EXE}...", flush=True)
proc = subprocess.Popen(
    [str(OBSCURA_EXE), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR),
    env=os.environ,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(4)
print(f"[1] PID={proc.pid}", flush=True)

from playwright.sync_api import sync_playwright

print("[2] Conectando Playwright...", flush=True)
p = sync_playwright().start()
browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
ctx = browser.contexts[0] if browser.contexts else browser.new_context()
page = ctx.new_page()

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
print(f"[3] Navegando a sofascore.com/basketball/{yesterday}...", flush=True)
page.goto(f'https://www.sofascore.com/basketball/{yesterday}',
          timeout=20000, wait_until='commit')
time.sleep(3)
print(f"[4] Title: {page.title()}", flush=True)

api_url = f'https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{yesterday}'

print("[5] Iniciando fetch con variable global...", flush=True)
page.evaluate(f"""
window.__r = null;
window.__d = false;
fetch('{api_url}', {{
    headers: {{ 'Accept': 'application/json', 'Referer': 'https://www.sofascore.com/' }}
}})
.then(r => r.ok ? r.json() : Promise.reject('HTTP ' + r.status))
.then(d => {{ window.__r = {{ total: d.events?.length, first: d.events?.[0]?.id }}; window.__d = true; }})
.catch(e => {{ window.__r = {{ error: String(e) }}; window.__d = true; }});
""")

print("[6] Esperando fetch...", flush=True)
try:
    page.wait_for_function("window.__d === true", timeout=15000)
    result = page.evaluate("window.__r")
    print(f"[7] Result: {json.dumps(result, indent=2, ensure_ascii=False)}", flush=True)
except Exception as e:
    print(f"[6] Error: {e}", flush=True)

print("[8] Cleanup...", flush=True)
page.close()
ctx.close()
browser.close()
p.stop()
proc.terminate()
try:
    proc.wait(timeout=3)
except:
    proc.kill()
print("[DONE]", flush=True)
