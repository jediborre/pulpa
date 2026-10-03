"""Test sin stealth ni verbose"""
import subprocess, time, os, sys, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
OBSCURA_EXE = ROOT / "tools" / "obscura" / "v0.1.5" / "obscura-fixed.exe"
OBSCURA_DIR = ROOT / "tools" / "obscura" / "v0.1.5"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
os.environ["SSL_CERT_FILE"] = str(CERT_PATH)

for exe in ["obscura.exe", "obscura-fixed.exe"]:
    subprocess.run(["taskkill", "/IM", exe, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

print("[1] Starting Obscura (with stealth)...", flush=True)
proc = subprocess.Popen(
    [str(OBSCURA_EXE), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR),
    env=os.environ,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(4)

if proc.poll() is not None:
    out, err = proc.communicate()
    print(f"Process died! Code: {proc.returncode}", flush=True)
    print(f"STDERR: {err.decode('utf-8', errors='ignore')[:500]}", flush=True)
    exit(1)

import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
if sock.connect_ex(('127.0.0.1', 9222)) != 0:
    print("Port 9222 NOT listening!", flush=True)
    out, err = proc.communicate(timeout=2)
    print(f"STDERR: {err.decode('utf-8', errors='ignore')[:500]}", flush=True)
    sock.close()
    exit(1)
sock.close()
print("[2] Port 9222 OK", flush=True)

from playwright.sync_api import sync_playwright
p = sync_playwright().start()
try:
    browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=8000)
    print("[3] CDP connected", flush=True)
except Exception as e:
    print(f"[3] CDP error: {e}", flush=True)
    exit(1)

ctx = browser.contexts[0] if browser.contexts else browser.new_context()
page = ctx.new_page()

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
print(f"[4] Navigating...", flush=True)
try:
    page.goto(f'https://www.sofascore.com/basketball/{yesterday}', timeout=15000, wait_until='commit')
    print("[4] OK", flush=True)
except Exception as e:
    print(f"[4] Error: {e}", flush=True)
    exit(1)

time.sleep(2)
print(f"[5] Title: {page.title()}", flush=True)

api_url = f'https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{yesterday}'
print("[6] Fetch via page.evaluate...", flush=True)
page.evaluate(f"""
window.__r = null; window.__d = false;
fetch('{api_url}', {{ headers: {{ 'Accept': 'application/json', 'Referer': 'https://www.sofascore.com/' }} }})
.then(r => r.ok ? r.json() : Promise.reject('HTTP ' + r.status))
.then(d => {{ window.__r = {{ ok: true, count: d.events?.length }}; window.__d = true; }})
.catch(e => {{ window.__r = {{ ok: false, error: String(e) }}; window.__d = true; }})
""")

try:
    page.wait_for_function("window.__d === true", timeout=30000)
    result = page.evaluate("window.__r")
    print(f"[7] {json.dumps(result, ensure_ascii=False)}", flush=True)
except Exception as e:
    print(f"[7] Error: {e}", flush=True)

page.close()
ctx.close()
browser.close()
p.stop()
proc.terminate()
try: proc.wait(timeout=3)
except: proc.kill()
print("[DONE]", flush=True)
