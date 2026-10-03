"""Test: verificar cookies via CDP comandos directos"""
import subprocess, time, os, sys, json, socket
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

proc = subprocess.Popen(
    [str(OBSCURA_EXE), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR), env=os.environ,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(4)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
if sock.connect_ex(('127.0.0.1', 9222)) != 0:
    print("Port not listening!", flush=True); exit(1)
sock.close()

from playwright.sync_api import sync_playwright
p = sync_playwright().start()
browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=8000)

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

# Test 1: Get cookies BEFORE navigation
print("\n[Test 1] Cookies BEFORE navigation:", flush=True)
cookies_before = browser.contexts[0].cookies()
print(f"  Playwright cookies: {len(cookies_before)}", flush=True)

# Navigate
page = browser.contexts[0].new_page()
try:
    page.goto(f'https://www.sofascore.com/basketball/{yesterday}', timeout=15000, wait_until='commit')
except:
    pass
time.sleep(3)
print(f"  Title: {page.title()}", flush=True)

# Test 2: Get cookies AFTER navigation via Playwright API
print("\n[Test 2] Cookies AFTER navigation (Playwright API):", flush=True)
cookies_after = browser.contexts[0].cookies()
print(f"  Count: {len(cookies_after)}", flush=True)
for c in cookies_after[:5]:
    print(f"  {c['name']}: domain={c['domain']}, path={c['path']}, httponly={c.get('httpOnly', False)}", flush=True)

# Test 3: Get cookies via CDP Network.getCookies
print("\n[Test 3] Cookies via CDP:", flush=True)
try:
    cdp_result = page._channel.send("Network.getCookies")
    cookies_cdp = cdp_result.get("cookies", [])
    print(f"  Count: {len(cookies_cdp)}", flush=True)
    for c in cookies_cdp[:5]:
        print(f"  {c['name']}: domain={c['domain']}, path={c['path']}, httponly={c.get('httpOnly')}", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)

# Test 4: document.cookie
print("\n[Test 4] document.cookie:", flush=True)
try:
    dc = page.evaluate("document.cookie")
    print(f"  '{dc}'", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)

page.close()
browser.close()
p.stop()
proc.terminate()
try: proc.wait(timeout=3)
except: proc.kill()
print("\n[DONE]", flush=True)
