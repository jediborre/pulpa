"""Test: original obscura + cross-origin fetch"""
import subprocess, time, os, sys, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
OBSCURA_DIR = ROOT / "tools" / "obscura" / "v0.1.5"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
os.environ["SSL_CERT_FILE"] = str(CERT_PATH)

subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

proc = subprocess.Popen(
    [str(OBSCURA_DIR / "obscura.exe"), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR), env=os.environ,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(4)

from playwright.sync_api import sync_playwright
p = sync_playwright().start()
browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=8000)
ctx = browser.contexts[0]
page = ctx.new_page()

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
page.goto(f'https://www.sofascore.com/basketball/{yesterday}', timeout=15000, wait_until='commit')
time.sleep(2)
print(f'Title: {page.title()}', flush=True)

api_url = f'https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{yesterday}'
page.evaluate(f"""
window.__r = null; window.__d = false;
fetch('{api_url}', {{ headers: {{ 'Accept': 'application/json', 'Referer': 'https://www.sofascore.com/' }} }})
.then(r => r.ok ? r.json() : Promise.reject('HTTP ' + r.status))
.then(d => {{ window.__r = {{ ok: true, count: d.events?.length }}; window.__d = true; }})
.catch(e => {{ window.__r = {{ ok: false, error: String(e) }}; window.__d = true; }})
""")

try:
    page.wait_for_function("window.__d === true", timeout=35000)
    result = page.evaluate("window.__r")
    print(f'Result: {json.dumps(result, ensure_ascii=False)}', flush=True)
except Exception as e:
    print(f'Error: {e}', flush=True)

page.close()
ctx.close()
browser.close()
p.stop()
proc.terminate()
try: proc.wait(timeout=3)
except: proc.kill()
print("DONE", flush=True)
