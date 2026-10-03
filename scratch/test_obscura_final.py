"""Test final del build v0.1.5 con fixes"""
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

print("[1] Iniciando obscura-fixed...", flush=True)
proc = subprocess.Popen(
    [str(OBSCURA_EXE), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR), env=os.environ,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(4)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
if sock.connect_ex(('127.0.0.1', 9222)) != 0:
    print("Port not listening!", flush=True)
    exit(1)
sock.close()
print("[2] Puerto OK", flush=True)

from playwright.sync_api import sync_playwright
p = sync_playwright().start()
browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=8000)
ctx = browser.contexts[0]
page = ctx.new_page()

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
print(f"[3] Navegando a {yesterday}...", flush=True)
try:
    page.goto(f'https://www.sofascore.com/basketball/{yesterday}', timeout=15000, wait_until='commit')
    print("[3] NAV OK", flush=True)
except Exception as e:
    print(f"[3] NAV ERROR: {e}", flush=True)
    exit(1)

time.sleep(2)
print(f"[4] Title: {page.title()}", flush=True)

api_url = f'https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{yesterday}'
print("[5] Fetching API via page.evaluate...", flush=True)
page.evaluate(f"""
window.__r = null; window.__d = false;
fetch('{api_url}', {{ headers: {{ 'Accept': 'application/json', 'Referer': 'https://www.sofascore.com/' }} }})
.then(r => r.ok ? r.json() : Promise.reject('HTTP ' + r.status))
.then(d => {{ window.__r = {{ ok: true, count: d.events?.length, first5: (d.events || []).filter(e=>e.status?.type==='finished').slice(0,5).map(e=>({{id:e.id,home:e.homeTeam?.name,away:e.awayTeam?.name,hs:e.homeScore?.current,as:e.awayScore?.current}})) }}; window.__d = true; }})
.catch(e => {{ window.__r = {{ ok: false, error: String(e) }}; window.__d = true; }})
""")

try:
    page.wait_for_function("window.__d === true", timeout=45000)
    result = page.evaluate("window.__r")
    print(f"[6] Result: {json.dumps(result, ensure_ascii=False, indent=2)}", flush=True)
except Exception as e:
    print(f"[6] ERROR: {e}", flush=True)

page.close()
ctx.close()
browser.close()
p.stop()
proc.terminate()
try: proc.wait(timeout=3)
except: proc.kill()
print("[DONE]", flush=True)
