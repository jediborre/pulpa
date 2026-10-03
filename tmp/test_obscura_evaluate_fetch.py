"""
Ubicación original: scratch/test_obscura_evaluate_fetch.py
Propósito / Qué hacía:
Solución de contorno para promesas asíncronas en evaluación JS con Obscura.
"""

"""Test: workaround para promesas usando variables globales"""
import subprocess
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
OBSCURA_EXE = ROOT / "tools" / "obscura" / "v0.1.5" / "obscura.exe"
OBSCURA_DIR = ROOT / "tools" / "obscura" / "v0.1.5"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"

os.environ["SSL_CERT_FILE"] = str(CERT_PATH)
sys.stdout.reconfigure(line_buffering=True)

print("[1] Iniciando Obscura...", flush=True)
subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

proc = subprocess.Popen(
    [str(OBSCURA_EXE), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR),
    env=os.environ,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(4)

from playwright.sync_api import sync_playwright

print("[2] Conectando Playwright CDP...", flush=True)
p = sync_playwright().start()
browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
ctx = browser.contexts[0] if browser.contexts else browser.new_context()
page = ctx.new_page()

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
print(f"[3] Navegando a sofascore.com/basketball/{yesterday}...", flush=True)

try:
    page.goto(f'https://www.sofascore.com/basketball/{yesterday}',
              timeout=20000, wait_until='commit')
except Exception as e:
    print(f"[3] Nav timeout (continuando)", flush=True)

time.sleep(3)
print(f"[4] Title: {page.title()}", flush=True)

api_url = f'https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{yesterday}'

print("[5] Iniciando fetch async con variable global...", flush=True)
js_start = f"""
window.__obscura_result = null;
window.__obscura_done = false;
fetch('{api_url}', {{
    headers: {{
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.sofascore.com/'
    }}
}})
.then(r => {{
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
}})
.then(d => {{
    const finished = (d.events || []).filter(e => e.status && e.status.type === 'finished');
    window.__obscura_result = {{
        total: (d.events || []).length,
        finished: finished.length,
        first3: finished.slice(0, 3).map(e => ({{
            id: e.id,
            home: e.homeTeam && e.homeTeam.name,
            away: e.awayTeam && e.awayTeam.name,
            hs: e.homeScore && e.homeScore.current,
            as: e.awayScore && e.awayScore.current,
            league: e.tournament && e.tournament.name
        }}))
    }};
    window.__obscura_done = true;
}})
.catch(e => {{
    window.__obscura_result = {{ error: e.message }};
    window.__obscura_done = true;
}});
"""

page.evaluate(js_start)
print("[6] Fetch iniciado, esperando resultado...", flush=True)

try:
    page.wait_for_function("window.__obscura_done === true", timeout=15000)
    print("[7] Fetch completado!", flush=True)
    result = page.evaluate("window.__obscura_result")
    print(f"[8] Result: {json.dumps(result, indent=2, ensure_ascii=False)}", flush=True)
    
    if result and not result.get('error') and result.get('finished', 0) > 0:
        print(f"\n[SUCCESS] {result['finished']} partidos terminados encontrados!", flush=True)
        print(f"Primeros 3:", flush=True)
        for match in result.get('first3', []):
            print(f"  - {match['home']} vs {match['away']}: {match['hs']}-{match['as']} ({match['league']})", flush=True)
    else:
        print(f"[INFO] No hay partidos o error: {result}", flush=True)
        
except Exception as e:
    print(f"[ERROR] {e}", flush=True)
    import traceback
    traceback.print_exc()

print("[9] Cleanup...", flush=True)
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
