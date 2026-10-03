"""
Debug granular: ver exactamente en qué paso falla el fetch FT.
Prueba el warmup URL directamente con Obscura.
"""
import asyncio
import sys
import io
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

# Match IDs a probar - Fenerbahce vs Esenler (Turquia, hoy)
TESTS = [
    ("16193682", "Rain Or Shine vs Barangay Ginebra"),   # PBA filipinas - hoy
    ("16204053", "Satria Muda vs Hangtuah Jakarta"),      # Indonesia - hoy
]

def test_warmup_only(match_id: str, name: str, backend: str = "obscura"):
    """Solo prueba el warmup URL — sin hacer peticiones API."""
    warmup_url = f"https://www.sofascore.com/basketball/match/{match_id}#id:{match_id}"
    
    print(f"\n{'='*60}")
    print(f"  Partido: {name} ({match_id})")
    print(f"  URL: {warmup_url}")
    print(f"  Backend: {backend}")
    
    start = time.time()
    try:
        with sync_playwright() as p:
            if backend == "obscura":
                print(f"  Conectando a Obscura CDP...")
                t1 = time.time()
                browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
                print(f"  CDP conectado en {time.time()-t1:.1f}s")
            else:
                print(f"  Lanzando Chromium...")
                t1 = time.time()
                browser = p.chromium.launch(headless=True)
                print(f"  Chromium lanzado en {time.time()-t1:.1f}s")
            
            ctx = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            page = ctx.new_page()
            
            print(f"  Navigando a warmup URL (timeout 20s)...")
            t2 = time.time()
            try:
                page.goto(warmup_url, wait_until="networkidle", timeout=20_000)
                print(f"  Warmup OK en {time.time()-t2:.1f}s")
            except Exception as e:
                print(f"  Warmup ERROR en {time.time()-t2:.1f}s: {e}")
            
            # Ahora prueba una sola petición API
            print(f"  Petición /event/{match_id}...")
            t3 = time.time()
            try:
                resp = ctx.request.get(
                    f"https://api.sofascore.com/api/v1/event/{match_id}",
                    headers={"Referer": "https://www.sofascore.com/"},
                    timeout=15_000
                )
                elapsed = time.time() - t3
                if resp.ok:
                    data = resp.json()
                    event = data.get("event", {})
                    status = event.get("status", {}).get("type", "?")
                    home_score = event.get("homeScore", {}).get("current", "?")
                    away_score = event.get("awayScore", {}).get("current", "?")
                    print(f"  API OK en {elapsed:.1f}s | status={status} | score={home_score}-{away_score}")
                else:
                    print(f"  API HTTP {resp.status} en {elapsed:.1f}s")
            except Exception as e:
                print(f"  API ERROR en {time.time()-t3:.1f}s: {e}")
            
            ctx.close()
            
    except Exception as e:
        print(f"  ❌ ERROR total: {e}")
    
    print(f"  Total: {time.time()-start:.1f}s")

def main():
    for match_id, name in TESTS:
        test_warmup_only(match_id, name, backend="obscura")
        print()

main()
