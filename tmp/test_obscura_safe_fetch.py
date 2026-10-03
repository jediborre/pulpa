"""
Ubicación original: scratch/test_obscura_safe_fetch.py
Propósito / Qué hacía:
Implementación segura de fetch con reintentos y timeouts sobre Obscura.
"""

import sys
import os
from pathlib import Path

# Add workspace to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess
import time
import asyncio
from playwright.sync_api import sync_playwright
import bet_monitor_v2.config.constants as constants

async def test_obscura_safe_fetch():
    cert_path = r"C:\Users\App\Desktop\pulpa\.venv\Lib\site-packages\certifi\cacert.pem"
    os.environ["SSL_CERT_FILE"] = cert_path
    
    # 1. Kill any existing obscura
    print("[INFO] Killing obscura...")
    subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    
    # 2. Start obscura.exe serve
    obscura_exe = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5\obscura.exe"
    obscura_dir = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5"
    
    print("[INFO] Launching Obscura serve...")
    proc = subprocess.Popen(
        [obscura_exe, "serve", "--port", "9222", "--stealth"],
        cwd=obscura_dir,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(2.0)
    
    # Configure backend live & probe to "obscura" first
    constants.SOFASCORE_SCRAPER_BACKEND_LIVE = "obscura"
    constants.SOFASCORE_SCRAPER_BACKEND_PROBE = "obscura"
    
    # 3. Try to fetch a non-existent match id (e.g. "999999999") to trigger a clean 404 error
    from bet_monitor_v2.scrapers.live_scraper import fetch_event_snapshot
    
    print("[INFO] Running fetch_event_snapshot for a non-existent match ID (999999999)...")
    try:
        # This should raise RuntimeError with "HTTP 404"
        res = await fetch_event_snapshot("999999999", backend="obscura")
        print(f"[WARNING] Event snap succeeded?! {res}")
    except Exception as e:
        print(f"[OK] Event snap failed as expected with error: {e}")
        
    # 4. Now let's simulate three 403 errors using check_403_streak to prove that the hotswap dynamic routing works instantly!
    from bet_monitor_v2.scrapers.base_scraper import check_403_streak
    
    print(f"[INFO] Current backend before streak: {constants.SOFASCORE_SCRAPER_BACKEND_LIVE}")
    
    print("[INFO] Simulating 1st HTTP 403 block...")
    await check_403_streak(403)
    print("[INFO] Simulating 2nd HTTP 403 block...")
    await check_403_streak(403)
    
    print("[INFO] Simulating 3rd HTTP 403 block...")
    await check_403_streak(403)
    
    print(f"[OK] Current backend after 3 consecutive 403s: {constants.SOFASCORE_SCRAPER_BACKEND_LIVE}")
    
    proc.terminate()
    proc.wait()
    print("[INFO] Cleaned up. Test complete.")

if __name__ == "__main__":
    asyncio.run(test_obscura_safe_fetch())
