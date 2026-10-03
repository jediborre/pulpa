"""
Ubicación original: scratch/test_obscura_ssl.py
Propósito / Qué hacía:
Prueba de inyección de certificados SSL (SSL_CERT_FILE) en peticiones de Obscura.
"""

import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def test_obscura_ssl():
    # 1. Locate the certifi CA bundle
    cert_path = r"C:\Users\App\Desktop\pulpa\.venv\Lib\site-packages\certifi\cacert.pem"
    if not os.path.exists(cert_path):
        print(f"[ERROR] certifi cacert.pem not found at: {cert_path}")
        return
    
    print(f"[INFO] Using CA bundle: {cert_path}")
    
    # 2. Set environment variable
    os.environ["SSL_CERT_FILE"] = cert_path
    
    # 3. Terminate any running obscura instance
    print("[INFO] Killing any running obscura.exe...")
    subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    
    # 4. Start obscura.exe serve
    obscura_exe = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5\obscura.exe"
    obscura_dir = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5"
    
    print("[INFO] Launching Obscura serve with SSL_CERT_FILE set...")
    # Launch in background
    proc = subprocess.Popen(
        [obscura_exe, "serve", "--port", "9222", "--stealth"],
        cwd=obscura_dir,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    time.sleep(2.0)
    
    if proc.poll() is not None:
        print(f"[ERROR] Obscura failed to start. Return code: {proc.poll()}")
        out, err = proc.communicate()
        print(f"Stdout: {out}")
        print(f"Stderr: {err}")
        return
        
    print("[OK] Obscura serve started successfully on port 9222!")
    
    # 5. Connect via Playwright and navigate to Sofascore
    with sync_playwright() as p:
        try:
            print("[INFO] Connecting Playwright to Obscura (CDP 127.0.0.1:9222)...")
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            print("[OK] Connected to Obscura!", flush=True)
            
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.new_page()
            
            # Target URL
            target_url = "https://www.sofascore.com/"
            print(f"[INFO] Navigating Obscura to {target_url}...")
            
            start_time = time.time()
            page.goto(target_url, timeout=30000, wait_until="domcontentloaded")
            print(f"[OK] Page loaded in {time.time() - start_time:.2f}s!")
            
            # Check title
            title = page.title()
            print(f"[INFO] Page Title: {title}")
            
            # Perform a test fetch to Sofascore API inside the page context
            api_url = "https://api.sofascore.com/api/v1/event/15415698"
            print(f"[INFO] Evaluating fetch inside Obscura context to {api_url}...")
            
            js_code = f"fetch('{api_url}').then(r => r.json())"
            res = page.evaluate(js_code)
            
            if res and "event" in res:
                event = res["event"]
                home = event.get("homeTeam", {}).get("name", "Unknown")
                away = event.get("awayTeam", {}).get("name", "Unknown")
                print(f"[OK] Fetch succeeded! Event: {home} vs {away}")
            else:
                print(f"[WARNING] Fetch returned empty or non-event response: {res}")
                
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
        finally:
            # Terminate obscura
            print("[INFO] Cleaning up: terminating obscura process...")
            proc.terminate()
            proc.wait()
            print("[OK] Test complete.")

if __name__ == "__main__":
    test_obscura_ssl()
