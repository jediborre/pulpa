import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def test_obscura_ssl_success():
    cert_path = r"C:\Users\App\Desktop\pulpa\.venv\Lib\site-packages\certifi\cacert.pem"
    if not os.path.exists(cert_path):
        print(f"[ERROR] certifi cacert.pem not found at: {cert_path}")
        return
    
    print(f"[INFO] Using CA bundle: {cert_path}")
    os.environ["SSL_CERT_FILE"] = cert_path
    
    print("[INFO] Killing any running obscura.exe...")
    subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    
    obscura_exe = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5\obscura.exe"
    obscura_dir = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5"
    
    print("[INFO] Launching Obscura serve with SSL_CERT_FILE set...")
    proc = subprocess.Popen(
        [obscura_exe, "serve", "--port", "9222"],
        cwd=obscura_dir,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2.0)
    
    if proc.poll() is not None:
        print(f"[ERROR] Obscura failed to start. Return code: {proc.poll()}")
        return
        
    print("[OK] Obscura serve started successfully on port 9222!")
    
    with sync_playwright() as p:
        try:
            print("[INFO] Connecting Playwright to Obscura (CDP 127.0.0.1:9222)...")
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            print("[OK] Connected to Obscura!", flush=True)
            
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.new_page()
            
            target_url = "https://httpbin.org/ip"
            print(f"[INFO] Navigating Obscura to {target_url}...")
            
            start_time = time.time()
            page.goto(target_url, timeout=10000, wait_until="domcontentloaded")
            print(f"[OK] Page loaded successfully in {time.time() - start_time:.2f}s!")
            
            # Use page.evaluate to fetch text to bypass Playwright's complex DOM querying
            print("[INFO] Evaluating document.body.innerText inside Obscura context...")
            text = page.evaluate("document.body.innerText")
            print(f"[OK] Retrieved data successfully!\n{text}")
            
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
        finally:
            print("[INFO] Cleaning up: terminating obscura process...")
            proc.terminate()
            proc.wait()
            print("[OK] Test complete.")

if __name__ == "__main__":
    test_obscura_ssl_success()
