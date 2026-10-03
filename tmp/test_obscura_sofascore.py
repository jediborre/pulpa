"""
Ubicación original: scratch/test_obscura_sofascore.py
Propósito / Qué hacía:
Prueba de navegación y extracción directa sobre páginas de SofaScore con Obscura.
"""

import sys
import os
from pathlib import Path
import subprocess
import json
from typing import Optional, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class ObscuraScraper:
    def __init__(self):
        self.obscura_exe = r"C:\Users\App\Desktop\pulpa\tools\obscura\v0.1.5\obscura.exe"
        cert_path = r"C:\Users\App\Desktop\pulpa\.venv\Lib\site-packages\certifi\cacert.pem"
        os.environ["SSL_CERT_FILE"] = cert_path
    
    def fetch_page(self, url: str, wait: int = 10, timeout: int = 30, dump: str = "text") -> Optional[str]:
        """Fetch a page using obscura."""
        print(f"[INFO] Fetching: {url}")
        
        result = subprocess.run(
            [self.obscura_exe, "fetch", url,
             "--stealth", "--wait", str(wait), "--timeout", str(timeout), "--dump", dump],
            env=os.environ,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=timeout + 15
        )
        
        if result.returncode == 0:
            print(f"[OK] Fetched {len(result.stdout)} chars")
            return result.stdout
        else:
            print(f"[ERROR] Failed: {result.stderr}")
            return None
    
    def fetch_sofascore_basketball(self) -> Optional[str]:
        """Fetch sofascore basketball page."""
        return self.fetch_page("https://www.sofascore.com/basketball")
    
    def fetch_sofascore_event(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Fetch sofascore event API data."""
        api_url = f"https://api.sofascore.com/api/v1/event/{match_id}"
        content = self.fetch_page(api_url, wait=5, timeout=20, dump="original")
        
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not parse JSON")
                return {"raw": content}
        return None
    
    def fetch_sofascore_matches(self, date: str = None) -> Optional[str]:
        """Fetch sofascore matches for a specific date."""
        url = "https://www.sofascore.com/basketball"
        if date:
            url = f"https://www.sofascore.com/basketball/{date}"
        return self.fetch_page(url)

def main():
    print("=" * 70)
    print("Obscura Sofascore Scraper")
    print("=" * 70)
    
    scraper = ObscuraScraper()
    
    print("\n[TEST 1] Fetching basketball main page...")
    content = scraper.fetch_sofascore_basketball()
    if content:
        print("\n--- Sample content (first 500 chars) ---")
        print(content[:500])
    
    print("\n" + "=" * 70)
    print("\n[TEST 2] Fetching event API (match 15415698)...")
    event_data = scraper.fetch_sofascore_event("15415698")
    if event_data:
        if "event" in event_data:
            event = event_data["event"]
            home = event.get("homeTeam", {}).get("name", "N/A")
            away = event.get("awayTeam", {}).get("name", "N/A")
            print(f"[INFO] Match: {home} vs {away}")
        else:
            print(f"[INFO] Response keys: {list(event_data.keys())}")
    
    print("\n" + "=" * 70)
    print("\n[TEST 3] Fetching HTML version...")
    html = scraper.fetch_page("https://www.sofascore.com/basketball", dump="html")
    if html:
        print(f"[INFO] HTML length: {len(html)} chars")
        if "<title>" in html:
            import re
            title = re.search(r"<title>(.*?)</title>", html)
            if title:
                print(f"[INFO] Title: {title.group(1)}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")

if __name__ == "__main__":
    main()
