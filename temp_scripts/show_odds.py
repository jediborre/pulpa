"""Show raw odds response + parsed output for a match."""
import sys, json
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
from playwright.sync_api import sync_playwright
from scraper import _parse_odds

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

match_id = "15935010"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)

    # Raw odds response  
    resp = ctx.request.get(f"{BASE}/event/{match_id}/odds/1/all", headers=extra_headers, timeout=15_000)
    raw = resp.json()
    print("=== RAW odds response ===")
    print(json.dumps(raw, indent=2)[:3000])
    print()

    # Parse it  
    parsed = _parse_odds(match_id, raw)
    print("=== PARSED odds ===")
    for p in parsed:
        print(json.dumps(p, indent=2))

    browser.close()
