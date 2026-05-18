"""Check the actual structure of the odds API response."""
import json, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # NBA match (upcoming)
    resp = ctx.request.get(
        "https://api.sofascore.com/api/v1/event/15935010/odds/1/all",
        headers=extra_headers, timeout=15_000
    )
    if resp.ok:
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False)[:3000])

    # Try provider 1 without /all
    resp2 = ctx.request.get(
        "https://api.sofascore.com/api/v1/event/15935010/odds/1/featured",
        headers=extra_headers, timeout=15_000
    )
    if resp2.ok:
        print(f"\n=== featured odds ===")
        print(json.dumps(resp2.json(), indent=2, ensure_ascii=False)[:2000])

    # Try winning-odds
    resp3 = ctx.request.get(
        "https://api.sofascore.com/api/v1/event/15935010/provider/1/winning-odds",
        headers=extra_headers, timeout=15_000
    )
    if resp3.ok:
        print(f"\n=== winning odds ===")
        print(json.dumps(resp3.json(), indent=2, ensure_ascii=False)[:2000])

    browser.close()
