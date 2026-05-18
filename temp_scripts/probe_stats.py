"""Probe the /event/{id}/statistics endpoint response structure."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# A Euroleague match ID
match_id = "14491671"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass

    resp = ctx.request.get(
        f"https://api.sofascore.com/api/v1/event/{match_id}/statistics",
        headers=extra_headers, timeout=20_000
    )
    print(f"Status: {resp.status}")
    if resp.ok:
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))

    browser.close()
