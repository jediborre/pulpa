"""Log ALL API requests from a real SofaScore match page to discover endpoints."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

match_id = "14491671"
BASE = "https://api.sofascore.com/api/v1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()

    requests_log = []
    def on_request(req):
        url = req.url
        if "api.sofascore.com" in url:
            requests_log.append(url)

    page.on("request", on_request)

    try:
        page.goto(
            f"https://www.sofascore.com/basketball/match/{match_id}",
            wait_until="networkidle",
            timeout=60_000,
        )
    except Exception as e:
        print(f"Navigation warning: {e}")

    import time; time.sleep(5)

    # Print UNIQUE API requests grouped by resource pattern
    unique = sorted(set(requests_log))
    print(f"\nTotal unique API requests: {len(unique)}")
    for url in unique:
        print(f"  {url}")
