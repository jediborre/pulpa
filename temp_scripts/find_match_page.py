"""Find the correct match page URL and log API calls."""
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

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()

    requests_log = []
    api_urls = set()

    def on_request(req):
        url = req.url
        if "api.sofascore.com" in url:
            api_urls.add(url.split("?")[0])

    page.on("request", on_request)

    # Try different match page URL formats
    urls_to_try = [
        f"https://www.sofascore.com/basketball/match/{match_id}",
        f"https://www.sofascore.com/match/basketball/{match_id}",
        f"https://www.sofascore.com/event/{match_id}",
        f"https://www.sofascore.com/basketball/event/{match_id}",
        f"https://www.sofascore.com/en/basketball/match/{match_id}",
    ]

    for url in urls_to_try:
        page = ctx.new_page()
        page_requests = []
        def _on_req(req):
            if "api.sofascore.com" in req.url:
                page_requests.append(req.url.split("?")[0])
        page.on("request", _on_req)

        print(f"\n=== Trying: {url} ===")
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            print(f"  Response: {resp.status if resp else 'None'}")
            import time; time.sleep(3)
        except Exception as e:
            print(f"  Error: {e}")

        print(f"  Final URL: {page.url}")
        print(f"  API requests: {len(page_requests)}")
        for u in sorted(set(page_requests)):
            print(f"    {u}")

        page.close()

    browser.close()
