"""Try more odds endpoint variations + check actual match page for API calls."""
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

match_id = "14491671"
BASE = "https://api.sofascore.com/api/v1"

more_endpoints = [
    # Common SofaScore odds patterns
    f"/event/{match_id}/odds/all",
    f"/event/{match_id}/odds/1x2",
    f"/event/{match_id}/odds/spread",
    f"/event/{match_id}/odds/over-under",
    f"/event/{match_id}/odds/h2h",
    f"/event/{match_id}/bet365",
    f"/event/{match_id}/betting",
    f"/event/{match_id}/prematch",
    f"/event/{match_id}/market",
    f"/event/{match_id}/markets",
    f"/event/{match_id}/handicap",
    # Try with /v1 removed
    f"/event/{match_id}/odds",
    # Try different base
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()

    # Listen for all XHR requests
    requests_log = []
    page.on("request", lambda r: requests_log.append(r.url))

    try:
        page.goto(
            f"https://www.sofascore.com/basketball/match/{match_id}",
            wait_until="networkidle",
            timeout=45_000,
        )
    except Exception:
        pass

    # Wait for dynamic loads
    import time; time.sleep(3)

    # Log all API requests the page made
    api_requests = [u for u in requests_log if "api.sofascore" in u and "odds" in u.lower()]
    print(f"=== Odds-related API requests from match page ({len(api_requests)}) ===")
    for url in api_requests[:20]:
        print(f"  {url}")

    # Also try the endpoints directly
    print(f"\n=== Direct endpoint probes ===")
    for ep in more_endpoints:
        url = f"{BASE}{ep}"
        # Remove double / if any
        url = url.replace("/event", "/event")
        resp = ctx.request.get(url, headers=extra_headers, timeout=15_000)
        print(f"  [{resp.status}] {ep}")

    browser.close()
