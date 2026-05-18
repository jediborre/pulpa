"""Final odds endpoint discovery attempt + check event JSON for odds fields."""
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

# Check event JSON for odds-related fields
odds_related = [
    "odds", "bet", "market", "handicap", "spread", "overunder",
    "bookmaker", "prediction", "expected", "implied", "price",
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass

    # Check event JSON thoroughly
    resp = ctx.request.get(
        f"{BASE}/event/{match_id}",
        headers=extra_headers, timeout=15_000
    )
    if resp.ok:
        ev = resp.json()
        # Check all keys recursively for odds-related terms
        def find_keys(obj, path=""):
            results = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full = f"{path}.{k}" if path else k
                    for term in odds_related:
                        if term in k.lower():
                            results.append((full, type(v).__name__))
                    if k in ("homeScore", "awayScore", "status", "tournament"):
                        continue
                    results.extend(find_keys(v, full))
            elif isinstance(obj, list) and len(obj) > 0:
                results.extend(find_keys(obj[0], f"{path}[0]"))
            return results

        print("=== Odds-related fields in event JSON ===")
        for field, typ in find_keys(ev.get("event", ev)):
            print(f"  {field} ({typ})")
        print(f"\nTop-level keys: {list(ev.get('event', {}).keys())}")

    # Try alternate API domains for odds
    alt_base = [
        "https://api-il.sofascore.com/api/v1",
        "https://api2.sofascore.com/api/v1",
    ]
    for ab in alt_base:
        resp = ctx.request.get(
            f"{ab}/event/{match_id}",
            headers=extra_headers, timeout=10_000
        )
        if resp.ok:
            print(f"\n[OK] {ab}/event/{match_id}")
        else:
            print(f"[{resp.status}] {ab}")

    # Try to load page with odds tab
    print(f"\n=== Navigating to match page ===")
    match_url = f"https://www.sofascore.com/event/{match_id}"
    try:
        page.goto(match_url, wait_until="domcontentloaded", timeout=30_000)
        import time; time.sleep(4)
        print(f"Final URL: {page.url}")

        # Try to find odds tab/data in page
        for selector in ['[data-testid="odds"]', '[href*="odds"]', 'button*odds', 'a*odds']:
            try:
                el = page.query_selector(selector)
                if el:
                    print(f"Found odds element: {selector} -> {el.text_content()[:100] if el.text_content() else 'empty'}")
            except:
                pass
    except Exception as e:
        print(f"Navigation error: {e}")

    # Check the page content for "Odds"
    try:
        content = page.content()
        if "Odds" in content:
            # Find context around "Odds"
            idx = content.find("Odds")
            print(f"\n'Odds' found at position {idx}")
            print(f"Context: ...{content[max(0,idx-200):idx+200]}...")
    except:
        pass

    browser.close()
