"""Click odds tab and capture ALL network activity including iframes."""
import json, re, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

match_id = "14491671"
all_requests = set()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA, viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()

    # Capture ALL requests at context level
    ctx.on("request", lambda req: all_requests.add(req.url.split("?")[0]))

    page.goto(
        f"https://www.sofascore.com/event/{match_id}",
        wait_until="domcontentloaded", timeout=30_000
    )
    time.sleep(3)

    print(f"Requests so far: {len(all_requests)}")

    # Find and click odds tab
    odds_clicked = False
    for selector in [
        'button:has-text("Odds")',
        'a:has-text("Odds")',
        'div:has-text("Odds")',
        '[data-testid*="odds"]',
        '[href*="odds"]',
    ]:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                print(f"Clicking odds tab: {selector}")
                el.click()
                time.sleep(3)
                odds_clicked = True
                break
        except:
            pass

    if not odds_clicked:
        # Try scroll to bottom where odds might be
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        for selector in [
            'button:has-text("Odds")',
            'a:has-text("Odds")',
        ]:
            try:
                el = page.query_selector(selector)
                if el:
                    print(f"Clicking odds tab (scrolled): {selector}")
                    el.click()
                    time.sleep(3)
                    break
            except:
                pass

    print(f"\nTotal unique requests: {len(all_requests)}")
    print(f"\n=== Requests with keyword matches ===")
    for url in sorted(all_requests):
        low = url.lower()
        if any(x in low for x in ["odds", "bet", "market", "handicap", "spread", "bookmaker", "prediction"]):
            print(f"  {url}")

    print(f"\n=== All API requests to api.sofascore.com ===")
    for url in sorted(all_requests):
        if "api.sofascore" in url:
            print(f"  {url}")

    # Check page content after clicking odds
    html = page.content()
    # Look for any data-* attributes with odds
    for match in re.finditer(r'data-[^=]+="[^"]{0,500}"', html):
        val = match.group(0)
        if "odd" in val.lower():
            print(f"\n  Data attr: {val[:200]}")

    # Also check for iframes
    iframes = page.query_selector_all("iframe")
    print(f"\nIframes on page: {len(iframes)}")
    for frame in iframes:
        src = frame.get_attribute("src") or ""
        if "bet" in src.lower() or "odd" in src.lower():
            print(f"  Odds iframe: {src[:200]}")

    # Save page HTML after odds tab clicked
    with open(ROOT / "temp_scripts" / "page_after_odds.html", "w", encoding="utf-8") as f:
        f.write(html)

    browser.close()
