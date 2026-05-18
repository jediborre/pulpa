"""Find odds by checking multiple match pages and searching page HTML for
any API URLs or embedded data."""

import json, re, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Various match IDs from different leagues
match_ids = [
    "14491671",  # ACB
    "14507685",  # Table tennis
    "14508205",  # Table tennis
    "14502824",  # Another
    "14507932",
    "14507735",
    "14507691",
]

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

BASE = "https://api.sofascore.com/api/v1"

all_api_calls = set()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()

    # First navigate to basketball landing
    page.goto("https://www.sofascore.com/basketball",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    for mid in match_ids:
        print(f"\n=== Match {mid} ===")

        # Direct API probes
        for ep in [
            f"/event/{mid}/odds",
            f"/event/{mid}/odds/all",
            f"/event/{mid}/betting",
        ]:
            try:
                resp = ctx.request.get(
                    f"{BASE}{ep}", headers=extra_headers, timeout=10_000
                )
                status = resp.status
                text_preview = ""
                if resp.ok:
                    body = resp.json() if resp.text else {}
                    text_preview = f"keys={list(body.keys())[:5]}" if isinstance(body, dict) else f"type={type(body).__name__}"
                print(f"  {ep} -> HTTP {status} {text_preview}")
            except Exception as e:
                print(f"  {ep} -> ERROR {e}")

        # Check the match page HTML for any embedded odds data or API URLs
        try:
            page.goto(
                f"https://www.sofascore.com/event/{mid}",
                wait_until="domcontentloaded", timeout=30_000
            )
            time.sleep(3)

            # Search HTML for odds-related content
            html = page.content()
            odds_patterns = re.findall(r'(?:odds|bet365|betting|bookmaker)[^<]{0,100}',
                                        html, re.IGNORECASE)
            if odds_patterns:
                print(f"  [HTML odds mentions: {len(odds_patterns)}]")
                for p in odds_patterns[:8]:
                    print(f"    {p.strip()[:150]}")

            # Search for __NEXT_DATA__ or __INITIAL_STATE__
            for pattern in [r'__NEXT_DATA__[^>]*>(.*?)<\/script>',
                            r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                            r'window\.__DATA__\s*=\s*({.*?});']:
                match = re.search(pattern, html, re.DOTALL)
                if match:
                    print(f"  [Found {pattern[:30]}...]")

            # Look for API URLs in embedded data
            for match in re.finditer(r'https://api\.sofascore\.com[^"\' ]+', html):
                url = match.group(0)
                if any(x in url.lower() for x in ["odds", "bet", "market", "handicap"]):
                    print(f"  [API in HTML] {url}")

        except Exception as e:
            print(f"  [Page error] {e}")

    browser.close()
