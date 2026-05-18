"""Interact with the real SofaScore match page to discover
per-quarter statistics and odds API endpoints via click events."""

import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Try multiple match IDs — some might have odds
match_ids = [
    "14491671",   # Basket Zaragoza vs Baxi Manresa (ACB)
    "14507685",   # Another recent match
    "14508205",   # Another
]

def intercept_page(page, ctx, match_id):
    """Load match page, capture ALL API requests, click on quarter/odds tabs."""
    api_calls = set()
    interval_timer = [0]

    def on_response(resp):
        url = resp.url
        if "api.sofascore.com" in url and "event" in url:
            api_calls.add(url.split("?")[0])
            # Log new calls we haven't seen
            print(f"  [API] {resp.status} {url[:120]}")

    page.on("response", on_response)

    try:
        page.goto(
            f"https://www.sofascore.com/event/{match_id}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        page.wait_for_timeout(3000)
    except Exception as e:
        print(f"  [Nav] {e}")

    print(f"  Final URL: {page.url[:100]}")

    # Try clicking on quarter tab buttons (if any)
    quarter_selectors = [
        'button:has-text("Q1")',
        'button:has-text("Q2")',
        'button:has-text("Q3")',
        'button:has-text("Q4")',
        '[data-testid*="quarter"]',
        '[data-testid*="period"]',
        'button[class*="period"]',
        'button[class*="quarter"]',
        'div[class*="period"] button',
        'div[class*="Period"]',
    ]
    for sel in quarter_selectors:
        try:
            buttons = page.query_selector_all(sel)
            for btn in buttons:
                try:
                    text = btn.text_content()[:30]
                    print(f"  [Click Q] '{sel}' -> '{text}'")
                    btn.click()
                    page.wait_for_timeout(1500)
                except:
                    pass
        except:
            pass

    # Try clicking on "Statistics" tab
    stat_selectors = [
        'a:has-text("Statistics")',
        'button:has-text("Statistics")',
        'div:has-text("Statistics")',
        '[data-testid*="statistics"]',
        'a[href*="statistics"]',
    ]
    for sel in stat_selectors:
        try:
            el = page.query_selector(sel)
            if el:
                print(f"  [Click Stats] '{sel}'")
                el.click()
                page.wait_for_timeout(2000)
                break
        except:
            pass

    # Try clicking on "Betting" / "Odds" tab
    odds_selectors = [
        'a:has-text("Betting")',
        'button:has-text("Betting")',
        'a:has-text("Odds")',
        'button:has-text("Odds")',
        'a[href*="betting"]',
        'a[href*="odds"]',
        '[data-testid*="betting"]',
        '[data-testid*="odds"]',
        'div[class*="betting"]',
        'div[role="tab"]:has-text("Betting")',
        'div[role="tab"]:has-text("Odds")',
    ]
    for sel in odds_selectors:
        try:
            el = page.query_selector(sel)
            if el:
                print(f"  [Click Odds] '{sel}'")
                el.click()
                page.wait_for_timeout(3000)
                break
        except:
            pass

    # Scroll down to see if there's more content
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(2000)

    return api_calls

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    for mid in match_ids:
        print(f"\n{'='*60}")
        print(f"=== Match: {mid} ===")
        ctx = browser.new_context(user_agent=STANDARD_UA)
        page = ctx.new_page()

        all_calls = intercept_page(page, ctx, mid)

        print(f"\n  Unique event API calls:")
        for url in sorted(all_calls):
            if "odds" in url.lower() or "statistics" in url.lower() or "period" in url.lower() or "quarter" in url.lower():
                print(f"    ** {url}")

        # Also try direct calls for different odds/statistics patterns
        extra_headers = {
            "Referer": "https://www.sofascore.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        BASE = "https://api.sofascore.com/api/v1"
        for ep in [
            f"/event/{mid}/statistics?period=1",
            f"/event/{mid}/statistics?periodId=1",
            f"/event/{mid}/statistics/1",
            f"/event/{mid}/statistics?quarter=1",
            f"/event/{mid}/statistics/byperiod",
            f"/event/{mid}/odds",
            f"/event/{mid}/odds/1x2",
            f"/event/{mid}/bet365",
        ]:
            try:
                resp = ctx.request.get(f"{BASE}{ep}", headers=extra_headers, timeout=10_000)
                if resp.ok:
                    # Check if period is different from ALL
                    body = resp.json() if resp.text else {}
                    if isinstance(body, dict):
                        stats = body.get("statistics", [])
                        periods = [s.get("period") for s in stats if isinstance(s, dict)]
                        print(f"  [PROBE OK] {ep} -> periods: {periods}")
                    else:
                        print(f"  [PROBE OK] {ep}")
            except:
                pass

        page.close()
        ctx.close()

    browser.close()
