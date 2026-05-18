"""Capture API calls when clicking specific quarter tabs on the stats page."""
import json, time
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
    ctx = browser.new_context(
        user_agent=STANDARD_UA,
        viewport={"width": 1920, "height": 1080},
    )
    page = ctx.new_page()

    api_calls = set()
    def on_response(resp):
        url = resp.url
        if "api.sofascore" in url and "event" in url:
            api_calls.add(url.split("?")[0])

    page.on("response", on_response)

    # Go to match page
    page.goto(
        f"https://www.sofascore.com/event/{match_id}",
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    time.sleep(5)
    print(f"Final URL: {page.url}")

    # Take a screenshot to see what's on the page
    page.screenshot(path=ROOT / "temp_scripts" / "match_page.png")
    print("Screenshot saved")

    # Scroll down in case stats are below the fold
    page.evaluate("window.scrollTo(0, 600)")
    time.sleep(2)

    page.screenshot(path=ROOT / "temp_scripts" / "match_page_scrolled.png")

    # Try various quarter/label elements
    click_targets = [
        # Period/quarter tabs inside statistics
        ("text=Q1", "text Q1"),
        ("text=Q2", "text Q2"),
        ("text=Q3", "text Q3"),
        ("text=Q4", "text Q4"),
        ("button:has-text('1')", "button 1"),
        ("button:has-text('2')", "button 2"),
        ("button:has-text('3')", "button 3"),
        ("button:has-text('4')", "button 4"),
    ]

    for selector, label in click_targets:
        try:
            el = page.query_selector(selector)
            if el:
                print(f"\nFound clickable: {label} ({selector})")
                el.click()
                time.sleep(2)
        except Exception as e:
            pass

    # Print all captured API calls
    print(f"\n\n=== All API calls captured ===")
    for url in sorted(api_calls):
        print(f"  {url}")

    # Check if any new calls appeared
    print(f"\n=== Calls with odds/period/stat ===")
    for url in sorted(api_calls):
        low = url.lower()
        if any(x in low for x in ["odds", "period", "stat", "quarter", "bet"]):
            print(f"  {url}")

    browser.close()
