"""Capture the REAL statistics API response from the browser during page load.
The user confirms per-quarter stats (rebounds, assists, etc.) ARE visible
on the page when clicking Q1. Let's find where they come from."""
import json, time, re
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

match_id = "14491671"

captured_responses = {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA, viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()

    def on_response(resp):
        url = resp.url
        if "api.sofascore.com" not in url:
            return
        # Only capture statistics responses
        if "statistics" in url.lower() or "incidents" in url.lower():
            try:
                body = resp.json()
                captured_responses[url] = body
            except:
                pass

    page.on("response", on_response)

    # Navigate to the match page
    page.goto(f"https://www.sofascore.com/event/{match_id}",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(5)

    # Print captured statistics responses
    print("=== Captured API responses ===")
    for url, body in captured_responses.items():
        print(f"\nURL: {url}")
        if isinstance(body, dict):
            if "statistics" in body:
                stats = body.get("statistics", [])
                print(f"  Statistics entries: {len(stats)}")
                for s in stats[:8]:
                    period = s.get("period", "?")
                    groups = s.get("groups", [])
                    print(f"  Period: {period}, Groups: {len(groups)}")
                    for g in groups[:5]:
                        items = g.get("statisticsItems", [])
                        print(f"    Group: {g.get('groupName')}, Items: {len(items)}")
                        for item in items[:5]:
                            print(f"      {item.get('name')}: home={item.get('home')} away={item.get('away')} "
                                  f"homeValue={item.get('homeValue')} awayValue={item.get('awayValue')}")
            elif "incidents" in body:
                incs = body.get("incidents", [])
                print(f"  Incidents: {len(incs)}")
                types = {}
                for i in incs:
                    t = i.get("incidentType", "?")
                    types[t] = types.get(t, 0) + 1
                print(f"  Types: {types}")
            else:
                print(f"  Keys: {list(body.keys())[:10]}")

    # Also try to click the Statistics tab and Q1, then capture any new data
    print("\n\n=== After clicking Statistics tab ===")
    try:
        page.click('a:has-text("Statistics")')
        time.sleep(2)

        # Click Q1
        page.click('text=Q1')
        time.sleep(2)

        # Check the page for any new data rendered
        # Look for the statistics table in the DOM
        stats_text = page.evaluate("""
            () => {
                const els = document.querySelectorAll('[class*="stat"]');
                return Array.from(els).slice(0, 20).map(el => el.textContent?.trim()).filter(Boolean);
            }
        """)
        print(f"Stats elements on page: {stats_text[:30]}")

        # Also check React props
        # Try to find the statistics data in the virtual DOM
        react_data = page.evaluate("""
            () => {
                try {
                    const root = document.getElementById('__NEXT_DATA__');
                    if (root) {
                        const data = JSON.parse(root.textContent);
                        const props = data.props?.pageProps?.initialProps;
                        // Check for statistics in props
                        if (props) {
                            const keys = Object.keys(props);
                            return 'Keys: ' + keys.join(', ');
                        }
                        return 'No initialProps';
                    }
                    return 'No __NEXT_DATA__';
                } catch(e) { return 'Error: ' + e.message; }
            }
        """)
        print(f"\nReact data: {react_data}")

    except Exception as e:
        print(f"Click error: {e}")

    browser.close()
