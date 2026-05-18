"""Extract full initialState structure to find odds data."""
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

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()

    page.goto(
        f"https://www.sofascore.com/event/{match_id}",
        wait_until="domcontentloaded", timeout=30_000
    )
    time.sleep(3)

    html = page.content()
    browser.close()

match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
if match:
    data = json.loads(match.group(1))
    state = data.get("props", {}).get("pageProps", {}).get("initialState", {})

    print("=== initialState top keys ===")
    for k, v in state.items():
        if isinstance(v, dict):
            print(f"  {k}: dict ({len(v)} keys) {list(v.keys())[:8]}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
        else:
            print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")

    # Full odds state
    print("\n=== initialState.odds (FULL) ===")
    print(json.dumps(state.get("odds", {}), indent=2, ensure_ascii=False)[:2000])

    # Save full state for exploration
    with open(ROOT / "temp_scripts" / "state_dump.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    print("\nFull state saved to temp_scripts/state_dump.json")
