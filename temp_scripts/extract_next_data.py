"""Extract __NEXT_DATA__ from SofaScore match page to find odds + other data."""
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

# Extract __NEXT_DATA__
match = re.search(r'<script id="__NEXT_DATA__"[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL)
if not match:
    match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)

if match:
    raw = match.group(1)
    data = json.loads(raw)
    print(f"Top-level keys: {list(data.keys())}")

    # Navigate to props.pageProps
    props = data
    for key in ["props", "pageProps"]:
        if isinstance(props, dict) and key in props:
            props = props[key]
        else:
            break

    print(f"\npageProps top keys: {list(props.keys())[:30]}")

    # Look for odds, betting data
    def find_key(obj, target, path="", depth=0, max_depth=6):
        results = []
        if depth > max_depth:
            return results
        if isinstance(obj, dict):
            for k, v in obj.items():
                full = f"{path}.{k}" if path else k
                if any(t in k.lower() for t in target):
                    results.append((full, type(v).__name__, str(v)[:200] if not isinstance(v, (dict, list)) else ""))
                results.extend(find_key(v, target, full, depth+1, max_depth))
        elif isinstance(obj, list) and depth < max_depth:
            for i, item in enumerate(obj[:20]):
                results.extend(find_key(item, target, f"{path}[{i}]", depth+1, max_depth))
        return results

    print("\n=== Odds-related keys ===")
    for field, typ, val in find_key(props, ["odds", "bet", "market", "handicap", "spread", "bookmaker"]):
        print(f"  {field} ({typ}) = {val[:250]}")

    print("\n=== Statistics/period-related keys ===")
    for field, typ, val in find_key(props, ["period", "quarter", "statistics"], max_depth=5):
        if "statisticsItems" in field or len(val) > 5:
            continue
        print(f"  {field} ({typ}) = {val[:250]}")
else:
    print("No __NEXT_DATA__ found")
    # Save the HTML for inspection
    with open(ROOT / "temp_scripts" / "page.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML saved to temp_scripts/page.html")
