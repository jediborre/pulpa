"""Extract and inspect initialProps from __NEXT_DATA__ for per-period stats."""
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
    time.sleep(4)

    html = page.content()
    browser.close()

match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
if match:
    data = json.loads(match.group(1))
    page_props = data.get("props", {}).get("pageProps", {})
    print(f"pageProps keys: {list(page_props.keys())}")

    initial_props = page_props.get("initialProps", {})
    print(f"\ninitialProps keys: {list(initial_props.keys())[:20]}")

    # Check for statistics data
    stats_data = initial_props.get("statistics")
    if stats_data:
        print(f"\nstatistics type: {type(stats_data).__name__}")
        if isinstance(stats_data, dict):
            print(f"statistics keys: {list(stats_data.keys())[:10]}")
            stats_list = stats_data.get("statistics", [])
            if stats_list:
                print(f"statistics[0] keys: {list(stats_list[0].keys())}")
                periods = [s.get("period") for s in stats_list if isinstance(s, dict)]
                print(f"Available periods: {periods}")
                # Show one stat per period
                for s in stats_list[:6]:
                    period = s.get("period")
                    groups = s.get("groups", [])
                    print(f"\n  Period {period}: {len(groups)} groups")
                    for g in groups[:3]:
                        items = g.get("statisticsItems", [])
                        print(f"    Group '{g.get('groupName')}': {len(items)} items")
                        for item in items[:3]:
                            print(f"      {item.get('name')}: {item.get('home')} vs {item.get('away')}")
        elif isinstance(stats_data, list):
            print(f"statistics list length: {len(stats_data)}")
            if stats_data and isinstance(stats_data[0], dict):
                print(f"first item keys: {list(stats_data[0].keys())}")
    else:
        # Search for anything containing stat/period/quarter
        print("\nNo 'statistics' key found. Searching for stat data...")
        def find_stat_keys(obj, path="", depth=0):
            results = []
            if depth > 5 or not isinstance(obj, (dict, list)):
                return results
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full = f"{path}.{k}" if path else k
                    if any(x in k.lower() for x in ["statistics", "stat", "period", "quarter", "stats"]):
                        results.append((full, type(v).__name__, 
                            list(v.keys())[:10] if isinstance(v, dict) else 
                            f"len={len(v)}" if isinstance(v, list) else str(v)[:100]))
                    results.extend(find_stat_keys(v, full, depth+1))
            else:  # list
                for i, item in enumerate(obj[:5]):
                    if isinstance(item, dict):
                        results.extend(find_stat_keys(item, f"{path}[{i}]", depth+1))
            return results

        for field, typ, val in find_stat_keys(initial_props):
            print(f"  {field} ({typ}) = {val}")

    # Save full initialProps for later inspection
    out_path = ROOT / "temp_scripts" / "full_initial_props.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(initial_props, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull initialProps saved to {out_path}")
else:
    print("No __NEXT_DATA__ found")
