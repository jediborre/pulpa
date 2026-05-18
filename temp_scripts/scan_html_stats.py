"""Search match page HTML for any embedded per-quarter statistics data."""
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
    time.sleep(5)

    # Get full page content
    html = page.content()

    # Search for any script tags with JSON data
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
    print(f"Total script tags: {len(scripts)}")

    # Look for large JSON blobs
    for i, script in enumerate(scripts):
        script = script.strip()
        if script.startswith("{"):
            try:
                data = json.loads(script)
                print(f"\nScript {i}: JSON dict with keys: {list(data.keys())[:15]}")
                # Check for stats-related keys
                def find_key(obj, target, path="", depth=0):
                    if depth > 6:
                        return
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            full = f"{path}.{k}" if path else k
                            if any(t in k.lower() for t in target):
                                val_preview = ""
                                if isinstance(v, dict):
                                    val_preview = f"dict({len(v)} keys) {list(v.keys())[:5]}"
                                elif isinstance(v, list):
                                    val_preview = f"list({len(v)} items)"
                                elif isinstance(v, (int, float)):
                                    val_preview = str(v)
                                elif isinstance(v, str) and len(v) < 100:
                                    val_preview = v
                                else:
                                    val_preview = f"{type(v).__name__}({len(str(v))})"
                                print(f"    {full} = {val_preview}")
                            find_key(v, target, full, depth+1)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj[:5]):
                            find_key(item, target, f"{path}[{i}]", depth+1)

                find_key(data, ["period", "quarter", "statisticsItems", "homeValue", "awayValue", "homeTotal", "awayTotal"])
            except:
                pass
        elif script.startswith("window.__"):
            print(f"\nWindow script: {script[:100]}...")

    # Also search for any data-* attributes with period/quarter stats
    data_attrs = re.findall(r'data-[^=]+="[^"]{0,300}"', html)
    for attr in data_attrs:
        if any(x in attr.lower() for x in ["period", "quarter", "statistic"]):
            print(f"\nData attr: {attr[:200]}")

    # Search HTML for any period-related data
    for match in re.finditer(r'"period[12]?"\s*:\s*\{[^}]{0,500}', html):
        print(f"\nPeriod data block: {match.group(0)[:200]}")

    browser.close()
