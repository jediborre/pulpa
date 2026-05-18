"""Check all available team data endpoints on SofaScore."""
import sys, time, json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

team_ids = {
    "Pistons": 3424,
    "Cavaliers": 3432,
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    for name, tid in team_ids.items():
        print(f"\n{'='*60}")
        print(f"=== {name} (ID={tid}) ===")
        print(f"{'='*60}")

        # 1. /team/{id}
        for suffix, label in [
            ("", "Team base"),
            ("/performance", "Performance"),
            ("/statistics", "Statistics (season)"),
            ("/players", "Players"),
            ("/seasons", "Seasons"),
        ]:
            url = f"{BASE}/team/{tid}{suffix}"
            resp = ctx.request.get(url, headers=extra_headers, timeout=15_000)
            if resp.ok:
                body = resp.json()
                print(f"\n--- {label} ({url.split('/')[-1] or 'team'}) ---")
                if isinstance(body, dict):
                    # Show keys and a sample of values
                    for k, v in body.items():
                        if isinstance(v, dict):
                            print(f"  {k}: {json.dumps(list(v.keys())[:10])}... ({len(v)} keys)")
                        elif isinstance(v, list):
                            print(f"  {k}: list[{len(v)}] sample={json.dumps(v[:2], default=str)[:200]}")
                        elif isinstance(v, str):
                            print(f"  {k}: '{v[:100]}'")
                        elif v is None:
                            print(f"  {k}: null")
                        else:
                            print(f"  {k}: {v}")
                else:
                    print(f"  {json.dumps(body, default=str)[:300]}")
            else:
                print(f"\n--- {label} --- failed (status {resp.status})")
            time.sleep(0.3)

    browser.close()
