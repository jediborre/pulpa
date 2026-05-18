"""Inspect all 33 top-level keys of the event JSON + try team endpoints for H2H history."""
import json, sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

MATCH_ID = "14491671"
BASE = "https://api.sofascore.com/api/v1"

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

out = {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ))
    page = ctx.new_page()
    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass

    # 1. Get full event JSON and inspect all keys
    resp = ctx.request.get(f"{BASE}/event/{MATCH_ID}", headers=extra_headers, timeout=15_000)
    if resp.ok:
        ev = resp.json().get("event", {})
        out["event_keys"] = {k: type(v).__name__ for k, v in ev.items()}
        # Show sample values for dict/list types
        for k, v in ev.items():
            if isinstance(v, dict):
                out[f"event_{k}_keys"] = list(v.keys())[:15]
                out[f"event_{k}_sample"] = str(v)[:300]
            elif isinstance(v, list):
                out[f"event_{k}_len"] = len(v)
                if v and isinstance(v[0], dict):
                    out[f"event_{k}_sample_keys"] = list(v[0].keys())[:15]
                elif v:
                    out[f"event_{k}_sample"] = str(v[0])[:200]

    # 2. Try team endpoints for H2H match history
    # Team IDs from event: home=3560, away=25134
    team_endpoints = [
        f"/team/3560/events/last/10",
        f"/team/3560/events/h2h/25134",
        f"/team/3560/h2h/25134",
    ]
    for ep in team_endpoints:
        try:
            resp = ctx.request.get(f"{BASE}{ep}", headers=extra_headers, timeout=15_000)
            if resp.ok:
                body = resp.json()
                out[f"team_ep_{ep}"] = {
                    "status": resp.status,
                    "keys": list(body.keys())[:10],
                    "sample": json.dumps(body, indent=2, ensure_ascii=False)[:2000],
                }
            else:
                out[f"team_ep_{ep}"] = {"status": resp.status, "keys": None}
        except Exception as e:
            out[f"team_ep_{ep}"] = {"status": "ERROR", "error": str(e)}

    browser.close()

with open(ROOT / "temp_scripts" / "probe_results2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

# Print summary
print("=== EVENT TOP-LEVEL KEYS ===")
for k, v in out.get("event_keys", {}).items():
    print(f"  {k}: {v}")

print("\n=== TEAM ENDPOINTS ===")
for k, v in out.items():
    if k.startswith("team_ep"):
        print(f"\n{k}:")
        if v.get("keys"):
            print(f"  Status: {v['status']}")
            print(f"  Keys: {v['keys']}")
            print(f"  Sample: {v['sample'][:500]}")
        else:
            print(f"  Status: {v.get('status')}")
