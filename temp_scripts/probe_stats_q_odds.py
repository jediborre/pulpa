"""Probe statistics by quarter and find odds endpoint."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

match_id = "14491671"
BASE = "https://api.sofascore.com/api/v1"

endpoints = [
    # Per-quarter statistics
    f"/event/{match_id}/statistics",
    f"/event/{match_id}/statistics?period=Q1",
    f"/event/{match_id}/statistics?period=Q2",
    f"/event/{match_id}/statistics?period=3",
    f"/event/{match_id}/statistics?period=4",

    # Odds / betting
    f"/event/{match_id}/odds",
    f"/event/{match_id}/betting/odds",
    f"/event/{match_id}/prematch-odds",
    f"/event/{match_id}/bookmaker",
    f"/event/{match_id}/predictions",

    # Another H2H format
    f"/event/{match_id}/h2h/teams",
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass

    for ep in endpoints:
        url = f"{BASE}{ep}"
        resp = ctx.request.get(url, headers=extra_headers, timeout=20_000)
        status = resp.status
        ok = resp.ok
        msg = ""
        if ok:
            body = resp.json()
            if isinstance(body, dict):
                msg = f"Keys: {list(body.keys())[:8]}"
                # Check for period/quarter data
                stats = body.get("statistics", [])
                if stats and isinstance(stats, list):
                    periods = [s.get("period") for s in stats if isinstance(s, dict)]
                    msg += f" | Periods: {periods}"
            elif isinstance(body, list):
                msg = f"List of {len(body)} items"
        else:
            msg = f"HTTP {status}"
        print(f"[{'OK' if ok else 'FAIL'}] {ep}")
        print(f"       {msg}\n")

    browser.close()
