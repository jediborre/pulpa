"""Deeper search: find the exact function that computes per-quarter stats."""
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

    js_urls = set()
    page.on("request", lambda req: js_urls.add(req.url) if ".js" in req.url.split("?")[0] else None)

    page.goto(f"https://www.sofascore.com/event/{match_id}",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    html = page.content()
    script_srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
    next_chunks = [s for s in script_srcs if "_next/static/chunks" in s]

    print(f"Chunks: {len(next_chunks)}")

    # Download the most relevant chunks for statistics
    stat_chunks = {}
    for url in next_chunks:
        try:
            full_url = url if url.startswith("http") else f"https://www.sofascore.com{url}"
            resp = ctx.request.get(full_url, timeout=15_000)
            if resp.ok:
                text = resp.text()
                # Only keep chunks containing stat-related keywords
                if any(kw in text for kw in [".statistics", "period1", "statisticsItems", "homeValue", "awayValue"]):
                    stat_chunks[url.split("/")[-1][:80]] = text
        except:
            pass

    print(f"Stat-related chunks: {len(stat_chunks)}")

    # Search each for per-period stat computation
    # Look for: statistics data being filtered/split by period
    patterns = [
        r'\.statistics[\s\S]{0,500}period',
        r'period[\s\S]{0,300}\.statistics',
        r'statisticsItems[\s\S]{0,500}period',
        r'period[12]?[\s\S]{0,300}homeValue',
        r'getPeriod[^)]*\)[\s\S]{0,200}stat',
        r'byPeriod[\s\S]{0,300}',
        r'selectedPeriod[\s\S]{0,500}stat',
        r'periodData[\s\S]{0,500}',
    ]

    for name, content in stat_chunks.items():
        for pat in patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            for m in matches[:3]:
                print(f"\n--- {name} | pattern: {pat[:30]} ---")
                print(f"  {m[:400]}")

    # Also look for the specific function that transforms statistics by period
    # Search for "ALL" period handling
    for name, content in stat_chunks.items():
        if "\"ALL\"" in content or "'ALL'" in content:
            idx = content.find("\"ALL\"")
            cxt = content[max(0,idx-200):idx+400]
            print(f"\n--- {name} | 'ALL' context ---")
            print(f"  {cxt}")

    browser.close()
