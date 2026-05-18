"""Extract JavaScript from the match page and search for per-quarter stats logic."""
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

    # Capture all JS file URLs
    js_urls = set()
    page.on("request", lambda req: js_urls.add(req.url) if ".js" in req.url.split("?")[0] else None)

    page.goto(
        f"https://www.sofascore.com/event/{match_id}",
        wait_until="domcontentloaded", timeout=30_000
    )
    time.sleep(5)

    # Get page source
    html = page.content()

    # Find all script src URLs
    script_srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
    print(f"Script sources in HTML: {len(script_srcs)}")

    # Also find _next/static/ chunks (Next.js chunks)
    next_chunks = [s for s in script_srcs if "_next/static/chunks" in s]
    print(f"Next.js chunks: {len(next_chunks)}")
    for url in next_chunks[:10]:
        print(f"  {url}")

    # Try fetching the main JS chunk that might have statistics logic
    main_chunk = None
    for url in next_chunks:
        if "framework" in url or "main" in url or "page" in url:
            main_chunk = url
            break

    # Also search for any JS that references "period" or "statisticsItems"
    # by fetching and scanning JS chunks
    import hashlib

    js_contents = {}
    for url in next_chunks:
        try:
            full_url = url if url.startswith("http") else f"https://www.sofascore.com{url}"
            resp = ctx.request.get(full_url, timeout=15_000)
            if resp.ok:
                text = resp.text()
                js_contents[url] = text
        except:
            pass

    print(f"\nDownloaded {len(js_contents)} JS chunks")

    # Search for per-period statistics keywords
    keywords = [
        "period", "statistics", "aggregat", "perPeriod", "periodStats",
        "statisticsItems", "reversedPeriodTime", "periodTimeSeconds",
        "computePeriodStats", "periodStatistics", "byPeriod"
    ]

    print("\n=== Searching JS for period/statistics logic ===")
    for url, content in js_contents.items():
        for kw in keywords:
            if kw in content.lower():
                # Find the context around the match
                idx = content.lower().find(kw)
                start = max(0, idx - 100)
                end = min(len(content), idx + 300)
                context = content[start:end]
                print(f"\n--- {url.split('/')[-1][:60]} | keyword: '{kw}' ---")
                print(f"  ...{context}...")
                break  # One match per file is enough

    browser.close()
