"""Find the bnp (API route builder) definition that constructs statistics URLs."""
import re, time, json
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

    page.goto(f"https://www.sofascore.com/event/{match_id}",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    html = page.content()
    script_srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
    next_chunks = [s for s in script_srcs if "_next/static/chunks" in s]

    # Look for the chunk that defines bnp (API builder)
    for url in next_chunks:
        try:
            full_url = url if url.startswith("http") else f"https://www.sofascore.com{url}"
            resp = ctx.request.get(full_url, timeout=15_000)
            if not resp.ok:
                continue
            text = resp.text()
            # Look for the definition of bnp or API route builder containing "event"+"statistics"
            if 'bnp=' in text or 'bnp={' in text or 'cWY' in text:
                # Find the bnp object definition
                idx = text.find('cWY')
                if idx < 0:
                    idx = text.find('bnp=')
                if idx < 0:
                    idx = text.find('bnp:')
                if idx >= 0:
                    context = text[max(0,idx-200):idx+1000]
                    # Find readable URLs/patterns
                    routes = re.findall(r'(?:[\"\'])(/[^\"\']*)(?:[\"\'])', context)
                    stat_routes = [r for r in routes if 'statistic' in r.lower() or 'event' in r.lower()]
                    print(f"\n=== {url.split('/')[-1][:60]} ===")
                    print(f"  bnp/cWY context found at idx={idx}")
                    for route in stat_routes[:20]:
                        print(f"  Route: {route}")

                    # Also print the readable part of context
                    printable = re.sub(r'[^\x20-\x7E]', ' ', context)
                    print(f"\n  Context snippet:")
                    print(f"  ...{printable[:1500]}...")
        except:
            pass

    browser.close()
