"""Find the exact API call pattern for per-quarter stats in the JS."""
import re, time
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
    next_chunks = [s for s in script_srcs if "_next/static/chunks" in s and s.endswith(".js")]

    # Download and search for the statistics API call builder
    for url in next_chunks:
        try:
            full_url = url if url.startswith("http") else f"https://www.sofascore.com{url}"
            resp = ctx.request.get(full_url, timeout=15_000)
            if not resp.ok:
                continue
            text = resp.text()
            # Look for "statistics" keyword combined with URL/building
            if "statistics" not in text:
                continue
            # Search for patterns that build statistics API URLs
            patterns = [
                r'statistics[^)]{0,50}\)', 
                r'"/statistics"',
                r"'/statistics'",
                r'`.*statistics.*`',
                r'bnp\.[a-z]*statistics',
                r'[a-zA-Z]+\.statistics\s*=',
                r'period.*statistics',
                r'statistics.*period',
            ]
            for pat in patterns:
                matches = re.findall(pat, text)
                for m in matches[:5]:
                    if 'statistics' in m and len(m) < 200:
                        # Get more context
                        idx = text.find(m)
                        ctx_text = text[max(0,idx-100):idx+200]
                        print(f"\n--- {url.split('/')[-1][:60]} ---")
                        print(f"  Pattern: {pat}")
                        print(f"  Match: {m[:150]}")
                        print(f"  Context: {ctx_text}")
        except:
            pass

    browser.close()
