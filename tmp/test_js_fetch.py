"""
Ubicación original: scratch/test_js_fetch.py
Propósito / Qué hacía:
Prueba de evaluación JavaScript en Playwright (page.evaluate) para extracción segura.
"""

# -*- coding: utf-8 -*-
import sys
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

async def main():
    match_id = "16244040"
    # Visit the SofaScore landing or match page to establish session/cookies in the tab
    warmup_url = f"https://www.sofascore.com/event/{match_id}"
    print(f"Launching headful real Chrome to test JS-Fetch inside page context for match {match_id}...")
    
    async with async_playwright() as p:
        try:
            # Launch real Google Chrome
            browser = await p.chromium.launch(channel="chrome", headless=False)
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()
            
            # Warm up by visiting the match page in the tab
            print("Warming up page session...")
            await page.goto(warmup_url, wait_until="domcontentloaded", timeout=20000)
            
            # Delay to let Cloudflare check and page JS initialize
            print("Waiting for page session validation...")
            await asyncio.sleep(4)
            
            # Execute JS Fetch inside the tab context
            print("Evaluating JS fetch in tab context...")
            js_code = f"""
            fetch('https://api.sofascore.com/api/v1/event/{match_id}')
                .then(res => {{
                    if (!res.ok) throw new Error('HTTP status ' + res.status);
                    return res.json();
                }})
            """
            
            try:
                event_json = await page.evaluate(js_code)
                print("\nJS FETCH SUCCESS!")
                event = event_json.get("event", {})
                print("Home Team:", event.get("homeTeam", {}).get("name"))
                print("Away Team:", event.get("awayTeam", {}).get("name"))
                print("Status Type:", event.get("status", {}).get("type"))
                print("Status Description:", event.get("status", {}).get("description"))
                print("Score:", event.get("homeScore", {}).get("current"), "-", event.get("awayScore", {}).get("current"))
            except Exception as js_err:
                print("\nJS FETCH FAILED!")
                print("Error during page evaluation:", js_err)
                
            await ctx.close()
            await browser.close()
            
        except Exception as e:
            print("\nOuter Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
