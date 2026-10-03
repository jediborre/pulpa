"""
Ubicación original: scratch/test_headful_chrome.py
Propósito / Qué hacía:
Prueba de Chrome en modo visible (headful) para depuración visual de captchas.
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
    warmup_url = f"https://www.sofascore.com/event/{match_id}"
    print(f"Launching headful real Google Chrome (headless=False) to fetch match {match_id}...")
    
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    async with async_playwright() as p:
        try:
            # Launch real Google Chrome in headful (visible) mode
            browser = await p.chromium.launch(channel="chrome", headless=False)
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await ctx.new_page()
            
            # Warm up by visiting the match page
            print("Warming up session by visiting match page...")
            await page.goto(warmup_url, wait_until="domcontentloaded", timeout=15000)
            
            # Brief delay to let cookies settle
            await asyncio.sleep(2)
            
            # Fetch the Event API JSON
            print("Fetching Event API JSON...")
            resp = await ctx.request.get(
                f"https://api.sofascore.com/api/v1/event/{match_id}",
                headers=extra_headers,
                timeout=15000
            )
            
            print("Response Status Code:", resp.status)
            print("Response OK:", resp.ok)
            if resp.ok:
                event_json = await resp.json()
                event = event_json.get("event", {})
                print("\nSUCCESS!")
                print("Home Team:", event.get("homeTeam", {}).get("name"))
                print("Away Team:", event.get("awayTeam", {}).get("name"))
                print("Status Type:", event.get("status", {}).get("type"))
                print("Status Description:", event.get("status", {}).get("description"))
                print("Score:", event.get("homeScore", {}).get("current"), "-", event.get("awayScore", {}).get("current"))
            else:
                print("Failed with status:", resp.status)
                
            await ctx.close()
            await browser.close()
            
        except Exception as e:
            print("\nError occurred:", e)

if __name__ == "__main__":
    asyncio.run(main())
