# -*- coding: utf-8 -*-
import sys
import asyncio
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright
from bet_monitor_v2.config.constants import SOFASCORE_SCRAPER_BACKEND

async def main():
    match_id = "16208417"
    ts = int(time.time() * 1000)
    url_event = f"https://api.sofascore.com/api/v1/event/{match_id}?_={ts}"
    url_incidents = f"https://api.sofascore.com/api/v1/event/{match_id}/incidents?_={ts}"
    
    print(f"Fetching {url_event} via Playwright (CDP/Obscura) with cache-busting...")
    
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    
    async with async_playwright() as p:
        # Connect over CDP
        cdp_url = "http://127.0.0.1:9222"
        browser = await p.chromium.connect_over_cdp(cdp_url)
        ctx = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # 1. Fetch Event
        resp_event = await ctx.request.get(url_event, headers=extra_headers)
        print("Event Response OK:", resp_event.ok, "Status:", resp_event.status)
        if resp_event.ok:
            event_json = await resp_event.json()
            event = event_json.get("event", {})
            print("Status Type:", event.get("status", {}).get("type"))
            print("Status Description:", event.get("status", {}).get("description"))
            print("Home Score:", event.get("homeScore", {}).get("current"))
            print("Away Score:", event.get("awayScore", {}).get("current"))
            print("Periods:")
            for i in range(1, 10):
                h = event.get("homeScore", {}).get(f"period{i}")
                a = event.get("awayScore", {}).get(f"period{i}")
                if h is not None or a is not None:
                    print(f"  Period {i}: Home {h} - Away {a}")
                    
        # 2. Fetch Incidents
        resp_inc = await ctx.request.get(url_incidents, headers=extra_headers)
        print("Incidents Response OK:", resp_inc.ok, "Status:", resp_inc.status)
        if resp_inc.ok:
            inc_json = await resp_inc.json()
            incidents = inc_json.get("incidents", [])
            print("Total Incidents:", len(incidents))
            if incidents:
                print("Newest Incident:", incidents[0])
                
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
