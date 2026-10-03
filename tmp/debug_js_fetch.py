"""
Ubicación original: scratch/debug_js_fetch.py
Propósito / Qué hacía:
Prueba de inyección de fetch() en el contexto del navegador para evadir Cloudflare.
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
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chrome", headless=False)
            ctx = await browser.new_context()
            page = await ctx.new_page()
            
            await page.goto(warmup_url, wait_until="domcontentloaded")
            await asyncio.sleep(3)
            
            # Helper to execute fetch and print result
            async def run_fetch(name, url):
                print(f"\n--- Fetching {name} ({url}) ---")
                js_code = f"""
                fetch('{url}')
                    .then(res => {{
                        console.log('{name} res ok:', res.ok, 'status:', res.status);
                        if (!res.ok) {{
                            throw new Error('HTTP ' + res.status);
                        }}
                        return res.json();
                    }})
                """
                try:
                    result = await page.evaluate(js_code)
                    print(f"{name} Result Type: {type(result)}")
                    print(f"{name} Result (first 100 chars): {str(result)[:100]}")
                    return result
                except Exception as e:
                    print(f"{name} JS Error caught in Python: {e}")
                    return None
                    
            await run_fetch("Event", f"https://api.sofascore.com/api/v1/event/{match_id}")
            await run_fetch("Incidents", f"https://api.sofascore.com/api/v1/event/{match_id}/incidents")
            await run_fetch("Graph", f"https://api.sofascore.com/api/v1/event/{match_id}/graph")
            
            await ctx.close()
            await browser.close()
        except Exception as outer:
            print("Outer Error:", outer)

if __name__ == "__main__":
    asyncio.run(main())
