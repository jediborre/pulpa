"""Probe INCIDENTS endpoint for full data including missed shots/attempts."""
import json, time
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

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball",
               wait_until="networkidle", timeout=45_000)
    time.sleep(2)

    # Get FULL incidents
    resp = ctx.request.get(
        f"{BASE}/event/{match_id}/incidents",
        headers=extra_headers, timeout=30_000
    )
    if resp.ok:
        data = resp.json()
        incidents = data.get("incidents", [])
        print(f"Total incidents from API: {len(incidents)}")

        # All unique incident types
        from collections import Counter
        types = Counter(i.get("incidentType","?") for i in incidents)
        print(f"\nIncident types:")
        for t, c in types.most_common():
            print(f"  {t}: {c}")

        # All keys found
        all_keys = set()
        for i in incidents:
            all_keys.update(i.keys())
        print(f"\nAll possible keys: {sorted(all_keys)}")

        # Sample one of each incident type
        seen_types = set()
        for i in incidents:
            typ = i.get("incidentType")
            if typ not in seen_types:
                seen_types.add(typ)
                ic = i.get("incidentClass","")
                fr = i.get("from","")
                itext = i.get("text","")
                print(f"\n--- {typ} ---")
                print(f"  incidentClass={ic}, from={fr}")
                print(f"  team={i.get('team')}, teamId={i.get('teamId')}")
                print(f"  homeScore={i.get('homeScore')}, awayScore={i.get('awayScore')}")
                print(f"  isHome={i.get('isHome')}, isScored={i.get('isScored')}")
                print(f"  isShot={i.get('isShot')}, pointValue={i.get('pointValue')}")
                print(f"  period={i.get('period')}, reversedPeriodTime={i.get('reversedPeriodTime')}")
                print(f"  time={i.get('time')}, timeSeconds={i.get('timeSeconds')}")
                print(f"  text={itext[:100] if itext else ''}")
                # Check for additional keys
                extra = [k for k in i.keys() if k not in (
                    'incidentType','incidentClass','from','team','teamId',
                    'homeScore','awayScore','isHome','isScored','isShot',
                    'pointValue','period','reversedPeriodTime','time',
                    'timeSeconds','text','id','player','isLive','addedTime')]
                if extra:
                    print(f"  EXTRA KEYS: {extra}")
                    for ek in extra:
                        print(f"    {ek}={i.get(ek)}")

        # Check for 'missed' shots
        print(f"\n\n=== Missed shot search ===")
        misses = [i for i in incidents if i.get('isScored') == False]
        print(f"Misses (isScored=False): {len(misses)}")
        for m in misses[:5]:
            print(f"  type={m.get('incidentType')} class={m.get('incidentClass')} isShot={m.get('isShot')} isScored={m.get('isScored')}")

        # Check for 'isShot' field
        shots = [i for i in incidents if i.get('isShot') == True]
        print(f"\nShots (isShot=True): {len(shots)}")
        from collections import Counter as Cnt
        shot_types = Cnt((s.get('incidentType'), s.get('incidentClass'), s.get('isScored')) for s in shots)
        print("Shot type breakdown:")
        for (typ, cls, scored), cnt in shot_types.most_common():
            print(f"  {typ}/{cls}: isScored={scored} count={cnt}")

    browser.close()
