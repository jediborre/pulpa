# -*- coding: utf-8 -*-
import http.client
import json

def fetch_direct():
    conn = http.client.HTTPSConnection("api.sofascore.com")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    # Fetch event metadata
    conn.request("GET", "/api/v1/event/16208417", headers=headers)
    res = conn.getresponse()
    print("Status code:", res.status)
    data = res.read()
    
    try:
        event_json = json.loads(data)
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
    except Exception as e:
        print("Parsing error:", e)
        print("Raw response first 500 chars:", data[:500])

if __name__ == "__main__":
    fetch_direct()
