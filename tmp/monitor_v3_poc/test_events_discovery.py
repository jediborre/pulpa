"""
Script auxiliar: test_events_discovery.py
Propósito: Probar diferentes variantes de endpoints de partidos programados y en vivo
para entender la estructura exacta que responde la API de SofaScore.
"""

import sys
import time
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROXIES = {'http': 'http://127.0.0.1:8000', 'https': 'http://127.0.0.1:8000'}
CERT = r'C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem'

session = requests.Session()
session.proxies.update(PROXIES)
session.verify = CERT

headers = {
    'User-Agent': 'com.sofascore.results/260921/022538',
    'x-timestamp': str(int(time.time() * 1000)),
    'Accept-Encoding': 'gzip',
}

# 1. Probar /sport/basketball/events/live
res_live = session.get('https://api.sofascore.com/api/v1/sport/basketball/events/live', headers=headers)
print(f"Live status: {res_live.status_code}")
if res_live.status_code == 200:
    live_events = res_live.json().get('events', [])
    print(f"Live events count: {len(live_events)}")
    if live_events:
        ev0 = live_events[0]
        tourn_id = ev0.get('tournament', {}).get('uniqueTournament', {}).get('id')
        print(f"Unique Tournament ID: {tourn_id}, Name: {ev0.get('tournament', {}).get('name')}")
        # Probar endpoint de torneo programado
        if tourn_id:
            url_tourn = f"https://api.sofascore.com/api/v1/unique-tournament/{tourn_id}/scheduled-events/2026-10-04"
            res_t = session.get(url_tourn, headers=headers)
            print(f"Unique tournament scheduled events: {res_t.status_code}")

# 2. Probar variantes de scheduled-events
test_urls = [
    "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-10-04",
    "https://api.sofascore.com/api/v1/sport/basketball/events/schedule/2026-10-04",
    "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events",
    "https://api.sofascore.com/api/v1/sport/basketball/events/2026-10-04",
]
for u in test_urls:
    r = session.get(u, headers=headers)
    print(f"{u} -> {r.status_code}")
