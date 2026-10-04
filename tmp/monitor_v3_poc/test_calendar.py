"""
Script auxiliar: test_calendar.py
Propósito: Explorar cómo SofaScore lista todos los partidos de la fecha o torneos del día.
"""

import sys
import time
import requests
import json

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

# Probar varios posibles endpoints de calendario
candidates = [
    "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-10-04",
    "https://api.sofascore.com/mobile/v4/sport/basketball/events/schedule/2026-10-04",
    "https://api.sofascore.com/api/v1/sport/basketball/categories",
    "https://api.sofascore.com/api/v1/sport/basketball/tournaments",
]
for c in candidates:
    r = session.get(c, headers=headers)
    print(f"{c} -> {r.status_code}")
    if r.status_code == 200:
        try:
            d = r.json()
            keys = list(d.keys())
            print(f"   Keys: {keys}")
        except Exception:
            pass
