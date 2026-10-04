"""
Script auxiliar: test_schedule_endpoint.py
Propósito: Probar si el endpoint general /sport/basketball/scheduled-events/{date}
funciona directamente con el JWT móvil y el proxy HTTP Toolkit.
"""

import sys
import time
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROXIES = {
    'http': 'http://127.0.0.1:8000',
    'https': 'http://127.0.0.1:8000'
}
CERT = r'C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem'

# Cargar un token de prueba de test_token_generation o generar uno
def test_schedule():
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.verify = CERT

    # Obtener token nuevo
    import uuid
    headers = {
        'User-Agent': 'com.sofascore.results/260921/022538',
        'x-timestamp': str(int(time.time() * 1000)),
        'Content-Type': 'application/json; charset=UTF8',
    }
    payload = {
        "deviceType": "android",
        "version": 260921,
        "sdk": 29,
        "language": "en",
        "country": "MX",
        "timezone": -18000,
        "advertisingId": str(uuid.uuid4()),
        "uuid": str(uuid.uuid4())
    }
    res_token = session.post("https://api.sofascore.com/api/v1/token/init", headers=headers, json=payload, timeout=10)
    if res_token.status_code != 200:
        print("Error obteniendo token:", res_token.status_code)
        return
    token = res_token.json()["token"]

    auth_headers = {
        'User-Agent': 'com.sofascore.results/260921/022538',
        'x-timestamp': str(int(time.time() * 1000)),
        'Authorization': f'Bearer {token}',
        'Accept-Encoding': 'gzip',
    }

    # Probar scheduled events para hoy
    today = "2026-10-04"
    url = f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{today}"
    t0 = time.time()
    res = session.get(url, headers=auth_headers, timeout=10)
    print(f"GET {url} -> Status {res.status_code} ({time.time()-t0:.2f}s)")
    if res.status_code == 200:
        events = res.json().get("events", [])
        print(f"Total eventos programados de baloncesto para {today}: {len(events)}")
        if events:
            ev = events[0]
            print("Ejemplo de evento:", ev.get("id"), ev.get("homeTeam", {}).get("name"), "vs", ev.get("awayTeam", {}).get("name"), "| Liga:", ev.get("tournament", {}).get("name"))
    else:
        print("Respuesta:", res.text[:200])

if __name__ == '__main__':
    test_schedule()
