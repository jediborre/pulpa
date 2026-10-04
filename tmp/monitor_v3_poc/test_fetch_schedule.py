"""
Script auxiliar: test_fetch_schedule.py
Propósito: Validar la descarga completa del calendario de baloncesto para hoy (scheduled-events),
partidos en vivo (live) y detalle de un partido (incidents, graph, statistics)
utilizando las cabeceras móviles auténticas capturadas a través de HTTP Toolkit.
"""

import sys
import time
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'com.sofascore.results/260921/022538',
    'x-timestamp': str(int(time.time() * 1000)),
    'Accept-Encoding': 'gzip',
    'Cache-Control': 'max-age=0',
    'Connection': 'Keep-Alive',
    'Host': 'api.sofascore.com',
}

PROXIES = {
    'http': 'http://127.0.0.1:8000',
    'https': 'http://127.0.0.1:8000'
}

CERT = r'C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem'

def test_pipeline():
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.verify = CERT
    session.headers.update(HEADERS)

    # 1. Live Events
    url_live = "https://api.sofascore.com/api/v1/sport/basketball/events/live"
    t0 = time.time()
    res_live = session.get(url_live, timeout=10)
    print(f"[Live {res_live.status_code}] ({time.time() - t0:.2f}s) {url_live}")
    
    if res_live.status_code != 200:
        print("Error al obtener partidos en vivo:", res_live.text[:200])
        return

    live_events = res_live.json().get('events', [])
    print(f"Total de partidos de baloncesto EN VIVO en este instante: {len(live_events)}")
    
    sample_id = None
    for le in live_events:
        eid = le['id']
        home = le.get('homeTeam', {}).get('name', 'Home')
        away = le.get('awayTeam', {}).get('name', 'Away')
        tourn = le.get('tournament', {}).get('name', 'Tourn')
        status = le.get('status', {}).get('description', '')
        hs = le.get('homeScore', {}).get('current', 0)
        as_ = le.get('awayScore', {}).get('current', 0)
        print(f"  * [{eid}] {tourn} | {home} {hs} - {as_} {away} ({status})")
        if not sample_id:
            sample_id = eid

    if not sample_id:
        print("No hay partidos en vivo en este momento.")
        return

    # 2. Test Detail Endpoints for sample match
    print(f"\n--- Probando Ráfaga de Detalle para Match ID {sample_id} ---")
    detail_endpoints = [
        f"https://api.sofascore.com/api/v1/event/{sample_id}",
        f"https://api.sofascore.com/api/v1/event/{sample_id}/incidents",
        f"https://api.sofascore.com/api/v1/event/{sample_id}/graph",
        f"https://api.sofascore.com/api/v1/event/{sample_id}/lineups",
        f"https://api.sofascore.com/api/v1/event/{sample_id}/statistics",
    ]

    for ep in detail_endpoints:
        t0 = time.time()
        r = session.get(ep, timeout=10)
        ms = (time.time() - t0) * 1000
        print(f"[{r.status_code}] ({ms:.1f}ms) {ep}")
        if r.status_code == 200:
            data = r.json()
            keys = list(data.keys())
            item_count = len(data.get(keys[0], [])) if keys and isinstance(data.get(keys[0]), list) else "N/A"
            print(f"     -> JSON Claves: {keys} | items: {item_count}")
        else:
            print(f"     -> Error: {r.text[:150]}")

if __name__ == '__main__':
    test_pipeline()
