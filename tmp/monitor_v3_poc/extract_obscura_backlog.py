"""
Script auxiliar: extract_obscura_backlog.py
Propósito: Demostrar la extracción completa de la información que Obscura no pudo extraer.
Toma los últimos 5 partidos históricos de matches.db y descarga sus 6 endpoints
(Metadata, Incidents/PBP, Graph/Momentum, H2H, Lineups, Statistics) midiendo tiempos y bytes.
"""

import sys
import time
import requests
import json
from pathlib import Path

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

MATCH_IDS = [
    ("15935071", "New York Knicks vs San Antonio Spurs"),
    ("16078015", "Criollos de Caguas vs Mets de Guaynabo"),
    ("16078014", "Gigantes de Carolina vs Atléticos de San Germán"),
    ("16256125", "Urupan vs Olivol Mundial"),
    ("15395177", "Montreal Alliance vs Brampton Honey Badgers")
]

def extract_matches():
    s = requests.Session()
    s.proxies.update(PROXIES)
    s.verify = CERT
    s.headers.update(HEADERS)

    results = []

    print(f"Iniciando extracción de {len(MATCH_IDS)} partidos históricos que Obscura no pudo procesar...\n")

    for match_id, match_name in MATCH_IDS:
        print(f"=== Extrayendo Match {match_id}: {match_name} ===")
        t_match_start = time.perf_counter()
        
        endpoints = {
            "metadata": f"https://api.sofascore.com/api/v1/event/{match_id}",
            "incidents": f"https://api.sofascore.com/api/v1/event/{match_id}/incidents",
            "graph": f"https://api.sofascore.com/api/v1/event/{match_id}/graph",
            "h2h": f"https://api.sofascore.com/api/v1/event/{match_id}/h2h",
            "lineups": f"https://api.sofascore.com/api/v1/event/{match_id}/lineups",
            "statistics": f"https://api.sofascore.com/api/v1/event/{match_id}/statistics",
        }

        match_data = {}
        all_ok = True

        for key, url in endpoints.items():
            s.headers['x-timestamp'] = str(int(time.time() * 1000))
            t0 = time.perf_counter()
            r = s.get(url, timeout=10)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if r.status_code == 200:
                body = r.json()
                match_data[key] = body
                print(f"   [{key:10s}] -> 200 OK ({elapsed_ms:5.1f}ms) | {len(r.content):6d} bytes")
            else:
                all_ok = False
                print(f"   [{key:10s}] -> {r.status_code} ERROR ({elapsed_ms:5.1f}ms)")

        t_total_match = (time.perf_counter() - t_match_start) * 1000
        
        # Resumen del partido extraído
        event_info = match_data.get("metadata", {}).get("event", {})
        status = event_info.get("status", {}).get("description", "N/A")
        home_score = event_info.get("homeScore", {}).get("current", "-")
        away_score = event_info.get("awayScore", {}).get("current", "-")
        pbp_count = len(match_data.get("incidents", {}).get("incidents", []))
        gp_count = len(match_data.get("graph", {}).get("graphPoints", []))

        print(f"   -> Marcador Final: {home_score} - {away_score} ({status})")
        print(f"   -> Jugadas PBP: {pbp_count} | Puntos Momentum: {gp_count}")
        print(f"   -> Tiempo total descarga ráfaga completa: {t_total_match:.1f}ms\n")

        results.append({
            "id": match_id,
            "name": match_name,
            "ok": all_ok,
            "time_ms": t_total_match,
            "pbp": pbp_count,
            "gp": gp_count
        })

    print("=" * 65)
    print("RESUMEN DE EXTRACCIÓN MASIVA (BACKLOG OBSCURA)")
    print("=" * 65)
    for res in results:
        status_sym = "✅" if res["ok"] else "❌"
        print(f"{status_sym} [{res['id']}] {res['name'][:40]:40s} | {res['time_ms']:5.1f}ms | PBP: {res['pbp']:3d} | GP: {res['gp']:3d}")

if __name__ == '__main__':
    extract_matches()
