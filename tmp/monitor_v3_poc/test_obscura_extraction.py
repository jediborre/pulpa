"""
Script auxiliar: test_obscura_extraction.py
Propósito: Probar la extracción de la información que Obscura no pudo extraer
(endpoints de partidos históricos y calendario que fallaban con 403 Forbidden)
utilizando el token JWT móvil de 6 meses y las cabeceras móviles capturadas.
"""

import sys
import time
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9."
    "eyJpYXQiOjE3OTEwOTQ1MjEsImV4cCI6MTgwNjg3NDUyMSwiaWQiOiI1OTdjZDI3NC1iYmM2LTRmYjUtYTkxOC00NmEzM2Q0MzE5YmYifQ."
    "MzuzpmP_EdP9Q4foBSusBgCLx-fGAQgRU6rL3n_L36zNfUbK5fw5FjPgkkqXZbuXVoEL2h2Q_sdJ-mWrfbYUvez1QNhBU-WugcHRawX38gLtqoUeTuBVP4GsNF1DH42V15sjKEZhOZpHtKQtBZZGmFubdgr5NOHg99DKhegMxt7di_CYrcu2id10Hf2xyxY8kHdrFYqet9vJM_40De5yHvd-4JoWTiF7opsmhFu8509gq-4b7crrIWgfn-5gRST_5p03hSBWvLp3dxGOM9GjBs20CnIfUazlN37AK_-JoURqJRThxyutplWp-jmL0RwmrOEXUI8vqSue1c_rWq1P5JG28k4gUrca_tneJDNzKxPL8aE37LOEDaK-cMIKTwel8CSI9Co1CYf2DcKjd3U-6yTyBI4OUs2r5_DrTbFYhM1d3E3Gb2u7y953TiowJTzwq2SZO95oKKk_KxTl0pr0hI3j-X8e3PWFaZXur7YU38wOwEMDIRanu1q7ie5ee0RwybFDLKAdxkhbXzaiiPjobVakL_X_TsOOFvtm1qsVO3AF0UBOlD9Gobr6gLjYKwkBktO8EEPrsRX-kHJqLKAmKVZ_sIb3yUmDtpshxTBklvh0OxoUPXQf9v_N2HIWiAqpv9WQRIzVPNUc4RrVGiP4rt4wLQwVR5-lxVdwxi6ipKM"
)

HEADERS = {
    'User-Agent': 'com.sofascore.results/260921/022538',
    'x-timestamp': str(int(time.time() * 1000)),
    'Accept-Encoding': 'gzip',
    'Cache-Control': 'max-age=0',
    'Connection': 'Keep-Alive',
    'Host': 'api.sofascore.com',
    'Authorization': f'Bearer {TOKEN}',
}

PROXIES = {
    'http': 'http://127.0.0.1:8000',
    'https': 'http://127.0.0.1:8000'
}

CERT = r'C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem'

session = requests.Session()
session.proxies.update(PROXIES)
session.verify = CERT
session.headers.update(HEADERS)

def run_tests():
    print("=" * 60)
    print("TEST 1: Descarga del calendario histórico (lo que fallaba en Obscura con 403)")
    print("=" * 60)
    
    dates_to_test = ["2026-05-28", "2026-06-11", "2026-10-03"]
    for d in dates_to_test:
        url = f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{d}"
        t0 = time.time()
        r = session.get(url, timeout=10)
        ms = (time.time() - t0) * 1000
        print(f"[{r.status_code}] ({ms:.1f}ms) {url}")
        if r.status_code == 200:
            events = r.json().get('events', [])
            print(f"   -> ÉXITO TOTAL: {len(events)} partidos encontrados para {d}!")
            for ev in events[:2]:
                print(f"      * {ev.get('homeTeam', {}).get('name')} vs {ev.get('awayTeam', {}).get('name')}")
        else:
            print(f"   -> Respuesta: {r.text[:120]}")

    print("\n" + "=" * 60)
    print("TEST 2: Extracción atómica de partido histórico que Obscura no podía descargar")
    print("Ejemplo: Match 15935071 (Knicks vs Spurs)")
    print("=" * 60)
    
    match_id = "15935071"
    match_endpoints = [
        ("Metadata y Marcadores", f"https://api.sofascore.com/api/v1/event/{match_id}"),
        ("Play-by-Play (Incidents)", f"https://api.sofascore.com/api/v1/event/{match_id}/incidents"),
        ("Momentum Graph", f"https://api.sofascore.com/api/v1/event/{match_id}/graph"),
        ("Head-to-Head (H2H)", f"https://api.sofascore.com/api/v1/event/{match_id}/h2h"),
        ("Lineups", f"https://api.sofascore.com/api/v1/event/{match_id}/lineups"),
        ("Estadísticas", f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"),
    ]

    for name, ep in match_endpoints:
        t0 = time.time()
        r = session.get(ep, timeout=10)
        ms = (time.time() - t0) * 1000
        print(f"[{r.status_code}] ({ms:.1f}ms) {name}: {ep}")
        if r.status_code == 200:
            data = r.json()
            keys = list(data.keys())
            print(f"   -> Datos recibidos: {keys}")
        else:
            print(f"   -> Fallo: {r.text[:120]}")

    print("\n" + "=" * 60)
    print("TEST 3: Consulta Directa SIN PROXY (Python puro hacia Cloudflare con JWT)")
    print("=" * 60)
    
    s_direct = requests.Session()
    s_direct.headers.update(HEADERS)
    for name, ep in match_endpoints[:3]:
        t0 = time.time()
        try:
            r = s_direct.get(ep, timeout=10)
            ms = (time.time() - t0) * 1000
            print(f"[{r.status_code}] ({ms:.1f}ms) DIRECTO {name}")
            if r.status_code == 200:
                print(f"   -> ÉXITO DIRECTO SIN PROXY: {list(r.json().keys())}")
            else:
                print(f"   -> Error: {r.text[:100]}")
        except Exception as e:
            print(f"   -> Error excepción: {e}")

if __name__ == '__main__':
    run_tests()
