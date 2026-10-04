"""
tmp/debug_connection/test_curl_cffi.py
Propósito: Probar si curl_cffi con impersonación TLS (evitando la huella de python-httpx)
puede consultar directamente api.sofascore.com sin necesidad de HTTP Toolkit en puerto 8000.
"""
import json
from pathlib import Path
from curl_cffi import requests

def test():
    tokens_file = Path("monitor_v3/config/tokens.json")
    with open(tokens_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    token = data["tokens"][0]["token"]
    
    headers = {
        "User-Agent": "com.sofascore.results/260921/022538",
        "Authorization": f"Bearer {token}",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    url = "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-09-01"
    
    impersonates = ["chrome", "chrome124", "safari17_0", "okhttp4_android"]
    for imp in impersonates:
        print(f"\nProbando con impersonate='{imp}'...")
        try:
            r = requests.get(url, headers=headers, impersonate=imp, timeout=10)
            print(f"Status ({imp}): {r.status_code}")
            if r.status_code == 200:
                events = r.json().get("events", [])
                print(f"¡ÉXITO ROTUNDO! Eventos encontrados: {len(events)}")
                return
            else:
                print(f"Respuesta ({imp}): {r.text[:200]}")
        except Exception as e:
            print(f"Error ({imp}): {e}")

if __name__ == "__main__":
    test()
