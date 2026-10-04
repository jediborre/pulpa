"""
PoC: Reproductor de Petición Capturada de la App Android SofaScore
Ubicación: tmp/monitor_v3_poc/replay_captured_request.py
Propósito:
    Permitir probar de forma inmediata las cabeceras reales extraídas durante la
    intercepción con HTTP Toolkit / Mitmproxy en la app de Android.
    
Instrucciones de uso:
    1. Reemplaza el diccionario CAPTURED_HEADERS con las cabeceras exactas obtenidas.
    2. Ejecuta: .venv\\Scripts\\python.exe tmp\\monitor_v3_poc\\replay_captured_request.py
    3. Si responde 200 OK y entrega JSON, la fórmula queda validada para monitor_v3.
"""

import sys
import time
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import httpx

# REEMPLAZAR ESTE DICCIONARIO CON LAS CABECERAS REALES CAPTURADAS EN HTTP TOOLKIT
CAPTURED_HEADERS = {
    # Ejemplo de estructura (actualizar con lo copiado de HTTP Toolkit):
    "User-Agent": "SofaScore/Android ...",
    "Accept": "application/json",
    # "X-So-...": "...",
    # "Authorization": "...",
}

# Match ID de prueba reciente (o el que hayas abierto en la app)
TEST_MATCH_ID = "15935071"

def test_replay():
    print("=" * 75)
    print("   PoC MONITOR V3: REPLAY DE PETICIÓN CAPTURADA DE ANDROID")
    print("=" * 75)
    
    if "..." in CAPTURED_HEADERS.get("User-Agent", ""):
        print("[AVISO] Aún no has pegado las cabeceras reales en CAPTURED_HEADERS.")
        print("        Pega los headers capturados en este script y vuelve a ejecutar.")
        print("=" * 75)
        return

    urls = {
        "Snapshot Evento": f"https://api.sofascore.com/api/v1/event/{TEST_MATCH_ID}",
        "Incidentes PBP": f"https://api.sofascore.com/api/v1/event/{TEST_MATCH_ID}/incidents",
        "Gráfica Momentum": f"https://api.sofascore.com/api/v1/event/{TEST_MATCH_ID}/graph",
    }
    
    with httpx.Client(headers=CAPTURED_HEADERS, timeout=10.0) as client:
        for name, url in urls.items():
            t0 = time.perf_counter()
            resp = client.get(url)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            if resp.status_code == 200:
                print(f"  [EXITO 200] {name:<18} | {elapsed_ms:>6.1f}ms | {len(resp.content)} bytes (JSON)")
                try:
                    data = resp.json()
                    print(f"              Llaves recibidas: {list(data.keys())[:5]}")
                except Exception:
                    pass
            else:
                print(f"  [{resp.status_code}]       {name:<18} | {elapsed_ms:>6.1f}ms | {resp.text[:120]}")
                
    print("=" * 75)

if __name__ == "__main__":
    test_replay()
