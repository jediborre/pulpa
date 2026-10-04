"""
tmp/debug_connection/test_direct.py
Propósito: Verificar conectividad directa con SofaScore API usando httpx / requests / curl_cffi
y los tokens JWT extraídos de Android, sin depender de proxy en el puerto 8000.
Hipótesis: Determinar si la API móvil requiere proxy local o si admite peticiones directas
con los headers móviles y tokens válidos.
"""
import asyncio
import json
import httpx
from pathlib import Path

async def test_direct_request():
    tokens_file = Path("monitor_v3/config/tokens.json")
    if not tokens_file.exists():
        print("tokens.json no existe")
        return
    with open(tokens_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokens = data.get("tokens", [])
    if not tokens:
        print("No hay tokens en tokens.json")
        return
        
    token = tokens[0]["token"]
    print(f"Probando token: ...{token[-15:]}")
    
    headers = {
        "User-Agent": "com.sofascore.results/260921/022538",
        "Authorization": f"Bearer {token}",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    
    url = "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-09-01"
    
    # 1. Direct httpx request (no proxy)
    print("\n--- 1. httpx DIRECTO (sin proxy) ---")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                events = resp.json().get("events", [])
                print(f"¡ÉXITO DIRECTO! Eventos obtenidos: {len(events)}")
            else:
                print(f"Respuesta ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"Error directo: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_request())
