"""
PoC: Prueba de Peticiones HTTP con Cabeceras Móviles (Android) vs Web contra SofaScore API
Ubicación: tmp/monitor_v3_poc/test_mobile_headers.py
Propósito:
    Evaluar la respuesta de las APIs de SofaScore (calendario, evento, incidentes, gráfica)
    ante diferentes configuraciones de red y cabeceras:
    1. Baseline sin cabeceras (Control)
    2. Cabeceras estándar de navegador Web Desktop (Chrome)
    3. Cabeceras de aplicación nativa Android (OkHttp / SofaScore Android)
    4. Variaciones con y sin Referer / Origin

Hipótesis:
    Las peticiones con cabeceras de aplicación móvil nativa pueden no ser sometidas
    a los mismos retos interactivos de JavaScript (Cloudflare Turnstile) que se le imponen
    a los navegadores web de escritorio, permitiendo la extracción directa en JSON.
"""

import sys
import time
import json
import sqlite3
from pathlib import Path

# Configurar salida UTF-8 para consola Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import httpx

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "matches.db"

# Obtener un match_id real reciente de matches.db para probar
def get_sample_match_id() -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT match_id FROM matches ORDER BY date DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            return str(row[0])
    except Exception:
        pass
    return "15935071" # Fallback a Knicks vs Spurs

# Perfiles de Cabeceras a evaluar
PROFILES = {
    "1. Control (Sin Cabeceras)": {},
    
    "2. Web Desktop (Chrome 133)": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
        "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    },
    
    "3. Android App (OkHttp Nativo)": {
        "User-Agent": "okhttp/4.12.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    },
    
    "4. SofaScore Android App (Custom UA)": {
        "User-Agent": "SofaScore/Android 12.3.1 (Linux; Android 14; Pixel 8 Build/UD1A.230803.041)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "X-Requested-With": "com.sofascore.app",
        "Connection": "Keep-Alive",
    },

    "5. Android Dalvik / VM": {
        "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 14; Pixel 8 Build/UD1A.230803.041)",
        "Accept": "application/json",
        "Connection": "Keep-Alive",
    }
}

def test_endpoint(client: httpx.Client, name: str, url: str, headers: dict) -> dict:
    start = time.perf_counter()
    try:
        resp = client.get(url, headers=headers, timeout=8.0, follow_redirects=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        server = resp.headers.get("server", "unknown")
        cf_ray = resp.headers.get("cf-ray", "none")
        content_type = resp.headers.get("content-type", "")
        
        is_json = "application/json" in content_type
        preview = ""
        if resp.status_code == 200:
            preview = f"OK ({len(resp.content)} bytes, JSON: {is_json})"
        elif resp.status_code == 403:
            preview = f"403 FORBIDDEN (Cloudflare Ray: {cf_ray})"
        else:
            preview = f"{resp.status_code} ({len(resp.content)} bytes)"
            
        return {
            "profile": name,
            "status": resp.status_code,
            "time_ms": elapsed,
            "server": server,
            "cf_ray": cf_ray,
            "preview": preview,
            "ok": resp.status_code == 200
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        return {
            "profile": name,
            "status": "ERROR",
            "time_ms": elapsed,
            "server": "error",
            "cf_ray": "none",
            "preview": f"Error: {type(exc).__name__} - {exc}",
            "ok": False
        }

def main():
    match_id = get_sample_match_id()
    print("=" * 75)
    print("   PoC MONITOR V3: EVALUACIÓN DE CABECERAS MÓVILES VS CLOUDFLARE")
    print(f"   Match ID de prueba: {match_id}")
    print(f"   Base de datos: {DB_PATH.name}")
    print("=" * 75)
    
    endpoints = {
        "Snapshot Evento": f"https://api.sofascore.com/api/v1/event/{match_id}",
        "Incidents (PBP)": f"https://api.sofascore.com/api/v1/event/{match_id}/incidents",
        "Graph Momentum": f"https://api.sofascore.com/api/v1/event/{match_id}/graph",
    }
    
    with httpx.Client() as client:
        for ep_name, ep_url in endpoints.items():
            print(f"\n[ENDPOINT] {ep_name}")
            print(f"URL: {ep_url}")
            print("-" * 75)
            
            for prof_name, headers in PROFILES.items():
                res = test_endpoint(client, prof_name, ep_url, headers)
                status_color = "[OK]" if res["ok"] else f"[{res['status']}]"
                print(f"  {status_color:<7} {prof_name:<38} | {res['time_ms']:>6.1f}ms | {res['preview']}")
                time.sleep(0.3) # Cortesía entre ráfagas

    print("\n" + "=" * 75)
    print("   Fin de la Prueba de Concepto Inicial.")
    print("=" * 75)

if __name__ == "__main__":
    main()
