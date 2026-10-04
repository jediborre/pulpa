"""
PoC: Análisis Estático de Cadenas, Endpoints y Cabeceras en Sofascore.apk
Ubicación: tmp/monitor_v3_poc/analyze_apk.py
Propósito:
    Inspeccionar los archivos DEX (código compilado) del APK oficial de SofaScore
    para extraer:
    1. URLs base y endpoints (/api/v1/..., api.sofascore.com, otros dominios móviles).
    2. Cabeceras HTTP personalizadas (X-So-..., User-Agent, Authorization, etc.).
    3. Nombres de clases de red y configuración de OkHttp/Retrofit.

Hipótesis:
    El APK contiene las URLs y cabeceras exactas que utiliza el cliente móvil,
    lo que nos permitirá replicar la petición en Python sin depender de conjeturas.
"""

import sys
import re
import zipfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
APK_PATH = ROOT / "tmp" / "monitor_v3_poc" / "Sofascore.apk"

def extract_strings_from_dex(dex_bytes: bytes, min_len: int = 4) -> list[str]:
    # Extraer strings ASCII/UTF-8 imprimibles del binario DEX
    pattern = re.compile(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}')
    matches = pattern.findall(dex_bytes)
    return [m.decode('latin1', errors='ignore') for m in matches]

def analyze_apk():
    print("=" * 75)
    print("   PoC MONITOR V3: ANÁLISIS ESTÁTICO DE SOFASCORE APK")
    print(f"   Archivo: {APK_PATH} ({APK_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 75)

    all_strings = set()

    with zipfile.ZipFile(APK_PATH, 'r') as z:
        namelist = z.namelist()
        dex_files = [f for f in namelist if f.endswith('.dex')]
        print(f"Total archivos en APK: {len(namelist)} | Archivos DEX de código: {len(dex_files)}")
        
        for dex_name in dex_files:
            dex_data = z.read(dex_name)
            strings = extract_strings_from_dex(dex_data)
            all_strings.update(strings)
            print(f"  * {dex_name}: {len(strings)} cadenas extraídas")

    print(f"\nTotal cadenas únicas en memoria: {len(all_strings)}")

    # 1. Búsqueda de URLs y dominios de SofaScore
    print("\n" + "=" * 75)
    print("[1] URLs Y ENDPOINTS DETECTADOS (sofascore)")
    print("=" * 75)
    url_pattern = re.compile(r'https?://[a-zA-Z0-9\.\-_]*sofascore[a-zA-Z0-9\.\-_/:]*')
    matched_urls = set()
    for s in all_strings:
        for u in url_pattern.findall(s):
            matched_urls.add(u)

    for u in sorted(matched_urls):
        if any(term in u for term in ["api", "event", "sport", "basket", "mobile", "feed", "ws", "v1"]):
            print(f"  -> {u}")

    # 2. Búsqueda de cabeceras HTTP personalizadas (X-...)
    print("\n" + "=" * 75)
    print("[2] CABECERAS HTTP SOSPECHOSAS (X-...)")
    print("=" * 75)
    header_pattern = re.compile(r'^[Xx]-[A-Za-z0-9\-_]{3,40}$')
    matched_headers = set()
    for s in all_strings:
        if header_pattern.match(s):
            matched_headers.add(s)

    for h in sorted(matched_headers):
        if any(term in h.lower() for term in ["sofa", "token", "auth", "client", "app", "key", "device", "sig", "version"]):
            print(f"  -> {h}")

    # 3. Búsqueda de User-Agent patterns
    print("\n" + "=" * 75)
    print("[3] PATRONES DE USER-AGENT DETECTADOS")
    print("=" * 75)
    for s in all_strings:
        if "sofascore" in s.lower() and ("android" in s.lower() or "okhttp" in s.lower() or "mobile" in s.lower()):
            if len(s) < 100:
                print(f"  -> {s}")

    # 4. Búsqueda de tokens o constantes de API
    print("\n" + "=" * 75)
    print("[4] CONSTANTES DE API / BACKEND")
    print("=" * 75)
    for s in all_strings:
        if any(k in s for k in ["api.sofascore.com", "api.sofascore.app", "/api/v1/event", "/api/v1/sport"]):
            if len(s) < 120:
                print(f"  -> {s}")

    print("\n" + "=" * 75)
    print("   Fin del Análisis Estático de Sofascore.apk.")
    print("=" * 75)

if __name__ == "__main__":
    analyze_apk()
