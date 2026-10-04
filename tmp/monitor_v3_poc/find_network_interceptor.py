"""
PoC: Búsqueda Profunda de Interceptores de Red en el código de SofaScore
Ubicación: tmp/monitor_v3_poc/find_network_interceptor.py
Propósito:
    Localizar en los archivos DEX de com.sofascore las cadenas asociadas a:
    - android-auth
    - OkHttp addHeader
    - Headers de petición y User-Agent
"""

import sys
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APK_PATH = ROOT / "tmp" / "monitor_v3_poc" / "Sofascore.apk"

with zipfile.ZipFile(APK_PATH, 'r') as z:
    dex_files = [f for f in z.namelist() if f.endswith('.dex')]
    
    print("Buscando en clases DEX...")
    for dex_name in dex_files:
        data = z.read(dex_name)
        
        # Buscar cadenas con android-auth
        if b"android-auth" in data:
            print(f"-> Encontrado 'android-auth' en {dex_name}!")
            idx = 0
            while True:
                idx = data.find(b"android-auth", idx)
                if idx == -1: break
                start = max(0, idx - 100)
                end = min(len(data), idx + 100)
                snippet = data[start:end]
                print("   Snippet:", bytes([b if 32 <= b <= 126 else 46 for b in snippet]).decode('latin1'))
                idx += 12
                
        # Buscar cadenas que tengan X- y sofascore
        matches = re.findall(rb'X-[A-Za-z0-9\-]{2,30}', data)
        if matches:
            for m in set(matches):
                if any(x in m.lower() for x in [b"sofa", b"auth", b"app", b"token"]):
                    print(f"   Header encontrado en {dex_name}: {m.decode('latin1')}")
