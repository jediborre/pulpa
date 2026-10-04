"""
PoC: Análisis de android-auth en classes6.dex
Ubicación: tmp/monitor_v3_poc/inspect_android_auth.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APK_PATH = ROOT / "tmp" / "monitor_v3_poc" / "Sofascore.apk"

with zipfile.ZipFile(APK_PATH, 'r') as z:
    data = z.read("classes6.dex")
    
    # Extraer strings en bloque
    pos = 0
    while True:
        pos = data.find(b"android-auth", pos)
        if pos == -1: break
        start = max(0, pos - 300)
        end = min(len(data), pos + 300)
        chunk = data[start:end]
        text = "".join([chr(b) if 32 <= b <= 126 else "\n" for b in chunk])
        print("=== BLOQUE ENCONTRADO ===")
        for line in text.split("\n"):
            line = line.strip()
            if line:
                print("  ", line)
        pos += 12
