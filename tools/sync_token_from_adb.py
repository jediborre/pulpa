"""
Sincronizador de Token JWT directo desde Android vía ADB

Propósito:
- Conectarse al teléfono vía ADB con 'run-as com.sofascore.results'.
- Leer el archivo de preferencias com.sofascore.results_preferences.xml.
- Extraer el AUTH_TOKEN (JWT oficial emitido directamente por SofaScore al teléfono).
- Decodificar los metadatos del JWT (fecha de emisión, expiración de 6 meses).
- Guardar el token en monitor_v3/config/tokens.json sin intermediarios ni proxies.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"
PACKAGE_NAME = "com.sofascore.results"
PREFS_FILE = "shared_prefs/com.sofascore.results_preferences.xml"


def find_adb() -> Path | None:
    which_adb = shutil.which("adb")
    if which_adb:
        return Path(which_adb)
    candidates = [
        Path(r"C:\Users\App\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def decode_jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            payload_b64 = parts[1]
            # Padding
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            data = base64.urlsafe_b64decode(payload_b64)
            return json.loads(data.decode("utf-8"))
    except Exception:
        pass
    return {}


def extract_token_from_device() -> str | None:
    adb = find_adb()
    if not adb:
        print("[ERROR] No se encontró el ejecutable 'adb.exe'.")
        return None

    cmd = [str(adb), "shell", "run-as", PACKAGE_NAME, "cat", PREFS_FILE]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if proc.returncode != 0:
        print(f"[ERROR] Error al leer preferencias del paquete: {proc.stderr.strip()}")
        return None

    content = proc.stdout
    # Buscar tag AUTH_TOKEN
    match = re.search(r'name=["\']AUTH_TOKEN["\']>([^<]+)<', content)
    if not match:
        print("[AVISO] No se encontró la clave AUTH_TOKEN en las preferencias.")
        return None

    return match.group(1).strip()


def sync_adb():
    print("=" * 65)
    print("EXTRACTOR DIRECTO DE JWT SOFASCORE VIA ADB (USB)")
    print("=" * 65)

    token = extract_token_from_device()
    if not token:
        print("\n[!] No se pudo extraer el token. Asegúrate de:")
        print("    1. Tener el teléfono conectado por USB con Depuración activada.")
        print("    2. Tener abierta la app SofaScore en el teléfono.")
        return False

    print(f"\n[+] ¡Token JWT extraído exitosamente de la memoria del teléfono!")
    print(f"    Longitud: {len(token)} caracteres")
    print(f"    Prefijo:  {token[:35]}...")
    print(f"    Sufijo:   ...{token[-25:]}")

    payload_info = decode_jwt_payload(token)
    if payload_info:
        iat = payload_info.get("iat")
        exp = payload_info.get("exp")
        device_id = payload_info.get("id")
        if iat:
            print(f"    Emitido:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(iat))}")
        if exp:
            exp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))
            dias = int((exp - time.time()) / 86400)
            print(f"    Expira:   {exp_str} (dentro de ~{dias} días)")
        if device_id:
            print(f"    ID Disp:  {device_id}")

    # Guardar en tokens.json
    TOKENS_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    # Cargar tokens existentes para acumular si hay otros válidos
    existing_tokens = []
    if TOKENS_JSON.exists():
        try:
            with open(TOKENS_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
                existing_tokens = [t for t in d.get("tokens", []) if t.get("token") != token]
        except Exception:
            pass

    new_item = {
        "token": token,
        "created_at": time.time(),
        "device_uuid": payload_info.get("id", "android-direct"),
        "advertising_id": "direct-adb",
        "failures": 0,
        "last_used": time.time(),
    }
    
    all_tokens = [new_item] + existing_tokens

    data = {
        "updated_at": time.time(),
        "count": len(all_tokens),
        "tokens": all_tokens,
    }

    with open(TOKENS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[ÉXITO] Token guardado en: {TOKENS_JSON}")
    print(f"        Total tokens activos en pool: {len(all_tokens)}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = sync_adb()
    sys.exit(0 if success else 1)
