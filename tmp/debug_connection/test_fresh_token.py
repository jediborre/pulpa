"""
tmp/debug_connection/test_fresh_token.py
Proposito: Extraer el AUTH_TOKEN actual de la app (sin regenerar) via run-as y probar
conexion DIRECTA a api.sofascore.com para determinar si un token recien emitido por la
app pasa Cloudflare desde la PC. Ademas prueba a traves del proxy HTTP Toolkit si esta
disponible.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ADB = Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe"
PREFS = "shared_prefs/com.sofascore.results_preferences.xml"
PKG = "com.sofascore.results"
UA = "com.sofascore.results/260921/022538"

import httpx


def fresh_token() -> str | None:
    p = subprocess.run([str(ADB), "shell", "run-as", PKG, "cat", PREFS],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r'name=["\']AUTH_TOKEN["\']>([^<]+)<', p.stdout)
    return m.group(1).strip() if m else None


def test(label, proxy, token):
    headers = {
        "User-Agent": UA,
        "x-timestamp": str(int(time.time() * 1000)),
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    url = "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-09-01"
    verify = str(Path(r"C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem")) if proxy else True
    try:
        with httpx.Client(proxy=proxy, verify=verify, timeout=20) as c:
            r = c.get(url, headers=headers)
        reason = ""
        try:
            b = r.json()
            if isinstance(b, dict) and "error" in b:
                reason = b["error"].get("reason", "")
        except Exception:
            pass
        print(f"  {label}: HTTP {r.status_code} {reason} ({len(r.content)}b)")
    except Exception as exc:
        print(f"  {label}: ERR {type(exc).__name__}: {exc}")


def main() -> None:
    tok = fresh_token()
    if not tok:
        print("No se pudo extraer AUTH_TOKEN")
        return
    print(f"Token fresco: ...{tok[-12:]}")
    test("DIRECTO", None, tok)
    test("PROXY 8000", "http://127.0.0.1:8000", tok)


if __name__ == "__main__":
    main()
