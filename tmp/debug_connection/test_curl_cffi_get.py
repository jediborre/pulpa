"""
tmp/debug_connection/test_curl_cffi_get.py
Proposito: Completar la PoC sin tls_client: con curl_cffi + JA3 OkHttp + UA firmado,
emitir token y descargar un partido completo (event, incidents, graph, statistics).
Confirma que toda la API movil funciona con una pila HTTP ligera.
"""
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from curl_cffi import requests as cffi

JA3 = "771,4865-4866-4867-49195-49196-52393-49199-49200-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-51-45-43-21,29-23-24,0"
VER6 = "260921"
PKG = "com.sofascore.results"


def signed_ua() -> str:
    bucket = int(time.time() // 100)
    return f"{PKG}/{VER6}/{hashlib.md5(f'{bucket}sofa2012'.encode()).hexdigest()[:6]}"


def headers(token=None):
    h = {
        "User-Agent": signed_ua(),
        "X-Timestamp": str(int(time.time() * 1000)),
        "app-version": VER6,
        "Cache-Control": "max-age=0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def main() -> None:
    s = cffi.Session(ja3=JA3, impersonate="chrome")
    h = headers()
    h["Content-Type"] = "application/json; charset=UTF8"
    payload = {
        "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
        "country": "MX", "timezone": -18000,
        "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
    }
    r = s.post("https://api.sofascore.com/api/v1/token/init", headers=h, json=payload, timeout=20)
    print("token/init:", r.status_code)
    tok = r.json()["token"]
    for ep in ("event/15935071", "event/15935071/incidents", "event/15935071/graph",
               "event/15935071/statistics", "event/15935071/lineups"):
        g = s.get(f"https://api.sofascore.com/api/v1/{ep}", headers=headers(tok), timeout=20)
        keys = ""
        try:
            keys = list(g.json().keys())[:5]
        except Exception:
            keys = g.text[:60]
        print(f"  {g.status_code} ({len(g.content)}b) {ep:<40} {keys}")


if __name__ == "__main__":
    main()
