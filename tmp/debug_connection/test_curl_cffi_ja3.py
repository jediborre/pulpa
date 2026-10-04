"""
tmp/debug_connection/test_curl_cffi_ja3.py
Proposito: PoC sin tls_client. Probar curl_cffi (libcurl) alimentado con el JA3 exacto
de OkHttp Android + UA firmado + cabeceras de la app, para ver si pasa el WAF de Fastly.
Demuestra que basta un cliente HTTP ligero con el ClientHello correcto.
"""
import hashlib
import sys
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from curl_cffi import requests as cffi

ROOT = Path(__file__).resolve().parents[2]
# JA3 OkHttp Android (ver docs/REVERSE_ENGINEERING_SOFASCORE.md). Incluye SNI (ext 0).
JA3 = "771,4865-4866-4867-49195-49196-52393-49199-49200-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-51-45-43-21,29-23-24,0"

PKG = "com.sofascore.results"
VER6 = "260921"


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
        "Connection": "Keep-Alive",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def main() -> None:
    print("JA3:", JA3)
    payload = {
        "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
        "country": "MX", "timezone": -18000,
        "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
    }
    h = headers()
    h["Content-Type"] = "application/json; charset=UTF8"

    for label, kw in (
        ("ja3 solo", {"ja3": JA3}),
        ("ja3 + impersonate chrome", {"ja3": JA3, "impersonate": "chrome"}),
        ("ja3 + http2", {"ja3": JA3, "http_version": "v2"}),
    ):
        try:
            r = cffi.post(
                "https://api.sofascore.com/api/v1/token/init",
                headers=h, json=payload, timeout=20, **kw,
            )
            tok = ""
            if r.status_code == 200:
                tok = r.json().get("token", "")
            print(f"  [{label}] HTTP {r.status_code} {tok[-12:] if tok else r.text[:100]}")
        except Exception as exc:
            print(f"  [{label}] ERR {type(exc).__name__}: {exc}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
