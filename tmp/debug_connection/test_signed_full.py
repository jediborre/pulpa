"""
tmp/debug_connection/test_signed_full.py
Proposito: Probar token/init y GET con el User-Agent firmado + el set completo de
cabeceras de los interceptores OkHttp (app-version, Cache-Control, Accept-Language,
X-Token-Refresh, X-Premium-Token) combinado con la huella TLS/HTTP2 OkHttp Android
(tls_client okhttp4_android_13) y con httpx HTTP/1.1. Hipotesis: el WAF exige la
firma de UA junto con el resto del contrato y HTTP/2.
"""
import hashlib
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx
import tls_client

PKG = "com.sofascore.results"
VER = "260921003"
VER6 = VER[:6]


def signed_ua() -> str:
    bucket = int(time.time() // 100)
    return f"{PKG}/{VER6}/{hashlib.md5(f'{bucket}sofa2012'.encode()).hexdigest()[:6]}"


def full_headers(token: str | None, premium: str | None = None) -> dict:
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
    if premium:
        h["X-Premium-Token"] = premium
    return h


def try_httpx() -> None:
    print("\n--- httpx HTTP/1.1 ---")
    with httpx.Client(timeout=20) as c:
        h = full_headers(None)
        h["Content-Type"] = "application/json; charset=UTF8"
        payload = {
            "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
            "country": "MX", "timezone": -18000,
            "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
        }
        r = c.post("https://api.sofascore.com/api/v1/token/init", headers=h, json=payload)
        print(f"  token/init: HTTP {r.status_code} {r.text[:120]}")


def try_tls_client(cid: str) -> None:
    print(f"\n--- tls_client {cid} ---")
    try:
        s = tls_client.Session(client_identifier=cid, random_tls_extension_order=False)
    except Exception as exc:
        print(f"  no disp: {exc}")
        return
    h = full_headers(None)
    h["Content-Type"] = "application/json; charset=UTF8"
    payload = {
        "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
        "country": "MX", "timezone": -18000,
        "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
    }
    try:
        r = s.post("https://api.sofascore.com/api/v1/token/init", headers=h, json=payload)
        print(f"  token/init: HTTP {r.status_code} {r.text[:120]}")
        if r.status_code == 200:
            tok = r.json().get("token")
            gh = full_headers(tok)
            g = s.get("https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-09-01", headers=gh)
            print(f"  GET: HTTP {g.status_code} {g.text[:120]}")
    except Exception as exc:
        print(f"  ERR {type(exc).__name__}: {exc}")


def main() -> None:
    print("UA:", signed_ua())
    try_httpx()
    for cid in ("okhttp4_android_13", "okhttp4_android_12", "chrome_131"):
        try_tls_client(cid)


if __name__ == "__main__":
    main()
