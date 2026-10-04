"""
tmp/debug_connection/test_mint_multi.py
Proposito: Verificar que POST /api/v1/token/init con huella OkHttp Android + UA firmado
puede emitir varios JWT consecutivos desde la PC (sin la app), y comprobar si hay
rate limiting en la emision. Base para la auto-regeneracion de tokens.
"""
import hashlib
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import tls_client

PKG = "com.sofascore.results"
VER6 = "260921"


def signed_ua() -> str:
    bucket = int(time.time() // 100)
    return f"{PKG}/{VER6}/{hashlib.md5(f'{bucket}sofa2012'.encode()).hexdigest()[:6]}"


def main() -> None:
    s = tls_client.Session(client_identifier="okhttp4_android_13", random_tls_extension_order=False)
    for i in range(1, 6):
        h = {
            "User-Agent": signed_ua(),
            "X-Timestamp": str(int(time.time() * 1000)),
            "app-version": VER6,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json; charset=UTF8",
        }
        payload = {
            "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
            "country": "MX", "timezone": -18000,
            "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
        }
        try:
            r = s.post("https://api.sofascore.com/api/v1/token/init", headers=h, json=payload)
            tok = ""
            if r.status_code == 200:
                tok = r.json().get("token", "")
            print(f"  [{i}] HTTP {r.status_code} token=...{tok[-12:] if tok else r.text[:80]}")
        except Exception as exc:
            print(f"  [{i}] ERR {type(exc).__name__}: {exc}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
