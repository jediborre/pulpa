"""
tmp/debug_connection/test_signed_ua.py
Proposito: Reproducir el User-Agent firmado de la app de SofaScore
  UA = "com.sofascore.results/260921/" + md5(str(unix_segundos//100) + "sofa2012")[:6]
y probar token/init y un GET directo (sin proxy). Hipotesis: el WAF de Fastly valida
esa firma temporal y por eso la PC era rechazada con el UA estatico.
"""
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx

PKG = "com.sofascore.results"
VERSION6 = "260921003"[:6]


def signed_ua(offset_buckets: int = 0) -> str:
    bucket = int(time.time() // 100) + offset_buckets
    digest = hashlib.md5(f"{bucket}sofa2012".encode()).hexdigest()
    return f"{PKG}/{VERSION6}/{digest[:6]}"


def main() -> None:
    print("UA actual:", signed_ua())
    print("UA -1   :", signed_ua(-1))
    print("UA +1   :", signed_ua(1))

    base_headers = {
        "x-timestamp": str(int(time.time() * 1000)),
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    with httpx.Client(timeout=20) as c:
        for off in (0, -1, 1):
            h = dict(base_headers)
            h["User-Agent"] = signed_ua(off)
            h["x-timestamp"] = str(int(time.time() * 1000))
            payload = {
                "deviceType": "android", "version": 260921, "sdk": 29,
                "language": "en", "country": "MX", "timezone": -18000,
                "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
            }
            h["Content-Type"] = "application/json; charset=UTF8"
            try:
                r = c.post("https://api.sofascore.com/api/v1/token/init", headers=h, json=payload)
                print(f"token/init off={off:+d}: HTTP {r.status_code} {r.text[:120]}")
                if r.status_code == 200:
                    tok = r.json().get("token")
                    print("  TOKEN OK ...", tok[-12:] if tok else None)
                    # probar GET firmado
                    gh = {
                        "User-Agent": signed_ua(off),
                        "x-timestamp": str(int(time.time() * 1000)),
                        "Authorization": f"Bearer {tok}",
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                    }
                    g = c.get(
                        "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-09-01",
                        headers=gh,
                    )
                    print(f"  GET off={off:+d}: HTTP {g.status_code} {g.text[:120]}")
            except Exception as exc:
                print(f"token/init off={off:+d}: ERR {type(exc).__name__}: {exc}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
