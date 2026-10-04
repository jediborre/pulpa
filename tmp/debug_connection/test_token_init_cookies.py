"""
tmp/debug_connection/test_token_init_cookies.py
Proposito: Reproducir POST /api/v1/token/init desde la PC (mismo endpoint que usa la
app para obtener el JWT) e inspeccionar todas las cabeceras de respuesta, en especial
Set-Cookie (posible cf_clearance u otras cookies de sesion) que la app reutiliza.
"""
import json
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx

UA = "com.sofascore.results/260921/022538"
URL = "https://api.sofascore.com/api/v1/token/init"


def main() -> None:
    headers = {
        "User-Agent": UA,
        "x-timestamp": str(int(time.time() * 1000)),
        "Content-Type": "application/json; charset=UTF8",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
        "Host": "api.sofascore.com",
    }
    payload = {
        "deviceType": "android",
        "version": 260921,
        "sdk": 29,
        "language": "en",
        "country": "MX",
        "timezone": -18000,
        "advertisingId": str(uuid.uuid4()),
        "uuid": str(uuid.uuid4()),
    }
    with httpx.Client(timeout=20) as c:
        r = c.post(URL, headers=headers, json=payload)
    print(f"HTTP {r.status_code}")
    for k, v in r.headers.items():
        print(f"  {k}: {v}")
    print("body:", r.text[:300])


if __name__ == "__main__":
    main()
