"""
tmp/debug_connection/test_okhttp_signed.py
Proposito: Validar TODOS los endpoints reales de produccion (monitor_v3/mobile_client)
con la combinacion ganadora: tls_client okhttp4_android_13 + User-Agent firmado
(md5(str(unix//100)+'sofa2012')[:6]) + cabeceras completas. Confirmar que se pasa el
WAF de Fastly y se obtiene JSON valido.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import tls_client

PKG = "com.sofascore.results"
VER = "260921003"
VER6 = VER[:6]
TZ = -6 * 3600


def signed_ua() -> str:
    bucket = int(time.time() // 100)
    return f"{PKG}/{VER6}/{hashlib.md5(f'{bucket}sofa2012'.encode()).hexdigest()[:6]}"


def headers(token: str | None) -> dict:
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
    s = tls_client.Session(client_identifier="okhttp4_android_13", random_tls_extension_order=False)
    # token/init
    h = headers(None)
    h["Content-Type"] = "application/json; charset=UTF8"
    payload = {
        "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
        "country": "MX", "timezone": -18000,
        "advertisingId": "00000000-0000-0000-0000-000000000000",
        "uuid": "00000000-0000-0000-0000-000000000000",
    }
    r = s.post("https://api.sofascore.com/api/v1/token/init", headers=h, json=payload)
    print("token/init:", r.status_code)
    if r.status_code != 200:
        print(r.text[:200])
        return
    tok = r.json()["token"]

    urls = [
        f"https://api.sofascore.com/api/v1/sport/basketball/2026-09-01/{TZ}/categories",
        "https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/2026-09-01",
        "https://api.sofascore.com/api/v1/event/15935071",
        "https://api.sofascore.com/api/v1/event/15935071/incidents",
        "https://api.sofascore.com/api/v1/event/15935071/graph",
        "https://api.sofascore.com/api/v1/event/15935071/statistics",
        "https://api.sofascore.com/api/v1/event/15935071/lineups",
        "https://api.sofascore.com/api/v1/event/15935071/h2h",
        "https://api.sofascore.com/api/v1/event/15935071/odds/1/all",
    ]
    for u in urls:
        try:
            g = s.get(u, headers=headers(tok))
            body = g.text
            n = len(body)
            try:
                js = json.loads(body)
                keys = list(js.keys())[:6] if isinstance(js, dict) else f"list({len(js)})"
            except Exception:
                keys = body[:80]
            print(f"  {g.status_code} ({n}b) {u.split('/api/v1/')[1][:55]:<55} keys={keys}")
        except Exception as exc:
            print(f"  ERR {type(exc).__name__}: {exc} {u}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
