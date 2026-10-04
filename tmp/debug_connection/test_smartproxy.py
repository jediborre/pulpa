"""
tmp/debug_connection/test_smartproxy.py
Proposito: Verificar si el proxy residencial Smartproxy del .env permite consultar
SofaScore usando los JWT de ADB, sin pasar por HTTP Toolkit (127.0.0.1:8000).
Hipotesis: El proxy residencial cambia la IP publica y por eso Cloudflare no emite
el reto "challenge"; HTTP Toolkit pudo haber estado encadenado a este upstream.
"""
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"

for line in (ROOT / ".env").read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if line.startswith("SOFASCORE_PROXY_URL="):
        os.environ["SOFASCORE_PROXY_URL"] = line.split("=", 1)[1].strip()

import httpx

UA = "com.sofascore.results/260921/022538"
DATE = "2026-09-01"
URL = f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{DATE}"


def token() -> str:
    data = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
    return data["tokens"][0]["token"]


def main() -> None:
    proxy = os.environ.get("SOFASCORE_PROXY_URL", "")
    print(f"Proxy: {proxy[:40]}...(len={len(proxy)})")
    headers = {
        "User-Agent": UA,
        "x-timestamp": str(int(time.time() * 1000)),
        "Authorization": f"Bearer {token()}",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    try:
        with httpx.Client(proxy=proxy, verify=True, timeout=25) as c:
            r = c.get(URL, headers=headers)
        print(f"HTTP {r.status_code} ({len(r.content)}b)")
        print(r.text[:400])
    except Exception as exc:
        print(f"ERR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
