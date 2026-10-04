"""
tmp/debug_connection/test_app_host.py
Proposito: Explorar el host api.sofascore.app (candidato al host real de la app movil)
para confirmar estructura de endpoints y si devuelve JSON valido usando los JWT de ADB
sin proxy. Se inspeccionan cabeceras de respuesta y cuerpo crudo.
"""
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"
UA = "com.sofascore.results/260921/022538"

import httpx


def token() -> str:
    data = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
    return data["tokens"][0]["token"]


PATHS = [
    "https://api.sofascore.app/api/v1/sport/basketball/scheduled-events/2026-09-01",
    "https://api.sofascore.app/api/v1/sport/basketball/events/live",
    "https://api.sofascore.app/api/v1/event/15935071",
    "https://api.sofascore.app/api/v1/config/all",
    "https://api.sofascore.app/api/v1/sport/basketball/categories",
    "https://api.sofascore.app/mobile/v4/sport/basketball/events/schedule/2026-09-01",
    "https://api.sofascore.app/api/v1/unique-tournament/132/scheduled-events/2026-09-01",
]


def main() -> None:
    tok = token()
    headers = {
        "User-Agent": UA,
        "x-timestamp": str(int(time.time() * 1000)),
        "Authorization": f"Bearer {tok}",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "Keep-Alive",
    }
    with httpx.Client(http2=False, timeout=12, follow_redirects=True) as c:
        for url in PATHS:
            try:
                r = c.get(url, headers=headers)
                body = r.content
                preview = body[:300].decode("utf-8", "replace")
                print(f"\n{r.status_code} ({len(body)}b) {url}")
                print(f"  ct={r.headers.get('content-type')} server={r.headers.get('server')} loc={r.headers.get('location')}")
                print(f"  body: {preview}")
                if body:
                    try:
                        js = r.json()
                        if isinstance(js, dict):
                            print(f"  json keys: {list(js.keys())[:12]}")
                    except Exception:
                        pass
            except Exception as exc:
                print(f"\nERR {url}: {type(exc).__name__}: {exc}")
            time.sleep(0.4)


if __name__ == "__main__":
    main()
