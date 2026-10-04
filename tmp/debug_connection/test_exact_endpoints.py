"""
tmp/debug_connection/test_exact_endpoints.py
Proposito: Probar en conexion DIRECTA (sin proxy) los endpoints EXACTOS que usa el
cliente movil de produccion (monitor_v3/core/mobile_client.py):
  - sport/basketball/{date}/{tz}/categories
  - category/{id}/scheduled-events/{date}
  - event/{id}, event/{id}/incidents, event/{id}/graph...
usando tokens frescos del pool y variantes de cabeceras (basic vs browser-like).
Hipotesis: Identificar que combinacion de token/host/cabeceras responde 200 sin
pasar por HTTP Toolkit.
"""
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"
import httpx

UA = "com.sofascore.results/260921/022538"
DATE = "2026-09-01"
TZ = -6 * 3600


def tokens() -> list[str]:
    data = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
    return [t["token"] for t in data.get("tokens", []) if t.get("token")]


def headers(tok: str, style: str) -> dict:
    h = {
        "User-Agent": UA,
        "x-timestamp": str(int(time.time() * 1000)),
        "Authorization": f"Bearer {tok}",
        "Accept-Encoding": "identity",
        "Connection": "Keep-Alive",
    }
    if style == "browser":
        h.update({
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com",
            "X-Requested-With": "com.sofascore.results",
        })
    return h


def probe(c: httpx.Client, url: str, tok: str, style: str) -> str:
    try:
        r = c.get(url, headers=headers(tok, style))
        reason = ""
        try:
            b = r.json()
            if isinstance(b, dict) and "error" in b:
                reason = b["error"].get("reason", "")
        except Exception:
            pass
        return f"{r.status_code} {reason} ({len(r.content)}b)"
    except Exception as exc:
        return f"ERR {type(exc).__name__}"


def main() -> None:
    toks = tokens()
    print(f"Tokens: {len(toks)}")
    urls = [
        f"https://api.sofascore.com/api/v1/sport/basketball/{DATE}/{TZ}/categories",
        f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{DATE}",
        "https://api.sofascore.com/api/v1/sport/basketball/events/live",
        "https://api.sofascore.com/api/v1/event/15935071",
    ]
    with httpx.Client(timeout=12) as c:
        for style in ("basic", "browser"):
            print(f"\n===== estilo={style} =====")
            for i, tok in enumerate(toks, 1):
                row = [f"T{i:02d}"] + [probe(c, u, tok, style) for u in urls]
                print("  " + " | ".join(row))
                time.sleep(0.3)


if __name__ == "__main__":
    main()
