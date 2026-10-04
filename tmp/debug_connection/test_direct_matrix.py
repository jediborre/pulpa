"""
tmp/debug_connection/test_direct_matrix.py
Proposito: Determinar si es viable consultar SofaScore DIRECTAMENTE (sin el proxy
HTTP Toolkit en 127.0.0.1:8000) usando los JWT extraidos por ADB. Se prueban varios
hosts, variantes de path y perfiles de cabeceras moviles, y se recorren todos los
tokens del pool para verificar si alguno responde 200.
Hipotesis: Si el token/token o el host correcto se usa con las cabeceras exactas de
la app Android, Cloudflare no deberia emitir el reto "challenge".
"""
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"

try:
    from curl_cffi import requests as cffi_requests
except Exception as exc:  # pragma: no cover
    cffi_requests = None
    print(f"[!] curl_cffi no disponible: {exc}")

import httpx

DATE = "2026-09-01"

HOSTS = [
    ("api.sofascore.com", f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{DATE}"),
    ("www.sofascore.com", f"https://www.sofascore.com/api/v1/sport/basketball/scheduled-events/{DATE}"),
    ("api.sofascore.app", f"https://api.sofascore.app/api/v1/sport/basketball/scheduled-events/{DATE}"),
    ("mobile/v4", f"https://api.sofascore.com/mobile/v4/sport/basketball/events/schedule/{DATE}"),
]

UA = "com.sofascore.results/260921/022538"


def load_tokens() -> list[str]:
    if not TOKENS_JSON.exists():
        return []
    data = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
    return [t["token"] for t in data.get("tokens", []) if t.get("token")]


def base_headers(token: str, with_extra: bool) -> dict:
    h = {
        "User-Agent": UA,
        "x-timestamp": str(int(time.time() * 1000)),
        "Authorization": f"Bearer {token}",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    if with_extra:
        h.update({
            "Accept": "application/json",
            "X-Requested-With": "com.sofascore.results",
        })
    return h


def try_curl_cffi(url: str, headers: dict) -> str:
    if cffi_requests is None:
        return "curl_cffi N/A"
    try:
        r = cffi_requests.get(url, headers=headers, impersonate="chrome", timeout=10)
        reason = ""
        try:
            body = r.json()
            if isinstance(body, dict) and "error" in body:
                reason = body["error"].get("reason", "")
        except Exception:
            pass
        return f"HTTP {r.status_code} {reason} ({len(r.content)}b)"
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}"


def try_httpx(url: str, headers: dict) -> str:
    try:
        with httpx.Client(http2=False, timeout=10) as c:
            r = c.get(url, headers=headers)
        reason = ""
        try:
            body = r.json()
            if isinstance(body, dict) and "error" in body:
                reason = body["error"].get("reason", "")
        except Exception:
            pass
        return f"HTTP {r.status_code} {reason} ({len(r.content)}b)"
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}"


def main() -> None:
    tokens = load_tokens()
    print(f"Tokens en pool: {len(tokens)}")
    if not tokens:
        return

    print("\n=== 1. Tokens x hosts (curl_cffi chrome, headers basicos) ===")
    for idx, tok in enumerate(tokens, 1):
        line = []
        for name, url in HOSTS:
            res = try_curl_cffi(url, base_headers(tok, with_extra=False))
            line.append(f"{name}={res}")
            time.sleep(0.4)
        print(f"  T{idx:02d} ...{tok[-8:]}: " + " | ".join(line))

    print("\n=== 2. Token #1 con cabeceras extra (httpx y curl_cffi) ===")
    tok = tokens[0]
    for name, url in HOSTS:
        print(f"  {name}")
        print(f"    httpx  : {try_httpx(url, base_headers(tok, True))}")
        print(f"    cffi   : {try_curl_cffi(url, base_headers(tok, True))}")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
