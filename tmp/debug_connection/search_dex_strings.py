"""
tmp/debug_connection/search_dex_strings.py
Proposito: Buscar en los DEX del APK de SofaScore cadenas relacionadas con la capa
de red (OkHttp/Retrofit) y cabeceras personalizadas, para identificar como la app
construye la peticion que pasa Cloudflare. Se imprimen cadenas unicas que contienen
los terminos clave.
"""
import re
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"

KEYS = [
    "x-timestamp", "addheader", "interceptor", "okhttp", "retrofit",
    "authorization", "cf_clearance", "cache-control", "user-agent",
    "x-so-", "x-client", "x-app", "x-device", "signature", "hmac",
    "bearer", "x-token", "x-premium", "cookie", "x-requested",
]


def strings_from(data: bytes, min_len: int = 4) -> list[str]:
    pat = re.compile(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}')
    return [m.decode("latin1", "ignore") for m in pat.findall(data)]


def main() -> None:
    hits: set[str] = set()
    with zipfile.ZipFile(APK) as z:
        for name in z.namelist():
            if not name.endswith(".dex"):
                continue
            for s in strings_from(z.read(name)):
                low = s.lower()
                if any(k in low for k in KEYS):
                    if len(s) <= 120:
                        hits.add(s)
    for s in sorted(hits):
        print(s)


if __name__ == "__main__":
    main()
