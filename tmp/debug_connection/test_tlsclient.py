"""
tmp/debug_connection/test_tlsclient.py
Proposito: Probar tls_client (bogdanfinn) con huellas OkHttp4 Android (las que usa la
app real de SofaScore) para ver si alguna permite pasar Cloudflare en api.sofascore.com
con el AUTH_TOKEN fresco extraido por ADB. Hipotesis: el reto es por fingerprint
TLS/HTTP2; la huella OkHttp nativa debe ser aceptada.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import tls_client

ADB = Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe"
PREFS = "shared_prefs/com.sofascore.results_preferences.xml"
PKG = "com.sofascore.results"
UA = "com.sofascore.results/260921/022538"
DATE = "2026-09-01"


def fresh_token() -> str | None:
    p = subprocess.run([str(ADB), "shell", "run-as", PKG, "cat", PREFS],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r'name=["\']AUTH_TOKEN["\']>([^<]+)<', p.stdout)
    return m.group(1).strip() if m else None


def main() -> None:
    tok = fresh_token()
    if not tok:
        print("sin token")
        return
    print(f"Token ...{tok[-12:]}")
    url = f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{DATE}"
    ids = [
        "okhttp4_android_10", "okhttp4_android_11", "okhttp4_android_12", "okhttp4_android_13",
        "chrome_131", "chrome_133",
    ]
    for cid in ids:
        try:
            s = tls_client.Session(client_identifier=cid, random_tls_extension_order=True)
        except Exception as exc:
            print(f"  {cid:<22}: no disponible ({exc})")
            continue
        headers = {
            "User-Agent": UA,
            "x-timestamp": str(int(time.time() * 1000)),
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        try:
            r = s.get(url, headers=headers, timeout_seconds=20)
            reason = ""
            try:
                b = r.json()
                if isinstance(b, dict) and "error" in b:
                    reason = b["error"].get("reason", "")
            except Exception:
                pass
            print(f"  {cid:<22}: HTTP {r.status_code} {reason} ({len(r.content)}b)")
        except Exception as exc:
            print(f"  {cid:<22}: ERR {type(exc).__name__}: {exc}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
