"""
tmp/debug_connection/test_android_tls.py
Proposito: Probar si curl_cffi con huellas TLS de Android/Chrome-Android (BoringSSL,
equivalente a OkHttp/Conscrypt) permite pasar Cloudflare en api.sofascore.com usando
el AUTH_TOKEN fresco extraido de la app por ADB. Hipotesis: el bloqueo es por huella
TLS, no por token ni IP.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from curl_cffi import requests as cffi

ADB = Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe"
PREFS = "shared_prefs/com.sofascore.results_preferences.xml"
PKG = "com.sofascore.results"
UA = "com.sofascore.results/260921/022538"
DATE = "2026-09-01"


def fresh_token() -> str | None:
    p = subprocess.run([str(ADB), "shell", "run-as", PKG, "cat", PREFS],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r'name=["\']AUTH_TOKEN["\']>(?:[^<]+)<', p.stdout)
    if not m:
        return None
    m2 = re.search(r'name=["\']AUTH_TOKEN["\']>([^<]+)<', p.stdout)
    return m2.group(1).strip() if m2 else None


def main() -> None:
    tok = fresh_token()
    if not tok:
        print("sin token")
        return
    print(f"Token ...{tok[-12:]}")
    url = f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{DATE}"
    targets = ["chrome_android", "chrome131_android", "chrome99_android", "chrome", "chrome136"]
    for imp in targets:
        headers = {
            "User-Agent": UA,
            "x-timestamp": str(int(time.time() * 1000)),
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        try:
            r = cffi.get(url, headers=headers, impersonate=imp, timeout=20)
            reason = ""
            try:
                b = r.json()
                if isinstance(b, dict) and "error" in b:
                    reason = b["error"].get("reason", "")
            except Exception:
                pass
            print(f"  {imp:<18}: HTTP {r.status_code} {reason} ({len(r.content)}b)")
        except Exception as exc:
            print(f"  {imp:<18}: ERR {type(exc).__name__}: {exc}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
