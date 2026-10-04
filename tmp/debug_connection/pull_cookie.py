"""
tmp/debug_connection/pull_cookie.py
Proposito: Extraer por ADB la base SQLite de cookies del WebView de SofaScore
(app_webview/Default/Cookies) de forma binaria segura en Windows, y volcar las
cookies asociadas a sofascore/cloudflare (especialmente cf_clearance) para evaluar
si permiten conexion directa sin proxy.
"""
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
ADB = Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe"
OUT = Path(__file__).resolve().parent / "mobile_cookies" / "Cookies"
REMOTE = "app_webview/Default/Cookies"
PKG = "com.sofascore.results"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(ADB), "exec-out", "run-as", PKG, "cat", REMOTE],
        capture_output=True,
    )
    data = proc.stdout
    print(f"stderr: {proc.stderr.decode('utf-8', 'replace').strip()}")
    print(f"bytes: {len(data)}")
    OUT.write_bytes(data)
    print(f"guardado en: {OUT}")


if __name__ == "__main__":
    main()
