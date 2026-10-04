"""
tmp/debug_connection/find_header_methods.py
Proposito: Localizar exactamente que clase/metodo del APK de SofaScore usa las
constantes de cabecera (X-Timestamp, Authorization, etc.) y las clases de cookies
(SCSWebviewCookieJar), parseando el bytecode DEX. Asi se identifica el interceptor
OkHttp y el "truco" de red de la app.
"""
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"
NEEDLES = ["X-Timestamp", "x-timestamp", "SCSWebviewCookieJar", "X-Premium-Token",
           "X-Token-Refresh", "android-auth", "cf_clearance", "android-auth"]


def main() -> None:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.dex import DEX

    with zipfile.ZipFile(APK) as z:
        for dn in [n for n in z.namelist() if n.endswith(".dex")]:
            try:
                d = DEX(z.read(dn))
            except Exception:
                continue
            for c in d.get_classes():
                for m in c.get_methods():
                    try:
                        code = m.get_code()
                        if not code:
                            continue
                        for ins in code.get_bc().get_instructions():
                            name = ins.get_name()
                            if name not in ("const-string", "const-string/jumbo"):
                                continue
                            val = ins.get_output()
                            if any(nd in str(val) for nd in NEEDLES):
                                print(f"{dn} | {c.get_name()} | {m.get_name()} | {val}")
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
