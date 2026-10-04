"""
tmp/debug_connection/fast_dex_search.py
Proposito: Parsear los DEX del APK de SofaScore con el parser crudo de androguard
(rapido, sin analisis de xrefs) para localizar clases de la capa de red
(interceptores OkHttp, cookie jars, builders de cabeceras) y listar sus metodos.
Tambien busca constantes string tipo cabecera/cookie.
"""
import re
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"
CLASS_KEYS = ["Interceptor", "Cookie", "Header", "OkHttp", "NetworkModule",
              "ApiClient", "Authenticator", "SofaScore"]
STR_KEYS = ["x-timestamp", "x-requested-with", "cache-control", "user-agent",
            "authorization", "x-premium", "x-token", "cookie", "x-android",
            "x-device", "x-client", "x-app", "accept-language"]


def main() -> None:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.dex import DEX
    with zipfile.ZipFile(APK) as z:
        dex_names = [n for n in z.namelist() if n.endswith(".dex")]
        for dn in dex_names:
            data = z.read(dn)
            try:
                d = DEX(data)
            except Exception as exc:
                print(f"[!] {dn}: {exc}")
                continue
            hits = []
            for c in d.get_classes():
                name = c.get_name()
                if any(k.lower() in name.lower() for k in CLASS_KEYS):
                    hits.append(name)
            if hits:
                print(f"\n===== {dn} ({len(hits)} clases red) =====")
                for h in sorted(set(hits)):
                    print(" ", h)
            # string constants tipo cabecera
            strs = set()
            for s in d.get_strings():
                val = s.get() if hasattr(s, "get") else s
                low = str(val).lower()
                if any(k in low for k in STR_KEYS) and len(str(val)) <= 60:
                    strs.add(str(val))
            if strs:
                print(f"\n--- {dn} strings cabecera ---")
                for s in sorted(strs):
                    print("  ", s)


if __name__ == "__main__":
    main()
