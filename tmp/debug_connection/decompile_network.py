"""
tmp/debug_connection/decompile_network.py
Proposito: Usar androguard para localizar en el APK de SofaScore las clases de la
capa de red (interceptores OkHttp, cookie jars, builders de cabeceras) y volcar los
metodos/cadenas relevantes. Objetivo: descubrir el "truco" (cabecera/cookie/firma)
que permite a la app pasar el WAF de Fastly.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"

KEYWORDS = ["Interceptor", "CookieJar", "Cookie", "Header", "OkHttp", "NetworkModule",
            "ApiClient", "Authenticator", "Signature", "SofaScoreClient"]


def main() -> None:
    from androguard.misc import AnalyzeAPK
    print("Cargando APK (puede tardar)...")
    a, d, dx = AnalyzeAPK(str(APK))
    print(f"clases: {len(list(dx.get_classes()))}")
    seen = 0
    for c in dx.get_classes():
        name = c.get_name()
        if any(k.lower() in name.lower() for k in KEYWORDS):
            methods = [m.get_name() for m in c.get_methods()]
            print(f"\nCLASS {name}")
            print(f"  methods: {methods[:40]}")
            seen += 1
            if seen > 120:
                print("... (truncado)")
                break


if __name__ == "__main__":
    main()
