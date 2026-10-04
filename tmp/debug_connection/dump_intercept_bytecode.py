"""
tmp/debug_connection/dump_intercept_bytecode.py
Proposito: Volcar el bytecode Dalvik completo de los metodos intercept de las clases
OkHttp de SofaScore (Lmmg;=User-Agent/X-Timestamp, Lnmg;=Authorization/X-Token-Refresh,
Llmg;=X-Premium-Token/Cache-Control) para reconstruir el formato exacto del
User-Agent y de las cabeceras de red de la app.
"""
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"
TARGETS = {"Lmmg;", "Lnmg;", "Llmg;"}


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
                if c.get_name() not in TARGETS:
                    continue
                for m in c.get_methods():
                    if m.get_name() != "intercept":
                        continue
                    print(f"\n===== {dn} {c.get_name()}.intercept =====")
                    code = m.get_code()
                    if not code:
                        continue
                    for ins in code.get_bc().get_instructions():
                        print("  ", ins.get_name(), ins.get_output())


if __name__ == "__main__":
    main()
