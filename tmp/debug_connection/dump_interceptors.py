"""
tmp/debug_connection/dump_interceptors.py
Proposito: Volcar todas las constantes string (cabeceras/cookies) usadas por los
metodos de las clases interceptoras OkHttp identificadas en SofaScore
(Llmg;=X-Premium-Token, Lmmg;=X-Timestamp, Lnmg;=X-Token-Refresh, Lx1h;=cookie jar,
Lp1i;=android-auth) para reconstruir el contrato exacto de red de la app.
"""
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"
TARGETS = {"Llmg;", "Lmmg;", "Lnmg;", "Lx1h;", "Lp1i;"}


def dump_class(c) -> None:
    print(f"\n===== {c.get_name()} =====")
    for m in c.get_methods():
        consts = []
        try:
            code = m.get_code()
            if code:
                for ins in code.get_bc().get_instructions():
                    if ins.get_name() in ("const-string", "const-string/jumbo"):
                        consts.append(str(ins.get_output()))
        except Exception:
            pass
        if consts:
            print(f"  {m.get_name()}: {consts}")


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
                if c.get_name() in TARGETS:
                    dump_class(c)
                # cookie jar real: metodos loadForRequest/saveFromResponse
                elif any(m.get_name() in ("loadForRequest", "saveFromResponse") for m in c.get_methods()):
                    dump_class(c)


if __name__ == "__main__":
    main()
