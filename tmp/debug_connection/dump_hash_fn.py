"""
tmp/debug_connection/dump_hash_fn.py
Proposito: Volcar el bytecode de la funcion de hash Lue0;->g([B)Ljava/lang/String; y
de funciones auxiliares (l6j;->J, w7i;->h, jdd;->i) usadas para construir el
User-Agent firmado de SofaScore. Objetivo: replicar exactamente el algoritmo.
"""
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

APK = Path(__file__).resolve().parents[1] / "monitor_v3_poc" / "Sofascore.apk"


def dump(c, method_name) -> None:
    for m in c.get_methods():
        if m.get_name() != method_name:
            continue
        print(f"\n===== {c.get_name()}.{method_name} =====")
        code = m.get_code()
        if not code:
            continue
        for ins in code.get_bc().get_instructions():
            print("  ", ins.get_name(), ins.get_output())


def main() -> None:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.dex import DEX

    want = {"Lue0;": ["g"], "Ll6j;": ["J"], "Lw7i;": ["h"], "Ljdd;": ["i"]}
    with zipfile.ZipFile(APK) as z:
        for dn in [n for n in z.namelist() if n.endswith(".dex")]:
            try:
                d = DEX(z.read(dn))
            except Exception:
                continue
            for c in d.get_classes():
                if c.get_name() in want:
                    for mn in want[c.get_name()]:
                        dump(c, mn)


if __name__ == "__main__":
    main()
