# =====================================================================
# VERIFICADOR DE CONSISTENCIA DE menu.bat
# ---------------------------------------------------------------------
# QUÉ VALIDA:
#   1. Las opciones del menú (`echo   N) ...`) no están duplicadas y van
#      1..N sin huecos.
#   2. Cada opción numérica tiene AL MENOS una ruta `if "%OPT%"=="N" goto X`.
#   3. Toda ruta apunta a un label `:X` que existe.
#   4. No hay rutas numéricas huérfanas (números sin entrada en el menú).
#   5. Referencias a "opcion N" dentro del texto apuntan a opciones válidas.
#
# PROBLEMA QUE ABORDA:
#   Tras años de parches, menu.bat tenía números repetidos (31, 32, 33, 35
#   dos veces) y el submenú Smart Backfill quedó inalcanzable porque la ruta
#   del 35 se resolvía antes como team_strength.
#
# USO: .venv\Scripts\python.exe tmp\menu_check\check_menu.py
# =====================================================================

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
MENU = ROOT / "menu.bat"

src = MENU.read_text(encoding="utf-8", errors="replace").splitlines()

# 1. Opciones del menú principal (solo el bloque antes de set /p OPT)
opts: dict[int, str] = {}
in_main = True
for line in src:
    s = line.strip()
    if s.startswith("set /p OPT="):
        in_main = False
    if in_main:
        m = re.match(r'echo\s+(\d+)\)\s*(.*)', s)
        if m:
            num = int(m.group(1))
            opts.setdefault(num, m.group(2))

# 2. Rutas numéricas del menú principal
routes: dict[int, list[str]] = {}
for line in src:
    s = line.strip()
    m = re.match(r'if "%OPT%"=="(\d+)" goto (\w+)', s)
    if m:
        routes.setdefault(int(m.group(1)), []).append(m.group(2))

# 3. Labels existentes
labels = {line.strip()[1:] for line in src if re.match(r"^\s*:[A-Za-z_]\w*", line)}

ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    ok = ok and cond
    print(("  ✅ " if cond else "  ❌ ") + msg)


nums = sorted(opts)
# La opción 0 es "Salir" (convención) y 99 es un alias histórico de INSTALAR.
ALIAS_NUMS = {99}
nums_main = [n for n in nums if n != 0]
print(f"Opciones en el menú: {len(nums_main)} (1..{max(nums_main) if nums_main else 0})")
print("\n[1] Sin duplicados y sin huecos")
dups = [n for n in nums if list(opts).count(n) > 1]
check(not dups, "números duplicados" if dups else "todos los números son únicos")
check(
    nums_main == list(range(1, (max(nums_main) if nums_main else 0) + 1)),
    "la numeración es continua 1..N",
)

print("\n[2] Cada opción tiene ruta")
missing = [n for n in nums_main if n not in routes or not routes[n]]
check(not missing, f"opciones sin ruta: {missing}" if missing else "todas las opciones tienen ruta")

print("\n[3] Las rutas apuntan a labels existentes")
bad = [(n, t) for n, targets in routes.items() for t in targets if t not in labels]
check(not bad, f"labels inexistentes: {bad}" if bad else "todos los destinos existen")

print("\n[4] Sin rutas huérfanas")
orphans = [n for n in routes if n not in opts and n not in ALIAS_NUMS]
check(not orphans, f"rutas sin entrada en el menú: {orphans}" if orphans else "no hay rutas huérfanas")

print("\n[5] Referencias 'opcion N' en el texto")
refs = set()
for line in src:
    for m in re.finditer(r"opci[oó]n\s+(\d+)", line, re.IGNORECASE):
        refs.add(int(m.group(1)))
invalid = sorted(r for r in refs if r not in opts)
check(not invalid, f"referencias a opciones inexistentes: {invalid}" if invalid else "todas las referencias son válidas")

print("\n=== RESUMEN ===")
print("✅ menu.bat consistente" if ok else "❌ hay inconsistencias (ver arriba)")
sys.exit(0 if ok else 1)
