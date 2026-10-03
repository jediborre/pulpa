"""
Ubicación original: tmp/db_audit/find_db_refs.py
Propósito / Qué hacía:
Busca todas las referencias a matches.db en el código fuente (Python, bat, yaml, etc.)
para redirigirlas limpiamente a la raíz del repositorio (/matches.db).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

found = []

for ext in ["*.py", "*.bat", "*.yaml", "*.yml", "*.json", "*.ts", "*.tsx", "*.js"]:
    for p in ROOT.rglob(ext):
        if any(part in p.parts for part in [".git", ".venv", "node_modules", "tools/obscura-src"]):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(text.splitlines(), 1):
                if "matches.db" in line:
                    found.append((str(p.relative_to(ROOT)), line_no, line.strip()))
        except Exception:
            pass

print(f"Total de referencias encontradas a matches.db: {len(found)}")
for rel, line_no, text in found:
    print(f"{rel}:{line_no}: {text}")
