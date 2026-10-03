"""
Ubicación original: tmp/monitor_migration/apply_monitor_migration.py
Propósito / Qué hace:
Script auxiliar de migración que actualiza las referencias de importación y rutas
tras renombrar `bet_monitor_v2` a `monitor_v2` y mover los componentes del Monitor V1
a `monitor_v1/`. Garantiza integridad de imports, rutas a base de datos y referencias en scripts batch.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 1. Actualizar imports internos dentro de monitor_v2/
monitor_v2_dir = ROOT / "monitor_v2"
for p in monitor_v2_dir.rglob("*.py"):
    text = p.read_text(encoding="utf-8")
    if "bet_monitor_v2" in text:
        text = text.replace("bet_monitor_v2", "monitor_v2")
        p.write_text(text, encoding="utf-8")
        print(f"Actualizado: {p.relative_to(ROOT)}")

# 2. Actualizar import en monitor_v2/main.py: match.bet_monitor -> monitor_v1.bet_monitor
v2_main = monitor_v2_dir / "main.py"
if v2_main.exists():
    c = v2_main.read_text(encoding="utf-8")
    if "from match.bet_monitor import _fetch_all_events_for_date_sync" in c:
        c = c.replace(
            "from match.bet_monitor import _fetch_all_events_for_date_sync",
            "from monitor_v1.bet_monitor import _fetch_all_events_for_date_sync"
        )
        v2_main.write_text(c, encoding="utf-8")
        print("Actualizado import de _fetch_all_events_for_date_sync en monitor_v2/main.py")

# 3. Crear monitor_v1/__init__.py
v1_init = ROOT / "monitor_v1" / "__init__.py"
if not v1_init.exists():
    v1_init.write_text('"""Paquete monitor_v1: Monitor original basado en Telegram Bot y Bet Monitor en hilo daemon."""\n', encoding="utf-8")
    print("Creado monitor_v1/__init__.py")

# 4. Crear monitor_v1/main.py
v1_main = ROOT / "monitor_v1" / "main.py"
v1_main_content = '''"""
Punto de entrada principal para el Monitor V1 (Telegram Bot y Bet Monitor en hilo secundario).
"""
import sys
from pathlib import Path

# Configurar sys.path para imports de match y monitor_v1
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "match") not in sys.path:
    sys.path.insert(0, str(_ROOT / "match"))
if str(_ROOT / "monitor_v1") not in sys.path:
    sys.path.insert(0, str(_ROOT / "monitor_v1"))

from monitor_v1.telegram_bot import main

if __name__ == "__main__":
    main()
'''
v1_main.write_text(v1_main_content, encoding="utf-8")
print("Creado monitor_v1/main.py")

# 5. Ajustar imports y sys.path en monitor_v1/telegram_bot.py
v1_tg = ROOT / "monitor_v1" / "telegram_bot.py"
if v1_tg.exists():
    text = v1_tg.read_text(encoding="utf-8")
    
    # Asegurar que sys.path incluye match/ para db, scraper, ml_tools
    path_setup_old = """BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent"""
    
    path_setup_new = """BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT / "match") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "match"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))"""

    if path_setup_old in text and 'str(PROJECT_ROOT / "match")' not in text:
        text = text.replace(path_setup_old, path_setup_new)
        v1_tg.write_text(text, encoding="utf-8")
        print("Actualizado sys.path en monitor_v1/telegram_bot.py")

# 6. Ajustar imports y sys.path en monitor_v1/bet_monitor.py
v1_bm = ROOT / "monitor_v1" / "bet_monitor.py"
if v1_bm.exists():
    text = v1_bm.read_text(encoding="utf-8")
    
    bm_setup_old = """BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))"""

    bm_setup_new = """BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT / "match") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "match"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))"""

    if bm_setup_old in text and 'str(PROJECT_ROOT / "match")' not in text:
        text = text.replace(bm_setup_old, bm_setup_new)
        v1_bm.write_text(text, encoding="utf-8")
        print("Actualizado sys.path en monitor_v1/bet_monitor.py")

# 7. Actualizar match/cli.py: import telegram_bot -> import monitor_v1.telegram_bot
p_cli = ROOT / "match" / "cli.py"
if p_cli.exists():
    c = p_cli.read_text(encoding="utf-8")
    if 'bot_mod = importlib.import_module("telegram_bot")' in c:
        c = c.replace(
            'bot_mod = importlib.import_module("telegram_bot")',
            'bot_mod = importlib.import_module("monitor_v1.telegram_bot")'
        )
        p_cli.write_text(c, encoding="utf-8")
        print("Actualizado import en match/cli.py")

# 8. Actualizar monitor_v1/test_keyboard_functions.py
p_tkf = ROOT / "monitor_v1" / "test_keyboard_functions.py"
if p_tkf.exists():
    c = p_tkf.read_text(encoding="utf-8")
    if "import telegram_bot" in c and "from monitor_v1" not in c:
        c = c.replace("import telegram_bot", "from monitor_v1 import telegram_bot")
        p_tkf.write_text(c, encoding="utf-8")
        print("Actualizado import en monitor_v1/test_keyboard_functions.py")

# 9. Actualizar menu.bat
p_menu = ROOT / "menu.bat"
if p_menu.exists():
    c = p_menu.read_text(encoding="utf-8", errors="ignore")
    c = c.replace(r"python bet_monitor_v2\main.py", r"python monitor_v2\main.py")
    c = c.replace(r"python match\telegram_bot.py", r"python monitor_v1\telegram_bot.py")
    c = c.replace(r"3) Iniciar Bot de Telegram", r"3) Iniciar Bot de Telegram (Monitor V1)")
    p_menu.write_text(c, encoding="utf-8")
    print("Actualizado menu.bat con rutas a monitor_v1 y monitor_v2")

# 10. Copiar FILTROS_LIGAS.md a docs/ si no existe
filtros_docs = ROOT / "docs" / "FILTROS_LIGAS.md"
filtros_v1 = ROOT / "monitor_v1" / "FILTROS_LIGAS.md"
if filtros_v1.exists() and not filtros_docs.exists():
    header = "> **Ubicación original:** match/FILTROS_LIGAS.md\n\n---\n\n"
    filtros_docs.write_text(header + filtros_v1.read_text(encoding="utf-8"), encoding="utf-8")
    print("Copiado FILTROS_LIGAS.md a docs/FILTROS_LIGAS.md con encabezado")

print("Migracion de monitores completada exitosamente.")
