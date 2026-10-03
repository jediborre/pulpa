"""
Ubicación original: tmp/monitor_migration/fix_telegram_bot_syspath.py
Propósito / Qué hace:
Reordena las declaraciones de sys.path en monitor_v1/telegram_bot.py para que precedan
a los imports de db, scraper, ml_tools y bet_monitor, preservando saltos de línea CRLF.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p_tg = ROOT / "monitor_v1" / "telegram_bot.py"

text = p_tg.read_text(encoding="utf-8")

old_block = """import db as db_mod
import ml_tools as ml_mod
import scraper as scraper_mod
import bet_monitor as bet_monitor_mod


class CustomHTTPXRequest(HTTPXRequest):
    \"\"\"HTTPXRequest with extended timeouts for slow networks and webhooks.\"\"\"
    def __init__(self, *args, **kwargs):
        # Set extended connect/pool/read/write timeouts (60 seconds each)
        kwargs.setdefault('connect_timeout', 60.0)
        kwargs.setdefault('pool_timeout', 60.0)
        kwargs.setdefault('read_timeout', 60.0)
        kwargs.setdefault('write_timeout', 60.0)
        super().__init__(*args, **kwargs)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT / "match") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "match"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))"""

new_block = """BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT / "match") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "match"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import db as db_mod
import ml_tools as ml_mod
import scraper as scraper_mod
import bet_monitor as bet_monitor_mod


class CustomHTTPXRequest(HTTPXRequest):
    \"\"\"HTTPXRequest with extended timeouts for slow networks and webhooks.\"\"\"
    def __init__(self, *args, **kwargs):
        # Set extended connect/pool/read/write timeouts (60 seconds each)
        kwargs.setdefault('connect_timeout', 60.0)
        kwargs.setdefault('pool_timeout', 60.0)
        kwargs.setdefault('read_timeout', 60.0)
        kwargs.setdefault('write_timeout', 60.0)
        super().__init__(*args, **kwargs)"""

# Normalize CRLF for matching
text_norm = text.replace("\r\n", "\n")
old_block_norm = old_block.replace("\r\n", "\n")
new_block_norm = new_block.replace("\r\n", "\n")

if old_block_norm in text_norm:
    text_norm = text_norm.replace(old_block_norm, new_block_norm)
    # Restore CRLF
    p_tg.write_text(text_norm.replace("\n", "\r\n"), encoding="utf-8")
    print("Reordenado sys.path en monitor_v1/telegram_bot.py correctamente")
else:
    print("ERROR: No se encontró old_block")
