"""
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
