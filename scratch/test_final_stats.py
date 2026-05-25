import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import asyncio
from pathlib import Path

# Cargar el path del workspace para importar los módulos de match
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.notifications.telegram_bot import build_all_models_stats_message

def test():
    print("=== TESTING INTEGRATED build_all_models_stats_message ===")
    msg = build_all_models_stats_message()
    print("\n" + msg)
    print("=========================================================")

if __name__ == "__main__":
    test()
