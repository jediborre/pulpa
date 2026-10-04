"""
tmp/debug_connection/test_bot_v3.py
Proposito: Probar las funciones de datos del nuevo bot v3 (build_status_text,
build_signals_text) sin arrancar el polling de Telegram.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications.bot_runner import build_status_text, build_signals_text, fetch_v3_stats

print("=== /status ===")
print(build_status_text())
print("\n=== /signals (hoy) ===")
print(build_signals_text())
print("\n=== stats raw ===")
print(fetch_v3_stats())
