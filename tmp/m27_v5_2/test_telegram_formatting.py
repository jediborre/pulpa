"""
tmp/m27_v5_2/test_telegram_formatting.py
Verificación de formato de mensajes de Telegram ante señales BET vs NO_BET.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitor_v3.notifications.telegram_bot import format_combined_bet_message, format_bet_message

def test():
    print("=== CASO 1: Un modelo tiene BET real (P>=75%), otros están en NO BET ===")
    preds_case_1 = {
        "m27_v5_2": {
            "signal": "BET_HOME",
            "pick": "HOME",
            "confidence": 0.78,
            "raw_model_signal": "BET",
        },
        "m27_v5_1": {
            "signal": "NO_BET",
            "pick": "HOME",
            "confidence": 0.65,
            "raw_model_signal": "NO_BET",
        },
        "v6_2": {
            "signal": "NO_BET",
            "pick": "AWAY",
            "confidence": 0.05,
            "raw_model_signal": "NO_BET",
        },
    }
    msg_1 = format_combined_bet_message(preds_case_1, "Lakers", "Celtics")
    print(msg_1)
    
    print("\n=== CASO 2: Ningún modelo tiene BET real (todos en NO BET como en el caso U15) ===")
    preds_case_2 = {
        "m27_v5_2": {
            "signal": "NO_BET",
            "pick": "AWAY",
            "confidence": 0.51,
            "raw_model_signal": "NO_BET",
        },
        "m27_v5_1": {
            "signal": "NO_BET",
            "pick": "AWAY",
            "confidence": 0.51,
            "raw_model_signal": "NO_BET",
        },
        "v6_2": {
            "signal": "NO_BET",
            "pick": "HOME",
            "confidence": 0.01,
            "raw_model_signal": "NO_BET",
        },
        "m27_v3": {
            "signal": "NO_BET",
            "pick": "HOME",
            "confidence": 0.54,
            "raw_model_signal": "NO_BET",
        },
    }
    msg_2 = format_combined_bet_message(preds_case_2, "Siguldas SS U15", "Akademia Pierniki U15")
    print(msg_2)

if __name__ == "__main__":
    test()
