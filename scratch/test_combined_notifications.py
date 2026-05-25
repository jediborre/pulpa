# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# Fix terminal encoding issues on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.notifications.telegram_bot import (
    format_combined_bet_message, 
    format_combined_final_message,
    _format_telegram_details
)

def test_combined_notifications():
    print("=== STARTING COMBINED NOTIFICATION FORMATTING TESTS ===")

    home_team = "CD Universitario Concepcion"
    away_team = "Sportiva Italiana"
    league = "Liga Nacional Femenina Chile, Fase regular"
    scheduled_ts = 1779662400 # 2026-05-24 14:00:00 (some timestamp)
    minute = 31
    q_key = "Q3"
    q_home = 6
    q_away = 11

    # Scenario 1: Both models low confidence (<30%), ordinary Q4
    # Expected title: 🟡 APUESTA Q4
    # Predictions: v6_2 (22%), m27_v3 (21%)
    predictions = {
        "v6_2": {
            "signal": "BET_HOME",
            "pick": "HOME",
            "confidence": 0.22
        },
        "m27_v3": {
            "signal": "BET_AWAY",
            "pick": "AWAY",
            "confidence": 0.21
        }
    }
    
    base_text = format_combined_bet_message(predictions, home_team, away_team)
    full_text = _format_telegram_details(
        base_text, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away
    )
    
    print("\n[TEST Scenario 1 Output]:")
    print(full_text)
    
    # Assertions
    assert "🟡 APUESTA Q4" in full_text
    assert "v6_2 22% → 🏠 CD Universitario Concepcion" in full_text
    assert "m27_v3 21% → ✈️ Sportiva Italiana" in full_text
    assert "CD Universitario Concepcion vs Sportiva Italiana" in full_text
    assert "Liga Nacional Femenina Chile, Fase regular" in full_text
    print("   [OK] Scenario 1 matches expectations perfectly!")

    # Scenario 2: One high confidence (>=30%), ordinary Q4
    # Expected title: 🟢 APUESTA Q4
    predictions_high = {
        "v6_2": {
            "signal": "BET_HOME",
            "pick": "HOME",
            "confidence": 0.22
        },
        "m27_v3": {
            "signal": "BET_AWAY",
            "pick": "AWAY",
            "confidence": 0.35
        }
    }
    base_text_high = format_combined_bet_message(predictions_high, home_team, away_team)
    print("\n[TEST Scenario 2 Base Text]:")
    print(base_text_high)
    assert "🟢 APUESTA Q4" in base_text_high
    print("   [OK] Scenario 2 (one high confidence) upgrades indicator to green 🟢.")

    # Scenario 3: Both low confidence (<30%), LATE Q4
    # Expected title: 🟡⚪️ APUESTA TARDIA Q4
    predictions_late = {
        "v6_2": {
            "signal": "BET_HOME_LATE",
            "pick": "HOME",
            "confidence": 0.22
        },
        "m27_v3": {
            "signal": "BET_AWAY_LATE",
            "pick": "AWAY",
            "confidence": 0.21
        }
    }
    base_text_late = format_combined_bet_message(predictions_late, home_team, away_team)
    print("\n[TEST Scenario 3 Base Text]:")
    print(base_text_late)
    assert "🟡⚪️ APUESTA TARDIA Q4" in base_text_late
    print("   [OK] Scenario 3 (late and low confidence) produces 🟡⚪️ APUESTA TARDIA Q4.")

    # Scenario 4: One high confidence, LATE Q4
    # Expected title: ⚪️ APUESTA TARDIA Q4
    predictions_late_high = {
        "v6_2": {
            "signal": "BET_HOME_LATE",
            "pick": "HOME",
            "confidence": 0.35
        },
        "m27_v3": {
            "signal": "BET_AWAY_LATE",
            "pick": "AWAY",
            "confidence": 0.21
        }
    }
    base_text_late_high = format_combined_bet_message(predictions_late_high, home_team, away_team)
    print("\n[TEST Scenario 4 Base Text]:")
    print(base_text_late_high)
    assert "⚪️ APUESTA TARDIA Q4" in base_text_late_high
    print("   [OK] Scenario 4 (late and one high confidence) produces ⚪️ APUESTA TARDIA Q4.")

    # Scenario 5: Combined Final Confirmation (one win, one loss)
    # Expected title: ✅ CD Universitario Concepcion vs Sportiva Italiana
    # Predictions: v6_2 (win), m27_v3 (loss)
    logs = [
        {
            "model_version": "v6_2",
            "confidence": 0.22,
            "picked_side": "HOME",
            "signal_type": "BET_HOME",
            "result": "win"
        },
        {
            "model_version": "m27_v3",
            "confidence": 0.21,
            "picked_side": "AWAY",
            "signal_type": "BET_AWAY",
            "result": "loss"
        }
    ]
    base_final_text = format_combined_final_message(logs, home_team, away_team)
    full_final_text = _format_telegram_details(
        base_final_text, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away, is_final=True
    )
    print("\n[TEST Scenario 5 Combined Final Confirmation Output]:")
    print(full_final_text)
    
    assert "🟡 APUESTA Q4" in full_final_text
    assert "✅ v6_2 22% → 🏠 CD Universitario Concepcion" in full_final_text
    assert "❌ m27_v3 21% → ✈️ Sportiva Italiana" in full_final_text
    assert "Do May 24, 4:40 pm | Q3 6 - 11" in full_final_text
    print("   [OK] Scenario 5 (Combined Final Confirmation) matches expectations perfectly!")

    # Scenario 6: Combined Final Confirmation with NO_BET (one win, one NO_BET)
    # Expected title: 🟡 APUESTA Q4
    # Predictions: v6_2 (win), m27_v3 (NO_BET)
    logs_with_nobet = [
        {
            "model_version": "v6_2",
            "confidence": 0.22,
            "picked_side": "HOME",
            "signal_type": "BET_HOME",
            "result": "win"
        },
        {
            "model_version": "m27_v3",
            "confidence": 0.04,
            "picked_side": "NONE",
            "signal_type": "NO_BET",
            "result": "push"
        }
    ]
    base_final_nobet = format_combined_final_message(logs_with_nobet, home_team, away_team)
    full_final_nobet = _format_telegram_details(
        base_final_nobet, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away, is_final=True
    )
    print("\n[TEST Scenario 6 Combined Final Confirmation with NO_BET Output]:")
    print(full_final_nobet)
    
    assert "🟡 APUESTA Q4" in full_final_nobet
    assert "✅ v6_2 22% → 🏠 CD Universitario Concepcion" in full_final_nobet
    assert "🔴 m27_v3 NO BET" in full_final_nobet
    assert "Do May 24, 4:40 pm | Q3 6 - 11" in full_final_nobet
    print("   [OK] Scenario 6 (Combined Final Confirmation with NO_BET) matches expectations perfectly!")

    print("\n=== ALL COMBINED NOTIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_combined_notifications()
