# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.utils.logger import COLOR_GREEN, COLOR_BRIGHT_RED, COLOR_RESET, COLOR_WARNING
from bet_monitor_v2.main import colorize_scores

def test_score_formatting():
    print("=== STARTING SCORE AND BRACKET FORMATTING TESTS ===")

    # Test 1: colorize_scores
    print("\n[TEST 1] Testing colorize_scores:")
    # Case A: Home Wins
    h, a = colorize_scores(40, 33)
    assert h == f"{COLOR_GREEN}40{COLOR_RESET}", f"Expected green for home, got {repr(h)}"
    assert a == f"{COLOR_BRIGHT_RED}33{COLOR_RESET}", f"Expected bright red for away, got {repr(a)}"
    print("   [OK] Home wins (40 - 33): Home score is green, Away score is red.")

    # Case B: Away Wins
    h, a = colorize_scores(19, 20)
    assert h == f"{COLOR_BRIGHT_RED}19{COLOR_RESET}", f"Expected bright red for home, got {repr(h)}"
    assert a == f"{COLOR_GREEN}20{COLOR_RESET}", f"Expected green for away, got {repr(a)}"
    print("   [OK] Away wins (19 - 20): Home score is red, Away score is green.")

    # Case C: Tie
    h, a = colorize_scores(10, 10)
    assert h == "10", f"Expected plain 10, got {repr(h)}"
    assert a == "10", f"Expected plain 10, got {repr(a)}"
    print("   [OK] Tie (10 - 10): Scores remain uncolored.")

    # Case D: Non-integer string or invalid type
    h, a = colorize_scores("N/A", None)
    assert h == "N/A", f"Expected 'N/A', got {repr(h)}"
    assert a == "None", f"Expected 'None', got {repr(a)}"
    print("   [OK] Non-integer fallback: Safely handles non-integer values.")

    # Test 2: Bracket Building Simulation
    print("\n[TEST 2] Testing bracket building simulation:")

    # Scenario A: Full information (period Q2, minute 21, quarter score 20 - 19)
    period_colored = "Q2"  # plain for Q2
    minute = 21
    q_home, q_away = 20, 19
    
    bracket_parts = [period_colored]
    if minute is not None:
        bracket_parts.append(f"MIN~{minute}")
    
    q_home_col, q_away_col = colorize_scores(q_home, q_away)
    bracket_parts.append(f"{q_home_col} - {q_away_col}")
    
    bracket_content = " | ".join(bracket_parts)
    bracket_label = f"[{bracket_content} ]" if bracket_content else ""
    
    expected_bracket = f"[Q2 | MIN~21 | {COLOR_GREEN}20{COLOR_RESET} - {COLOR_BRIGHT_RED}19{COLOR_RESET} ]"
    assert bracket_label == expected_bracket, f"Expected: {repr(expected_bracket)}\nGot:      {repr(bracket_label)}"
    print(f"   [OK] Full info bracket matches: {bracket_label}")

    # Scenario B: Missing quarter score (only period and minute)
    bracket_parts = [f"{COLOR_GREEN}Q4{COLOR_RESET}"]
    minute = 35
    bracket_parts.append(f"MIN~{minute}")
    bracket_content = " | ".join(bracket_parts)
    bracket_label = f"[{bracket_content} ]" if bracket_content else ""
    expected_bracket = f"[{COLOR_GREEN}Q4{COLOR_RESET} | MIN~35 ]"
    assert bracket_label == expected_bracket, f"Expected: {repr(expected_bracket)}\nGot:      {repr(bracket_label)}"
    print(f"   [OK] Missing quarter score bracket matches: {bracket_label}")

    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_score_formatting()
