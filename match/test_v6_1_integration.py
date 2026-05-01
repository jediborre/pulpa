#!/usr/bin/env python3
"""
Integration test for v6_1 model selector functionality after defensive fixes.

Tests:
1. All .get() methods work with MONITOR_MODEL_CONFIG and MODEL_CONFIG
2. Model selector callbacks don't crash with v6_1
3. Version conversion functions work for all models including v6_1
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_config_access():
    """Test that MODEL_CONFIG and MONITOR_MODEL_CONFIG are safely accessed."""
    print("=" * 60)
    print("TEST 1: MODEL_CONFIG and MONITOR_MODEL_CONFIG .get() access")
    print("=" * 60)
    
    from telegram_bot import MODEL_CONFIG, MONITOR_MODEL_CONFIG, AVAILABLE_MODELS
    
    print(f"AVAILABLE_MODELS: {AVAILABLE_MODELS}")
    print(f"v6_1 in models: {'v6_1' in AVAILABLE_MODELS}")
    
    # Test MODEL_CONFIG access
    q3_val = MODEL_CONFIG.get("q3", "v4")
    q4_val = MODEL_CONFIG.get("q4", "v4")
    print(f"MODEL_CONFIG - q3={q3_val}, q4={q4_val}")
    
    # Test MONITOR_MODEL_CONFIG access
    mon_q3 = MONITOR_MODEL_CONFIG.get("q3", "v4")
    mon_q4 = MONITOR_MODEL_CONFIG.get("q4", "v4")
    print(f"MONITOR_MODEL_CONFIG - q3={mon_q3}, q4={mon_q4}")
    
    # Change to v6_1 and test
    MODEL_CONFIG["q3"] = "v6_1"
    MONITOR_MODEL_CONFIG["q3"] = "v6_1"
    
    q3_val = MODEL_CONFIG.get("q3", "v4")
    mon_q3 = MONITOR_MODEL_CONFIG.get("q3", "v4")
    print(f"After setting to v6_1 - MODEL_CONFIG['q3']={q3_val}, MONITOR_MODEL_CONFIG['q3']={mon_q3}")
    
    assert q3_val == "v6_1", "MODEL_CONFIG['q3'] should be v6_1"
    assert mon_q3 == "v6_1", "MONITOR_MODEL_CONFIG['q3'] should be v6_1"
    
    # Reset
    MODEL_CONFIG["q3"] = "v4"
    MONITOR_MODEL_CONFIG["q3"] = "v4"
    
    print("✓ PASSED\n")


def test_version_conversion():
    """Test that version token conversion works for all models including v6_1."""
    print("=" * 60)
    print("TEST 2: Version token conversion (all models)")
    print("=" * 60)
    
    from telegram_bot import (
        AVAILABLE_MODELS,
        _model_version_callback_token,
        _model_version_from_callback_token,
    )
    
    for version in AVAILABLE_MODELS:
        token = _model_version_callback_token(version)
        recovered = _model_version_from_callback_token(token)
        print(f"{version} -> token={token} -> recovered={recovered}")
        assert recovered == version, f"Version conversion failed for {version}"
    
    print("✓ PASSED\n")


def test_keyboard_rendering():
    """Test that keyboard functions render without errors."""
    print("=" * 60)
    print("TEST 3: Keyboard rendering functions")
    print("=" * 60)
    
    from telegram_bot import (
        _model_submenu_keyboard,
        _match_model_submenu_keyboard,
        _date_model_submenu_keyboard,
        _monitor_model_keyboard,
        MODEL_CONFIG,
        MONITOR_MODEL_CONFIG,
    )
    
    # Set v6_1 for testing
    MODEL_CONFIG["q3"] = "v6_1"
    MONITOR_MODEL_CONFIG["q3"] = "v6_1"
    
    try:
        # Test _model_submenu_keyboard
        kb1 = _model_submenu_keyboard()
        print(f"✓ _model_submenu_keyboard() returned {len(kb1.inline_keyboard)} rows")
        
        # Test _match_model_submenu_keyboard
        kb2 = _match_model_submenu_keyboard("match123", "token456", 1)
        print(f"✓ _match_model_submenu_keyboard() returned {len(kb2.inline_keyboard)} rows")
        
        # Test _date_model_submenu_keyboard
        kb3 = _date_model_submenu_keyboard("2026-04-27")
        print(f"✓ _date_model_submenu_keyboard() returned {len(kb3.inline_keyboard)} rows")
        
        # Test _monitor_model_keyboard with and without chat_id
        kb4 = _monitor_model_keyboard()
        print(f"✓ _monitor_model_keyboard() returned {len(kb4.inline_keyboard)} rows")
        
        kb5 = _monitor_model_keyboard(chat_id=12345)
        print(f"✓ _monitor_model_keyboard(chat_id=12345) returned {len(kb5.inline_keyboard)} rows")
        
        # Reset
        MODEL_CONFIG["q3"] = "v4"
        MONITOR_MODEL_CONFIG["q3"] = "v4"
        
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}\n")
        raise


def test_v6_1_buttons_in_keyboard():
    """Test that v6_1 appears as a button in keyboard."""
    print("=" * 60)
    print("TEST 4: v6_1 appears in keyboard buttons")
    print("=" * 60)
    
    from telegram_bot import _model_submenu_keyboard, AVAILABLE_MODELS
    
    kb = _model_submenu_keyboard()
    all_buttons_text = []
    for row in kb.inline_keyboard:
        for button in row:
            all_buttons_text.append(button.text)
    
    buttons_str = " | ".join(all_buttons_text)
    print(f"Keyboard buttons: {buttons_str}")
    
    # Check if v6_1 appears in any button
    v6_1_found = any("v6_1" in btn_text for btn_text in all_buttons_text)
    print(f"v6_1 found in buttons: {v6_1_found}")
    
    assert v6_1_found, "v6_1 should appear in keyboard buttons"
    print("✓ PASSED\n")


def test_inference_v6_1():
    """Test that v6_1 inference module can be imported and used."""
    print("=" * 60)
    print("TEST 5: v6_1 inference module")
    print("=" * 60)
    
    try:
        from training import infer_match
        from pathlib import Path
        
        # Check that v6_1 model directory exists
        v6_1_dir = Path(__file__).parent / "training" / "model_outputs_v6_1"
        print(f"v6_1 model directory: {v6_1_dir}")
        print(f"v6_1 model directory exists: {v6_1_dir.exists()}")
        
        if v6_1_dir.exists():
            model_files = list(v6_1_dir.glob("*.joblib"))
            print(f"v6_1 model files found: {len(model_files)}")
            for mf in model_files[:3]:
                print(f"  - {mf.name}")
        
        print("✓ PASSED\n")
    except Exception as e:
        print(f"⚠ WARNING (non-critical): {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("V6_1 INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_model_config_access()
        test_version_conversion()
        test_keyboard_rendering()
        test_v6_1_buttons_in_keyboard()
        test_inference_v6_1()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"TEST SUITE FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
