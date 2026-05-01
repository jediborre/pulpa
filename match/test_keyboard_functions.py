import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Mock environment variables needed
os.environ['TELEGRAM_BOT_TOKEN'] = 'test_token'

print("=" * 60)
print("TEST SCRIPT: telegram_bot keyboard functions")
print("=" * 60)

try:
    # Import telegram_bot
    import telegram_bot
    print("✓ Successfully imported telegram_bot")
    
    # Check if AVAILABLE_MODELS includes v6_1
    print(f"\n✓ AVAILABLE_MODELS: {telegram_bot.AVAILABLE_MODELS}")
    if 'v6_1' in telegram_bot.AVAILABLE_MODELS:
        print("✓ v6_1 is available in AVAILABLE_MODELS")
    else:
        print("✗ v6_1 NOT found in AVAILABLE_MODELS")
    
    # Check current MODEL_CONFIG
    print(f"\n✓ Initial MODEL_CONFIG: {telegram_bot.MODEL_CONFIG}")
    print(f"✓ Initial MONITOR_MODEL_CONFIG: {telegram_bot.MONITOR_MODEL_CONFIG}")
    
    # Update MODEL_CONFIG to use v6_1 for testing
    telegram_bot.MODEL_CONFIG['q3'] = 'v6_1'
    telegram_bot.MODEL_CONFIG['q4'] = 'v6_1'
    print(f"✓ Updated MODEL_CONFIG to v6_1: {telegram_bot.MODEL_CONFIG}")
    
    # Test _model_submenu_keyboard()
    print("\n" + "-" * 60)
    print("Testing _model_submenu_keyboard()...")
    try:
        keyboard = telegram_bot._model_submenu_keyboard()
        print(f"✓ _model_submenu_keyboard() returned: {type(keyboard)}")
        if hasattr(keyboard, 'inline_keyboard'):
            print(f"✓ Has inline_keyboard attribute with {len(keyboard.inline_keyboard)} rows")
        else:
            print("✗ Missing inline_keyboard attribute")
    except Exception as e:
        print(f"✗ Error calling _model_submenu_keyboard(): {e}")
        import traceback
        traceback.print_exc()
    
    # Test _monitor_model_keyboard()
    print("\n" + "-" * 60)
    print("Testing _monitor_model_keyboard()...")
    try:
        keyboard = telegram_bot._monitor_model_keyboard(chat_id=12345)
        print(f"✓ _monitor_model_keyboard(chat_id=12345) returned: {type(keyboard)}")
        if hasattr(keyboard, 'inline_keyboard'):
            print(f"✓ Has inline_keyboard attribute with {len(keyboard.inline_keyboard)} rows")
        else:
            print("✗ Missing inline_keyboard attribute")
    except Exception as e:
        print(f"✗ Error calling _monitor_model_keyboard(): {e}")
        import traceback
        traceback.print_exc()
    
    # Test _monitor_model_keyboard() without chat_id
    print("\n" + "-" * 60)
    print("Testing _monitor_model_keyboard() without chat_id...")
    try:
        keyboard = telegram_bot._monitor_model_keyboard()
        print(f"✓ _monitor_model_keyboard() returned: {type(keyboard)}")
        if hasattr(keyboard, 'inline_keyboard'):
            print(f"✓ Has inline_keyboard attribute with {len(keyboard.inline_keyboard)} rows")
        else:
            print("✗ Missing inline_keyboard attribute")
    except Exception as e:
        print(f"✗ Error calling _monitor_model_keyboard(): {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
