"""Fix corrupted emoji bytes in _monitor_keyboard and add missing config button."""
import re

path = "match/telegram_bot.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

# Find the corrupted lines by looking for the callback_data anchors
# and replace the entire InlineKeyboardButton(...) call with correct text.

# Pattern for the broken "Señales de hoy" line (any chars before "Señales")
old_signals = re.sub(
    r'rows\.append\(\[InlineKeyboardButton\("[^"]*Se\u00f1ales de hoy"',
    'PLACEHOLDER_SIGNALS',
    content,
    count=1,
)

# We'll do a targeted replacement using the known callback_data values
# as anchors, since the text before the string may be corrupted.

# Replace corrupted "Señales de hoy" button
content = re.sub(
    r'rows\.append\(\[InlineKeyboardButton\("[^\x00-\x7F\u2600-\u27FF]*(?:\ufffd)*\s*Se\xf1ales de hoy",\s*callback_data="monitor:signals_today"\)\]\)',
    'rows.append([InlineKeyboardButton("\U0001f4ca Se\u00f1ales de hoy", callback_data="monitor:signals_today")])',
    content,
)

# Replace corrupted "Bitácora reciente" button 
content = re.sub(
    r'rows\.append\(\[InlineKeyboardButton\("[^\x00-\x7F\u2600-\u27FF]*(?:\ufffd)*\U0001f4dc\s*Bit\u00e1cora reciente",\s*callback_data="monitor:log"\)\]\)',
    'rows.append([InlineKeyboardButton("\U0001f4dc Bit\u00e1cora reciente", callback_data="monitor:log")])',
    content,
)

# Now inject the Config button between signals_today and log buttons
config_btn = '    rows.append([InlineKeyboardButton("\u2699\ufe0f Config simulaci\u00f3n", callback_data="monitor:betcfg")])\n'
signals_marker = '    rows.append([InlineKeyboardButton("\U0001f4ca Se\u00f1ales de hoy", callback_data="monitor:signals_today")])\n'
log_marker     = '    rows.append([InlineKeyboardButton("\U0001f4dc Bit\u00e1cora reciente", callback_data="monitor:log")])\n'

if config_btn.strip() not in content:
    content = content.replace(
        signals_marker + log_marker,
        signals_marker + config_btn + log_marker,
    )
    print("Inserted Config button")
else:
    print("Config button already present")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done. Verifying...")
with open(path, encoding="utf-8") as f:
    data = f.read()
idx = data.find('monitor:signals_today')
print(repr(data[data.rfind('\n', 0, idx)+1 : data.find('\n', idx)+1]))
idx2 = data.find('monitor:betcfg")')
if idx2 != -1:
    print(repr(data[data.rfind('\n', 0, idx2)+1 : data.find('\n', idx2)+1]))
else:
    print("WARNING: betcfg button NOT found")
idx3 = data.find('monitor:log")')
print(repr(data[data.rfind('\n', 0, idx3)+1 : data.find('\n', idx3)+1]))
