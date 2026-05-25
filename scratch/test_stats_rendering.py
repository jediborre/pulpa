import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import asyncio
from pathlib import Path

# Cargar el path del workspace para importar los módulos de match
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.notifications.telegram_bot import _get_model_stats_dict

def visual_len(s: str) -> int:
    emojis = ["✅", "❌", "🟢", "🟡", "⚪", "🔴"]
    clean_s = s.replace("\ufe0f", "")
    length = 0
    for char in clean_s:
        if char in emojis:
            length += 2
        else:
            length += 1
    return length

def pad_cell(text: str, target_width: int) -> str:
    v_len = visual_len(text)
    padding_needed = max(0, target_width - v_len)
    return text + (" " * padding_needed)

def test_build_message():
    models = ["v6_2", "m27_v3"]
    all_stats = {m: _get_model_stats_dict(m) for m in models}

    BET_CATS = ["🟢", "🟡", "⚪️", "🟡⚪️"]
    COL_W = 24  # visual width per column

    header_parts = [pad_cell(f"{m} Stats", COL_W) for m in models]
    header = "    ".join(header_parts).rstrip()

    table_lines = [header]

    for cat in BET_CATS:
        for outcome, emoji in (("win", "✅"), ("loss", "❌")):
            cells = []
            for m in models:
                st  = all_stats[m][cat]
                n   = st["win"] if outcome == "win" else st["loss"]
                tot = st["win"] + st["loss"]
                
                # Para alinear el número, si la categoría tiene 3 emojis (longitud 3), usamos menos espacio inicial
                # ✅🟢  5   70%   vs   ✅🟡⚪️ 4   50%
                clean_cat = cat.replace("\ufe0f", "")
                emoji_spacing = " " if len(clean_cat) > 1 else "  "
                
                if tot > 0:
                    pct  = int(round(n * 100.0 / tot))
                    cell = f"{emoji}{cat}{emoji_spacing}{n}   {pct}%"
                else:
                    cell = f"{emoji}{cat}{emoji_spacing}0   0%"
                cells.append(pad_cell(cell, COL_W))
            table_lines.append("    ".join(cells).rstrip())

    # ── fila NO_BET ────────────────────────────────────────────────────────
    nobet_cells = []
    for m in models:
        count = all_stats[m]["🔴"]["count"]
        # Calcular total de todos los logs del modelo
        total_all = count
        for cat in BET_CATS:
            total_all += all_stats[m][cat]["win"] + all_stats[m][cat]["loss"]
        
        if total_all > 0:
            pct = int(round(count * 100.0 / total_all))
            cell = f"🔴 {count}   {pct}%"
        else:
            cell = f"🔴 0   0%"
        nobet_cells.append(pad_cell(cell, COL_W))
    table_lines.append("    ".join(nobet_cells).rstrip())

    table_content = "\n".join(table_lines)
    print("\n--- NEW VISUAL ALIGNED TABLE ---")
    print(table_content)
    print("--------------------------------")

if __name__ == "__main__":
    test_build_message()
