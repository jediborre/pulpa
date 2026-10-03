"""
Ubicación original: tmp/db_audit/apply_db_redirections.py
Propósito / Qué hace:
Script auxiliar de migración que actualiza de manera segura y precisa las referencias
a matches.db para que apunten a la raíz del proyecto (/matches.db), respetando saltos de línea (CRLF/LF)
y codificación UTF-8 en archivos de producción y auxiliares de tmp/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 1. match/telegram_bot.py
p_tg = ROOT / "match" / "telegram_bot.py"
if p_tg.exists():
    text = p_tg.read_text(encoding="utf-8")
    if 'BASE_DIR / _db_p' in text or 'BASE_DIR / "matches.db"' in text:
        text = text.replace('BASE_DIR / _db_p', 'PROJECT_ROOT / _db_p')
        text = text.replace('BASE_DIR / "matches.db"', 'PROJECT_ROOT / "matches.db"')
        p_tg.write_text(text, encoding="utf-8")
        print("Actualizado match/telegram_bot.py")

# 2. match/analyze_inference_debug.py
p_aid = ROOT / "match" / "analyze_inference_debug.py"
if p_aid.exists():
    text = p_aid.read_text(encoding="utf-8")
    old_line = 'DB_PATH = Path(__file__).resolve().parent / "matches.db"'
    new_line = 'DB_PATH = Path(__file__).resolve().parents[1] / "matches.db"'
    if old_line in text:
        text = text.replace(old_line, new_line)
        p_aid.write_text(text, encoding="utf-8")
        print("Actualizado match/analyze_inference_debug.py")

# 3. match/scripts/compare_scrapers.py
p_cs = ROOT / "match" / "scripts" / "compare_scrapers.py"
if p_cs.exists():
    text = p_cs.read_text(encoding="utf-8")
    old_target = 'default=str(ROOT / "match" / "matches.db")'
    new_target = 'default=str(ROOT / "matches.db")'
    if old_target in text:
        text = text.replace(old_target, new_target)
        p_cs.write_text(text, encoding="utf-8")
        print("Actualizado match/scripts/compare_scrapers.py")

# 4. menu.bat
p_menu = ROOT / "menu.bat"
if p_menu.exists():
    text = p_menu.read_text(encoding="utf-8", errors="ignore")
    old_backfill = r"python match\scripts\backfill.py match\matches.db --all --backend chrome --session-rotate 20"
    new_backfill = r"python match\scripts\backfill.py matches.db --all --backend chrome --session-rotate 20"
    if old_backfill in text:
        text = text.replace(old_backfill, new_backfill)
        p_menu.write_text(text, encoding="utf-8")
        print("Actualizado menu.bat")

# 5. tmp/ files
tmp_files = [
    "backfill_h2h_masivo.py",
    "backfill_h2h_and_revaluate.py",
    "compare_h2h_sources.py",
    "validate_h2h_features.py",
    "nba_bet_margin_analysis.py",
    "nba_margin_analysis.py",
    "check_db_schema.py",
    "check_9_h2h_results.py",
    "check_ft_failed.py",
    "check_h2h_stats.py",
    "check_leagues.py",
    "check_league_bets.py",
    "check_match.py",
    "check_match2.py",
    "check_match_status.py",
    "check_q4_scores.py",
    "check_q4_status.py",
    "check_zenit.py",
    "diag_leagues.py",
    "get_ft_ids.py",
    "search_fast.py",
    "search_matches.py",
    "search_teams.py",
    "test_borrego.py",
    "check_data_location.py",
    "db_query.py",
    "explore_comebacks.py",
    "explore_db_schema.py",
    "explore_game_patterns.py",
]

for filename in tmp_files:
    file_path = ROOT / "tmp" / filename
    if not file_path.exists():
        continue
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    modified = False

    # Replacements for paths inside match/
    replacements = [
        ('str(ROOT / "match" / "matches.db")', 'str(ROOT / "matches.db")'),
        ('ROOT / "match" / "matches.db"', 'ROOT / "matches.db"'),
        ('match/matches.db', 'matches.db'),
        (r'match\matches.db', 'matches.db'),
        (r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db", r"matches.db"),
        (r"c:\Users\App\Desktop\pulpa\match\matches.db", r"matches.db"),
        (r"os.path.join(os.path.dirname(__file__), '..', 'match', 'matches.db')", r"os.path.join(os.path.dirname(__file__), '..', 'matches.db')"),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            modified = True

    if modified:
        file_path.write_text(text, encoding="utf-8")
        print(f"Actualizado tmp/{filename}")

print("Migracion de rutas a matches.db en raiz completada con exito.")
