"""
Ubicación original: temp_scripts/backfill_h2h_masivo.py
Propósito / Qué hacía:
Backfill masivo de historial H2H desde SofaScore para ~23,000 partidos pendientes en matches.db.
"""

"""
Backfill masivo de H2H para los 23k partidos restantes.
Prioriza ligas con más datos, excluye ligas de mujeres.
"""

import sys
import os
import time
import random
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Disable proxy
os.environ['SOFASCORE_PROXY_URL'] = ''

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match" / "training"))

import match.scraper as scraper
import match.db as db_mod

DB_PATH = str(ROOT / "match" / "matches.db")

WAIT_BASE_SECS = 30
WAIT_JITTER_SECS = 15

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def get_matches_needing_h2h_prioritized():
    """Get matches without H2H, prioritized by league size, excluding women leagues."""
    print(f"  [1/4] Consultando liga sizes...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get league counts (excluding women leagues)
    leagues = conn.execute("""
        SELECT league, COUNT(*) as cnt
        FROM matches
        WHERE league NOT LIKE '%women%' 
          AND league NOT LIKE '%WNBA%'
          AND league NOT LIKE '%FIBA Women%'
          AND league NOT LIKE '%EuroLeague Women%'
        GROUP BY league
        ORDER BY cnt DESC
    """).fetchall()
    
    league_priority = {r['league']: i for i, r in enumerate(leagues)}
    print(f"  [2/4] {len(leagues)} ligas encontradas")
    
    print(f"  [3/4] Buscando partidos sin H2H (puede tardar 1-2 min)...")
    # Get matches without H2H - optimized with LEFT JOIN
    matches = conn.execute("""
        SELECT m.match_id, m.home_team, m.away_team, m.date, m.league, m.custom_id
        FROM matches m
        LEFT JOIN match_h2h h ON h.match_id = m.match_id AND h.q1_home IS NOT NULL
        WHERE m.custom_id IS NOT NULL
          AND h.match_id IS NULL
    """).fetchall()
    
    print(f"  [4/4] Filtrando y priorizando {len(matches)} partidos...")
    
    result = []
    for r in matches:
        league = r['league'] or ''
        # Skip women leagues
        if any(x in league.lower() for x in ['women', 'wnba', 'fiba women', 'euroleague women']):
            continue
        # Skip 2-half leagues (NCAA, NAIA, etc.)
        if any(x in league for x in ['NCAA', 'NAIA', 'NJCAA']):
            continue
        # Skip Nostra.lt-RKL (2 halves)
        if 'Nostra.lt' in league or 'RKL' in league:
            continue
        # Skip NBA (already has H2H data)
        if 'NBA' in league:
            continue
        
        priority = league_priority.get(league, 9999)
        result.append({
            'match_id': r['match_id'],
            'home_team': r['home_team'],
            'away_team': r['away_team'],
            'date': r['date'],
            'league': league,
            'custom_id': r['custom_id'],
            'priority': priority,
        })
    
    # Sort by priority (league size) then by date
    result.sort(key=lambda x: (x['priority'], x['date']))
    
    conn.close()
    print(f"  Listo: {len(result)} partidos a procesar\n")
    return result


def download_h2h_for_match(match_id, custom_id):
    """Download H2H by clicking the H2H button and intercepting the response."""
    conn = db_mod.get_conn(DB_PATH)
    db_mod.init_db(conn)
    
    h2h_rows = []
    debug_info = {}
    
    try:
        from playwright.sync_api import sync_playwright
        
        _headless = os.environ.get("SOFASCORE_HEADLESS", "1").strip() in ("1", "true", "yes")
        
        warmup_url = f"https://www.sofascore.com/event/{match_id}"
        
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=_headless)
            ctx = browser.new_context(user_agent=scraper.STANDARD_UA)
            page = ctx.new_page()
            
            # Warmup
            try:
                page.goto(warmup_url, wait_until="domcontentloaded", timeout=10_000)
            except Exception as e:
                debug_info["warmup_error"] = str(e)[:100]
            time.sleep(3)
            
            debug_info["page_url"] = page.url[:120]
            
            # Click H2H button
            try:
                h2h_selectors = [
                    'text="Head to Head"',
                    'text="H2H"',
                    '[data-testid="h2h"]',
                    'button:has-text("H2H")',
                ]
                clicked = False
                for selector in h2h_selectors:
                    try:
                        if page.locator(selector).first.is_visible():
                            page.locator(selector).first.click()
                            clicked = True
                            break
                    except:
                        continue
                
                if clicked:
                    # Wait for H2H response
                    try:
                        with page.expect_response(
                            lambda r: "/h2h/events" in r.url and r.ok,
                            timeout=8000
                        ) as response_info:
                            pass
                        h2h_response = response_info.value
                        if h2h_response:
                            h2h_data = h2h_response.json()
                            events = h2h_data.get("events", []) if isinstance(h2h_data, dict) else (h2h_data or [])
                            for entry in events:
                                if not isinstance(entry, dict):
                                    continue
                                hs = entry.get("homeScore") or {}
                                as_ = entry.get("awayScore") or {}
                                ts_val = entry.get("startTimestamp", 0)
                                dt_val = datetime.fromtimestamp(ts_val, tz=timezone.utc) if ts_val else None
                                h2h_rows.append({
                                    "match_id": str(match_id),
                                    "h2h_match_id": str(entry.get("id") or ""),
                                    "date": dt_val.strftime("%Y-%m-%d") if dt_val else "",
                                    "timestamp": ts_val or None,
                                    "home_team": (entry.get("homeTeam") or {}).get("name", ""),
                                    "away_team": (entry.get("awayTeam") or {}).get("name", ""),
                                    "home_score": hs.get("current", hs.get("normaltime")),
                                    "away_score": as_.get("current", as_.get("normaltime")),
                                    "q1_home": hs.get("period1"), "q1_away": as_.get("period1"),
                                    "q2_home": hs.get("period2"), "q2_away": as_.get("period2"),
                                    "q3_home": hs.get("period3"), "q3_away": as_.get("period3"),
                                    "q4_home": hs.get("period4"), "q4_away": as_.get("period4"),
                                    "tournament": (entry.get("tournament") or {}).get("name", ""),
                                })
                            debug_info["h2h_count"] = len(h2h_rows)
                    except Exception as e:
                        debug_info["wait_response_error"] = str(e)[:100]
                else:
                    debug_info["click_error"] = "H2H button not found"
                    
            except Exception as e:
                debug_info["click_error"] = str(e)[:100]
            
            ctx.close()
            browser.close()
    
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg:
            debug_info["error"] = "403"
        elif "404" in error_msg:
            debug_info["error"] = "404"
        else:
            debug_info["error"] = error_msg[:100]
    finally:
        conn.close()
    
    # Save to DB
    if h2h_rows:
        conn = db_mod.get_conn(DB_PATH)
        db_mod.init_db(conn)
        db_mod.save_match_h2h(conn, str(match_id), h2h_rows)
        conn.commit()
        conn.close()
        return True, len(h2h_rows), None
    
    return False, 0, debug_info.get("error", "unknown")


def progress_bar(current, total, width=50):
    """Simple ASCII progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct * 100:.1f}%)"


def main():
    # Kill residual Chrome
    print(f"{BLUE}[1/3] Matando procesos residuales de Chrome...{RESET}")
    subprocess.run(["taskkill", "/IM", "chrome.exe", "/F"],
                   capture_output=True, text=True)
    time.sleep(2)
    
    print(f"\n{BOLD}{'='*70}")
    print(f"  BACKFILL MASIVO DE H2H (Priorizado por Liga)")
    print(f"{'='*70}{RESET}\n")
    
    matches = get_matches_needing_h2h_prioritized()
    total = len(matches)
    
    if total == 0:
        print(f"{GREEN}No hay partidos que necesiten backfill de H2H.{RESET}")
        return
    
    print(f"{BOLD}Partidos a procesar: {total}{RESET}")
    print(f"Espera entre partidos: {WAIT_BASE_SECS}s + jitter 0-{WAIT_JITTER_SECS}s")
    print(f"Modo: {'Visible (primeros 3)' if total > 3 else 'Visible'}\n")
    
    success_count = 0
    fail_count = 0
    total_h2h_rows = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5
    interrupted = False
    
    try:
        for i, m in enumerate(matches):
            match_id = m["match_id"]
            custom_id = m["custom_id"]
            home = m["home_team"][:25]
            away = m["away_team"][:25]
            date = m["date"]
            league = m["league"][:30]
            
            bar = progress_bar(i, total)
            print(f"  {bar}")
            print(f"  [{i+1}/{total}] {BOLD}{match_id}{RESET} | {date} | {league}")
            print(f"           {home} vs {away}")
            
            # First 3 visible
            if i < 3:
                os.environ['SOFASCORE_HEADLESS'] = '0'
            else:
                os.environ['SOFASCORE_HEADLESS'] = '1'
            
            ok, count, error = download_h2h_for_match(match_id, custom_id)
            
            if ok:
                success_count += 1
                total_h2h_rows += count
                consecutive_errors = 0
                print(f"    {GREEN}OK - {count} partidos H2H descargados{RESET}")
            else:
                fail_count += 1
                consecutive_errors += 1
                
                # Color based on error type
                if error == "403":
                    print(f"    {RED}FAIL - HTTP 403 (Forbidden){RESET}")
                elif error == "404":
                    print(f"    {RED}FAIL - HTTP 404 (Not Found){RESET}")
                else:
                    print(f"    {YELLOW}FAIL - {error}{RESET}")
                
                # Pause if too many consecutive errors
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n{RED}{BOLD}[ALERTA] {consecutive_errors} errores consecutivos{RESET}")
                    print(f"{RED}Posible bloqueo de IP. Reinicia tu internet si es necesario.{RESET}")
                    input(f"{YELLOW}Presiona ENTER para continuar o Ctrl+C para detener...{RESET}\n")
                    consecutive_errors = 0
            
            # Wait between matches
            if i < total - 1:
                wait = WAIT_BASE_SECS + random.uniform(0, WAIT_JITTER_SECS)
                wait = int(wait)
                mins, secs = divmod(wait, 60)
                print(f"    Esperando {mins}m {secs}s...\n")
                time.sleep(wait)
    
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\n{YELLOW}[Ctrl+C detectado - Finalizando gracefully...]{RESET}")
    
    print(f"\n{BOLD}{'='*70}")
    print(f"  RESULTADOS")
    print(f"{'='*70}{RESET}\n")
    print(f"  Exitosos: {GREEN}{success_count}/{total}{RESET}")
    print(f"  Fallidos: {RED}{fail_count}/{total}{RESET}")
    print(f"  Total filas H2H: {total_h2h_rows}")
    if interrupted:
        print(f"  {YELLOW}[INTERRUMPIDO]{RESET}")
    
    print(f"\n{BOLD}{'='*70}")
    if interrupted:
        print(f"  BACKFILL INTERRUMPIDO - Los datos ya descargados se guardaron")
    else:
        print(f"  BACKFILL COMPLETADO")
    print(f"{'='*70}{RESET}\n")


if __name__ == "__main__":
    main()
