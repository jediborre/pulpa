"""
Ubicación original: temp_scripts/backfill_h2h_and_revaluate.py
Propósito / Qué hacía:
Backfill de registros H2H faltantes y reevaluación de la ganancia de señal en modelos.
"""

"""
Backfill H2H for matches that only have the empty H2H_SUMMARY row.

Uses the correct endpoint: /event/{custom_id}/h2h/events
Called via JavaScript fetch() inside a Chrome session (no proxy).

After backfill, re-evaluates all m27_v3 bets and shows before/after comparison.
"""

import sys
import os
import time
import random
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Disable proxy for this script
os.environ['SOFASCORE_PROXY_URL'] = ''

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match" / "training"))

import match.scraper as scraper
import match.db as db_mod
import infer_match

DB_PATH = str(ROOT / "match" / "matches.db")

WAIT_BASE_SECS = 60
WAIT_JITTER_SECS = 30


def get_all_m27_v3_matches():
    """Get all m27_v3 matches with results."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT DISTINCT
            l.match_id,
            s.home_team,
            s.away_team,
            s.event_date,
            s.league,
            m.custom_id,
            m.home_team_id,
            m.away_team_id,
            l.signal_type,
            l.picked_side,
            l.confidence,
            l.result
        FROM bet_monitor_log_v2 l
        JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
        JOIN matches m ON m.match_id = l.match_id
        WHERE l.model_version = 'm27_v3'
          AND l.result IN ('win', 'hit', 'loss', 'miss')
          AND m.custom_id IS NOT NULL
        ORDER BY s.event_date ASC
    """).fetchall()

    result = [dict(r) for r in rows]
    conn.close()
    return result


def has_complete_h2h(match_id):
    """Check if match already has complete H2H data."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM match_h2h
        WHERE match_id = ? AND q1_home IS NOT NULL AND q1_away IS NOT NULL
    """, (match_id,)).fetchone()
    conn.close()
    return row[0] > 0 if row else False


def download_h2h_for_match(match_id, custom_id, home_team, away_team):
    """Download H2H by clicking the H2H button and intercepting the page's own fetch.
    If H2H already exists in DB, skip download and return cached data."""
    conn = db_mod.get_conn(DB_PATH)
    db_mod.init_db(conn)

    # Check if already has complete H2H
    if has_complete_h2h(match_id):
        conn.close()
        return True, 0, "cached", {}

    h2h_rows: list[dict] = []
    debug_info = {}

    try:
        from playwright.sync_api import sync_playwright
        import time as time_mod

        _headless = os.environ.get("SOFASCORE_HEADLESS", "1").strip() in ("1", "true", "yes")
        _proxy = os.environ.get("SOFASCORE_PROXY_URL", "").strip()

        warmup_url = f"https://www.sofascore.com/event/{match_id}"

        with sync_playwright() as p:
            launch_kwargs = {"channel": "chrome", "headless": _headless}
            if _proxy:
                launch_kwargs["proxy"] = {"server": _proxy}
            browser = p.chromium.launch(**launch_kwargs)
            ctx = browser.new_context(user_agent=scraper.STANDARD_UA)
            page = ctx.new_page()

            # Warmup
            try:
                page.goto(warmup_url, wait_until="domcontentloaded", timeout=10_000)
            except Exception as e:
                debug_info["warmup_error"] = str(e)[:100]
            time_mod.sleep(3)

            debug_info["page_url"] = page.url[:120]
            debug_info["cookie_count"] = len(ctx.cookies())

            # Try to click the H2H button/accordion
            try:
                # Look for H2H section header or button
                h2h_selectors = [
                    'text="Head to Head"',
                    'text="H2H"',
                    '[data-testid="h2h"]',
                    'button:has-text("H2H")',
                    '.h2h-section',
                    '[aria-label*="H2H"]',
                ]
                clicked = False
                for selector in h2h_selectors:
                    try:
                        if page.locator(selector).first.is_visible():
                            page.locator(selector).first.click()
                            clicked = True
                            debug_info["clicked"] = selector
                            break
                    except:
                        continue
                
                if not clicked:
                    # Try scrolling down to find H2H section
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    time_mod.sleep(1)
                    for selector in h2h_selectors:
                        try:
                            if page.locator(selector).first.is_visible():
                                page.locator(selector).first.click()
                                clicked = True
                                debug_info["clicked"] = selector + " (after scroll)"
                                break
                        except:
                            continue
                
                debug_info["h2h_button_found"] = clicked
                
                # Wait for H2H response using expect_response (blocks until response arrives)
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
                    
            except Exception as e:
                debug_info["click_error"] = str(e)[:100]

            ctx.close()
            browser.close()

        complete = sum(
            1 for h in h2h_rows
            if h.get("q1_home") is not None and h.get("q1_away") is not None
        )

        if complete > 0:
            db_mod.save_match_h2h(conn, str(match_id), h2h_rows)
            conn.commit()
            return True, complete, None, debug_info
        else:
            return False, 0, f"sin H2H completo", debug_info

    except Exception as e:
        return False, 0, str(e), debug_info
    finally:
        conn.close()


def re_evaluate_match(match_id):
    """Re-evaluate a match with the H2H fix. Returns dict with old vs new."""
    conn = db_mod.get_conn(DB_PATH)
    db_mod.init_db(conn)

    try:
        match_data = db_mod.get_match(conn, str(match_id))
        if not match_data:
            return None

        result = infer_match.score_m27_v3(match_data, conn, str(match_id))

        p_home = result.get("p_home_win", 0.5)
        confidence = result.get("confidence", 0.0)
        predicted = result.get("predicted_winner", "home")
        h2h_avail = result.get("h2h_available", False)

        # Determine signal
        lean_thr = 0.11
        bet_thr = 0.18
        if confidence >= bet_thr:
            signal = "BET"
        elif confidence >= lean_thr:
            signal = "LEAN"
        else:
            signal = "NO_BET"

        return {
            "p_home": p_home,
            "confidence": confidence,
            "predicted": predicted,
            "h2h_available": h2h_avail,
            "signal": signal,
        }
    except Exception as e:
        return None
    finally:
        conn.close()


def progress_bar(current, total, width=50):
    """Simple ASCII progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct * 100:.1f}%)"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reeval-only", action="store_true",
                        help="Solo re-evaluar sin descargar H2H (para probar cambios de inferencia)")
    args = parser.parse_args()

    # Kill residual Chrome processes (like menu.bat does)
    if not args.reeval_only:
        print("[1/3] Matando procesos residuales de Chrome...")
        subprocess.run(["taskkill", "/IM", "chrome.exe", "/F"],
                       capture_output=True, text=True)
        time.sleep(2)

    print("=" * 70)
    print("  BACKFILL H2H PARA PARTIDOS m27_v3")
    if args.reeval_only:
        print("  [MODO: SOLO RE-EVALUAR]")
    print("=" * 70)
    print()

    matches = get_all_m27_v3_matches()
    total = len(matches)

    if total == 0:
        print("No hay partidos m27_v3 para procesar.")
        return

    print(f"Partidos a procesar: {total}")
    if not args.reeval_only:
        print(f"Espera entre partidos: {WAIT_BASE_SECS}s + jitter 0-{WAIT_JITTER_SECS}s")
    print()

    interrupted = False
    original = {}

    # Phase 1: Download H2H (skip if --reeval-only)
    if args.reeval_only:
        print("  [FASE 1 OMITIDA - Solo re-evaluando con datos existentes]")
        print()
    else:
        print("=" * 70)
        print("  FASE 1: Descargando H2H")
        print("=" * 70)
        print()

        success_count = 0
        fail_count = 0
        total_h2h_rows = 0
        interrupted = False

        # Load original results for inline re-eval
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        original = {}
        rows = conn.execute("""
            SELECT match_id, signal_type, picked_side, confidence, result, inference_minute
            FROM bet_monitor_log_v2
            WHERE model_version = 'm27_v3'
              AND result IN ('win', 'hit', 'loss', 'miss')
        """).fetchall()
        for r in rows:
            original[str(r["match_id"])] = {
                "signal": r["signal_type"] or "",
                "pick": r["picked_side"] or "",
                "confidence": r["confidence"] or 0.0,
                "result": r["result"] or "",
                "minute": r["inference_minute"] or 0,
            }
        conn.close()

        VISIBLE_FIRST = 3

        try:
            for i, m in enumerate(matches):
                match_id = m["match_id"]
                custom_id = m["custom_id"]
                home = m["home_team"][:25]
                away = m["away_team"][:25]
                date = m["event_date"]

                bar = progress_bar(i, total)
                print(f"  {bar}")
                print(f"  [{i+1}/{total}] mid={match_id} cid={custom_id} {date} {home} vs {away}")

                if i < VISIBLE_FIRST:
                    os.environ['SOFASCORE_HEADLESS'] = '0'
                    print(f"    [ventana visible {i+1}/{VISIBLE_FIRST}]")
                elif i == VISIBLE_FIRST:
                    os.environ['SOFASCORE_HEADLESS'] = '1'
                    print("    [cambiando a invisible para el resto]")

                ok, count, err, dbg = download_h2h_for_match(
                    match_id, custom_id, m["home_team"], m["away_team"]
                )

                if ok:
                    success_count += 1
                    total_h2h_rows += count
                    if err == "cached":
                        print(f"    [CACHED] ya tiene H2H - re-evaluando...")
                    else:
                        print(f"    OK - {count} H2H descargados")

                    # Re-evaluar al instante
                    new = re_evaluate_match(match_id)
                    if new:
                        old = original.get(str(match_id))
                        if old:
                            old_sig = old["signal"]
                            old_conf = old["confidence"] if old["confidence"] <= 1.0 else old["confidence"] / 100
                            old_pick = old["pick"] or "?"
                            new_sig = new["signal"]
                            new_conf = new["confidence"]
                            new_pick = new["predicted"].upper()
                            h2h_flag = "H2H" if new.get("h2h_available") else "noH2H"
                            result = old["result"]
                            # Viejo
                            if old_pick == "HOME":
                                old_correct = result in ("win", "hit")
                            else:
                                old_correct = result in ("loss", "miss")
                            old_icon = "W" if old_correct else "L"
                            # Nuevo
                            if new_pick == old_pick:
                                new_correct = old_correct
                            else:
                                new_correct = not old_correct
                            new_icon = "W" if new_correct else "L"
                            print(f"      VIEJO: {old_sig} {old_pick} {old_conf*100:.0f}% {old_icon} | NUEVO: {new_sig} {new_pick} {new_conf*100:.0f}% {new_icon} [{h2h_flag}]")
                else:
                    fail_count += 1
                    print(f"    FAIL - mid={match_id} cid={custom_id}: {err}")
                    if dbg:
                        page_url = dbg.get("page_url", "")
                        cookies = dbg.get("cookie_count", "?")
                        clicked = dbg.get("clicked", "")
                        h2h_count = dbg.get("h2h_count", "?")
                        wait_err = dbg.get("wait_response_error", "")
                        click_err = dbg.get("click_error", "")
                        warmup_err = dbg.get("warmup_error", "")
                        print(f"      url={page_url} cookies={cookies} clicked={clicked} h2h={h2h_count}")
                        if wait_err:
                            print(f"      wait_response: {wait_err}")
                        if click_err:
                            print(f"      click: {click_err}")
                        if warmup_err:
                            print(f"      warmup: {warmup_err}")

                # Wait between matches (except last, and skip if cached)
                if i < total - 1 and err != "cached":
                    wait = WAIT_BASE_SECS + random.uniform(0, WAIT_JITTER_SECS)
                    wait = int(wait)
                    mins, secs = divmod(wait, 60)
                    print(f"    Esperando {mins}m {secs}s...")
                    time.sleep(wait)
                    print()

        except KeyboardInterrupt:
            interrupted = True
            print("\n\n  [Ctrl+C detectado - Finalizando gracefully...]")

        print()
        print(f"  Resultados Fase 1:")
        print(f"    Exitosos: {success_count}/{total}")
        print(f"    Fallidos: {fail_count}/{total}")
        print(f"    Total filas H2H: {total_h2h_rows}")
        if interrupted:
            print(f"    [INTERRUMPIDO por Ctrl+C]")
        print()

    # Phase 2: Re-evaluate
    print("=" * 70)
    print("  FASE 2: Re-evaluando partidos con H2H")
    print("=" * 70)
    print()

    # Load original results if not already loaded (for --reeval-only mode)
    if args.reeval_only:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        original = {}
        rows = conn.execute("""
            SELECT match_id, signal_type, picked_side, confidence, result, inference_minute
            FROM bet_monitor_log_v2
            WHERE model_version = 'm27_v3'
              AND result IN ('win', 'hit', 'loss', 'miss')
        """).fetchall()
        for r in rows:
            original[str(r["match_id"])] = {
                "signal": r["signal_type"] or "",
                "pick": r["picked_side"] or "",
                "confidence": r["confidence"] or 0.0,
                "result": r["result"] or "",
                "minute": r["inference_minute"] or 0,
            }
        conn.close()

    # Re-evaluate all matches that now have H2H
    re_eval = {}
    for i, m in enumerate(matches):
        match_id = str(m["match_id"])
        re_eval[match_id] = re_evaluate_match(match_id)

        bar = progress_bar(i + 1, total)
        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  {bar} - re-evaluados: {i + 1}/{total}")

    print()

    # Phase 3: Comparison
    print("=" * 70)
    print("  FASE 3: Comparacion Before/After (solo MIN=27)")
    print("=" * 70)
    print()

    # Count how many have minute=27
    min27_count = sum(1 for r in original.values() if r["minute"] == 27)
    print(f"  Partidos con inferencia en MIN=27: {min27_count}/{len(original)}")
    print()

    # Original stats (only MIN=27)
    orig_bet = [r for r in original.values() if "BET" in r["signal"] and r["minute"] == 27]
    orig_w = sum(1 for r in orig_bet if r["result"] in ("win", "hit"))
    orig_l = sum(1 for r in orig_bet if r["result"] in ("loss", "miss"))
    orig_total = orig_w + orig_l
    orig_wr = orig_w / orig_total * 100 if orig_total else 0

    print(f"  ORIGINAL (sin H2H completo, MIN=27):")
    print(f"    W={orig_w} L={orig_l} WR={orig_wr:.1f}% (n={orig_total})")
    print()

    # New stats (only MIN=27)
    new_bet_w = 0
    new_bet_l = 0
    new_bet_total = 0
    newly_activated = 0
    newly_activated_w = 0
    newly_activated_l = 0
    changed_pick = 0
    confidence_up = 0
    confidence_down = 0

    for match_id, new in re_eval.items():
        if new is None:
            continue

        old = original.get(match_id)
        if not old:
            continue

        # Only compare MIN=27
        if old["minute"] != 27:
            continue

        old_sig = old["signal"]
        old_pick = old["pick"]
        old_conf = old["confidence"]
        if old_conf <= 1.0:
            old_conf *= 100

        new_sig = new["signal"]
        new_pick = new["predicted"].upper()
        new_conf = new["confidence"] * 100
        result = old["result"]

        # Check if newly activated
        was_no_bet = "BET" not in old_sig
        now_bet = new_sig in ("BET", "LEAN")

        if was_no_bet and now_bet:
            newly_activated += 1
            # Determine if correct
            if new_pick == old_pick:
                correct = result in ("win", "hit")
            else:
                correct = result in ("loss", "miss")
            if correct:
                newly_activated_w += 1
            else:
                newly_activated_l += 1

        # Check if still bet
        if now_bet:
            if new_pick == old_pick:
                correct = result in ("win", "hit")
            else:
                correct = result in ("loss", "miss")
                changed_pick += 1
            if correct:
                new_bet_w += 1
            else:
                new_bet_l += 1
            new_bet_total += 1

        # Confidence change
        if new_conf > old_conf + 1:
            confidence_up += 1
        elif new_conf < old_conf - 1:
            confidence_down += 1

    new_wr = new_bet_w / new_bet_total * 100 if new_bet_total else 0

    print(f"  CON FIX H2H (MIN=27):")
    print(f"    W={new_bet_w} L={new_bet_l} WR={new_wr:.1f}% (n={new_bet_total})")
    print()

    print(f"  CAMBIOS (MIN=27):")
    min27_reeval = sum(1 for mid in re_eval if re_eval[mid] and original.get(mid, {}).get("minute") == 27)
    print(f"    Partidos re-evaluados: {min27_reeval}")
    print(f"    Confianza subio: {confidence_up}")
    print(f"    Confianza bajo: {confidence_down}")
    print(f"    Pick cambio: {changed_pick}")
    print()

    print(f"  NUEVAMENTE ACTIVADOS (NO_BET -> BET/LEAN):")
    na_total = newly_activated_w + newly_activated_l
    na_wr = newly_activated_w / na_total * 100 if na_total else 0
    print(f"    W={newly_activated_w} L={newly_activated_l} WR={na_wr:.1f}% (n={na_total})")
    print()

    # H2H availability (MIN=27 only)
    h2h_yes = sum(1 for mid, r in re_eval.items() if r and r.get("h2h_available") and original.get(mid, {}).get("minute") == 27)
    h2h_no = sum(1 for mid, r in re_eval.items() if r and not r.get("h2h_available") and original.get(mid, {}).get("minute") == 27)
    print(f"  H2H DISPONIBLE (MIN=27):")
    print(f"    Con H2H: {h2h_yes}")
    print(f"    Sin H2H: {h2h_no}")
    print()

    print("=" * 70)
    if interrupted:
        print("  BACKFILL INTERRUMPIDO - Los datos ya descargados se guardaron")
    else:
        print("  BACKFILL COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
