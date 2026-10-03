"""
Ubicación original: scratch/obscura_fetch_date.py
Propósito / Qué hacía:
Scraper independiente usando Obscura en modo serve + Playwright para una fecha completa.
"""

"""
Scraper independiente usando Obscura (modo serve + Playwright).
NO modifica scraper.py - usa obscura.exe directamente.

Uso:
    python obscura_fetch_date.py --date 2026-05-28
    python obscura_fetch_date.py --yesterday --limit 3
"""
import sys
import os
import json
import subprocess
import argparse
import socket
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "match"))

OBSCURA_EXE = ROOT / "tools" / "obscura" / "v0.1.5" / "obscura.exe"
OBSCURA_DIR = ROOT / "tools" / "obscura" / "v0.1.5"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"

def setup_env():
    os.environ["SSL_CERT_FILE"] = str(CERT_PATH)

def wait_for_port(port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def start_obscura(port=9222):
    print("[obscura] Deteniendo instancia previa...")
    subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    
    print(f"[obscura] Iniciando en puerto {port}...")
    proc = subprocess.Popen(
        [str(OBSCURA_EXE), "serve", "--port", str(port), "--stealth"],
        cwd=str(OBSCURA_DIR),
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    if not wait_for_port(port, timeout=15):
        print("[obscura] ERROR: No se pudo iniciar")
        proc.kill()
        return None
    
    print(f"[obscura] Listo en puerto {port}")
    return proc

def stop_obscura(proc):
    if proc:
        print("[obscura] Deteniendo...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[obscura] Detenido")

def fetch_finished_matches_via_playwright(date_str: str, port: int = 9222) -> list[dict]:
    """Usa Playwright + Obscura para obtener partidos terminados."""
    from playwright.sync_api import sync_playwright
    
    print(f"[fetch] Conectando a obscura en puerto {port}...")
    
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}", timeout=10000)
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        page = ctx.new_page()
        
        try:
            print(f"[fetch] Navegando a sofascore.com/basketball/{date_str}...")
            page.goto(f"https://www.sofascore.com/basketball/{date_str}", 
                     timeout=20000, wait_until="domcontentloaded")
            time.sleep(2)
            
            api_url = f"https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/{date_str}"
            print(f"[fetch] Llamando API: {api_url}")
            
            js_code = f"""
            async () => {{
                const r = await fetch('{api_url}', {{
                    headers: {{
                        'Accept': 'application/json',
                        'Referer': 'https://www.sofascore.com/'
                    }}
                }});
                if (!r.ok) return {{ error: r.status }};
                const d = await r.json();
                return {{
                    count: d.events?.length || 0,
                    events: (d.events || []).filter(e => e.status?.type === 'finished').map(e => ({{
                        id: e.id,
                        home: e.homeTeam?.name,
                        away: e.awayTeam?.name,
                        homeScore: e.homeScore?.current,
                        awayScore: e.awayScore?.current,
                        league: e.tournament?.name
                    }}))
                }};
            }}
            """
            
            result = page.evaluate(js_code)
            print(f"[fetch] Resultado: {result}")
            
            if not result or "error" in result:
                print(f"[fetch] Error: {result}")
                return []
            
            matches = []
            for ev in result.get("events", []):
                matches.append({
                    "match_id": str(ev.get("id", "")),
                    "event_date": date_str,
                    "home_team": ev.get("home", ""),
                    "away_team": ev.get("away", ""),
                    "home_score": ev.get("homeScore"),
                    "away_score": ev.get("awayScore"),
                    "league": ev.get("league", ""),
                })
            
            print(f"[fetch] Encontrados {len(matches)} partidos terminados")
            return matches
            
        finally:
            try:
                page.close()
            except:
                pass
            try:
                ctx.close()
            except:
                pass
            try:
                browser.close()
            except:
                pass

def fetch_single_match_via_playwright(match_id: str, port: int = 9222) -> dict | None:
    """Descarga datos completos de un partido usando Playwright + Obscura."""
    from playwright.sync_api import sync_playwright
    
    print(f"[fetch] Descargando partido {match_id}...")
    
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}", timeout=10000)
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        page = ctx.new_page()
        
        try:
            warmup_url = f"https://www.sofascore.com/event/{match_id}"
            page.goto(warmup_url, timeout=15000, wait_until="domcontentloaded")
            time.sleep(2)
            
            event_api = f"https://api.sofascore.com/api/v1/event/{match_id}"
            incidents_api = f"https://api.sofascore.com/api/v1/event/{match_id}/incidents"
            graph_api = f"https://api.sofascore.com/api/v1/event/{match_id}/graph"
            
            js_code = f"""
            async () => {{
                const [event, incidents, graph] = await Promise.all([
                    fetch('{event_api}').then(r => r.ok ? r.json() : null),
                    fetch('{incidents_api}').then(r => r.ok ? r.json() : null),
                    fetch('{graph_api}').then(r => r.ok ? r.json() : null)
                ]);
                return {{ event, incidents, graph }};
            }}
            """
            
            result = page.evaluate(js_code)
            
            if not result or not result.get("event"):
                print(f"[fetch] No se obtuvo event data para {match_id}")
                return None
            
            event_json = result.get("event", {})
            incidents_json = result.get("incidents", {})
            graph_json = result.get("graph", {})
            
            ev = event_json.get("event", event_json)
            home = ev.get("homeTeam", {}).get("name", "Unknown")
            away = ev.get("awayTeam", {}).get("name", "Unknown")
            
            hs = ev.get("homeScore", {})
            as_ = ev.get("awayScore", {})
            home_total = hs.get("current", hs.get("normaltime", 0))
            away_total = as_.get("current", as_.get("normaltime", 0))
            
            quarters = {}
            for i in range(1, 5):
                h = hs.get(f"period{i}")
                a = as_.get(f"period{i}")
                if h is not None and a is not None:
                    quarters[f"Q{i}"] = {"home": h, "away": a}
            
            incidents = incidents_json.get("incidents", []) if incidents_json else []
            graph_points = graph_json.get("graphPoints", []) if graph_json else []
            
            print(f"  [OK] {home} {home_total}-{away_total} {away}")
            print(f"       Quarters: {quarters}")
            print(f"       Incidents: {len(incidents)} | Graph: {len(graph_points)}")
            
            return {
                "match_id": match_id,
                "event": event_json,
                "incidents": incidents_json,
                "graph": graph_json,
                "home_team": home,
                "away_team": away,
                "home_score": home_total,
                "away_score": away_total,
                "quarters": quarters,
            }
            
        except Exception as e:
            print(f"[fetch] Error descargando {match_id}: {e}")
            return None
        finally:
            try:
                page.close()
            except:
                pass
            try:
                ctx.close()
            except:
                pass
            try:
                browser.close()
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description="Descargar partidos usando Obscura + Playwright")
    parser.add_argument("--date", help="Fecha YYYY-MM-DD")
    parser.add_argument("--yesterday", action="store_true", help="Usar ayer")
    parser.add_argument("--limit", type=int, help="Límite de partidos")
    parser.add_argument("--port", type=int, default=9222, help="Puerto de obscura")
    
    args = parser.parse_args()
    
    setup_env()
    
    if args.yesterday:
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    elif args.date:
        date_str = args.date
    else:
        date_str = input("Fecha (YYYY-MM-DD): ").strip()
    
    print(f"\n{'='*60}")
    print(f"Obscura Fetch Date (Playwright) - {date_str}")
    print(f"{'='*60}\n")
    
    proc = start_obscura(args.port)
    if not proc:
        return
    
    try:
        matches = fetch_finished_matches_via_playwright(date_str, args.port)
        
        if not matches:
            print("[done] No hay partidos para descargar")
            return
        
        if args.limit:
            matches = matches[:args.limit]
        
        print(f"\n[info] Descargando {len(matches)} partidos...\n")
        
        success = 0
        failed = 0
        
        for i, match in enumerate(matches, 1):
            print(f"\n[{i}/{len(matches)}] {match['home_team']} vs {match['away_team']}")
            print(f"  Score: {match['home_score']}-{match['away_score']} | League: {match['league']}")
            
            result = fetch_single_match_via_playwright(match["match_id"], args.port)
            
            if result:
                success += 1
            else:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Resumen: OK={success} FAIL={failed} Total={len(matches)}")
        print(f"{'='*60}")
        
    finally:
        stop_obscura(proc)

if __name__ == "__main__":
    main()
