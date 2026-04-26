import sqlite3
import urllib.request
import urllib.error
import json
import time
import sys
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOTDIR / "match" / "matches.db"
BASE_URL = "http://127.0.0.1:8000"
VERSIONS = ['v2', 'v4', 'v6', 'v9', 'v12', 'v13', 'v15', 'v16', 'v17']

def get_dates(limit=100):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT DISTINCT substr(date, 1, 10) as d FROM matches WHERE d IS NOT NULL ORDER BY d DESC LIMIT ?", (limit,))
        dates = [row[0] for row in c.fetchall() if row[0]]
        conn.close()
        return dates
    except Exception as e:
        print(f"Error leyendo DB: {e}")
        return []

def precompute(limit):
    dates = get_dates(limit)
    if not dates:
        print("No se encontraron fechas.")
        return
        
    print(f"==================================================")
    print(f"[*] INICIANDO PRE-CALCULO MASIVO DE CACHE")
    print(f"Fechas a procesar: {len(dates)} (desde {dates[-1]} hasta {dates[0]})")
    print(f"Modelos: {len(VERSIONS)} {VERSIONS}")
    print(f"Total de iteraciones: {len(dates) * len(VERSIONS)}")
    print(f"Asegurate de que FastAPI ('python api.py') este corriendo en {BASE_URL}")
    print(f"==================================================\n")
    
    total_calls = len(dates) * len(VERSIONS)
    done_calls = 0
    
    for date in dates:
        print(f"\n--- Procesando fecha: {date} ---")
        for v in VERSIONS:
            done_calls += 1
            print(f"[{done_calls}/{total_calls}] Computando {v}... ", end="", flush=True)
            url = f"{BASE_URL}/api/compute/{v}/{date}?force=true"
            t0 = time.time()
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=300) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode('utf-8'))
                        print(f"OK ({len(data)} bets) en {time.time()-t0:.1f}s")
                    else:
                        print(f"ERROR HTTP {response.status}")
            except urllib.error.URLError as e:
                print(f"ERROR RED: {e.reason} (esta encendido api.py?)")
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == '__main__':
    limit = 60 # Por defecto procesamos los ultimos 60 dias (2 meses) para no hacerlo infinito
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    precompute(limit)
