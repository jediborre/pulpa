# =====================================================================
# ¿QUÉ MERCADOS IN-PLAY LLEGAN Y EN QUÉ MINUTOS?
# =====================================================================

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) CLAVES PRESENTES EN odds_live_json (por minuto):")
claves_por_min = {}
vacias = Counter()
con_datos = Counter()
for r in conn.execute("SELECT game_minute, odds_live_json FROM live_snapshots_v3 "
                       "WHERE odds_live_json IS NOT NULL AND odds_live_json != ''"):
    mn = r["game_minute"]
    try:
        d = json.loads(r["odds_live_json"])
    except Exception:
        continue
    if not d:
        vacias[mn] += 1
        continue
    con_datos[mn] += 1
    claves_por_min.setdefault(mn, Counter()).update(d.keys())

for mn in sorted(set(list(claves_por_min) + list(vacias))):
    c = claves_por_min.get(mn, Counter())
    print(f"   min {mn:>5.1f} | con datos {con_datos.get(mn, 0):>4} | vacios {vacias.get(mn, 0):>4} "
          f"| claves: {dict(c)}")

print()
print("2) ¿ALGUNA FILA TIENE q4_ml_home?")
n = conn.execute("SELECT COUNT(*) n FROM live_snapshots_v3 "
                 "WHERE odds_live_json LIKE '%q4_ml_home%'").fetchone()["n"]
print(f"   filas con q4_ml_home: {n}")

print()
print("3) ¿SE LLAMA AL ENDPOINT DE CUOTAS IN-PLAY? (logs)")
import subprocess
for pat in ("odds/1/all", "odds_live", "CUOTAS", "cuotas in-play"):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Select-String -Path 'C:\\Users\\App\\Desktop\\pulpa\\logs\\monitor_v3_stdout.log' "
         f"-Pattern '{pat}' -SimpleMatch | Select-Object -Last 3 | ForEach-Object {{ $_.Line }}"],
        capture_output=True, text=True, timeout=120).stdout.strip()
    print(f"   patron '{pat}':")
    print("      " + (out.replace("\n", "\n      ") if out else "(nada en el log)"))

print()
print("4) ESTADO DEL SERVIDOR DE CUOTAS: ¿se capturo alguna vez el mercado Q4?")
print("   (misma consulta sobre match_odds para el historico pre-partido)")
for r in conn.execute("""
    SELECT odds_type, market_period, COUNT(*) n FROM match_odds
    WHERE market_period LIKE '%quarter%' GROUP BY odds_type, market_period"""):
    print("   ", dict(r))
conn.close()
