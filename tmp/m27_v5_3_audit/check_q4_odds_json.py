# =====================================================================
# ¿QUÉ HAY EN odds_live_json? -> ¿cuota del mercado de Q4 por minuto?
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

filas = conn.execute("""
    SELECT id, match_id, game_minute, home_score, away_score, odds_live_json, created_at
    FROM live_snapshots_v3 WHERE odds_live_json IS NOT NULL AND odds_live_json != ''
    ORDER BY id DESC LIMIT 3""").fetchall()
for f in filas:
    print(f"--- id {f['id']} match {f['match_id']} min {f['game_minute']} "
          f"({f['home_score']}-{f['away_score']})")
    try:
        d = json.loads(f["odds_live_json"])
    except Exception as exc:
        print("    no parsea:", exc)
        continue
    print("    tipo:", type(d).__name__)
    if isinstance(d, dict):
        print("    claves:", list(d.keys())[:20])
        # buscar mercados
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"    lista '{k}' con {len(v)} entradas; claves del 1º: {list(v[0].keys())}")
                print("    1º:", json.dumps(v[0], ensure_ascii=False)[:400])
                for m in v:
                    nm = str(m.get("marketName") or m.get("name") or "")
                    if "4th" in nm.lower() or "quarter" in nm.lower():
                        print("    >>> MERCADO DE CUARTO:", json.dumps(m, ensure_ascii=False)[:300])
                break
    elif isinstance(d, list):
        print("    entradas:", len(d))
        print("    1ª:", json.dumps(d[0], ensure_ascii=False)[:400])
    print()

print("=== RESUMEN: ¿hay mercado de Q4 (4th quarter) capturado? ===")
n = 0
con_q4 = 0
ejemplos = []
for r in conn.execute("SELECT id, game_minute, odds_live_json FROM live_snapshots_v3 "
                       "WHERE odds_live_json IS NOT NULL AND odds_live_json != ''"):
    n += 1
    try:
        d = json.loads(r["odds_live_json"])
    except Exception:
        continue
    markets = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "marketName" in v[0]:
                markets = v
                break
    elif isinstance(d, list):
        markets = d
    for m in markets:
        nm = str(m.get("marketName") or "")
        if "4th quarter" in nm.lower() or nm.lower().startswith("4th"):
            con_q4 += 1
            if len(ejemplos) < 3:
                ejemplos.append((r["id"], r["game_minute"], m))
            break
print(f"   snapshots totales: {n} | con mercado de Q4 (4th quarter): {con_q4} "
      f"({con_q4 * 100 / max(n, 1):.1f} %)")
for e in ejemplos:
    print("   ejemplo:", json.dumps(e[2], ensure_ascii=False)[:400])
conn.close()
