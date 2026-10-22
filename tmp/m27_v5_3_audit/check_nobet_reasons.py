# =====================================================================
# ¿QUE MOTIVOS HAY GUARDADOS EN LAS FILAS NO BET?
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

filas = conn.execute("""
    SELECT id, match_id, signal_type, confidence, result, created_at, raw_json, inference_json
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type='NO_BET'
    ORDER BY id DESC LIMIT 400""").fetchall()
print(f"filas NO_BET de m27_v5_3 (muestra): {len(filas)}")

claves = Counter()
for r in filas:
    try:
        raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
    except Exception:
        raw = {}
    claves.update(raw.keys())
print()
print("claves del raw_json:", dict(claves))

print()
print("ejemplo de raw_json (NO_BET):")
for r in filas[:2]:
    try:
        raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
    except Exception:
        raw = {}
    print(f"   id {r['id']} conf {r['confidence']}:")
    for k, v in raw.items():
        if k == "features":
            continue
        print(f"      {k}: {str(v)[:120]}")

print()
print("MOTIVOS (block_reason) en las filas NO BET:")
motivos = Counter()
sin_motivo = 0
for r in filas:
    try:
        raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
    except Exception:
        raw = {}
    b = raw.get("block_reason")
    if b:
        motivos[str(b)[:80]] += 1
    else:
        sin_motivo += 1
for m, n in motivos.most_common(15):
    print(f"   {n:>4} | {m}")
print(f"   {sin_motivo:>4} | (sin block_reason) -> no cumplio el umbral")

print()
print("¿CUANTAS HAY EN TOTAL (todas las fechas) y por dia?")
for r in conn.execute("""
    SELECT substr(created_at,1,10) d, COUNT(*) n,
           SUM(CASE WHEN raw_json LIKE '%block_reason%' THEN 1 ELSE 0 END) con_raw
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type='NO_BET'
    GROUP BY d ORDER BY d"""):
    print(f"   {r['d']}: {r['n']:>4} NO_BET")
conn.close()
