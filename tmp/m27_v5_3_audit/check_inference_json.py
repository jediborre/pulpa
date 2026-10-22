# =====================================================================
# EL MOTIVO DE DESCARTE ESTA EN inference_json (no en raw_json)
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
    SELECT id, match_id, confidence, signal_type, created_at, inference_json, raw_json
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type='NO_BET'
    ORDER BY id DESC LIMIT 300""").fetchall()
print(f"muestra: {len(filas)} filas NO_BET")

print()
print("1) ESTRUCTURA de inference_json (ejemplo):")
r = filas[0]
inf = json.loads(r["inference_json"]) if r["inference_json"] else {}
print("   claves:", list(inf.keys()))
for k, v in inf.items():
    if isinstance(v, dict):
        print(f"      {k}: (dict) {list(v.keys())[:12]}")
    else:
        print(f"      {k}: {str(v)[:100]}")

print()
print("2) MOTIVOS HALLADOS:")
motivos = Counter()
conf_min = None
for r in filas:
    inf = json.loads(r["inference_json"]) if r["inference_json"] else {}
    q4 = inf.get("q4") or inf
    b = q4.get("block_reason") or inf.get("block_reason")
    if b:
        motivos[str(b)[:90]] += 1
    else:
        motivos["(sin motivo: no cumplio umbral)"] += 1
    if q4.get("min_conf_threshold"):
        conf_min = q4.get("min_conf_threshold")
for m, n in motivos.most_common(20):
    print(f"   {n:>4} | {m}")
print("   umbral en el config:", conf_min)

print()
print("3) ¿QUÉ HAY EN inference_json['q4']? (primer registro con block_reason)")
for r in filas:
    inf = json.loads(r["inference_json"]) if r["inference_json"] else {}
    q4 = inf.get("q4") or {}
    if q4.get("block_reason"):
        print("   ", json.dumps({k: v for k, v in q4.items() if k != "features"},
                               ensure_ascii=False)[:900])
        break

print()
print("4) ¿Y UNO SIN MOTIVO (no cumplio umbral)?")
for r in filas:
    inf = json.loads(r["inference_json"]) if r["inference_json"] else {}
    q4 = inf.get("q4") or {}
    if not q4.get("block_reason"):
        print("   ", json.dumps({k: v for k, v in q4.items() if k != "features"},
                               ensure_ascii=False)[:900])
        break
conn.close()
