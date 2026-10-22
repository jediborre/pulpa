# =====================================================================
# DESGLOSE DE DESCARTES: por motivo, con el bucket de cercania al umbral
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
    SELECT id, match_id, signal_type, confidence, inference_json, created_at
    FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' ORDER BY id""").fetchall()

motivos = Counter()
cercania = Counter()
sin_motivo_ejemplos = []
for r in filas:
    sig = r["signal_type"] or ""
    if "BET" in sig and "NO_BET" not in sig:
        continue
    try:
        inf = json.loads(r["inference_json"]) if r["inference_json"] else {}
    except Exception:
        inf = {}
    raw = inf.get("raw_json") or {}
    bloqueado = raw.get("block_reason")
    razon = inf.get("reason")
    conf = r["confidence"] or 0.0
    if bloqueado:
        motivos["LIGA/FORMATO: " + str(bloqueado)[:70]] += 1
    elif razon:
        motivos["NO DISPONIBLE: " + str(razon)[:60]] += 1
    else:
        motivos["no cumplio el umbral"] += 1
        if conf >= 0.70:
            cercania["0.70-0.75 (casi)"] += 1
        elif conf >= 0.60:
            cercania["0.60-0.70"] += 1
        else:
            cercania["<0.60"] += 1
        if len(sin_motivo_ejemplos) < 3:
            sin_motivo_ejemplos.append((r["id"], conf, raw.get("min_conf_threshold"),
                                         raw.get("is_green"), raw.get("toxic_league_blocked")))

print("DESCARTES POR MOTIVO (todas las fechas):")
for m, n in motivos.most_common(25):
    print(f"   {n:>4} | {m}")
print()
print("los que NO cumplieron el umbral, por cercania:")
for m, n in sorted(cercania.items()):
    print(f"   {n:>4} | {m}")
print()
print("ejemplos de NO BET sin motivo:", sin_motivo_ejemplos)

print()
print("TOTALES POR TIPO DE FILA:")
c = Counter()
for r in filas:
    sig = r["signal_type"] or ""
    if "BET" in sig and "NO_BET" not in sig:
        c["apuesta real"] += 1
    else:
        c["descartado (NO BET)"] += 1
for k, n in c.items():
    print(f"   {k:<22} {n:>4}")

print()
print("¿CUANTOS PARTIDOS DISTINTOS VIO EL MODELO?")
n_m = conn.execute("SELECT COUNT(DISTINCT match_id) n FROM bet_monitor_log_v3 "
                    "WHERE model_version='m27_v5_3'").fetchone()["n"]
print(f"   {n_m}")
conn.close()
