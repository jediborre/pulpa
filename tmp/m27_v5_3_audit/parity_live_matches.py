# =====================================================================
# PARIDAD ESTRICTA EN PARTIDOS VIVOS REALES
#   Para cada fila de model_inference_v3 de hoy: se recalculan los
#   features por el camino OFFLINE (eventos de la BD) al MISMO minuto y
#   se comparan con los features que el monitor vio EN VIVO.
#   Si difieren -> el modelo en vivo opera con un estado distinto al
#   que se entreno.
# =====================================================================

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.m27_v5_3.features as F  # noqa: E402
from models.common.data_loader import get_db_connection, load_match  # noqa: E402

conn = get_db_connection()
conn.row_factory = sqlite3.Row

filas = conn.execute("""
    SELECT i.match_id, i.game_minute, i.snapshot_minute, i.features_json, i.p_cal, i.pick
    FROM model_inference_v3 i
    WHERE i.model_version='m27_v5_3' AND i.features_json IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()
print(f"filas vivas a comparar: {len(filas)}")

peores = {}
n_cmp = 0
n_ok = 0
detalle = []
for r in filas:
    mid = str(r["match_id"])
    try:
        feats = json.loads(r["features_json"])
    except Exception:
        continue
    if not isinstance(feats, dict):
        continue
    md = load_match(conn, mid)
    if not md:
        continue
    t = float(r["snapshot_minute"] or r["game_minute"])
    ctx = F.ctx_from_payload(md, t, F.get_state())
    if ctx is None:
        continue
    try:
        v = F.snapshot_features(ctx, F.get_state())
    except Exception as exc:
        print(f"   {mid} min {t}: error al calcular features offline: {exc}")
        continue
    diffs = {}
    for j, fname in enumerate(F.FEATURE_NAMES):
        if fname not in feats or feats[fname] is None:
            continue
        a = float(feats[fname])
        b = float(v[j])
        if np.isnan(a) and np.isnan(b):
            d = 0.0
        elif np.isnan(a) != np.isnan(b):
            d = float("inf")
        else:
            d = abs(a - b)
        diffs[fname] = d
    n_cmp += 1
    fin = {k: d for k, d in diffs.items() if np.isfinite(d)}
    max_d = max(fin.values()) if fin else 0.0
    if max_d < 1e-6:
        n_ok += 1
    else:
        worse = sorted(fin.items(), key=lambda kv: -kv[1])[:4]
        detalle.append((mid, t, max_d, worse))
    for k, d in fin.items():
        peores[k] = max(peores.get(k, 0.0), d)

print(f"comparados: {n_cmp} | IDENTICOS (max|diff| < 1e-6): {n_ok} "
      f"({n_ok * 100 / max(n_cmp, 1):.1f} %)")
print()
if peores:
    print("diferencias maximas por feature (solo los que difieren):")
    for k, v in sorted(peores.items(), key=lambda kv: -kv[1])[:15]:
        if v > 1e-9:
            print(f"   {k:<24} {v:.6f}")
else:
    print("NINGUNA diferencia: vivo == offline al mismo minuto")
print()
if detalle:
    print("PEORES CASOS:")
    for mid, t, mx, worse in detalle[:10]:
        print(f"   {mid} min {t:.2f} max|diff| {mx:.4f} -> " +
              ", ".join(f"{k}={d:.3f}" for k, d in worse))
conn.close()
