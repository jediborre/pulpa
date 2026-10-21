# =====================================================================
# ¿EL PATH VIVO TIENE LAS FEATURES DE MERCADO QUE EL MODELO USABA?
#   Compara, para los mismos partidos y minutos, las features de mercado
#   (has_ml, mkt_p_home, odds_log_ratio, expected_ft_margin...) entre:
#     - lo que el monitor vio EN VIVO (features_json guardado)
#     - lo que se recalcula OFFLINE desde la BD (con las cuotas pre-partido)
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.m27_v5_3.features as F  # noqa: E402
from models.common.data_loader import get_db_connection, load_match  # noqa: E402

MKT = ["has_ml", "mkt_p_home", "odds_log_ratio", "expected_ft_margin",
       "mkt_margin_spread", "total_line", "cur_vs_expected"]

conn = get_db_connection()
conn.row_factory = sqlite3.Row
state = F.get_state()

filas = conn.execute("""
    SELECT i.match_id, i.game_minute, i.snapshot_minute, i.features_json
    FROM model_inference_v3 i
    WHERE i.model_version='m27_v5_3' AND i.features_json IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()

print("1) FEATURES DE MERCADO: VIVO vs OFFLINE (mismo partido y minuto)")
print(f"   {'feature':<22}{'vivo con dato':>14}{'offline con dato':>19}{'max|diff|':>12}")
for fname in MKT:
    if fname not in F.FEATURE_NAMES:
        continue
    j = F.FEATURE_NAMES.index(fname)
    n_vivo = n_off = n_cmp = 0
    maxd = 0.0
    for r in filas:
        feats = json.loads(r["features_json"])
        if fname not in feats or feats[fname] is None:
            continue
        a = feats[fname]
        md = load_match(conn, str(r["match_id"]))
        if not md:
            continue
        ctx = F.ctx_from_payload(md, float(r["snapshot_minute"] or r["game_minute"]), state)
        if ctx is None:
            continue
        v = F.snapshot_features(ctx, state)
        b = v[j]
        if a is not None and not (isinstance(a, float) and np.isnan(a)):
            n_vivo += 1
        if not np.isnan(b):
            n_off += 1
        if not (isinstance(a, float) and np.isnan(a)) and not np.isnan(b):
            n_cmp += 1
            maxd = max(maxd, abs(float(a) - float(b)))
    print(f"   {fname:<22}{n_vivo:>14,}{n_off:>19,}{maxd:>12.4f}")

print()
print("2) ¿QUE VE EL MODELO EN VIVO? ejemplo de una fila viva")
r = filas[len(filas) // 2]
feats = json.loads(r["features_json"])
md = load_match(conn, str(r["match_id"]))
print(f"   partido {r['match_id']} min {r['snapshot_minute']}")
for fname in MKT:
    if fname in F.FEATURE_NAMES:
        print(f"      {fname:<22} vivo = {feats.get(fname)}")
print("   --- offline ---")
ctx = F.ctx_from_payload(md, float(r["snapshot_minute"]), state)
v = F.snapshot_features(ctx, state)
for fname in MKT:
    if fname in F.FEATURE_NAMES:
        print(f"      {fname:<22} off  = {v[F.FEATURE_NAMES.index(fname)]}")
print()
print("   odds del payload (load_match):", type(md.get("odds")).__name__,
      len(md.get("odds") or []) if md.get("odds") else 0)
conn.close()
