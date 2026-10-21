# =====================================================================
# LA DIFERENCIA DE MARCADOR EN VIVO vs BD: direccion y efecto
#   1) signed diff del marcador (vivo - offline) al mismo minuto
#   2) ¿la API va por detras de los eventos o la BD esta incompleta?
#   3) ¿cambia la prediccion? (p_cal vivo vs p_cal recalculado)
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
from models.m27_v5_3.predict import _load_production_models, _ensemble  # noqa: E402

conn = get_db_connection()
conn.row_factory = sqlite3.Row
state = F.get_state()
models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]

filas = conn.execute("""
    SELECT i.match_id, i.game_minute, i.snapshot_minute, i.features_json, i.p_cal, i.pick,
           i.confidence
    FROM model_inference_v3 i
    WHERE i.model_version='m27_v5_3' AND i.features_json IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()

print(f"{'match':<11}{'min':>6}{'cur vivo':>10}{'cur bd':>8}{'dif':>6}"
      f"{'p vivo':>8}{'p bd':>8}{'dif':>8}{'eventos BD':>12}")
difs_score = []
difs_p = []
n = 0
for r in filas:
    mid = str(r["match_id"])
    t = float(r["snapshot_minute"] or r["game_minute"])
    feats = json.loads(r["features_json"])
    md = load_match(conn, mid)
    if not md:
        continue
    ctx = F.ctx_from_payload(md, t, state)
    if ctx is None:
        continue
    v = F.snapshot_features(ctx, state)
    d_score = float(feats.get("cur_score_diff", 0)) - float(v[F.FEATURE_NAMES.index("cur_score_diff")])

    # prediccion offline al mismo minuto
    X2 = np.vstack([v, F.mirror(v)])
    pe = _ensemble(models, X2)
    p_raw = 0.5 * (pe[0] + 1.0 - pe[1])
    p_off = float(np.clip(models["calibrator"].predict(np.array([p_raw]))[0], lo, hi))
    d_p = float(r["p_cal"]) - p_off

    n_ev = conn.execute("SELECT COUNT(*) c FROM play_by_play WHERE match_id=?",
                        (mid,)).fetchone()["c"]
    difs_score.append(d_score)
    difs_p.append(d_p)
    n += 1
    if n <= 18 or abs(d_score) >= 5:
        print(f"{mid:<11}{t:>6.2f}{float(feats.get('cur_score_diff', 0)):>10.1f}"
              f"{float(v[F.FEATURE_NAMES.index('cur_score_diff')]):>8.1f}{d_score:>6.1f}"
              f"{float(r['p_cal']):>8.4f}{p_off:>8.4f}{d_p:>8.4f}{n_ev:>12,}")

ds = np.array(difs_score)
dp = np.array(difs_p)
print()
print(f"n comparados: {n}")
print(f"diferencia de marcador (vivo - BD): media {ds.mean():+.2f} | mediana {np.median(ds):+.1f} "
      f"| min {ds.min():+.1f} max {ds.max():+.1f} | % con |dif|>=3: {(np.abs(ds) >= 3).mean() * 100:.0f} %")
print(f"diferencia de p_cal  (vivo - BD): media {dp.mean():+.4f} | mediana {np.median(dp):+.4f} "
      f"| min {dp.min():+.4f} max {dp.max():+.4f} | % con |dif|>=0.05: {(np.abs(dp) >= 0.05).mean() * 100:.0f} %")
print()
print("CORRELACION entre la diferencia de marcador y la de p_cal: "
      f"{np.corrcoef(ds, dp)[0, 1]:.3f}")
conn.close()
