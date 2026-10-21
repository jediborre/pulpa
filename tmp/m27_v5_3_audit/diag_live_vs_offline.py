# =====================================================================
# TEST DECISIVO: para las apuestas VIVAS, ¿que predice el backtest
# sobre ESOS MISMOS partidos al mismo minuto?
#   - Si el offline dice 80 % y el vivo 69 % -> problema del camino vivo
#   - Si el offline tambien dice ~69 %      -> la poblacion viva es mas dura
# =====================================================================

import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import IDX, mirror  # noqa: E402
from models.m27_v5_3.predict import _load_production_models, _ensemble  # noqa: E402

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p_off = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)

# indice por (match, minuto) -> fila del dataset
idx_by = {}
for i, m in enumerate(meta):
    idx_by[(str(m["mid"]), round(float(m["t"]), 1))] = i

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT b.*, m.league FROM bet_monitor_log_v3 b
    LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
    WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
      AND b.result IN ('win','loss') ORDER BY b.id""").fetchall()

print(f"{'fecha':<11}{'min':>5}{'conf viva':>11}{'conf offline':>14}{'lado off':>10}"
      f"{'y':>4}{'result':>8}  partido")
n_enc = 0
w_live = w_off = 0
diff_conf = []
for r in rows:
    mid = str(r["match_id"])
    minute = r["inference_minute"] or 27.0
    best = None
    for cand in (round(float(minute), 1), 30.0, 29.0, 28.0, 27.0, 26.0):
        if (mid, cand) in idx_by:
            best = idx_by[(mid, cand)]
            break
    if best is None:
        print(f"{r['created_at'][:10]:<11}{float(minute):>5.1f}{r['confidence']:>11.4f}"
              f"{'(no esta en el dataset)':>14}")
        continue
    n_enc += 1
    po = float(p_off[best])
    yo = int(y[best])
    conf_o = max(po, 1 - po)
    side_o = "HOME" if po >= 0.5 else "AWAY"
    won_live = r["result"] == "win"
    won_off = (side_o == "HOME" and yo == 1) or (side_o == "AWAY" and yo == 0)
    w_live += won_live
    w_off += won_off
    diff_conf.append(r["confidence"] - conf_o)
    print(f"{r['created_at'][:10]:<11}{float(minute):>5.1f}{r['confidence']:>11.4f}"
          f"{conf_o:>14.4f}{side_o:>10}{yo:>4}{r['result']:>8}  {str(r['league'])[:26]}")

if n_enc:
    print()
    print(f"apuestas con equivalente en el dataset: {n_enc}")
    print(f"   WR vivo        : {w_live * 100 / n_enc:.1f} %")
    print(f"   WR offline     : {w_off * 100 / n_enc:.1f} %")
    print(f"   diferencia de confianza (vivo - offline): media {np.mean(diff_conf):+.4f} "
          f"| max abs {np.max(np.abs(diff_conf)):.4f}")
    igual = sum(1 for d in diff_conf if abs(d) < 0.005)
    print(f"   apuestas con la MISMA confianza (|d|<0.005): {igual}/{n_enc}")
conn.close()
