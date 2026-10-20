# =====================================================================
# COMPARATIVA del modelo recien reentrenado (con el backfill de lineas
# completo) contra el anterior (dataset construido con ~7k lineas).
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import IDX  # noqa: E402

MD = ROOT / "models" / "m27_v5_3" / "model_outputs"
s = json.loads((MD / "m27_v5_3_summary.json").read_text(encoding="utf-8"))

print("MODELO ACTUAL EN DISCO")
print(f"  creado: {s['created_at']}")
print(f"  split: {s['split']}")
for k in ("auc_train", "auc_calib", "auc_test", "auc_gap", "brier_test", "logloss_test",
          "acc_test", "auc_anchor_t30", "auc_intra", "mae_q4_margin"):
    print(f"  {k:<16} {s.get(k)}")
op = s["operating"]
print(f"  operativo: thr {op['thr']} margen {op['min_margin']} | apuestas {op['bets']:,} "
      f"| WR {op['wr']} % | ROI140 {op['roi_140']:+} % | ROI183 {op['roi_183']:+} %")
print(f"  auto-seleccionado: {op.get('auto_selected')}")
print()
print("  UMBRALES (ancla t=30):")
for t in s["tiers_anchor"]:
    n = t["bets"]
    u140 = n * t["roi_140"] / 100
    print(f"    thr {t['thr']:.2f}: apuestas {n:>6,} | WR {t['wr']:>5.2f}% "
          f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi_140']:>+6.2f}% | u@1.40 {u140:>6.0f}")
print()
print(f"  ABLACION por grupo: {json.dumps(s['ablation_by_group'], ensure_ascii=False)}")
print(f"  AUC base: {s['ablation_base_auc']}")
print()
print("  WALK-FORWARD:")
for k, v in s["walk_forward_cuts"].items():
    print(f"    {k}: AUC {v['auc']} bets {v['bets']:,} WR {v['wr']} % ROI140 {v['roi_140']:+} %")
print()
print("  RESILIENCIA POR MINUTO:")
for r in s["resilience_by_minute"]:
    print(f"    {r['t']}: filas {r['n_rows']:,} AUC {r['auc']} bets {r['bets']:,} WR {r['wr']} %")
print()
print("  CALIBRACION (decil | n | p | y | bias):")
for c in s["calibration_matrix"]:
    print(f"    {c['decile']:<12} n={c['n']:>6,} p={c['p_mean']:.4f} y={c['y_mean']:.4f} "
          f"bias={c['bias']:+.4f} {c['diag']}")
print()

# Cobertura de las features de linea en el dataset recien construido
D = None
try:
    import pickle
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X = D["X"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in D["meta"]])
    rowA = (mrank >= 0) & (mrank < int(len(elig) * 0.70))
    print("  COBERTURA DE LAS FEATURES DE LINEA (bloque de train):")
    for feat in ("mkt_margin_spread", "total_line", "mkt_p_home", "has_ml"):
        col = X[rowA, IDX[feat]]
        nan = int(np.isnan(col).sum())
        n = len(col)
        print(f"    {feat:<20} con dato {100 - nan * 100 / n:>5.1f} % "
              f"({n - nan:,} de {n:,})")
except Exception as exc:
    print("  (no se pudo leer el dataset:", exc, ")")

# Ligas bloqueadas por medicion
cal = json.loads((MD / "league_calibration.json").read_text(encoding="utf-8"))
bloq = [c for c in cal if c.get("block_candidate")]
print()
print(f"  LIGAS BLOQUEADAS POR MEDICION: {len(bloq)}")
for c in bloq:
    print(f"    {c['league']:<40} WR {c['wr']:>5.1f}%  CI {c['wr_ci95']}")
