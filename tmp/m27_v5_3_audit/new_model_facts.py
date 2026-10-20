# =====================================================================
# CIFRAS EXACTAS DEL MODELO REENTRENADO para el documento de findings
# =====================================================================

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
anchor = np.array([m["anchor"] for m in meta])
lg = np.array([m["lg"] for m in meta])
cats = np.array([classify_stage(x)[0] for x in lg])
n_el = len(elig)
a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
rowA = (mrank >= 0) & (mrank < a_end)
rowB = (mrank >= a_end) & (mrank < b_end)
rowC = mrank >= b_end

print(f"filas totales dataset      : {len(y):,}")
print(f"partidos elegibles         : {n_el:,}")
print(f"A {int(rowA.sum()):,} | B {int(rowB.sum()):,} | C {int(rowC.sum()):,}")
print(f"anclas en C                : {int((rowC & anchor).sum()):,}")
print(f"anclas totales             : {int(anchor.sum()):,}")
print(f"holdout desde              : partido #{elig[int(n_el * 0.82)]}")
print()
print("composicion del train (A) por categoria de fase:")
for cat, n in Counter(cats[rowA].tolist()).most_common():
    print(f"  {cat:<16}{n:>8,}")
print()
print("datos que quedaron fuera del train (excluidos por la regla de fases):")
for cat in ("AMISTOSOS", "SEGUNDA_FASE", "PROMOCION", "PRELIMINAR", "PUESTO"):
    print(f"  {cat:<16}{int((rowA & (cats == cat)).sum()):>8,} filas train")
print(f"  {'TOTAL excluido':<16}{int((rowA & np.isin(cats, ['AMISTOSOS','SEGUNDA_FASE','PROMOCION','PRELIMINAR','PUESTO'])).sum()):>8,} filas")
print()
s = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json")
               .read_text(encoding="utf-8"))
print(f"created_at: {s['created_at']}")
print(f"resilience_by_minute:")
for r in s["resilience_by_minute"]:
    print(f"  {r['t']}: filas {r['n_rows']:,} AUC {r['auc']} bets {r['bets']:,} WR {r['wr']}%")
print(f"firing_policy: {s['firing_policy']}")
print(f"ablation_by_group: {s['ablation_by_group']}")
print(f"ablation_base_auc: {s['ablation_base_auc']}")
print(f"calibration_matrix (decil, n, p_mean, y_mean, bias):")
for c in s["calibration_matrix"]:
    print(f"  {c['decile']:<12} n={c['n']:>6} p={c['p_mean']:.4f} y={c['y_mean']:.4f} bias={c['bias']:+.4f} {c['diag']}")
