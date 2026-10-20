# =====================================================================
# Lee del summary los numeros de resiliencia / firing policy / calibracion
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
s = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json")
               .read_text(encoding="utf-8"))
print("created_at:", s["created_at"], "| features:", s["features_count"])
print("operating:", json.dumps(s["operating"], ensure_ascii=False))
print()
print("resilience_by_minute:")
for r in s["resilience_by_minute"]:
    print(f"  {r['t']}: filas {r['n_rows']:,} | AUC {r['auc']} | bets {r['bets']:,} | WR {r['wr']} %")
print()
print("firing_policy:", json.dumps(s["firing_policy"], ensure_ascii=False))
print()
print("calibration_matrix:")
for c in s["calibration_matrix"]:
    print(f"  {c['decile']:<12} n={c['n']:>6,} p={c['p_mean']:.4f} y={c['y_mean']:.4f} "
          f"bias={c['bias']:+.4f} {c['diag']}")
print()
print("ablation_by_group:", json.dumps(s["ablation_by_group"], ensure_ascii=False))
print("ablation_base_auc:", s["ablation_base_auc"])
print()
print("walk_forward:")
for k, v in s["walk_forward_cuts"].items():
    print(f"  {k}: AUC {v['auc']} bets {v['bets']:,} WR {v['wr']} % ROI140 {v['roi_140']:+} %")
