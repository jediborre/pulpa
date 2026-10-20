# =====================================================================
# LEE EL RESUMEN DEL MODELO REENTRENADO (m27_v5_3_summary.json)
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "models" / "m27_v5_3" / "model_outputs"

s = json.loads((MD / "m27_v5_3_summary.json").read_text(encoding="utf-8"))
cfg = json.loads((MD / "model_config.json").read_text(encoding="utf-8"))

print("CONFIG desplegada")
print(f"  min_conf={cfg['min_conf']}  min_margin={cfg['min_margin']}  "
      f"snapshot_range={cfg.get('snapshot_range')}")
print(f"  chosen_on={cfg.get('chosen_on')}")
print(f"  features={len(cfg.get('features', []))}")
print()
print("METRICAS")
for k in ("auc_train", "auc_calib", "auc_test", "auc_gap", "brier_train", "brier_test",
          "logloss_test", "acc_test", "auc_anchor_t30", "auc_intra", "mae_q4_margin"):
    print(f"  {k:<18} {s.get(k)}")
print(f"  split: {s.get('split')}")
print()
op = s["operating"]
print(f"OPERATIVO: thr={op['thr']} margen={op['min_margin']} apuestas={op['bets']:,} "
      f"WR {op['wr']}% ROI140 {op['roi_140']:+}% ROI183 {op['roi_183']:+}%")
print(f"  chosen: {op.get('chosen')}")
print(f"  auto-seleccionado: {op.get('auto_selected')}  (suelo WR {op.get('wr_floor')})")
print()
for key in ("tiers_all_rows", "tiers_anchor"):
    print(f"UMBRALES ({key})")
    for t in s[key]:
        print(f"  thr {t['thr']:.2f}: apuestas {t['bets']:>7,} | WR {t['wr']:>5.2f}% "
              f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi_140']:>+6.2f}% | ROI183 {t['roi_183']:>+6.2f}%")
    print()
print("WALK-FORWARD")
for k, v in s["walk_forward_cuts"].items():
    print(f"  {k}: AUC {v['auc']} bets {v['bets']:,} WR {v['wr']}% ROI140 {v['roi_140']:+}%")
print()
cal = json.loads((MD / "league_calibration.json").read_text(encoding="utf-8"))
bloq = [c for c in cal if c.get("block_candidate")]
print(f"LIGAS BLOQUEADAS POR MEDICION (IC95 sup < 71.4%): {len(bloq)}")
for c in bloq:
    print(f"  {c['league']:<40} WR {c['wr']:>5.1f}%  CI {c['wr_ci95']}  bets {c['bets']}")
