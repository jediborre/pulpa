# =====================================================================
# RESUMEN DEL REENTRENAMIENTO CON EL CLASIFICADOR ARREGLADO
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "models" / "m27_v5_3" / "model_outputs"

s = json.loads((MD / "m27_v5_3_summary.json").read_text(encoding="utf-8"))
cfg = json.loads((MD / "model_config.json").read_text(encoding="utf-8"))

print("1) METRICAS DEL MODELO REENTRENADO (holdout C):")
for k in ("created_at", "auc_train", "auc_calib", "auc_test", "auc_gap", "brier_test",
          "logloss_test", "acc_test", "auc_anchor_t30", "auc_intra", "mae_q4_margin"):
    print(f"   {k:<16} {s.get(k)}")
print(f"   split          {s.get('split')}")
op = s.get("operating") or {}
print(f"   operativo      thr {op.get('thr')} margen {op.get('min_margin')} | "
      f"apuestas {op.get('bets'):,} | WR {op.get('wr')} % | ROI140 {op.get('roi_140')} % "
      f"| ROI183 {op.get('roi_183')} %")
print(f"   config         min_conf {cfg.get('min_conf')} min_margin {cfg.get('min_margin')}")

print()
print("2) UMBRALES (ancla t=30) — comparacion con el modelo anterior:")
print(f"   {'thr':>6}{'apuestas':>10}{'WR':>8}{'IC95':>18}{'u@1.40':>10}{'u@1.83':>10}")
for t in s.get("tiers_anchor", []):
    n = t["bets"]
    w = n * t["wr"] / 100
    print(f"   {t['thr']:>6.2f}{n:>10,}{t['wr']:>7.1f}%"
          f"   [{t['wr_ci95'][0]:>5.1f}, {t['wr_ci95'][1]:>5.1f}]"
          f"{w * 1.40 - n:>10.0f}{w * 1.83 - n:>10.0f}")

print()
print("3) ABLACION POR GRUPO:")
print(f"   base AUC {s.get('ablation_base_auc')}")
for g, v in (s.get("ablation_by_group") or {}).items():
    print(f"   {g:<14} {v:+.4f}")

print()
print("4) WALK-FORWARD:")
for k, v in (s.get("walk_forward_cuts") or {}).items():
    print(f"   {k}: AUC {v.get('auc')} | bets {v.get('bets'):,} | WR {v.get('wr')} % | "
          f"ROI140 {v.get('roi_140')} %")

print()
print("5) CALIBRACION POR DECIL:")
for c in s.get("calibration_matrix", []):
    print(f"   {str(c['decile']):<12} n={c['n']:>6,} p={c['p_mean']:.4f} "
          f"y={c['y_mean']:.4f} bias={c['bias']:+.4f} {c.get('diag', '')}")

print()
print("6) LIGAS BLOQUEADAS POR MEDICION (WR < equilibrio con n minimo):")
cal = json.loads((MD / "league_calibration.json").read_text(encoding="utf-8"))
bloq = [c for c in cal if c.get("block_candidate")]
print(f"   {len(bloq)} de {len(cal)} ligas en la calibracion")
for c in bloq[:10]:
    print(f"      {c['league'][:50]:<52} n={c.get('bets')} WR {c.get('wr')}")

print()
print("7) ROBUSTEZ POR MINUTO:")
for r in s.get("resilience_by_minute", []):
    print(f"   {r['t']}: filas {r['n_rows']:,} | AUC {r['auc']} | bets {r['bets']:,} | "
          f"WR {r['wr']} %")
