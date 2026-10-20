# =====================================================================
# LEE Y PRESENTA EL RESULTADO DE LA ABLACION (phase_exclusion_ablation.json)
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / "tmp" / "m27_v5_3_audit" / "phase_exclusion_ablation.json")
                  .read_text(encoding="utf-8"))

base = data[0]
for r in data:
    print("=" * 96)
    print(r["variant"])
    print(f"  filas de entrenamiento: {r['train_rows']:,} (descartadas {r['dropped_train_rows']:,})")
    for label in ("C_completo", "C_filtrado"):
        if label not in r:
            continue
        d = r[label]
        print(f"  {label}: n={d['n']:,} | AUC {d['auc']:.4f} | Brier {d['brier']:.4f} | "
              f"LogLoss {d['logloss']:.4f} | Acc {d['acc'] * 100:.2f}%")
        for t in d["tiers"]:
            print(f"      thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
                  f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi_140']:>+6.2f}% | ROI183 {t['roi_183']:>+6.2f}%")
    if r is not base:
        for label in ("C_completo", "C_filtrado"):
            if label not in r or label not in base:
                continue
            a, b = base[label], r[label]
            print(f"  vs V0 en {label}: dAUC {b['auc'] - a['auc']:+.4f} | "
                  f"dBrier {b['brier'] - a['brier']:+.4f} | dLogLoss {b['logloss'] - a['logloss']:+.4f} | "
                  f"dAcc {(b['acc'] - a['acc']) * 100:+.2f} pp")
            for t0, t1 in zip(a["tiers"], b["tiers"]):
                print(f"      thr {t0['thr']:.2f}: apuestas {t0['bets']:>6,} -> {t1['bets']:>6,} | "
                      f"WR {t0['wr']:>5.2f}% -> {t1['wr']:>5.2f}% | "
                      f"ROI140 {t0['roi_140']:>+6.2f}% -> {t1['roi_140']:>+6.2f}% | "
                      f"ROI183 {t0['roi_183']:>+6.2f}% -> {t1['roi_183']:>+6.2f}%")
    print()
