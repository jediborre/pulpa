# =====================================================================
# DIFF REAL ENTRE m27_v5_2 y m27_v5_3
#   - especificacion de features
#   - datos de entrenamiento (dataset)
#   - calibracion y umbral
# =====================================================================

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]


def spec_de_features(path: Path) -> list[tuple[str, str]]:
    """Extrae (nombre, regla de espejo) de la FEATURE_SPEC del modulo."""
    txt = path.read_text(encoding="utf-8")
    m = re.search(r"FEATURE_SPEC[^=]*=\s*\[(.*?)\n\]", txt, re.S)
    if not m:
        return []
    out = []
    for nombre, regla in re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', m.group(1)):
        out.append((nombre, regla))
    return out


f2 = spec_de_features(ROOT / "models" / "m27_v5_2" / "features.py")
f3 = spec_de_features(ROOT / "models" / "m27_v5_3" / "features.py")

n2 = [a for a, _ in f2]
n3 = [a for a, _ in f3]
print(f"1) FEATURES: v5_2 = {len(n2)} | v5_3 = {len(n3)}")
solo2 = [x for x in n2 if x not in n3]
solo3 = [x for x in n3 if x not in n2]
print(f"   solo en v5_2 ({len(solo2)}): {solo2}")
print(f"   solo en v5_3 ({len(solo3)}): {solo3}")
comunes = [x for x in n2 if x in n3]
print(f"   comunes: {len(comunes)}")
cambios_regla = [(a, dict(f2)[a], dict(f3)[a]) for a in comunes if dict(f2)[a] != dict(f3)[a]]
print(f"   comunes con regla de espejo distinta: {cambios_regla}")

print()
print("2) ¿QUE SIGNIFICA CADA FEATURE? (v5_3)")
for nombre, regla in f3:
    print(f"   {nombre:<26} {regla}")

print()
print("3) CALIBRACION Y UMBRAL (model_config.json):")
for modelo in ("m27_v5_2", "m27_v5_3"):
    p = ROOT / "models" / modelo / "model_outputs" / "model_config.json"
    if not p.exists():
        p = ROOT / "models" / modelo / "model_config.json"
    if not p.exists():
        print(f"   {modelo}: sin model_config.json")
        continue
    cfg = json.loads(p.read_text(encoding="utf-8"))
    print(f"   {modelo}:")
    for k in ("min_conf", "min_margin", "p_clip", "block_toxic_leagues", "calibrator",
              "calibration", "snapshot_minute", "weights"):
        if k in cfg:
            print(f"      {k}: {cfg[k]}")

print()
print("4) RESUMEN DEL ENTRENAMIENTO (summary json):")
for modelo in ("m27_v5_2", "m27_v5_3"):
    base = ROOT / "models" / modelo / "model_outputs"
    cands = list(base.glob("*summary*.json")) if base.exists() else []
    if not cands:
        print(f"   {modelo}: sin summary")
        continue
    s = json.loads(cands[0].read_text(encoding="utf-8"))
    print(f"   {modelo} ({cands[0].name}):")
    for k in ("created_at", "auc_test", "auc_train", "brier_test", "logloss_test",
              "auc_anchor_t30", "n_train", "n_test", "split", "operating"):
        if k in s:
            v = s[k]
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in list(v.items())[:6]}
            print(f"      {k}: {v}")
    if "tiers_anchor" in s:
        for t in s["tiers_anchor"]:
            if abs(t["thr"] - 0.75) < 0.001:
                print(f"      thr 0.75 -> n={t['bets']:,} WR {t['wr']}%")
