# =====================================================================
# Claves disponibles en el meta del dataset
# =====================================================================

import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
print("claves del dataset:", sorted(D.keys()))
print("claves de meta:", sorted(D["meta"][0].keys()))
for m in D["meta"][:3]:
    print("  ", m)
