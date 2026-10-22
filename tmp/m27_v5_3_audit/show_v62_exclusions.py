# =====================================================================
# CATEGORIAS DE EXCLUSION DEL MODELO v6_2 (precedente) y a que fase
# pertenece "championship round" / "promotion round".
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]

p = ROOT / "match" / "training" / "v6_2_league_name_exclusions.json"
if not p.exists():
    print("no existe:", p)
    for cand in ROOT.rglob("v6_2_league_name_exclusions.json"):
        print("  encontrado:", cand)
    raise SystemExit(0)

d = json.loads(p.read_text(encoding="utf-8"))
for c in d.get("categories", []):
    pats = [str(x) for x in c.get("patterns", [])]
    print(f"\n== {c.get('name')} ({len(pats)} patrones) ==")
    for x in pats:
        print("   ", x)
