# =====================================================================
# RESUMEN EJECUTIVO DE LA LISTA NEGRA -> LIGAS CONCRETAS (top por volumen)
# ---------------------------------------------------------------------
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/blacklist_summary.py
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "tmp" / "m27_v5_3_audit" / "blacklist_resolution.json")
                  .read_text(encoding="utf-8"))

for nombre, d in DATA.items():
    items = [x for x in d["detalle"] if x["partidos_reales"] > 0]
    items.sort(key=lambda x: -x["partidos_reales"])
    print()
    print("=" * 100)
    print(f"{nombre}  [{d['nivel']}]   {d['ligas']:,} liga(s) | {d['partidos']:,} partidos")
    print(f"  regla: {d['regla']}")
    print("=" * 100)
    for x in items[:14]:
        print(f"  {x['league'][:58]:<60}{x['partidos_reales']:>8,} part.  "
              f"m_cnt={x['match_count'] or 0:>5}  pts={(x['avg_total_points'] or 0):>6.1f}  "
              f"Q4={(x['avg_q4_total_points'] or 0):>5.1f}")
    if len(items) > 14:
        print(f"  ... y {len(items) - 14} liga(s) mas")
