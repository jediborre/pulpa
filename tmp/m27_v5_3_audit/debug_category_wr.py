# =====================================================================
# DEBUG: por que el desglose por categoria salia con WR 100 %
# =====================================================================

import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}

cats = Counter()
for r in conn.execute("""
    SELECT match_id, result FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')"""):
    cat, _ = classify_stage(liga_de.get(str(r["match_id"])) or "")
    cats[(cat, r["result"])] += 1

print("cats:")
for k, v in sorted(cats.items()):
    print("   ", k, v)

por_cat = defaultdict(lambda: [0, 0])
for (cat, res), n in cats.items():
    por_cat[cat][0] += n
    por_cat[cat][1] += 1 if res == "win" else 0

print()
print("por_cat (con el += explicito):")
for cat, (n, w) in sorted(por_cat.items(), key=lambda kv: -kv[1][0]):
    print(f"   {cat:<14} {n:>3} bets | WR {w * 100 / n:>5.1f}%")
conn.close()
