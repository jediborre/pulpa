# =====================================================================
# ESTADO VIVO AHORA: bets recientes y efecto del arreglo de fases
# =====================================================================

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import BLOCKED_STAGE_CATEGORIES  # noqa: E402

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}

print("1) ULTIMAS 20 APUESTAS DE m27_v5_3 (con su categoria y si ahora se bloquea):")
for r in conn.execute("""
    SELECT b.id, b.match_id, b.signal_type, b.picked_side, b.confidence, b.result,
           b.created_at, m.league
    FROM bet_monitor_log_v3 b LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
    WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
    ORDER BY b.created_at DESC LIMIT 20"""):
    cat, _ = classify_stage(r["league"] or "")
    bloq = "BLOQUEADA AHORA" if cat in BLOCKED_STAGE_CATEGORIES else ""
    print(f"   {r['created_at'][:16]} | {str(r['league'])[:34]:<34} | {r['picked_side']:<5} "
          f"conf {r['confidence']:.4f} | {str(r['result']):<5} | {cat:<14} {bloq}")

print()
print("2) REGISTRO COMPLETO vs REGISTRO CON LAS REGLAS ACTUALES:")
tot = [0, 0]
act = [0, 0]
por_cat = Counter()
for r in conn.execute("""
    SELECT match_id, result FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')"""):
    cat, _ = classify_stage(liga_de.get(str(r["match_id"])) or "")
    por_cat[(cat, r["result"])] += 1
    tot[0] += 1
    tot[1] += r["result"] == "win"
    if cat not in BLOCKED_STAGE_CATEGORIES:
        act[0] += 1
        act[1] += r["result"] == "win"
print(f"   todo el registro        : {tot[1]}/{tot[0]} = {tot[1] * 100 / max(tot[0], 1):.1f} %")
print(f"   con las reglas actuales : {act[1]}/{act[0]} = {act[1] * 100 / max(act[0], 1):.1f} %")
print("   por categoria:")
agg = {}
for (cat, res), n in por_cat.items():
    a = agg.setdefault(cat, [0, 0])
    a[0] += n
    a[1] += res == "win"
for cat, (n, w) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    marca = "  <-- bloqueada ahora" if cat in BLOCKED_STAGE_CATEGORIES else ""
    print(f"      {cat:<14} {n:>3} bets | {w:>3} ganadas | WR {w * 100 / max(n, 1):>5.1f} %{marca}")

print()
print("3) HOY (2026-10-21) Y AYER:")
for r in conn.execute("""
    SELECT substr(created_at,1,10) d, COUNT(*) n,
           SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss') GROUP BY d ORDER BY d"""):
    print(f"   {r['d']}: {r['n']} bets | {r['w']} ganadas | WR {r['w'] * 100 / r['n']:.1f} %")
conn.close()
