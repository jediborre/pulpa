# =====================================================================
# CHEQUEO DIRECTO: distribucion de resultados reales de m27_v5_3
# =====================================================================

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) RESULT por signal_type:")
for r in conn.execute("""
    SELECT signal_type, result, COUNT(*) n FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' GROUP BY signal_type, result ORDER BY signal_type"""):
    print(f"   {r['signal_type']:<12} {str(r['result']):<8} {r['n']:>4}")

print()
print("2) RESULT de las BET (con el texto crudo):")
c = Counter()
for r in conn.execute("""
    SELECT result FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'"""):
    c[repr(r["result"])] += 1
print("  ", dict(c))

print()
print("3) Las 53 apuestas BET con su resultado:")
for r in conn.execute("""
    SELECT id, match_id, signal_type, picked_side, confidence, result, created_at
    FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
    ORDER BY created_at"""):
    print(f"   {r['id']:>5} {r['created_at'][:10]} {r['signal_type']:<10} "
          f"{r['picked_side']:<5} {r['confidence']:.4f} -> {r['result']!r}")
conn.close()
