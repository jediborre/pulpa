# =====================================================================
# NOMBRES REALES DE market_period (SofaScore) vistos en match_odds
#   Sirve para mapear el mercado del ganador de Q4 a la clave interna.
# =====================================================================

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("market_period distintos en match_odds:")
for r in conn.execute(
    "SELECT market_period, market_name, COUNT(*) n FROM match_odds "
    "GROUP BY market_period, market_name ORDER BY n DESC LIMIT 40"
):
    print(f"   {str(r['market_period']):<28} {str(r['market_name']):<22} {r['n']:>8,}")

print()
print("market_name distintos:")
for r in conn.execute(
    "SELECT market_name, COUNT(*) n FROM match_odds GROUP BY market_name ORDER BY n DESC LIMIT 20"
):
    print(f"   {str(r['market_name']):<40} {r['n']:>8,}")

print()
print("periodos que mencionan cuarto/half (para el mercado del ganador de Q4):")
for r in conn.execute(
    "SELECT market_period, COUNT(*) n FROM match_odds "
    "WHERE lower(COALESCE(market_period,'')) LIKE '%quarter%' "
    "   OR lower(COALESCE(market_period,'')) LIKE '%half%' "
    "   OR lower(COALESCE(market_period,'')) LIKE '%q%' "
    "GROUP BY market_period ORDER BY n DESC LIMIT 30"
):
    print(f"   {str(r['market_period']):<40} {r['n']:>8,}")
conn.close()
