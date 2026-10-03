"""
Debug del minuto inferido en HT para AD Brusque vs BT/Tatuí.
Descarga datos completos y analiza qué minuto detecta.
"""
import asyncio
import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from match.scraper import fetch_match_by_id
from match.training.infer_match import _infer_minute_from_pbp

MATCH_ID = "16220335"  # AD Brusque vs BT/Tatuí

def main():
    print(f"Fetch completo + minuto inferido: AD Brusque vs BT/Tatuí ({MATCH_ID})\n")

    data = fetch_match_by_id(MATCH_ID, fetch_h2h=False, fetch_statistics=False,
                              fetch_lineups=False, fetch_team_data=False, backend="obscura")

    # Status
    event = data.get("event", {}) or {}
    status = event.get("status", {})
    print(f"status_type: {status.get('type')}")
    print(f"status_description: {status.get('description')}")

    # Score / quarters
    score = data.get("score", {}) or {}
    quarters = score.get("quarters", {}) or {}
    print(f"\nScore global: {score.get('home')} - {score.get('away')}")
    print(f"Quarters: {json.dumps(quarters, indent=2)}")

    # Minuto inferido
    minute = _infer_minute_from_pbp(data)
    print(f"\nMinuto inferido (_infer_minute_from_pbp): {minute}")

    # Graph points tail
    gp = data.get("graph_points") or []
    print(f"Graph points: {len(gp)}")
    if gp:
        last3 = gp[-3:]
        print(f"Últimos 3 graph points: {json.dumps(last3, indent=2)}")

    # Last few PBP incidents
    incidents = data.get("incidents", {}).get("incidents", []) or []
    print(f"\nTotal incidents: {len(incidents)}")
    if incidents:
        last5 = incidents[-5:]
        print(f"Últimos 5 incidents:")
        for inc in last5:
            print(f"  {json.dumps(inc)}")

main()
