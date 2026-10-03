"""
Snapshot en vivo del partido AD Brusque vs BT/Tatuí para ver qué datos
retorna SofaScore en HT y por qué el monitor lo muestra como Q4.
"""
import asyncio
import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from match.scraper import fetch_event_snapshot as ss_fetch_snapshot

MATCH_ID = "16220335"  # AD Brusque vs BT/Tatuí

def main():
    print(f"Snapshot en vivo: AD Brusque vs BT/Tatuí ({MATCH_ID})\n")
    
    snap = ss_fetch_snapshot(MATCH_ID)
    
    print("=== SNAPSHOT RAW ===")
    print(json.dumps(snap, indent=2, ensure_ascii=False))
    
    print("\n=== CAMPOS CLAVE ===")
    print(f"  status_type:        {snap.get('status_type')}")
    print(f"  status_description: {snap.get('status_description')}")
    print(f"  home_score:         {snap.get('home_score')}")
    print(f"  away_score:         {snap.get('away_score')}")
    print(f"  time_current:       {snap.get('time_current')}")
    print(f"  time_period:        {snap.get('time_period')}")

main()
