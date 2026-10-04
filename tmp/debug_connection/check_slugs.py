"""
tmp/debug_connection/check_slugs.py
Proposito: Inspeccionar event_slug/custom_id/home_slug/away_slug en matches.db para
confirmar el formato correcto del link de SofaScore y ver por que el fallback 404.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT match_id, home_team, away_team, home_slug, away_slug, event_slug, custom_id "
    "FROM matches WHERE home_team LIKE '%ASVEL%' OR home_team LIKE '%Al-Ula%' "
    "OR home_team LIKE '%Gravelines%' OR away_team LIKE '%ASVEL%' "
    "ORDER BY date DESC LIMIT 8"
).fetchall()
for r in rows:
    d = dict(r)
    slug_combined = f"{d['home_slug']}-{d['away_slug']}"
    print(f"id={d['match_id']} | {d['home_team']} vs {d['away_team']}")
    print(f"   event_slug={d['event_slug']}  custom_id={d['custom_id']}")
    print(f"   home={d['home_slug']} away={d['away_slug']} combinado={slug_combined}")
