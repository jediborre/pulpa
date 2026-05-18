"""Test compute_h2h_side_stats."""
import sys, json, tempfile, os
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from scraper import fetch_match
from db import get_conn, init_db, save_match, compute_h2h_side_stats

# Fetch and save
match_id = "15935010"
result = fetch_match(url=f"https://www.sofascore.com/event/{match_id}", match_id=match_id, fetch_team_data=True)

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
conn = get_conn(tmp.name)
init_db(conn)
save_match(conn, match_id, result)

# Compute side stats
side = compute_h2h_side_stats(conn, match_id)
print(json.dumps(side, indent=2))

conn.close()
os.unlink(tmp.name)
