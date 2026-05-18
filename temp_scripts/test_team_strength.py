"""Full test: fetch match with team_strength and save to DB."""
import sys, json, time
from pathlib import Path
import tempfile, os

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from scraper import fetch_match, fetch_team_strength
from db import get_conn, init_db, save_match

match_id = "15935010"
url = f"https://www.sofascore.com/event/{match_id}"

# Fetch match with team_strength
result = fetch_match(url=url, match_id=match_id, fetch_team_data=True)

print("=== team_strength ===")
for ts in result.get("team_strength", []):
    print(json.dumps(ts, indent=2))

# Save to temp DB to verify
tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
conn = get_conn(tmp.name)
init_db(conn)
save_match(conn, match_id, result)

# Verify in DB
rows = conn.execute(
    "SELECT team_id, team_name, position, wins, losses, form, perf_points FROM team_strength WHERE match_id = ?",
    (match_id,),
).fetchall()
print("\n=== From DB ===")
for r in rows:
    print(dict(r))

conn.close()
os.unlink(tmp.name)
print("\nOK")
