"""Quick test of backfill."""
import sys, tempfile, os
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
from scraper import fetch_match
from db import get_conn, init_db, save_match

# First save 2 matches normally
tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
conn = get_conn(tmp.name)
init_db(conn)

for mid in ["15935010", "15935009"]:
    result = fetch_match(url=f"https://www.sofascore.com/event/{mid}", match_id=mid, fetch_team_data=True)
    save_match(conn, mid, result)

conn.close()

# Now backfill using our script
sys.argv = ["backfill.py", tmp.name, "--ids", "15935010", "15935009", "--delay", "0.5"]
exec(open(ROOT / "match" / "scripts" / "backfill.py").read())

os.unlink(tmp.name)
