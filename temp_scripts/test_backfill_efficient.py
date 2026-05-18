"""Quick smoke test of efficient backfill."""
import sys, tempfile, os
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
from scraper import fetch_match_by_id
from db import get_conn, init_db, save_match

# First save 2 matches the old way
tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
conn = get_conn(tmp.name)
init_db(conn)

for mid in ["15935010", "15935009"]:
    data = fetch_match_by_id(mid, fetch_team_data=True)
    save_match(conn, mid, data)
conn.close()

# Verify team_strength exists
conn = get_conn(tmp.name)
rows = conn.execute("SELECT count(*) FROM team_strength").fetchone()
print(f"Before backfill - team_strength rows: {rows[0]}")

# Now test the backfill
sys.argv = ["backfill.py", tmp.name, "--ids", "15935010", "15935009", "--delay", "0.3"]
exec(open(ROOT / "match" / "scripts" / "backfill.py").read())

# Verify still good after backfill
rows = conn.execute("SELECT count(*) FROM team_strength").fetchone()
print(f"After backfill - team_strength rows: {rows[0]}")
conn.close()
os.unlink(tmp.name)
