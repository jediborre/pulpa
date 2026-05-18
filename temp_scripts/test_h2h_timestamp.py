"""Test: H2H with timestamp."""
import sys, json, tempfile, os
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from scraper import fetch_match
from db import get_conn, init_db, save_match

match_id = "15935010"
result = fetch_match(url=f"https://www.sofascore.com/event/{match_id}", match_id=match_id)

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
conn = get_conn(tmp.name)
init_db(conn)
save_match(conn, match_id, result)

rows = conn.execute(
    "SELECT h2h_match_id, date, timestamp, home_team, home_score, away_score FROM match_h2h WHERE match_id = ? ORDER BY timestamp",
    (match_id,),
).fetchall()

print(f"H2H rows: {len(rows)}")
for r in rows:
    from datetime import datetime, timezone
    ts_str = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if r["timestamp"] else "N/A"
    print(f"  {r['h2h_match_id']}: {r['date']} | ts={r['timestamp']} ({ts_str}) | {r['home_team']} {r['home_score']}-{r['away_score']}")

conn.close()
os.unlink(tmp.name)
