import sqlite3

conn = sqlite3.connect("matches.db")
conn.row_factory = sqlite3.Row

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t["name"] for t in tables])

cols = conn.execute("PRAGMA table_info(eval_match_results)").fetchall()
print("eval_match_results columns:", [c["name"] for c in cols])

rows = conn.execute("SELECT * FROM eval_match_results WHERE match_id='15736636'").fetchall()
print(f"\nRows for match 15736636: {len(rows)}")
for row in rows:
    print(dict(row))

if not rows:
    print("\nNot found. Recent rows:")
    recent = conn.execute(
        "SELECT match_id, event_date, home_team, away_team, updated_at FROM eval_match_results ORDER BY updated_at DESC LIMIT 15"
    ).fetchall()
    for r in recent:
        print(dict(r))
    
    total = conn.execute("SELECT COUNT(*) as n FROM eval_match_results").fetchone()["n"]
    print(f"\nTotal rows in eval_match_results: {total}")

conn.close()
