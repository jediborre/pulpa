import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "match" / "matches.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r["name"] for r in c.fetchall()]
print("TABLES:", tables)

for t in tables:
    c.execute(f"SELECT COUNT(*) FROM {t}")
    cnt = c.fetchone()[0]
    c.execute(f"PRAGMA table_info({t})")
    cols = [cc["name"] for cc in c.fetchall()]
    print(f"  {t}: {cnt} rows, cols[:8]: {cols[:8]}")

# Check what versions exist in eval_match_results
c.execute("PRAGMA table_info(eval_match_results)")
eval_cols = [cc["name"] for cc in c.fetchall()]
versions_found = set()
for col in eval_cols:
    if col.startswith("q3_signal__"):
        versions_found.add(col.replace("q3_signal__", ""))
print("\nVersions in eval_match_results:", sorted(versions_found))

conn.close()
