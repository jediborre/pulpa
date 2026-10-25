import sqlite3
conn = sqlite3.connect(r"C:\Users\App\Desktop\pulpa\matches.db")
# Fix backfilled name to match existing convention
n = conn.execute("UPDATE matches SET league='NBL1, South' WHERE league='NBL1 South'").rowcount
conn.execute("UPDATE quarter_scores_v2 SET match_id=match_id WHERE 1=0")  # no-op safety
# also fix quarter_scores? match_id unchanged, so no need
conn.commit()
# verify
r = conn.execute("SELECT MIN(date) mn, MAX(date) mx, COUNT(*) n FROM matches WHERE league='NBL1, South'").fetchone()
print("NBL1, South now:", r[2], r[0], "->", r[1])
r = conn.execute("SELECT COUNT(*) n FROM matches WHERE league='NBL1 South'").fetchone()
print("NBL1 South (old name) remaining:", r[0])
conn.close()
