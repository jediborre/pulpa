import sqlite3
import os
import glob

# Search for matches.db in common locations
db_paths = [
    r"c:\Users\App\Desktop\pulpa\match\matches.db",
    r"C:\match\matches.db",
    r"matches.db"
]

print("Scanning for matches.db files...")
for p in db_paths:
    if os.path.exists(p):
        print(f"Found: {p} (size={os.path.getsize(p)} bytes)")
    else:
        print(f"Not found: {p}")

# Also search using glob
found_dbs = glob.glob(r"c:\Users\App\Desktop\pulpa\**\*.db", recursive=True)
print("All found .db files:", found_dbs)
