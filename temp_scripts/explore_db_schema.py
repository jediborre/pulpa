"""Deep exploration of matches.db schema and data."""
import sqlite3
import json
from collections import Counter

DB = r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# --- TABLES ---
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [t[0] for t in cur.fetchall()]
print("=== TABLAS ===")
for t in tables:
    print(f"  {t}")

# --- SCHEMA PER TABLE ---
for t in tables:
    cur.execute(f"PRAGMA table_info({t})")
    cols = cur.fetchall()
    print(f"\n=== {t} ===")
    for c in cols:
        print(f"  {c[1]:30s} {c[2]:15s} nullable={not c[3]}")

# --- SAMPLE DATA PER TABLE ---
print("\n\n=== MUESTRAS ===")
for t in tables:
    cur.execute(f"SELECT * FROM {t} LIMIT 2")
    rows = cur.fetchall()
    if not rows:
        print(f"\n{t}: (vacio)")
        continue
    col_names = [d[0] for d in cur.description]
    print(f"\n{t} ({len(col_names)} cols, 2 filas):")
    for row in rows:
        for name, val in zip(col_names, row):
            val_str = str(val)[:120]
            print(f"  {name:30s} = {val_str}")
        print("  ---")

# --- PLAY_BY_PLAY: analyze structure of JSON contents ---
print("\n\n=== PLAY_BY_PLAY ESTRUCTURA ===")
cur.execute("SELECT match_id, play_by_play FROM matches LIMIT 20")
for mid, pbp_json in cur.fetchall():
    pbp = json.loads(pbp_json) if pbp_json else {}
    if not pbp:
        continue
    # Show quarters available
    quarters = list(pbp.keys())
    # For first quarter, show first 3 plays structure
    first_q = quarters[0]
    plays = pbp[first_q]
    print(f"\nMatch: {mid}")
    print(f"  Quarters: {quarters}")
    if plays:
        print(f"  Plays in {first_q}: {len(plays)}")
        play_keys = list(plays[0].keys())
        print(f"  Play fields: {play_keys}")
        for i, play in enumerate(plays[:3]):
            print(f"  Play {i}:")
            for k, v in play.items():
                print(f"    {k}: {v}")
    break  # Just first match for depth

# --- GRAPH_POINTS: analyze structure ---
print("\n\n=== GRAPH_POINTS ESTRUCTURA ===")
cur.execute("SELECT match_id, graph_points FROM matches LIMIT 10")
for mid, gp_json in cur.fetchall():
    gp = json.loads(gp_json) if gp_json else []
    print(f"\nMatch: {mid}")
    print(f"  Graph points count: {len(gp)}")
    if gp:
        print(f"  First point keys: {list(gp[0].keys())}")
        print(f"  First 3 points:")
        for p in gp[:3]:
            print(f"    minute={p.get('minute')}, value={p.get('value')}")
        print(f"  Last 3 points:")
        for p in gp[-3:]:
            print(f"    minute={p.get('minute')}, value={p.get('value')}")
        # Full sequence of values
        values = [p.get('value') for p in gp]
        minutes = [p.get('minute') for p in gp]
        print(f"  Min values: {values}")
        print(f"  Min minutes: {minutes}")
    break

conn.close()
