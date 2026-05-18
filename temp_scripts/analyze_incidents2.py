"""Analyze incident period vs reversedPeriodTime relationship."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
d = json.load(open(ROOT / "temp_scripts" / "full_initial_props.json", "r", encoding="utf-8"))
incidents = d.get("incidents", [])

# Check period field
print("=== Period field values ===")
for i in incidents[:15]:
    print(f"  reversedPeriodTime={i.get('reversedPeriodTime')}, period={i.get('period')}, time={i.get('time')}")

# And the period-type incidents
print("\n=== Period-type incidents ===")
for i in incidents:
    if i.get("incidentType") == "period":
        print(f"  reversedPeriodTime={i.get('reversedPeriodTime')}, period={i.get('period')}, time={i.get('time')}")

# Show unique (reversedPeriodTime, period) pairs
pairs = set()
for i in incidents:
    rpt = i.get("reversedPeriodTime")
    p = i.get("period")
    pairs.add((rpt, p))
print(f"\nUnique (reversedPeriodTime, period) pairs: {sorted(pairs)}")

# The 'from' field indicates scoring type
print("\n=== 'from' field values ===")
from collections import Counter
from_vals = Counter(i.get("from", "?") for i in incidents if i.get("incidentType") != "period")
print(dict(from_vals))
