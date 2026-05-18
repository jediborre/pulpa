"""Analyze all incident types from initialProps."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
d = json.load(open(ROOT / "temp_scripts" / "full_initial_props.json", "r", encoding="utf-8"))
incidents = d.get("incidents", [])

# Pairs
pairs = set()
for i in incidents:
    rpt = i.get("reversedPeriodTime")
    p = i.get("period")
    if rpt is not None and p is not None:
        pairs.add((rpt, p))
print("Valid (reversedPeriodTime, period) pairs:", sorted(pairs))

# All incident types
print("All incidentTypes:", sorted(set(i.get("incidentType","?") for i in incidents)))

# Non-goal/period
others = [i for i in incidents if i.get("incidentType") not in ("goal","period")]
print(f"Non-goal/period incidents: {len(others)}")
for o in others[:15]:
    print(f"  type={o.get('incidentType')} class={o.get('incidentClass')} from={o.get('from')}")

# Check 'from' field distribution
from collections import Counter
from_vals = Counter(i.get("from","?") for i in incidents if i.get("incidentType") != "period")
print("\n'from' field distribution:", dict(from_vals))
