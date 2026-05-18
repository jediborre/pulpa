"""Analyze incident data structure for per-quarter computation."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
d = json.load(open(ROOT / "temp_scripts" / "full_initial_props.json", "r", encoding="utf-8"))
incidents = d.get("incidents", [])

# All possible keys
all_keys = set()
for i in incidents:
    all_keys.update(i.keys())
print("All possible keys in incidents:", sorted(all_keys))
print(f"Total incidents: {len(incidents)}")

# Incident types
from collections import Counter
types = Counter(i.get("incidentType") for i in incidents)
print("Incident types:", dict(types))
print("By reversedPeriodTime:", dict(sorted(Counter(i.get("reversedPeriodTime") for i in incidents if i.get("reversedPeriodTime")).items())))

# Sample one incident per reversedPeriodTime
seen = set()
for i in incidents:
    rpt = i.get("reversedPeriodTime")
    if rpt and rpt not in seen:
        seen.add(rpt)
        print(f"\nreversedPeriodTime={rpt}:")
        print(f"  incidentType={i.get('incidentType')}, incidentClass={i.get('incidentClass')}")
        print(f"  team={i.get('team')}, homeScore={i.get('homeScore')}, awayScore={i.get('awayScore')}")
        print(f"  isScored={i.get('isScored')}, pointValue={i.get('pointValue')}")
        print(f"  time={i.get('time')}, reversedPeriodTimeSeconds={i.get('reversedPeriodTimeSeconds')}")

# Check: does the incidents endpoint now return fouls/timeouts/turnovers?
# Compare with the incidents from the API vs initialProps
incident_types_detail = Counter()
for i in incidents:
    it = i.get("incidentType", "?")
    ic = i.get("incidentClass", "?")
    incident_types_detail[f"{it}/{ic}"] += 1
print("\nIncident type/class breakdown:")
for k, v in sorted(incident_types_detail.items()):
    print(f"  {k}: {v}")
