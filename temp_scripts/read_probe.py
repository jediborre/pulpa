"""Read probe results from file."""
import json
with open(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\temp_scripts\probe_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for k in sorted(data.keys()):
    if k.startswith("_event"):
        v = data[k]
        if isinstance(v, dict):
            print(f"{k}: dict with keys {list(v.keys())[:20]}")
        elif isinstance(v, list):
            print(f"{k}: list with {len(v)} items")
            if v and isinstance(v[0], dict):
                print(f"  first item keys: {list(v[0].keys())[:15]}")
        else:
            print(f"{k}: {v}")

if "_lineups" in data:
    print("\n=== LINEUPS ===")
    lu = data["_lineups"]
    print(f"Keys: {list(lu.keys())}")
    print(f"confirmed: {lu.get('confirmed')}")
    for side in ["home", "away"]:
        team = lu.get(side, {})
        players = team.get("players", [])
        print(f"\n{side}: {len(players)} players")
        for p in players[:3]:
            pl = p.get("player", {})
            print(f"  {pl.get('name')} (id={pl.get('id')}, pos={pl.get('position')}, #={pl.get('jerseyNumber')})")
            extra = [k for k in p.keys() if k != "player"]
            if extra:
                print(f"  extra keys: {extra}")
        # Check if any player has statistics
        for p in players[:1]:
            pkeys = list(p.keys())
            print(f"  player entry keys: {pkeys}")

if "_statistics" in data:
    print("\n=== TEAM STATISTICS ===")
    stats = data["_statistics"]
    for period_group in stats.get("statistics", []):
        print(f"Period: {period_group['period']}")
        for g in period_group.get("groups", []):
            print(f"  Group: {g['groupName']}")
            for item in g.get("statisticsItems", [])[:3]:
                hv = item.get("homeValue")
                av = item.get("awayValue")
                print(f"    {item['name']}: home={hv} away={av} (key={item.get('key')})")

if "_h2h" in data:
    print("\n=== H2H ===")
    print(json.dumps(data["_h2h"], indent=2, ensure_ascii=False)[:1500])

print("\n=== EVENT: homeTeam/awayTeam fields ===")
for side in ["homeTeam", "awayTeam"]:
    key = f"_event_{side}_keys"
    if key in data:
        print(f"{side} keys: {data[key]}")
    for sus in ["rating", "id", "statistics", "players"]:
        sk = f"_event_{side}_{sus}"
        if sk in data:
            print(f"  {sus}: {data[sk]}")
