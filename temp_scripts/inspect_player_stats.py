"""Inspect player statistics embedded in lineups response."""
import json

with open(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\temp_scripts\probe_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lu = data["_lineups"]
for side in ["home", "away"]:
    team = lu.get(side, {})
    players = team.get("players", [])
    print(f"\n{'='*60}")
    print(f"  {side.upper()} — {len(players)} players")
    print(f"{'='*60}")
    for p in players[:5]:  # first 5 players
        pl = p.get("player", {})
        name = pl.get("name", "?")
        stats = p.get("statistics", {})
        print(f"\n  {name} (id={pl.get('id')}, #={p.get('shirtNumber')}, pos={pl.get('position')})")
        print(f"    substitute: {p.get('substitute')}")
        if isinstance(stats, dict):
            print(f"    stats keys: {list(stats.keys())}")
            for sk, sv in stats.items():
                print(f"      {sk}: {sv}")
        else:
            print(f"    stats: {stats}")
