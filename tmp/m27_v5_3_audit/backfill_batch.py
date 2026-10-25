import asyncio, sys, json, subprocess, time
from pathlib import Path
ROOT = Path(r"C:\Users\App\Desktop\pulpa")
sys.path.insert(0, str(ROOT))

# Load search results
state = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "search_all_state.json", encoding="utf-8"))

# Cold-start leagues from v5_3_3 analysis
D = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "league_threshold_analysis_v5_3_3.json", encoding="utf-8"))
cold = set(l["league"] for l in D["leagues"] if l["matches_train"] == 0)

# Prioritize: leagues with >1 season AND in cold-start list, sorted by seasons desc
candidates = []
for lg, info in state.items():
    if lg not in cold:
        continue
    seasons = info.get("seasons", [])
    if len(seasons) < 2:
        continue
    tid = info.get("tid")
    if not tid:
        continue
    candidates.append((len(seasons), tid, lg, info["name"]))

candidates.sort(reverse=True)
print("Candidates to backfill:", len(candidates))
for n, tid, lg, name in candidates[:15]:
    print("  %d seasons | tid=%d | %s" % (n, tid, lg))

# Save candidate list
json.dump([{"tid": t, "league": lg, "name": n, "seasons": s} for _, t, lg, n in []], open(ROOT / "tmp" / "m27_v5_3_audit" / "candidates.json", "w"), indent=2)
# Actually save properly
cand_list = []
for n, tid, lg, name in candidates:
    info = state[lg]
    cand_list.append({"tid": tid, "league": lg, "name": name, "seasons": [s[1] for s in info["seasons"]]})
json.dump(cand_list, open(ROOT / "tmp" / "m27_v5_3_audit" / "candidates.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("saved candidates.json with", len(cand_list), "entries")
