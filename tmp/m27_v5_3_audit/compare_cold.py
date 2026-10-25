import json
D3 = json.load(open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\league_threshold_analysis_v5_3_3.json", encoding="utf-8"))
D4 = json.load(open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\league_threshold_analysis_v5_3_4.json", encoding="utf-8"))
# map league -> trained
m3 = {l["league"]: l["matches_train"] for l in D3["leagues"]}
m4 = {l["league"]: l["matches_train"] for l in D4["leagues"]}
# leagues that went from 0 to >0
gained = []
for lg, tr in m4.items():
    if tr > 0 and m3.get(lg, 0) == 0:
        gained.append((lg, tr))
gained.sort(key=lambda x: x[1], reverse=True)
out = ["Ligas que ganaron entrenamiento en v5_3_4 (%d):" % len(gained)]
for lg, tr in gained:
    out.append("  %6d | %s" % (tr, lg))
# also leagues that disappeared from cold-start (no longer in D4 or trained>0)
out.append("\nTotal leagues v5_3_3: %d, v5_3_4: %d" % (len(m3), len(m4)))
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\gained.txt", "w", encoding="utf-8").write("\n".join(out))
print("done: %d gained" % len(gained))
