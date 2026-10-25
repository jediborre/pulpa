s = open(r"C:\Users\App\Desktop\pulpa\menu.bat", encoding="utf-8", errors="replace").read()
out = []
out.append("BACKFILL count: %d" % s.count("BACKFILL"))
out.append("backfill count: %d" % s.lower().count("backfill"))
# find lines with goto
for i, l in enumerate(s.split("\n")):
    if "goto" in l.lower() and "BACKFILL" in l.upper():
        out.append("%d: %s" % (i+1, l.strip()[:100]))
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\menu_search.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
