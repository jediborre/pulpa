s = open(r"C:\Users\App\Desktop\pulpa\AGENTS.md", encoding="utf-8").read()
out = []
for line in s.split("\n"):
    if "docs/" in line:
        out.append(line.strip()[:120])
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\refs.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
