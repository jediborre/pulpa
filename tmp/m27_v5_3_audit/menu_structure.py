s = open(r"C:\Users\App\Desktop\pulpa\menu.bat", encoding="utf-8", errors="replace").read()
ls = s.split("\n")
out = []
# find lines with choice / if / goto patterns
for i, l in enumerate(ls):
    st = l.strip()
    if "choice" in st.lower() or "goto" in st.lower() or "echo" in st.lower() and ("." in st or ")" in st):
        if "backfill" in st.lower() or "smart" in st.lower() or "hist" in st.lower():
            out.append("%d: %s" % (i+1, st[:120]))
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\menu_structure.txt", "w", encoding="utf-8").write("\n".join(out) if out else "none")
print("done")
