import json, re
from pathlib import Path
ROOT = Path(r"C:\Users\App\Desktop\pulpa")
moved = json.loads((ROOT / "tmp" / "m27_v5_3_audit" / "reorg_map.json").read_text(encoding="utf-8"))

# files to update
targets = []
for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix in (".md", ".py", ".bat", ".yaml", ".yml", ".txt"):
        if ".git" in str(p) or "node_modules" in str(p):
            continue
        targets.append(p)

total_repl = 0
for p in targets:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        continue
    orig = text
    for fname, cat in moved.items():
        # Pattern 1: docs/fname -> docs/cat/fname
        text = text.replace("docs/" + fname, "docs/" + cat + "/" + fname)
        # Pattern 2: (fname) or (./fname) in markdown links
        text = re.sub(r"\(\./" + re.escape(fname) + r"\)", "(" + cat + "/" + fname + ")", text)
        text = re.sub(r"\(" + re.escape(fname) + r"\)", "(" + cat + "/" + fname + ")", text)
    if text != orig:
        p.write_text(text, encoding="utf-8")
        total_repl += 1
print("updated files:", total_repl)