import json, re
from pathlib import Path
ROOT = Path(r"C:\Users\App\Desktop\pulpa")
moved = json.loads((ROOT / "tmp" / "m27_v5_3_audit" / "reorg_map.json").read_text(encoding="utf-8"))

# Only scan these directories
scan_dirs = [ROOT / "docs", ROOT, ROOT / "monitor_v3", ROOT / "models", ROOT / "tmp", ROOT / "tools", ROOT / "match"]
files = []
for d in scan_dirs:
    if not d.exists(): continue
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in (".md", ".py", ".bat", ".yaml", ".yml"):
            if "__pycache__" in str(p) or "node_modules" in str(p):
                continue
            files.append(p)

total_repl = 0
for p in files:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        continue
    orig = text
    for fname, cat in moved.items():
        text = text.replace("docs/" + fname, "docs/" + cat + "/" + fname)
        text = re.sub(r"\(\./" + re.escape(fname) + r"\)", "(" + cat + "/" + fname + ")", text)
        text = re.sub(r"\(" + re.escape(fname) + r"\)", "(" + cat + "/" + fname + ")", text)
    if text != orig:
        p.write_text(text, encoding="utf-8")
        total_repl += 1
print("updated files:", total_repl)