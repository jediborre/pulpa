s = open(r"C:\Users\App\Desktop\pulpa\monitor_v3\models\evaluator.py", encoding="utf-8").read()
out = [l for l in s.split("\n") if "m27_v5_3" in l or "ACTIVE_MODELS" in l or "registry" in l.lower() or "predict" in l.lower()]
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\eval_grep.txt", "w", encoding="utf-8").write("\n".join(out))
print("done", len(out))
