s = open(r"C:\Users\App\Desktop\pulpa\monitor_v3\config\constants.py", encoding="utf-8").read()
out = []
for line in s.split("\n"):
    if ("SHADOW_MODELS" in line or "STATUS_MODELS" in line or "MODEL_INFERENCE" in line
            or "INFERENCE_WINDOW" in line or "m27_v5_3" in line or "ACTIVE_MODELS" in line
            or "INFERENCE_MINUTE" in line):
        out.append(line.strip())
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\const_lines.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
