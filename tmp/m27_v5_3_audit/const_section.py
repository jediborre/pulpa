s = open(r"C:\Users\App\Desktop\pulpa\monitor_v3\config\constants.py", encoding="utf-8").read()
i = s.find("ACTIVE_MODELS")
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\const.txt", "w", encoding="utf-8").write(s[i:i+700])
print("done")
