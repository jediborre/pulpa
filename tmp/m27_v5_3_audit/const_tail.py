s = open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\const.txt", encoding="utf-8").read()
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\const_tail.txt", "w", encoding="utf-8").write(s[600:])
print("done")
