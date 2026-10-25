s = open(r"C:\Users\App\Desktop\pulpa\monitor_v3\models\evaluator.py", encoding="utf-8").read()
# find run_model_inference
i = s.find("def run_model_inference")
open(r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\eval_snippet.txt", "w", encoding="utf-8").write(s[i:i+1800])
print("done")
