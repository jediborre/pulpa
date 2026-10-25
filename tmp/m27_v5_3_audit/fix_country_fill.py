import json
p = r"C:\Users\App\Desktop\pulpa\tmp\m27_v5_3_audit\country_fill.json"
d = json.load(open(p, encoding="utf-8"))
d["Road to Basketball Africa League"] = "International"
d["Serie B Interregionale"] = "Italy"
d["B Ethniki"] = "Greece"
# also fix mojibake Türkiye
if "T�rkiye" in d.values():
    for k, v in list(d.items()):
        if v == "T�rkiye":
            d[k] = "Türkiye"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("done")
