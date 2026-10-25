p = r"C:\Users\App\Desktop\pulpa\models\m27_v5_3_3\features.py"
s = open(p, encoding="utf-8").read()
old = '''    m = match_data.get("match") or {}
    league = str(m.get("league") or match_data.get("league") or "")
    info = state.lc_map.get(league)
    if info:
        return info'''
new = '''    m = match_data.get("match") or {}
    league = str(m.get("league") or match_data.get("league") or "")
    league = state._clean_league_name(league)
    info = state.lc_map.get(league)
    if info:
        return info'''
assert old in s
s = s.replace(old, new)
# ctx_from_payload
old2 = '''    league = str(m.get("league") or match_data.get("league") or "")
    if not ht or not at:
        return None'''
new2 = '''    league = str(m.get("league") or match_data.get("league") or "")
    league = state._clean_league_name(league)
    if not ht or not at:
        return None'''
assert old2 in s
s = s.replace(old2, new2)
# refresh_from_db: normalize r[2]
old3 = '''            self.update(r[3], r[4], date_ord(r[1]), r[2], r[5] - r[6], r[7] - r[8], r[9] - r[10], r[11] - r[12])'''
new3 = '''            self.update(r[3], r[4], date_ord(r[1]), self._clean_league_name(r[2]), r[5] - r[6], r[7] - r[8], r[9] - r[10], r[11] - r[12])'''
assert old3 in s
s = s.replace(old3, new3)
open(p, "w", encoding="utf-8").write(s)
print("patched")
