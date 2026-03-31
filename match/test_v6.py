import sys
sys.path.insert(0, ".")

import importlib
db_mod = importlib.import_module("db")
infer_match = importlib.import_module("training.infer_match")
infer_match.scraper_mod.fetch_event_snapshot = lambda mid: None

conn = db_mod.get_conn("matches.db")
data = db_mod.get_match(conn, "14439265")

feat = infer_match._build_features(conn, data, "v6", "q3")
print("Q3 features count:", len(feat))
print("mc_home_win_prob:", feat.get("mc_home_win_prob"))
prob = infer_match._predict_prob("v6", "q3", "ensemble_avg_prob", feat)
print("Q3 prob home win:", round(prob, 4))

feat4 = infer_match._build_features(conn, data, "v6", "q4")
print("Q4 features count:", len(feat4))
prob4 = infer_match._predict_prob("v6", "q4", "ensemble_avg_prob", feat4)
print("Q4 prob home win:", round(prob4, 4))
conn.close()
print("OK")
