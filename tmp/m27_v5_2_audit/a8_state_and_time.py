"""
tmp/m27_v5_2_audit/a8_state_and_time.py
(1) Estructura de v5_2_production_state.pkl: ¿existe team_priors[t]["q_profile"]["q4_net_avg"]? (d_strat_q4_net en vivo)
(2) Semantica de time_seconds vs minuto real: marcador reconstruido con score_upto(27) vs marcador real Q1+Q2+Q3.
"""
import sys, pickle, random, numpy as np
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
st = pickle.load(open("models/m27_v5_2/model_outputs/v5_2_production_state.pkl","rb"))
print("keys", list(st.keys()))
tp = st["team_priors"]; print("n teams", len(tp)); k = next(iter(tp)); print(k, tp[k])
pp = st["player_priors"]; k2 = next(iter(pp)); print("n players", len(pp), repr(k2), pp[k2])
print("lc_map sample", list(st["lc_map"].items())[:2])
has_q = sum(1 for v in tp.values() if v.get("q_profile")); print("teams con q_profile:", has_q)
print("teams con q4_net_avg:", sum(1 for v in tp.values() if isinstance(v.get("q_profile"), dict) and "q4_net_avg" in v["q_profile"]))
from models.common.data_loader import get_db_connection, load_match
from models.common.pbp_utils import score_upto, pbp_events_upto
conn = get_db_connection()
rows = conn.execute("select m.match_id from matches m where m.date>='2025-06-01' and m.status_type='finished' and exists(select 1 from match_events e where e.match_id=m.match_id) and m.league like '%Liga%' limit 3000").fetchall()
random.seed(3); random.shuffle(rows)
r = []
for (mid,) in rows[:150]:
    md = load_match(conn, mid)
    qs = md["score"]["quarters"]
    if not all(q in qs for q in ("Q1","Q2","Q3","Q4")): continue
    true3 = sum(qs[q]["home"]+qs[q]["away"] for q in ("Q1","Q2","Q3"))
    true2 = sum(qs[q]["home"]+qs[q]["away"] for q in ("Q1","Q2"))
    h,a = score_upto(md, 27); h30,a30 = score_upto(md, 30); h20,a20 = score_upto(md, 20)
    evs = [e for q in md["events"].values() for e in q if e.get("time_seconds") is not None]
    mx = max(e["time_seconds"] for e in evs) if evs else None
    r.append((true2, true3, h+a, h30+a30, h20+a20, mx, qs["Q4"]["home"]+qs["Q4"]["away"]+true3))
r = np.array(r, dtype=float)
print("N", len(r))
print("mean true_HT %.1f | true_3Q %.1f | score_upto27 %.1f | score_upto30 %.1f | score_upto20 %.1f | max time_s %.0f | final total %.1f" % tuple(r.mean(0)))
print("share score_upto30 ~ true_3Q (±2):", np.mean(np.abs(r[:,3]-r[:,1])<=2), " share score_upto27 within 3Q-?:", np.mean(r[:,2] < r[:,1]))
