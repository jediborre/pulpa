"""
tmp/m27_v5_2_audit/a5_parity.py
Paridad REAL de valores (no solo forma): para N partidos del test, compara el vector que genera el extractor
en vivo (models/m27_v5_2/features.py con load_match(DB), target_minute=27) contra la fila del cache de entrenamiento.
Hipotesis: recent_2m_run_diff, rubber_band_gap, white_flag, expected_ft_margin, pbp_3pt, ts, h2h, odds difieren
sistematicamente entre train y live (train-serving skew).
"""
import sys, pickle, random, numpy as np, pandas as pd
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
from models.common.data_loader import get_db_connection, load_match
from models.m27_v5_2.features import extract_features_v5_2, FEATURE_NAMES_27
d = pickle.load(open("tmp/m27_v5/dataset_v5_clean_cache.pkl","rb"))
names = list(d["feature_names"]); idx = [names.index(n) for n in FEATURE_NAMES_27]
meta = d["meta"]; X = d["X"]
pos = {m["mid"]: i for i, m in enumerate(meta)}
cand = [i for i, m in enumerate(meta) if m["yr"] >= 2025 and m["arch"] == "fiba_men_10m"]
random.seed(1); random.shuffle(cand)
conn = get_db_connection()
rows = []
for i in cand[:600]:
    mid = meta[i]["mid"]
    md = load_match(conn, mid)
    if not md: continue
    md.setdefault('pbp', {})
    if not md.get('play_by_play'): md['play_by_play'] = {'_': []}
    v = extract_features_v5_2(md, target_minute=27.0)
    rows.append((mid, v, X[i, idx]))
    if len(rows) >= 250: break
live = np.array([r[1] for r in rows]); trn = np.array([r[2] for r in rows])
print("N", len(rows))
out = []
for j, n in enumerate(FEATURE_NAMES_27):
    a, b = live[:, j], trn[:, j]
    c = np.corrcoef(a, b)[0,1] if a.std() > 0 and b.std() > 0 else float("nan")
    out.append((n, round(a.mean(),3), round(b.mean(),3), round(a.std(),3), round(b.std(),3), round(c,3), round((a==0).mean(),2), round((b==0).mean(),2)))
print(pd.DataFrame(out, columns=["feat","live_mean","train_mean","live_std","train_std","corr","live_zero","train_zero"]).to_string(index=False))

