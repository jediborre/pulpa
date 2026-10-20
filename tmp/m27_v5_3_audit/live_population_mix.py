# =====================================================================
# ¿EL MIX DE POBLACIONES (con/sin mercado) EN VIVO EXPLICA EL WR DE 72.5 %?
# ---------------------------------------------------------------------
# OFFLINE (bloque C, anclas, umbral 0.75):
#   con mercado  :  293 apuestas (13.5 %) | WR 74.1 %
#   sin mercado  : 1.879 apuestas (86.5 %) | WR 83.3 %
#   global       : 2.172 apuestas           | WR 82.0 %
# LIVE (18-19 oct): 40 apuestas, WR 72.5 %. ¿Cuántas eran con mercado?
#   Si el mix en vivo es mas "con mercado" que el offline, parte del gap
#   del WR live esta explicado por la poblacion, no por el modelo.
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

OFFLINE = {"con": (293, 74.1), "sin": (1879, 83.3)}
OFFLINE_GLOBAL = (2172, 82.04)

bets = [dict(r) for r in conn.execute(
    """
    SELECT l.model_version, l.signal_type, l.picked_side, l.confidence, l.result,
           l.match_id, l.inference_json, s.league
    FROM bet_monitor_log_v3 l
    LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
    WHERE l.model_version = 'm27_v5_3' AND l.signal_type LIKE '%BET%'
      AND l.signal_type NOT LIKE '%NO_BET%'
    """)]

print(f"apuestas live de m27_v5_3: {len(bets)}")
con_ml = 0
sin_ml = 0
w_con = w_sin = 0
detalle = []
for b in bets:
    raw = {}
    try:
        raw = (json.loads(b["inference_json"] or "{}").get("raw_json") or {})
    except Exception:
        pass
    has_ml = bool(raw.get("has_ml"))
    gana = b["result"] == "win"
    if has_ml:
        con_ml += 1
        w_con += 1 if gana else 0
    else:
        sin_ml += 1
        w_sin += 1 if gana else 0
    detalle.append((b["match_id"], b["result"], has_ml, round(float(b["confidence"] or 0), 3),
                    (b["league"] or "")[:34]))

n = len(bets)
w = sum(1 for b in bets if b["result"] == "win")
print(f"  global live      : {w}/{n} = {w * 100 / n:.1f} %")
print(f"  con mercado      : {w_con}/{con_ml} = {w_con * 100 / con_ml if con_ml else 0:.1f} %")
print(f"  sin mercado      : {w_sin}/{sin_ml} = {w_sin * 100 / sin_ml if sin_ml else 0:.1f} %")
print(f"  mix live con mercado: {con_ml * 100 / n:.1f} %  (offline: 13.5 %)")

esperado = (con_ml * OFFLINE["con"][1] + sin_ml * OFFLINE["sin"][1]) / n if n else 0
print(f"\n  WR esperado si el MIX live rindiera como el offline: {esperado:.1f} %")
print(f"  WR real live: {w * 100 / n:.1f} %  -> el mix explica "
      f"{esperado - w * 100 / n:+.1f} pp del gap frente al {OFFLINE_GLOBAL[1]:.1f} % offline")

print("\n  detalle:")
for mid, res, has, conf, lg in detalle:
    print(f"    {mid:<12} {res:<5} {'con mercado' if has else 'sin mercado':<12} conf={conf}  {lg}")

print("\n  Comparacion con el offline (mismas categorias):")
for tag, (n_off, wr_off) in OFFLINE.items():
    print(f"    {tag:<6} offline n={n_off:>5,} WR {wr_off:.1f} % | live n="
          f"{(con_ml if tag == 'con' else sin_ml):>3} WR "
          f"{(w_con if tag == 'con' else w_sin) * 100 / (con_ml if tag == 'con' else sin_ml) if (con_ml if tag == 'con' else sin_ml) else 0:.1f} %")
conn.close()
