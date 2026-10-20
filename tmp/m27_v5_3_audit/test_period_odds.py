# =====================================================================
# PRUEBA de summarize_live_odds con el mercado del ganador de Q4
# =====================================================================

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import summarize_live_odds  # noqa: E402

PAYLOAD = {
    "markets": [
        # ML del partido in-play
        {"isLive": True, "marketGroup": "Home/Away", "marketPeriod": "Match",
         "choices": [{"fractionalValue": "2/3"}, {"fractionalValue": "4/1"}]},
        # ML del 4to cuarto in-play -> el mercado que apostamos
        {"isLive": True, "marketGroup": "Home/Away", "marketPeriod": "4th quarter",
         "choices": [{"fractionalValue": "1/1"}, {"fractionalValue": "1/1"}]},
        # 2do cuarto y 3er cuarto
        {"isLive": True, "marketGroup": "Home/Away", "marketPeriod": "2nd quarter",
         "choices": [{"fractionalValue": "5/4"}, {"fractionalValue": "3/4"}]},
        {"isLive": True, "marketGroup": "Home/Away", "marketPeriod": "3rd quarter",
         "choices": [{"fractionalValue": "1/2"}, {"fractionalValue": "2/1"}]},
        # Handicap con linea
        {"isLive": True, "marketGroup": "Point spread", "marketPeriod": "4th quarter",
         "choiceGroup": "-3.5",
         "choices": [{"name": "Home (-3.5)", "fractionalValue": "10/11"},
                     {"name": "Away (+3.5)", "fractionalValue": "10/11"}]},
        # Total
        {"isLive": True, "marketGroup": "Over/Under", "marketPeriod": "4th quarter",
         "choiceGroup": "48.5",
         "choices": [{"name": "Over", "fractionalValue": "5/4"},
                     {"name": "Under", "fractionalValue": "3/4"}]},
        # PRE-partido (isLive False) -> debe ignorarse
        {"isLive": False, "marketGroup": "Home/Away", "marketPeriod": "Match",
         "choices": [{"fractionalValue": "9/10"}, {"fractionalValue": "9/10"}]},
        # mercado suspendido -> ignorado
        {"isLive": True, "suspended": True, "marketGroup": "Home/Away",
         "marketPeriod": "4th quarter",
         "choices": [{"fractionalValue": "1/3"}, {"fractionalValue": "3/1"}]},
    ]
}

out = summarize_live_odds(PAYLOAD)
print(json.dumps(out, indent=2, ensure_ascii=False))
print()
ok = True
checks = [
    ("ml_home", 1.667), ("ml_away", 5.0),
    ("q4_ml_home", 2.0), ("q4_ml_away", 2.0),
    ("q2_ml_home", 2.25), ("q2_ml_away", 1.75),
    ("q3_ml_home", 1.5), ("q3_ml_away", 3.0),
]
for k, expected in checks:
    got = out.get(k)
    good = got is not None and abs(got - expected) < 1e-6
    ok = ok and good
    print(f"  {'OK  ' if good else 'FALLA'} {k:<14} esperado {expected:<8} obtenido {got}")
print(f"  spread_line = {out.get('spread_line')}  total_line = {out.get('total_line')}")
print()
print("RESULTADO:", "OK" if ok else "FALLA")
