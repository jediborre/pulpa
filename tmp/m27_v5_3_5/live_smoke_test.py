"""
tmp/m27_v5_3_5/live_smoke_test.py
Prueba de humo REAL contra SofaScore: toma partidos en vivo (Q3/Q4), descarga el payload con el cliente de monitor_v3
(fetch_full_match), verifica que odds pre-partido no se contaminen con in-play, resume odds_live y corre m27_v5_3_5.
No escribe en matches.db.
"""
import asyncio, sys, json
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
import models
from monitor_v3.core.mobile_client import get_mobile_client
async def main():
    c = get_mobile_client()
    evs = await c.get_live_events()
    cands = [e for e in evs if (e.get("status") or {}).get("description") in ("3rd quarter", "4th quarter", "Halftime")]
    print("live:", len(evs), "Q2-Q4:", len(cands))
    for e in cands[:8]:
        mid = str(e["id"])
        try:
            p = await c.fetch_full_match(mid)
        except Exception as ex:
            print(mid, "error", ex); continue
        m = p["match"]
        ml = [(o["odds_type"], o["market_period"], o["home_value"], o["away_value"], o.get("line")) for o in p.get("odds", [])]
        r = models.predict(match_id=mid, model_version="m27_v5_3_5", match_data=p, snapshot_minute=27.0)
        rj = r.raw_json or {}
        print(f"\n{mid} {m['league']} | {m['home_team']} vs {m['away_team']} | {m['status_description']} played={m.get('game_seconds_played')} score={p['score']['home']}-{p['score']['away']}")
        print("  odds pre:", ml, "| odds_live:", p.get("odds_live"))
        print(f"  v5_3: available={r.available} signal={r.signal} p_home={r.p_home_win} min_used={rj.get('snapshot_minute_used')} missing={rj.get('features_missing')} reason={r.reason}")
        if r.available:
            f = rj["features"]; print("  feats:", {k: f[k] for k in ("cur_score_diff","q3_partial_diff","recent_pts_diff_2m","odds_log_ratio","d_ft_margin15","expected_ft_margin","gp_val")})
asyncio.run(main())
