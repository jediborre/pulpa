# =====================================================================
# SONDEO: como llama SofaScore a los mercados por periodo (Q4) y hay
#         mercado Home/Away de "4th quarter" IN-PLAY?
# =====================================================================

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client, summarize_live_odds

    conn = sqlite3.connect(str(ROOT / "matches.db"))
    rows = conn.execute(
        "SELECT match_id, league FROM matches WHERE status_type='inprogress' "
        "ORDER BY date DESC LIMIT 6").fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT match_id, league FROM matches ORDER BY date DESC LIMIT 6").fetchall()
    conn.close()

    client = get_mobile_client()
    try:
        live_events = await client.get_live_events()
    except Exception as exc:
        print("ERROR get_live_events:", exc)
        live_events = []
    print(f"eventos en vivo: {len(live_events)}")
    cand = []
    for ev in live_events:
        eid = str(ev.get("id") or "")
        lg = (((ev.get("tournament") or {}).get("category") or {}).get("name") or "?")
        if eid:
            cand.append((eid, lg))

    # Partidos que el monitor ya evaluo: sus mercados in-play se guardaron.
    conn2 = sqlite3.connect(str(ROOT / "matches.db"))
    for (m_id,) in conn2.execute(
        "SELECT DISTINCT match_id FROM live_snapshots_v3 ORDER BY id DESC LIMIT 6"
    ):
        cand.append((str(m_id), "(ya monitorizado)"))
    conn2.close()

    for mid, league in cand[:10]:
        try:
            resp = await client.request("GET", f"event/{mid}/odds/1/all")
            odds_json = resp if isinstance(resp, dict) else {}
        except Exception as exc:
            print(f"  {mid} ({league}): ERROR {exc}")
            continue
        markets = odds_json.get("markets") or []
        live = [m for m in markets if m.get("isLive")]
        print(f"\n  {mid} ({league}): {len(markets)} mercados, {len(live)} in-play")
        seen = set()
        for m in live:
            key = (m.get("marketGroup"), m.get("marketPeriod"))
            if key in seen:
                continue
            seen.add(key)
            ch = m.get("choices") or []
            fr = [c.get("fractionalValue") for c in ch[:2]]
            print(f"     group={str(m.get('marketGroup')):<16} period={str(m.get('marketPeriod')):<22} "
                  f"frac={fr}")
        print(f"     -> summarize_live_odds: {json.dumps(summarize_live_odds(odds_json), ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())
