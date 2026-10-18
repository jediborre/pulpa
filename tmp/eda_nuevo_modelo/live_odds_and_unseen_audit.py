"""
tmp/eda_nuevo_modelo/live_odds_and_unseen_audit.py
==================================================
Propósito:
1. Consultar en tiempo real `sport/basketball/events/live` en la API móvil de SofaScore
   para ver todos los partidos actualmente en vivo (especialmente Q3 y Q4), obtener
   la fecha real del servidor y auditar todos los mercados de momios en vivo (`/event/{id}/odds/1/all`,
   `/event/{id}/odds/1/featured`, etc.), buscando específicamente momios de Ganador Q4,
   Hándicap Q4, Total Q4 y Ganador FT en vivo.
2. Consultar en modo SOLO LECTURA (`mode=ro`) las fechas más recientes en `matches.db`
   para identificar exactamente qué días recientes ("desde antier") no estaban en el
   entrenamiento ni en la base cuando se entrenó `m27_v4`.
"""
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client


def frac_to_dec(frac_str: str) -> float:
    try:
        if "/" in str(frac_str):
            n, d = str(frac_str).split("/")
            return round(1.0 + float(n) / float(d), 3)
        return round(float(frac_str), 3)
    except Exception:
        return 0.0


async def inspect_live_matches_and_odds():
    print("=" * 85)
    print("1. PARTIDOS EN VIVO ACTUALMENTE Y AUDITORÍA DE MOMIOS EN VIVO (Q3 / Q4 / FT)")
    print("=" * 85)

    client = get_mobile_client()
    live_events = await client.get_live_events()
    print(f"Total partidos de baloncesto EN VIVO ahora mismo: {len(live_events)}")

    server_dates = set()
    q3_q4_events = []
    other_live_events = []

    for ev in live_events:
        eid = ev.get("id")
        st = ev.get("status") or {}
        desc = st.get("description", "")
        code = st.get("code", 0)
        tourn = (ev.get("tournament") or {}).get("name", "")
        cat = ((ev.get("tournament") or {}).get("category") or {}).get("name", "")
        lg = f"{cat} - {tourn}" if cat else tourn
        home = (ev.get("homeTeam") or {}).get("name", "?")
        away = (ev.get("awayTeam") or {}).get("name", "?")
        hs = (ev.get("homeScore") or {}).get("current", 0)
        as_ = (ev.get("awayScore") or {}).get("current", 0)
        start_ts = ev.get("startTimestamp", 0)
        if start_ts:
            dt_utc = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            server_dates.add(dt_utc[:10])
        else:
            dt_utc = "?"

        info = {
            "id": eid,
            "league": lg,
            "home": home,
            "away": away,
            "score": f"{hs}-{as_}",
            "status_desc": desc,
            "status_code": code,
            "start_utc": dt_utc,
            "homeScore": ev.get("homeScore") or {},
            "awayScore": ev.get("awayScore") or {},
        }
        if any(k in desc.upper() for k in ["3RD", "4TH", "Q3", "Q4", "HALFTIME", "PAUSE"]):
            q3_q4_events.append(info)
        else:
            other_live_events.append(info)

    print(f"Fechas UTC activas en el servidor SofaScore: {sorted(server_dates)}")
    print(f"\nPartidos en Q3 / Q4 / Descanso ({len(q3_q4_events)}):")
    for e in q3_q4_events:
        hs_d = e["homeScore"]
        as_d = e["awayScore"]
        q_str = " | ".join(
            f"Q{i}:{hs_d.get(f'period{i}','-')}-{as_d.get(f'period{i}','-')}"
            for i in range(1, 5)
            if f"period{i}" in hs_d or f"period{i}" in as_d
        )
        print(f"  - [{e['status_desc']}] ID={e['id']} | {e['league']} | {e['home']} vs {e['away']} ({e['score']}) [{q_str}]")

    print(f"\nOtros partidos en vivo (Q1/Q2/etc.) ({len(other_live_events)}):")
    for e in other_live_events[:15]:
        print(f"  - [{e['status_desc']}] ID={e['id']} | {e['league']} | {e['home']} vs {e['away']} ({e['score']})")

    # Consultar momios en todos los partidos de Q3/Q4 y una muestra de otros en vivo
    probe_list = q3_q4_events + other_live_events[:10]
    print(f"\nConsultando endpoints de momios (/event/{{id}}/odds/1/all) en {len(probe_list)} partidos en vivo...")

    for e in probe_list[:18]:
        eid = e["id"]
        await asyncio.sleep(0.5)
        r_odds = await client.request("GET", f"event/{eid}/odds/1/all")
        st_code = r_odds.status_code if r_odds else 0
        if st_code == 200:
            data = r_odds.json()
            markets = data.get("markets", [])
            print(f"\n  >>> ID={eid} [{e['status_desc']}] {e['league']}: {e['home']} vs {e['away']} ({e['score']}) -> {len(markets)} mercados:")
            for m in markets:
                m_name = m.get("marketName", "")
                m_group = m.get("marketGroup", "")
                m_period = m.get("marketPeriod", "")
                is_live = m.get("isLive", False)
                suspended = m.get("isSuspended", False)
                line = m.get("choiceGroup", "")
                choices = m.get("choices", [])
                ch_str = ", ".join(
                    f"{c.get('name')}: {frac_to_dec(c.get('fractionalValue'))} (ini:{frac_to_dec(c.get('initialFractionalValue'))})"
                    for c in choices
                )
                print(
                    f"      * [{ 'LIVE' if is_live else 'PRE' }{'/SUSP' if suspended else ''}] "
                    f"name='{m_name}' | group='{m_group}' | period='{m_period}' | line='{line}' -> {ch_str}"
                )
        else:
            print(f"  - ID={eid} [{e['status_desc']}] {e['league']} -> Sin momios (HTTP {st_code})")


def check_db_recent_dates():
    print("\n" + "=" * 85)
    print("2. ÚLTIMAS FECHAS EN matches.db (MODO SOLO LECTURA)")
    print("=" * 85)
    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    rows = conn.execute("""
        SELECT substr(date, 1, 10) as d, COUNT(*) as n
        FROM matches
        WHERE date >= '2026-03-01' OR date >= '2025-03-01'
        GROUP BY d
        ORDER BY d DESC
        LIMIT 25
    """).fetchall()
    for d, n in rows:
        print(f"  {d}: {n} partidos en matches.db")
    conn.close()


if __name__ == "__main__":
    check_db_recent_dates()
    asyncio.run(inspect_live_matches_and_odds())
