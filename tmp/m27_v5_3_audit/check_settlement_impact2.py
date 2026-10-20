# =====================================================================
# 1) ¿La API (endpoint del EVENTO) sirve los cuartos que faltan?
# 2) ¿Cuántas liquidaciones se hicieron con un Q4 PROVISIONAL?
#    (la v2 guarda lo que el monitor vio en vivo; la legacy es el dato final)
# =====================================================================

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row

    block("1) ¿LA API SIRVE LOS CUARTOS QUE FALTAN? (endpoint del evento)")
    muestras = [r["match_id"] for r in conn.execute(
        "SELECT g.match_id FROM (SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)<4) g "
        "JOIN matches m ON m.match_id=g.match_id ORDER BY m.date DESC LIMIT 6")]
    client = get_mobile_client()
    for mid in muestras:
        try:
            resp = await client.request("GET", f"event/{mid}")
            if hasattr(resp, "json"):
                resp = resp.json()
        except Exception as exc:
            print(f"   {mid}: ERROR {exc}")
            continue
        if not isinstance(resp, dict):
            print(f"   {mid}: respuesta no-dict ({type(resp).__name__})")
            continue
        ev = (resp or {}).get("event") or {}
        if not ev:
            print(f"   {mid}: sin 'event' en la respuesta (claves: {list(resp)[:6]})")
            continue
        hs = ev.get("homeScore") or {}
        per = hs.get("periods") or []
        det = ", ".join(f"P{p.get('period')}:{p.get('home')}-{p.get('away')}" for p in per)
        print(f"   {mid}: periodos -> {det or '(ninguno)'}")
        if not per:
            print(f"        homeScore crudo: {json.dumps(hs, ensure_ascii=False)[:300]}")

    block("2) LIQUIDACIONES HECHAS CON UN Q4 PROVISIONAL")
    conn.execute("DROP TABLE IF EXISTS temp._pivot")
    conn.execute("""
    CREATE TEMP TABLE _pivot AS
    SELECT q.match_id,
      MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4h, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4a
    FROM quarter_scores q GROUP BY q.match_id""")
    div = list(conn.execute(
        "SELECT p.match_id, p.q4h, p.q4a, v.q4_home vq4h, v.q4_away vq4a FROM _pivot p "
        "JOIN quarter_scores_v2 v ON v.match_id=p.match_id "
        "WHERE p.q4h IS NOT NULL AND v.q4_home IS NOT NULL AND (p.q4h<>v.q4_home OR p.q4a<>v.q4_away)"))
    print(f"  partidos con Q4 distinto (legacy vs v2): {len(div):,}")
    cambios_signo = [r for r in div if (r["q4h"] > r["q4a"]) != (r["vq4h"] > r["vq4a"])]
    print(f"  de los cuales cambia el GANADOR del Q4: {len(cambios_signo):,}")
    Bets = {}
    for t in ("bet_monitor_log_v3", "bet_monitor_log_v2"):
        for r in conn.execute(f"SELECT match_id, model_version, signal_type, picked_side, result FROM {t}"):
            Bets.setdefault(r["match_id"], []).append((t[-2:], dict(r)))
    afectadas = [r for r in div if r["match_id"] in Bets]
    print(f"  con apuestas registradas: {len(afectadas):,}")
    mal = 0
    for r in afectadas:
        for _, b in Bets[r["match_id"]]:
            if b["signal_type"] in ("NO_BET",) or not str(b["signal_type"]).endswith(("HOME", "AWAY")):
                continue
            if b["result"] in ("win", "loss"):
                # ¿el resultado guardado coincide con el Q4 legacy?
                pick_home = b["picked_side"] == "HOME"
                legacy_home_wins = r["q4h"] > r["q4a"]
                esperado = "win" if pick_home == legacy_home_wins else "loss"
                if r["q4h"] == r["q4a"]:
                    esperado = "push"
                if esperado != b["result"]:
                    mal += 1
                    print(f"     MAL LIQUIDADA {r['match_id']} {b['model_version']}: "
                          f"guardado={b['result']} | legacy Q4={r['q4h']}-{r['q4a']} "
                          f"(v2={r['vq4h']}-{r['vq4a']})")
    print(f"\n  liquidaciones que CAMBIARIAN con el dato definitivo (legacy): {mal}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
