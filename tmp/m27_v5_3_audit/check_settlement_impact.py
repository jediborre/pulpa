# =====================================================================
# 1) ¿La divergencia legacy/v2 ya liquidó apuestas MAL?
# 2) ¿La API sirve los cuartos que faltan en los 2.876 partidos de 2026?
# =====================================================================

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


block("1) LIQUIDACIONES AFECTADAS POR LA DIVERGENCIA")
conn.execute("DROP TABLE IF EXISTS temp._pivot")
conn.execute("""
CREATE TEMP TABLE _pivot AS
SELECT q.match_id,
  MAX(CASE WHEN q.quarter='Q1' THEN q.home END) q1h, MAX(CASE WHEN q.quarter='Q1' THEN q.away END) q1a,
  MAX(CASE WHEN q.quarter='Q2' THEN q.home END) q2h, MAX(CASE WHEN q.quarter='Q2' THEN q.away END) q2a,
  MAX(CASE WHEN q.quarter='Q3' THEN q.home END) q3h, MAX(CASE WHEN q.quarter='Q3' THEN q.away END) q3a,
  MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4h, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4a
FROM quarter_scores q GROUP BY q.match_id""")
rows = list(conn.execute(
    "SELECT p.*, v.q4_home vq4h, v.q4_away vq4a FROM _pivot p JOIN quarter_scores_v2 v "
    "ON v.match_id=p.match_id"))
div = [r for r in rows if r["q4h"] is not None and r["vq4h"] is not None
       and (r["q4h"], r["q4a"]) != (r["vq4h"], r["vq4a"])]
print(f"  partidos con Q4 distinto entre legacy y v2: {len(div):,}")
bets = {r["match_id"] for r in conn.execute(
    "SELECT DISTINCT match_id FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3'")}
bets2 = {r["match_id"] for r in conn.execute(
    "SELECT DISTINCT match_id FROM bet_monitor_log_v2")}
afectadas = [r for r in div if r["match_id"] in bets or r["match_id"] in bets2]
print(f"  de esos, con apuesta registrada (v2 o v3): {len(afectadas):,}")
for r in afectadas[:10]:
    print(f"     {r['match_id']}: legacy Q4={r['q4h']}-{r['q4a']} | v2 Q4={r['vq4h']}-{r['vq4a']}"
          + ("  <-- cambia el ganador" if (r["q4h"] > r["q4a"]) != (r["vq4h"] > r["vq4a"]) else ""))
    res = conn.execute("SELECT model_version, signal_type, picked_side, result FROM bet_monitor_log_v3 "
                        "WHERE match_id=?", (r["match_id"],)).fetchall()
    for x in res:
        print(f"        apuesta {dict(x)}")
conn.close()


async def main() -> None:
    block("2) ¿LA API SIRVE LOS CUARTOS QUE FALTAN? (muestra de los 2.876 de 2026)")
    from monitor_v3.core.mobile_client import get_mobile_client
    c = sqlite3.connect(str(ROOT / "matches.db"))
    c.row_factory = sqlite3.Row
    muestras = [r["match_id"] for r in c.execute(
        "SELECT g.match_id FROM (SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)<4) g "
        "JOIN matches m ON m.match_id=g.match_id ORDER BY m.date DESC LIMIT 5")]
    c.close()
    client = get_mobile_client()
    for mid in muestras:
        try:
            resp = await client.request("GET", f"event/{mid}/statistics")
        except Exception as exc:
            print(f"   {mid}: ERROR {exc}")
            continue
        if not isinstance(resp, dict):
            print(f"   {mid}: respuesta vacia")
            continue
        stats = resp.get("statistics") or []
        periods = [s for s in stats if str(s.get("period") or "").startswith("Q")]
        det = ", ".join(f"{s.get('period')}:{s.get('home')}-{s.get('away')}" for s in periods)
        print(f"   {mid}: periodos en la API -> {det or '(ninguno)'}")


if __name__ == "__main__":
    asyncio.run(main())
