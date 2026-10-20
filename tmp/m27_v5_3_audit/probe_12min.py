# =====================================================================
# SONDEO: ¿estas ligas son de 12 minutos? (period_length de la API)
# ---------------------------------------------------------------------
# Candidatos que la firma de anotacion (Q4 medio) sugiere 12 min pero que
# `leagues_classification` tiene marcadas como q_dur=10. Si lo son, el
# modelo m27_v5_3 las aceptaria en inferencia con priors de 10 min.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/probe_12min.py
# =====================================================================

import asyncio
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CANDIDATOS = [
    "The Basketball League, Regular Season",
    "Chinese Taipei P League",
    "Chinese Taipei T1 League",
    "Super 20, Playoffs",
    "NBL1, North",
    "CIBACOPA , Segunda Vuelta",
]
CONTROL = ["NBA", "Liga ACB", "Turkey KBSL"]


async def main() -> None:
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row
    from monitor_v3.core.mobile_client import get_mobile_client

    client = get_mobile_client()
    for league in CANDIDATOS + CONTROL:
        mids = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches WHERE league = ? AND status_type='finished' "
            "ORDER BY date DESC LIMIT 4", (league,)).fetchall()]
        if not mids:
            print(f"  {league[:50]:<52} (sin partidos en la BD)")
            continue
        pl = None
        mid_usado = None
        for mid in mids:
            try:
                data = await client.fetch_full_match(mid)
            except Exception:
                continue
            m = (data or {}).get("match") or {}
            if m.get("period_length"):
                pl = m.get("period_length")
                mid_usado = mid
                break
        if pl is None:
            print(f"  {league[:50]:<52} sin dato de period_length en {len(mids)} partido(s) "
                  f"probados: {mids[:2]}")
            continue
        print(f"  {league[:50]:<52} period_length={pl} (match {mid_usado})  "
              f"-> {'12 MIN' if pl == 720 else ('10 MIN' if pl == 600 else '?')}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
