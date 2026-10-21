# =====================================================================
# BACKFILL SELECTIVO DE ESTADISTICAS (FALTAS, TIROS LIBRES) POR CUARTO
# ---------------------------------------------------------------------
# 1) Cuantos partidos hay en las ligas "rentables" (calibracion del modelo)
#    -> cuantos requests costaria el backfill
# 2) ¿Que cobertura real tiene el endpoint 'statistics' en esas ligas?
# 3) Que的事件 hay disponibles para un detector (por cuarto)
# =====================================================================

import asyncio
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

cal = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" /
                  "league_calibration.json").read_text(encoding="utf-8"))
print("1) LIGAS DE LA CALIBRACION (41) — las candidatas a 'rentables':")
print(f"   {'liga':<46}{'partidos':>10}{'bets':>7}{'WR':>8}{'ROI140':>9}{'bloqueo':>10}")
rentables = []
for c in sorted(cal, key=lambda x: -x.get("bets", 0)):
    wr = c.get("wr") or 0
    ok = wr >= 71.4 and c.get("bets", 0) >= 30
    if ok:
        rentables.append(c)
    print(f"   {c['league'][:45]:<46}{c.get('matches', 0):>10,}{c.get('bets', 0):>7}"
          f"{wr:>7.1f}%{c.get('roi_140', 0):>8.1f}%{str(c.get('block_candidate')):>10}")
tot_matches = sum(c.get("matches", 0) for c in rentables)
tot_bets = sum(c.get("bets", 0) for c in rentables)
print(f"\n   ligas por encima del equilibrio (71.4 % @1.40) con >=30 bets: {len(rentables)}")
print(f"   partidos acumulados en esas ligas: {tot_matches:,} | bets historicos: {tot_bets:,}")
print(f"   backfill a 1 req/s (statistics): {tot_matches / 3600:.1f} h")
print(f"   backfill a 3 req/s:              {tot_matches / 3 / 3600:.1f} h")
print(f"   (referencia: TODO el dataset elegible son 184.206 partidos = {184206 / 3600:.0f} h)")

print()
print("2) ¿CUANTOS PARTIDOS HAY REALMENTE EN ESAS LIGAS EN LA BASE?")
n_real = 0
for c in rentables:
    n_real += conn.execute("SELECT COUNT(*) n FROM matches WHERE league=? AND status_type='finished'",
                            (c["league"],)).fetchone()["n"]
print(f"   partidos en la BD de esas ligas: {n_real:,} -> {n_real / 3600:.1f} h a 1 req/s")


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()
    print()
    print("3) COBERTURA REAL DEL ENDPOINT 'statistics' EN ESAS LIGAS (muestra):")
    print(f"   {'liga':<34}{'año':>6}{'con stats':>11}{'con faltas':>12}{'periodos':>20}")
    total = con_stats = con_faltas = 0
    t0 = time.time()
    muestras = 0
    for c in rentables[:8]:
        liga = c["league"]
        for yr in ("2025", "2024", "2023"):
            rows = conn.execute("""
                SELECT match_id FROM matches WHERE league=? AND status_type='finished'
                  AND substr(date,1,4)=? ORDER BY date DESC LIMIT 2""", (liga, yr)).fetchall()
            for r in rows:
                try:
                    resp = await client.request("GET", f"event/{r['match_id']}/statistics")
                    resp = resp.json() if hasattr(resp, "json") else resp
                except Exception:
                    total += 1
                    continue
                total += 1
                stats = (resp or {}).get("statistics") or []
                periods = [s.get("period") for s in stats]
                tiene_f = any(
                    any(str(it.get("name", "")).lower() == "fouls"
                        for it in (g.get("statisticsItems") or []))
                    for s in stats for g in (s.get("groups") or []))
                if stats:
                    con_stats += 1
                if tiene_f:
                    con_faltas += 1
                muestras += 1
                print(f"   {liga[:33]:<34}{yr:>6}{'si' if stats else 'NO':>11}"
                      f"{'si' if tiene_f else 'NO':>12}{str(periods)[:19]:>20}")
                if muestras >= 24:
                    break
            if muestras >= 24:
                break
        if muestras >= 24:
            break
    dt = time.time() - t0
    print(f"\n   muestra: {total} partidos | con estadisticas: {con_stats} "
          f"({con_stats * 100 / max(total, 1):.0f} %) | con FALTAS: {con_faltas} "
          f"({con_faltas * 100 / max(total, 1):.0f} %)")
    print(f"   tiempo: {dt:.1f}s para {total} requests -> {total / dt:.2f} req/s")

    print()
    print("4) QUE EVENTOS HAY POR CUARTO (para un detector):")
    r = conn.execute("SELECT match_id FROM matches WHERE status_type='finished' "
                     "AND date >= '2026-01-01' LIMIT 1").fetchone()
    if r:
        resp = await client.request("GET", f"event/{r['match_id']}/statistics")
        resp = resp.json() if hasattr(resp, "json") else resp
        for s in (resp or {}).get("statistics") or []:
            print(f"   periodo {s.get('period')}:")
            for g in s.get("groups") or []:
                items = {it.get("name"): it.get("value") for it in (g.get("statisticsItems") or [])}
                print(f"      {g.get('groupName')}: {items}")


asyncio.run(main())
conn.close()
