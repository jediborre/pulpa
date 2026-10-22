# =====================================================================
# ¿ALGUNA APUESTA VIVA CAE EN UNA FASE QUE EL FILTRO DEBERIA BLOQUEAR?
#   (la liga del partido en vivo se clasifica con el MISMO clasificador
#    que uso el train -> si dice "bloqueada" y se aposto, hay un bug)
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import (  # noqa: E402
    BLOCKED_STAGE_CATEGORIES, block_reason, is_blocked_league,
)

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}

print("1) TODAS LAS APUESTAS VIVAS DE m27_v5_3, CON LA FASE DETECTADA:")
print(f"   {'fecha':<12}{'liga':<44}{'fase':<18}{'categoria':<18}{'apuesta':>7}")
fallos = 0
for r in conn.execute("""
    SELECT match_id, picked_side, confidence, result, created_at, signal_type
    FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
    ORDER BY created_at"""):
    lg = liga_de.get(str(r["match_id"]), "?")
    cat, motivo = classify_stage(lg)
    bloq = is_blocked_league(lg)
    if bloq:
        fallos += 1
    print(f"   {r['created_at'][:10]:<12}{str(lg)[:43]:<44}{str(motivo)[:17]:<18}"
          f"{cat[:17]:<18}{r['result']:>7}"
          + ("   <== DEBERIA HABER SIDO BLOQUEADA" if bloq else ""))

print()
print(f"   apuestas en fases bloqueadas: {fallos}")

print()
print("2) ¿QUE HACE EL CLASIFICADOR CON LOS NOMBRES POLACOS DE FASE?")
for nombre in ("Poland 2nd Basketball League, Runda Zasadnicza",
               "Poland 2nd Basketball League",
               "1. A SKL", "1. SKL, Women", "Prva Liga",
               "2nd League", "LBP, Group B", "Czech NBL",
               "BNXT League", "Basketligaen", "Champions League, Group C",
               "EuroCup, Women, Group B", "LNB, Women",
               "LMB, Clausura", "Mexico LNBP, Playoffs"):
    cat, motivo = classify_stage(nombre)
    marca = "BLOQUEADA" if cat in BLOCKED_STAGE_CATEGORIES else "-"
    print(f"   {nombre[:48]:<50} {cat:<18} {marca:<10} {motivo[:40]}")

print()
print("3) ¿QUE PARTIDOS HAY EN LA BASE CON 'Runda Zasadnicza' Y COMO LOS CLASIFICA?")
for r in conn.execute("""SELECT league, COUNT(*) n FROM matches
                         WHERE league LIKE '%Zasadnicza%' GROUP BY league"""):
    cat, motivo = classify_stage(r["league"])
    print(f"   {r['league'][:50]:<52} n={r['n']:>5} | {cat:<16} {motivo[:40]}"
          + ("  <== BLOQUEADA" if cat in BLOCKED_STAGE_CATEGORIES else ""))

print()
print("4) ¿CUANTOS PARTIDOS HAY EN LA BASE POR CATEGORIA DE FASE?")
from collections import Counter  # noqa: E402
cnt = Counter()
for lg in liga_de.values():
    cat, _ = classify_stage(lg)
    cnt[cat] += 1
for cat, n in cnt.most_common():
    marca = " <-- BLOQUEADA" if cat in BLOCKED_STAGE_CATEGORIES else ""
    print(f"   {cat:<20} {n:>7,}{marca}")
conn.close()
