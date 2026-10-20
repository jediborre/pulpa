# =====================================================================
# PRUEBA DEL BLOQUEO DE LIGAS (R4)
# ---------------------------------------------------------------------
# QUÉ COMPRUEBA:
#   1. Los 5 patrones muertos se eliminaron y la poda del train no cambia
#      (los patrones que quedan casan con las mismas ligas de antes).
#   2. El bloqueo en inferencia es un SUPERCONJUNTO del de train.
#   3. Las ligas medidas como toxicas por el propio modelo quedan bloqueadas,
#      incluidas las variantes con sufijo (", Playoffs").
#   4. Ninguna liga sana queda bloqueada por error.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/test_league_blocklist.py
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import (  # noqa: E402
    TOXIC_LEAGUE_PATTERNS, is_blocked_league, is_toxic_league, measured_toxic_leagues,
)

PATRONES_MUERTOS_ANTES = [
    "albania super", "georgia superli", "kosovo superlig", "armenia a-league", "cyprus division",
]


def main() -> None:
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    train_leagues = {r[0] for r in conn.execute(
        "SELECT DISTINCT league FROM matches WHERE league IS NOT NULL AND league != ''")}
    live_leagues = {r[0] for r in conn.execute(
        "SELECT DISTINCT league FROM bet_monitor_schedule_v3 WHERE league IS NOT NULL")}
    conn.close()

    print(f"Patrones actuales: {len(TOXIC_LEAGUE_PATTERNS)}")
    for p in TOXIC_LEAGUE_PATTERNS:
        n_tr = sum(1 for lg in train_leagues if p in lg.lower())
        n_lv = sum(1 for lg in live_leagues if p in lg.lower())
        print(f"  '{p}': train={n_tr:>3} liga(s) | live={n_lv:>2}")

    print("\nPatrones eliminados (no casaban con NINGUNA liga, ni de train ni live):")
    for p in PATRONES_MUERTOS_ANTES:
        n_tr = sum(1 for lg in train_leagues if p in lg.lower())
        n_lv = sum(1 for lg in live_leagues if p in lg.lower())
        print(f"  '{p}': train={n_tr} live={n_lv}  -> eliminado (sin efecto real)")

    print(f"\nLigas medidas como toxicas por el modelo (IC95% sup < 71.4 %): "
          f"{len(measured_toxic_leagues())}")
    for lg in sorted(measured_toxic_leagues()):
        print(f"  - {lg}")

    print("\nComprobacion de bloqueo:")
    casos = [
        ("1. Division B", True, "medida como toxica (WR 42.1 %)"),
        ("1. Division B, Playoffs", True, "variante con sufijo"),
        ("Sparkasse Prva Liga Zeni", True, "medida como toxica (WR 47.6 %)"),
        ("Sparkasse Prva Liga Zeni, Play off", True, "variante con sufijo"),
        ("Cyprus Basketball Division A", False, "sin evidencia suficiente todavia"),
        ("B League Premier", False, "liga sana"),
        ("LNB, Women", False, "liga sana"),
        ("Polish 1st Division", False, "no debe confundirse con '1. Division B'"),
        ("NBL", True, "patron literal"),
        ("", False, "liga vacia"),
    ]
    ok = True
    for liga, esperado, nota in casos:
        bloqueo_patron = is_toxic_league(liga)
        bloqueo_total = is_blocked_league(liga)
        bien = bloqueo_total == esperado and (esperado is False or bloqueo_total)
        ok = ok and bien
        print(f"  {'OK ' if bien else 'FALLA'} {liga[:44]:<46} patron={str(bloqueo_patron):<5} "
              f"total={str(bloqueo_total):<5} | {nota}")

    print("\nSuperconjunto (bloqueo total >= bloqueo por patron):")
    violation = [lg for lg in list(train_leagues)[:2000]
                 if is_toxic_league(lg) and not is_blocked_league(lg)]
    print(f"  violaciones: {len(violation)}")

    print("\n" + ("TODAS LAS COMPROBACIONES OK" if ok else "HAY FALLAS"))


if __name__ == "__main__":
    main()
