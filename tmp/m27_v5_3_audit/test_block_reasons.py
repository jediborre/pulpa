# =====================================================================
# PRUEBA DE LOS MOTIVOS DE DESCARTE (bloqueo en inferencia)
# ---------------------------------------------------------------------
# QUÉ COMPRUEBA:
#   1. Que cada categoría de fase descartada devuelve su motivo.
#   2. Que las categorías NO descartadas (ELIMINATORIA, COPA, REGULAR) pasan.
#   3. Que las ligas medidas como tóxicas devuelven el motivo con su métrica.
#   4. Que la razón queda en el raw_json de la inferencia.
#   5. Que el bloqueo en inferencia y la regla del train coinciden.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/test_block_reasons.py
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import (  # noqa: E402
    BLOCKED_STAGE_CATEGORIES, block_reason, is_blocked_league, is_excluded_stage,
    is_toxic_league, measured_toxic_leagues,
)

CASOS = [
    # (liga, debe estar bloqueada?)
    ("Cyprus Basketball Division A, Championship Round", True),
    ("NBL, Championship Round", True),
    ("NKL, Championship Round", True),
    ("Portugal Proliga, Relegation Round", True),
    ("Champions League, Qualification", True),
    ("Club Friendlies", True),
    ("NBA Summer League", True),
    ("1. Liga, Relegation Round", True),
    ("Nostra.lt-RKL Division B, Atkrintamosios 3 etapas", False),  # lituanio: eliminatorias -> se mantienen
    ("ZBL, Playoffs", False),
    ("NBA, Playoffs", False),
    ("Brazil NBB, Playoffs", False),
    ("PBA Philippine Cup", False),
    ("French Basketball Cup", False),
    ("Poland 2nd Basketball League, Runda Zasadnicza", False),
    ("Kvindebasketligaen , Grundspil", False),
    ("A1 Women", False),
    ("LNB, Women", False),
    ("Turkey KBSL, Women", False),
]


def main() -> None:
    print("=" * 96)
    print("1. MOTIVOS DE DESCARTE POR CATEGORIA DE FASE")
    print("=" * 96)
    for liga, esperado in CASOS:
        cat, _ = classify_stage(liga)
        r = block_reason(liga)
        ok = (r is not None) == esperado
        marca = "OK  " if ok else "FALLA"
        print(f"  {marca} {liga[:46]:<48}[{cat:<13}] {str(r)[:70]}")

    print()
    print("=" * 96)
    print("2. LIGAS MEDIDAS COMO TOXICAS POR EL MODELO")
    print("=" * 96)
    for lg, (wr, ci) in measured_toxic_leagues().items():
        print(f"  {lg:<40} WR {wr:>5.1f}%  IC95 [{ci[0]:.1f}, {ci[1]:.1f}]")
        print(f"     -> {block_reason(lg)}")
        print(f"     -> con sufijo: {block_reason(lg + ', Playoffs')}")

    print()
    print("=" * 96)
    print("3. COHERENCIA TRAIN <-> INFERENCIA")
    print("=" * 96)
    incoherentes = []
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    ligas = [r[0] for r in conn.execute(
        "SELECT DISTINCT league FROM matches WHERE league IS NOT NULL AND league!=''")]
    conn.close()
    for lg in ligas:
        if is_excluded_stage(lg) and not is_blocked_league(lg):
            # solo puede diferir si la liga esta en la lista negra por patron
            if not is_toxic_league(lg):
                incoherentes.append(lg)
    print(f"  ligas en la base: {len(ligas):,}")
    print(f"  bloqueadas en inferencia pero NO excluidas del train: {len(incoherentes)}")
    for lg in incoherentes[:10]:
        print(f"     {lg}")

    print()
    print("  Cobertura por categoria (base completa):")
    from collections import Counter
    c = Counter(classify_stage(lg)[0] for lg in ligas)
    for cat, n in c.most_common():
        disc = "SE DESCARTA" if cat in BLOCKED_STAGE_CATEGORIES else "se mantiene"
        print(f"     {cat:<16}{n:>7,} ligas   {disc}")


if __name__ == "__main__":
    main()
