# =====================================================================
# IMPACTO DEL ARREGLO DEL CLASIFICADOR DE FASES
#   Compara la categoria de cada liga de la base con los marcadores
#   antiguos y los nuevos, y dice cuantas cambian y de que a que.
# =====================================================================

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common import competition_stage as CS  # noqa: E402
from models.m27_v5_3.features import BLOCKED_STAGE_CATEGORIES  # noqa: E402

# Listas ANTES del arreglo (para comparar el impacto).
CS.ELIMINATORIA_MARKERS_VIEJO = [m for m in CS.ELIMINATORIA_MARKERS if m not in ("final", "finals")]
CS.SEGUNDA_FASE_MARKERS_VIEJO = [m for m in CS.SEGUNDA_FASE_MARKERS
                                 if m not in ("runda zasadnicza", "runda de championship",
                                              "ruda mistrzowska", "runda pierwsza")]
CS.PROMOCION_MARKERS_VIEJO = [m for m in CS.PROMOCION_MARKERS
                               if not any(k in m for k in ("promovare", "aufsteiger", "absteiger",
                                                            "ascenso", "descenso", "awansu",
                                                            "spadkowa"))]


def clasificar_viejo(league: str) -> str:
    """Replica del clasificador ANTES del arreglo (orden y marcadores viejos)."""
    base, _, rest = str(league or "").partition(",")
    fase = rest
    txt_fase = CS._norm(fase)
    txt_base = CS._norm(base)
    txt_full = CS._norm(league)
    for cat, markers, _m in (
        ("ELIMINATORIA", CS.ELIMINATORIA_MARKERS_VIEJO, ""),
        ("COPA", CS.COPA_MARKERS, ""),
        ("SEGUNDA_FASE", CS.SEGUNDA_FASE_MARKERS_VIEJO, ""),
        ("PROMOCION", CS.PROMOCION_MARKERS_VIEJO, ""),
        ("PUESTO", CS.PUESTO_MARKERS, ""),
        ("PRELIMINAR", CS.PRELIMINAR_MARKERS, ""),
        ("AMISTOSOS", CS.AMISTOSOS_MARKERS, ""),
    ):
        if CS._has(txt_fase, markers):
            return cat
    for cat, markers, _m in (
        ("ELIMINATORIA", CS.ELIMINATORIA_MARKERS_VIEJO, ""),
        ("COPA", CS.COPA_MARKERS, ""),
        ("SEGUNDA_FASE", CS.SEGUNDA_FASE_MARKERS_VIEJO, ""),
        ("PROMOCION", CS.PROMOCION_MARKERS_VIEJO, ""),
        ("PUESTO", CS.PUESTO_MARKERS, ""),
        ("PRELIMINAR", CS.PRELIMINAR_MARKERS, ""),
        ("AMISTOSOS", CS.AMISTOSOS_MARKERS, ""),
    ):
        if CS._has(txt_base, markers):
            return cat
    return "REGULAR"


conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

cambios = Counter()
detalle = Counter()
partidos_cambio = Counter()
for r in conn.execute("SELECT league, COUNT(*) n FROM matches WHERE league IS NOT NULL GROUP BY league"):
    lg = r["league"]
    viejo = clasificar_viejo(lg)
    nuevo, _mot = CS.classify_stage(lg)
    if viejo != nuevo:
        cambios[(viejo, nuevo)] += r["n"]
        detalle[(viejo, nuevo, lg)] += r["n"]
    # el cambio de DECISION importa: bloqueado antes vs despues
    if (viejo in BLOCKED_STAGE_CATEGORIES) != (nuevo in BLOCKED_STAGE_CATEGORIES):
        partidos_cambio[(viejo, nuevo)] += r["n"]

print("1) CATEGORIAS QUE CAMBIAN (partidos afectados):")
for (viejo, nuevo), n in cambios.most_common():
    print(f"   {viejo:<15} -> {nuevo:<15} {n:>8,} partidos"
          + ("   <== CAMBIA LA DECISION (bloqueado <-> permitido)" if
             (viejo in BLOCKED_STAGE_CATEGORIES) != (nuevo in BLOCKED_STAGE_CATEGORIES) else ""))

print()
print("2) TOTAL QUE CAMBIA DE DECISION:")
tot = sum(partidos_cambio.values())
for (viejo, nuevo), n in partidos_cambio.most_common():
    print(f"   {viejo:<15} -> {nuevo:<15} {n:>8,}")
print(f"   TOTAL: {tot:,} partidos")

print()
print("3) TOP LIGAS QUE CAMBIAN:")
for (viejo, nuevo, lg), n in detalle.most_common(20):
    print(f"   {n:>7,} | {viejo:<12} -> {nuevo:<12} | {lg[:60]}")

print()
print("4) DISTRIBUCION FINAL DE CATEGORIAS (con el arreglo):")
cnt = Counter()
for r in conn.execute("SELECT league, COUNT(*) n FROM matches WHERE league IS NOT NULL GROUP BY league"):
    cat, _ = CS.classify_stage(r["league"])
    cnt[cat] += r["n"]
for cat, n in cnt.most_common():
    marca = "  <-- BLOQUEADA" if cat in BLOCKED_STAGE_CATEGORIES else ""
    print(f"   {cat:<15} {n:>8,}{marca}")
conn.close()
