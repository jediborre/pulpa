# =====================================================================
# SELECCION CANONICA DE ESTADISTICAS DE EQUIPO (por cuarto)
# ---------------------------------------------------------------------
# La misma lista se usa en:
#   - la captura en vivo del monitor (live_snapshots_v3.team_stats_json)
#   - el backfill historico (team_stats_v3)
# para que la feature en vivo y la del entrenamiento sean EXACTAMENTE la
# misma (paridad).
#
# Son las stats que el endpoint /event/{id}/statistics devuelve por periodo
# (1Q, 2Q, 3Q, 4Q, ALL). El agregado 'ALL' solo se conoce al final del
# partido: sirve para etiquetas, no como feature en vivo.
# =====================================================================

from __future__ import annotations

from typing import Any

# Periodos que se guardan (el cuarto en curso se anade aparte si aplica)
PERIODOS = ("Q1", "Q2", "Q3", "Q4", "ALL")

# Stats relevantes para el detector de momentum (con y sin mercado).
# Los nombres cubren las dos fuentes: /event/{id}/statistics (agregado por
# periodo) e incidents (reconstruccion por cuarto, que anade los tiros
# libres, los tiempos muertos y las perdidas por cuarto).
STATS_INTERESANTES = (
    "Fouls",
    "Timeouts",
    "Turnovers",
    "Points",
    "Free throws",
    "Free throws made",
    "2 pointers",
    "2 pointers made",
    "3 pointers",
    "3 pointers made",
    "Field goals",
    "Rebounds",
    "Offensive rebounds",
    "Assists",
    "Steals",
    "Blocks",
    "Time spent in lead",
    "Biggest lead",
    "Lead changes",
    "Max points in a row",
)

# El endpoint /statistics usa "1Q" y el reconstructor de incidents usa "Q1".
_ALIAS_PERIODO = {
    "1Q": "Q1", "2Q": "Q2", "3Q": "Q3", "4Q": "Q4",
    "1ST QUARTER": "Q1", "2ND QUARTER": "Q2", "3RD QUARTER": "Q3", "4TH QUARTER": "Q4",
    "1ST PERIOD": "Q1", "2ND PERIOD": "Q2", "3RD PERIOD": "Q3", "4TH PERIOD": "Q4",
    "ALL": "ALL",
}


def normalizar_periodo(periodo: str | None) -> str:
    """Lleva cualquier nombre de periodo al canonico Q1..Q4 / ALL."""
    p = str(periodo or "ALL").strip().upper()
    return _ALIAS_PERIODO.get(p, p)


def compact_team_stats(rows: list[dict[str, Any]] | None,
                       periodos: tuple[str, ...] = PERIODOS) -> list[list]:
    """Convierte las filas de _parse_team_statistics / _parse_period_stats en
    una lista compacta [periodo, stat, local, visitante].

    Acepta las dos fuentes (el agregado por periodo del endpoint /statistics y
    la reconstruccion por cuarto desde incidents) y normaliza el nombre del
    periodo, de modo que la fila capturada en vivo y la del backfill
    historico son EXACTAMENTE la misma.
    """
    if not rows:
        return []
    out: list[list] = []
    vistos: set[tuple[str, str]] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        periodo = normalizar_periodo(r.get("period"))
        if periodo not in periodos:
            continue
        stat = str(r.get("stat_name") or "")
        if stat not in STATS_INTERESANTES:
            continue
        clave = (periodo, stat)
        if clave in vistos:
            continue
        vistos.add(clave)
        out.append([periodo, stat, r.get("home_value"), r.get("away_value")])
    return out


def stats_to_dicts(filas: list[list] | None) -> list[dict[str, Any]]:
    """Convierte la lista compacta de vuelta a filas con clave (para entrenar)."""
    out: list[dict[str, Any]] = []
    for f in filas or []:
        if len(f) < 4:
            continue
        out.append({"period": f[0], "stat_name": f[1],
                    "home_value": f[2], "away_value": f[3]})
    return out
