"""
Clasificador de FASE DE COMPETICION (multilingue) y de MOTIVOS DE DESCARTE.

Por que existe:
  `leagues_classification.stage_detail` normaliza la fase, pero tiene agujeros:
  "Championship Round" cae en `Playoffs` en unas ligas (NBL, Basketligaen, NKL,
  Korisliiga, Portugal Proliga) y en `Regular Season` en otras (LNA, United
  League, Bulgaria NBL, Stoiximan GBL). Con 183 ligas que contienen
  "championship" en el nombre, la decision no puede depender de una heuristica
  de una sola fuente.

  Aqui se clasifica la fase con un diccionario de marcadores en multiples
  idiomas sobre el texto ORIGINAL de la fase (`stage`), y se usa
  `stage_detail` como segunda opinion. Cuando las dos fuentes no coinciden
  manda el marcador explicito, que es mas especifico.

Ademas centraliza los MOTIVOS de descarte (juvenil, college, amistoso,
eliminatoria, copa, 12 minutos, liga medida como toxica) para que el modelo
pueda decir POR QUE descarta un partido en vez de un PASS generico.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------
# Marcadores de fase por idioma. Se comparan en minusculas.
# --------------------------------------------------------------------------
ELIMINATORIA_MARKERS = [
    # en / es / pt / it / fr / de genericos
    "playoff", "play-off", "play off", "playoffs", "play-offs", "knockout",
    "knock-out", "knock out", "eliminatoria", "eliminatorias", "fase final",
    "final phase", "quarterfinal", "quarter-final", "quarter finals",
    "semifinal", "semi-final", "semi finals", "semis", "round of 16",
    "final four", "final 4", "final eight", "final 8", "final six", "final 6",
    "final 16", "elite 8", "superfinal", "super-final", "super final",
    "3rd place", "3rd place game", "bronze", "consolation", "first round knockout",
    "central playoffs", "central play-offs", "5th place playoff", "national finals",
    "1/8", "1/4", "1/2", "1/8 final", "1/4 final", "1/2 final",
    # es
    "cuartos de final", "cuartos final", "octavos de final", "semifinal",
    "fase eliminatoria", "ronda eliminatoria", "final nacional", "cuadrangular final",
    # pt
    "quartas de final", "oitavas de final", "meias-finais", "finais",
    "fase final", "play-offs", "final nacional", "3ª fase - final",
    "3ª fase - meias-finais", "disputa do 5º", "classificatória",
    # it
    "quarti", "semifinali", "finale", "finali", "play-off",
    # fr
    "phase finale", "quarts de finale", "demi-finales", "demies finales",
    "finale", "8e de finale", "16e de finale", "32e de finale",
    # de
    "viertelfinale", "halbfinale", "finale", "ausscheidungsspiele",
    # pl / cs / sk
    "ćwierćfinał", "ćwierćfinały", "półfinał", "półfinały", "finał",
    "semifinále", "finále", "čtvrtfinále", "čtvrtfinále", "play-offy",
    # hr / sr / ba
    "polufinale", "četvrtfinale", "četvrtfinala", "završni turnir", "zavrsni turnir",
    "poluzavršni", "izlučne utakmice", "izlucne utakmice", "plej-of", "play of",
    # lt
    "pusfinaliai", "ketvirtfinaliai", "atkrintamosios", "finalas", "įkrintamosios",
    # ru / bg
    "плей-офф", "плейофф", "полуфинал", "четвертьфинал", "финал",
    "финал 4-х", "заключителен", "хгас",
    # el
    "play off", "τελικός", "ημιτελικά", "προημιτελικά",
    # tr
    "play-off", "playoff maçları", "eleme", "final maçları",
    # hu / ro / bg / id / nl
    "play-off", "playoffuri", "döntő", "knockout", "play-off",
    # otros idiomas / variantes
    "klasman maçları", "półfinały", "finał", "fɪnal", "финал",
    "octavos", "quartas", "play offs", "playoffs women", "superfinals",
    # "final"/"finals" sueltos: la final de un torneo. OJO: una final de
    # PROMOCION (p.ej. rumano "turneu final de promovare") la captura antes
    # la categoria PROMOCION, que se comprueba antes que esta.
    "final", "finals",
]

COPA_MARKERS = [
    "cup", "copa", "pokal", "coupe", "coppa", "kupa", "kup", "taça", "taca",
    "trofeo", "trophy", "supercopa", "supercup", "superkupa", "supercupës",
    "superkupë", "super cup", "super kupa", "korábbi", "kupası", "kupa e",
    "bba cup", "cupa", "poklop", "pokal kvinder", "pokal herrer", "copa a2",
    "moldova cup", "kupa e shqiperise", "kupa shqipërise", "κύπελλο",
    "kvarnfinale", "kop", "puchar", "cupa de", "beker", "coupe de",
]

# Segunda fase / ronda de campeonato con grupos: NO es eliminatoria.
SEGUNDA_FASE_MARKERS = [
    "championship round", "championship league", "main round", "main round,",
    "second phase", "second round", "2ª fase", "2a fase", "2. fase",
    "fase 2", "intergroup", "inter group", "inter-group", "group stage",
    "fase de grupos", "championship", "ronda de campeonato", "2. krug",
    "main round group",
    # pl: "runda zasadnicza" = la ronda decisiva (campeonato/promocion) con
    # grupo; es el equivalente de "championship round" en ingles.
    "runda zasadnicza", "runda zasadnicza", "runda de championship",
    "runda mistrzowska", "runda pierwsza",
]

# Promotion / relegation / playout: formato de supervivencia, no eliminatoria.
PROMOCION_MARKERS = [
    "promotion", "relegation", "playout", "play-out", "play out", "placement",
    "classificação", "classificacao", "classificação", "liga za obstanek",
    "konservecja", "1. krug", "2. krug", "sparreggio", "spareggio",
    "promotion round", "relegation round", "rel./prom", "rel/prom",
    "liga de acesso", "playout group", "zone de maintien", "torneo de promoción",
    "turnir za promociju", "përshkallë", "lūg", "izjas", "playout a",
    # ro: turneu/faza de promovare (final de promociune)
    "final de promovare", "turneu final de promovare", "faza de promovare",
    "faza de retrogradare", "seria de promovare",
    # de: Auf-/Absteiger (auf-/absteigerrunde)
    "aufsteiger", "absteiger", "aufsteigerrunde", "absteigerrunde",
    # es: fase de ascenso / de descenso
    "fase de ascenso", "fase de descenso", "ronda de ascenso", "ronda de descenso",
    # pl: runda awansu / spadkowa
    "runda awansu", "runda spadkowa", "play-off awansu",
]

# Eliminatorias de puesto (3-8, 9-12): no compiten por el titulo.
PUESTO_MARKERS = [
    "3rd place", "3rd place game", "bronze", "class. game", "class. games",
    "classification 5", "classification 7", "classification 9", "classification 11",
    "classification 13", "classification 15", "classificacao", "classificación",
    "consolation", "playoff maçları", "izlučne utakmice",
    "disputa do 5º", "5º lugar", "9-12", "5-8", "7-8", "5-6", "9-10", "11-12",
    "13-14", "15-16", "placement matches", "placement 1-4",
    "placement 5-8", "placement 9-12", "lugar", "colocação",
]

PRELIMINAR_MARKERS = [
    "qualifying round", "qualifying", "qualifier", "qualifiers", "qualification",
    "pre-qualifier", "pre-qualifiers", "pre-qual.", "preliminar", "prelimINARY",
    "pre-season", "preseason", "amistoso",
]

AMISTOSOS_MARKERS = [
    "friendly", "friendlies", "friendly game", "amistoso", "exhibition",
    "tournoi", "tour", "invitation", "summer league", "all star", "all-star",
    "olympic tournament", "tournament phase", "gby_lwmyt_gbrym",
]

# Marcadores de la FASE REGULAR: grupos, regiones, genero, temporada.
REGULAR_MARKERS = [
    "group", "groups", "south", "east", "west", "north", "norte", "sur", "sul",
    "nord", "sud", "oeste", "este", "seccion", "sección", "women", "men",
    "fase i", "fase ii", "1ª fase", "2ª fase", "primeira fase", "faza i",
    "faza ii", "runda", "grundspil", "reguliarus", "temporada", "split",
    "apertura", "clausura", "zk", "conference", "division", "k.k.m",
    "poule", "girone", "skupina", "grupa", "grupo", "grubu", "grup", "группа",
    "regular season", "regular", "temporada regular", "fase regular",
    "all", "aller", "retour", "round robin", "liga", "league", "sezonas",
    "etapa regular", "fase regular", "1. liga", "2. liga", "1. krug",
    "poule a", "poule b", "zona", "zone", "bucket", "sede", "juegos", "playoff group",
    "main round group", "first group", "second group", "islands group",
    "rhodes group", "group stage", "premier league", "super league", "championship league",
]

# Etiquetas que el clasificador de `leagues_classification` produce y que
# NORMALIZAN a un bucket. Se usan para comparar y detectar agujeros.
STAGE_DETAIL_BUCKET = {
    "playoffs": "ELIMINATORIA",
    "quarterfinal": "ELIMINATORIA",
    "final / championship": "ELIMINATORIA",
    "semifinal / final four": "ELIMINATORIA",
    "play-in": "PRELIMINAR",
    "cup stage": "COPA",
    "group stage": "SEGUNDA_FASE",
    "relegation / playout": "PROMOCION",
    "bronze / 3rd place": "PUESTO",
    "regular season": "REGULAR",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _has(text: str, markers: list[str]) -> bool:
    return any(m in text for m in markers)


def classify_stage(league: str, stage: str = "") -> tuple[str, str]:
    """(categoria, motivo legible) a partir del nombre de liga y su fase.

    Prioridad: eliminatoria > copa > segunda fase > promotion > puesto >
    preliminar > amistoso > regular.  Se mira primero en la FASE y, si no
    aparece, en el nombre completo de la liga (muchas ligas llevan la fase
    en el nombre: "Serie B, Playoffs", "Cyprus Basketball Division A,
    Championship Round").
    """
    base, _, rest = str(league or "").partition(",")
    fase = stage or rest
    txt_fase = _norm(fase)
    txt_base = _norm(base)
    txt_full = _norm(league)

    # PROMOCION va ANTES que ELIMINATORIA: una "final de promovare" (rumano),
    # una "final de ascenso" o una "runda de descenso" llevan la palabra
    # "final" pero son un formato de supervivencia, no una eliminatoria.
    for cat, markers, motivo in (
        ("PROMOCION", PROMOCION_MARKERS, "fase de promotion/relegation (formato de supervivencia)"),
        ("ELIMINATORIA", ELIMINATORIA_MARKERS, "fase eliminatoria (fases de eliminatorias y finales)"),
        ("COPA", COPA_MARKERS, "torneo de copa"),
        ("SEGUNDA_FASE", SEGUNDA_FASE_MARKERS, "segunda fase / ronda de campeonato con grupos"),
        ("PUESTO", PUESTO_MARKERS, "fase de colocacion (partidos por puesto, no por titulo)"),
        ("PRELIMINAR", PRELIMINAR_MARKERS, "fase preliminar / clasificatoria"),
        ("AMISTOSOS", AMISTOSOS_MARKERS, "torneo amistoso / invitacion"),
    ):
        if _has(txt_fase, markers):
            return cat, motivo
    for cat, markers, motivo in (
        ("PROMOCION", PROMOCION_MARKERS, "fase de promotion/relegation (formato de supervivencia)"),
        ("ELIMINATORIA", ELIMINATORIA_MARKERS, "fase eliminatoria (fases de eliminatorias y finales)"),
        ("COPA", COPA_MARKERS, "torneo de copa"),
        ("SEGUNDA_FASE", SEGUNDA_FASE_MARKERS, "segunda fase / ronda de campeonato con grupos"),
        ("PUESTO", PUESTO_MARKERS, "fase de colocacion (partidos por puesto, no por titulo)"),
        ("PRELIMINAR", PRELIMINAR_MARKERS, "fase preliminar / clasificatoria"),
        ("AMISTOSOS", AMISTOSOS_MARKERS, "torneo amistoso / invitacion"),
    ):
        if _has(txt_base, markers):
            return cat, motivo
    if not txt_fase and not _has(txt_full, REGULAR_MARKERS):
        return "REGULAR", "temporada regular (sin fase declarada)"
    if _has(txt_fase, REGULAR_MARKERS) or _has(txt_base, REGULAR_MARKERS):
        return "REGULAR", "temporada regular (grupo/region/genero)"
    return "REGULAR", "temporada regular"


def classify_stage_detail(detail: str) -> str:
    """Bucket equivalente de `leagues_classification.stage_detail`."""
    return STAGE_DETAIL_BUCKET.get(_norm(detail), "REGULAR")
