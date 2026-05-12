"""League policy tiers for m27_v1 based on current ROI diagnostics."""

from __future__ import annotations


def normalize_league_name(name: str | None) -> str:
    return " ".join(str(name or "").split())


BLACKLIST = frozenset(
    {
        normalize_league_name("Liga 1 Masculin, Faza II"),
        normalize_league_name("Germany BBL"),
        normalize_league_name("EYBL U17, CHALLENGE CUP"),
    }
)

WATCHLIST = frozenset(
    {
        normalize_league_name("CIBACOPA , Segunda Vuelta"),
        normalize_league_name("LMB, Apertura"),
        normalize_league_name("Colombia LPB"),
        normalize_league_name("France Pro A"),
        normalize_league_name("Élite 2"),
        normalize_league_name("Meridianbet KLS"),
        normalize_league_name("EYBL U20, CHALLENGE CUP"),
        normalize_league_name("Liga Nacional Femenina Chile, Fase regular"),
    }
)

WHITELIST_SOFT = frozenset(
    {
        normalize_league_name("B1 League"),
        normalize_league_name("BNXT League"),
        normalize_league_name("Puerto Rico BSN"),
        normalize_league_name("Israeli National League Basketball"),
        normalize_league_name("China CBA"),
        normalize_league_name("Serie A2, Women, Playoffs"),
        normalize_league_name("LF Challenge, Regular Season"),
    }
)


def get_tier(name: str | None) -> str:
    normalized = normalize_league_name(name)
    if normalized in BLACKLIST:
        return "blacklist"
    if normalized in WATCHLIST:
        return "watchlist"
    if normalized in WHITELIST_SOFT:
        return "whitelist_soft"
    return "unclassified"


def get_action(name: str | None) -> str:
    tier = get_tier(name)
    if tier == "blacklist":
        return "bloquear"
    if tier == "watchlist":
        return "vigilar"
    if tier == "whitelist_soft":
        return "preferir"
    return "sin_regla"


def is_blacklisted(name: str | None) -> bool:
    return get_tier(name) == "blacklist"