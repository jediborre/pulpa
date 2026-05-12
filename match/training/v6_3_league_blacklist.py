"""V6.3 manual league blacklist.

Drop-in filter for both the ROI report and live inference.

Usage (inference):
    from training.v6_3_league_blacklist import V63LeagueBlacklist

    bl = V63LeagueBlacklist()
    blocked, reason = bl.is_blocked(league_name)
    if blocked:
        return {"skip": True, "reason": reason}
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Blacklist — exact league names (case/whitespace-insensitive match).
# ---------------------------------------------------------------------------
_RAW = """
SLB
NBA
LBF, Women
EYBL U17, SUPERFINAL - 02./05.04.2026 - Warsaw, Poland
NBL1 Central, Women
Liga 1 Masculin, Faza II
Serie A2, Women, Playoffs
Big V
Iceland Women's Basketball Premier League, Playoffs
Super League, Playoffs
Israeli Basketball Super League
Basketligaen, Championship round
Swedish Basketball League, Playoffs
PBA Commissioner's Cup
Superleague
1. Division A, Playoffs
1. Division, Women, Playoffs
Austrian Basketball Superliga, Championship Round
Extraliga, Women, Playoffs
Iceland Basketball Premier League, Playoffs
Latvia LBL, Playoffs
LNBM Mozzart, Playoffs
Poland 1st Division Basketball, Playoffs
Women's National Basketball League, Regular Season
Rapid League, Regular Season
ABA League 2, Playoffs
Austrian Women's Basketball Bundesliga, Playoffs
Champions League Asia-East, Group B
Korisliiga, Women, Playoffs
Liga e Parë
Portugal Proliga, Relegation Round
Super League
Superliga, Women
Swedish Basketball League Women, Playoffs
WNBA Preseason
1 Liga Kobiet, Ćwierćfinały
Croatian Premier League
United League, Playoffs
Korean Basketball League, Playoffs
NBL1, North
Basket Liga, Women, Playoffs
France LFB Women, Playoffs
Liga Nationala, Relegation Playoffs
Slovenian Second Basketball League, Četrtfinale
Swedish Basketball Superettan
Uruguay LUB, Championship Round
Zweite Liga, Playoffs
1. Liga, Relegation Round
1. SKL, Women, Playoffs
1st Division Women, Úrslitakeppni
Africa League, Group Kalahari
Austrian Basketball Superliga, Qualifying Round
Belgian Basketball 2nd Division, Playoffs
Campeonato Nacional da 1ª Divisao Feminina, 2ª Fase - Grupo Manutenção Norte
Champions League Asia-East, Group A
CN1 Basquetebol, 3ª Fase - Meias-Finais Norte
Cup, Women
Estonian National Women's Basketball League
European North Basketball League, Playoffs
LNB, Women, Playoffs
Poland 2nd Basketball League, Ćwierćfinały
Prvenstvo Hrvatske za dječake , Skupina 4
Prvenstvo Hrvatske za djevojčice , Poluzavršni turnir skupina A
Prvenstvo Hrvatske za djevojčice , Poluzavršni turnir skupina B
Rwanda Basketball League Women, Regular Season
Serie B, Play-in
Superleague, Rel/Prom Playoffs
Swedish Basketball Superettan, Playoffs
Switzerland LNA
United Cup, Knockout Stage
Azerbaijan Basketball League, PLAY-IN
Cyprus Basketball Division A, Championship Round
Emir Cup, Knock-out
Enovos League, Regular Season
France LF2, Women, Phase 1 - Poule A
I A MCKL, Playoffs
Premier League, Playoffs
Premier League, Women, Placement matches 9-12
Princ Caffe Superliga
Serie A2, Women, Group A
Slovenian Second Basketball League, Polfinale
Super League, Playout
Superleague, Playoffs
Top Division Women, Regular Season
1. Liga, Playoffs
1st Division, Championship round
A league , Playoffs
A1 Women, Playoffs
CBI U19 Feminino, Grupo F
Champions League Americas, Knockout stage
CIBACOPA , Primera Vuelta
Enovos League, Relegation Group
Estonian-Latvian Basketball League, Playoffs
Euroleague Women, Playoffs
Extraliga, Women, Placement 5-8
FIBA Europe Cup, Playoffs
France LFB Women, Relegation Round
Hungary Divison A Women, Placement Matches
Hungary Divison A Women, Playoffs
Leaders Cup
Liga Ouro, Quartas
Liga Señal Colombia de Baloncesto , I Fase - Eliminatoria
LNB, Women, Playouts
LSB - Estadual Amador, Classificação
National League, Relegation Round
Nostra.lt-RKL Division B, Ketvirtfinalia
Nostra.lt-RKL Division B, Pusfinaliai
Prva Liga, Playoffs
Prvenstvo Hrvatske za dječake , Skupina 2
Prvenstvo Hrvatske za dječake , Skupina 3
Prvenstvo RH za kadetkinje , Završni turnir
Super League, Placement 5-8
Swiss Basketball League Women, Playoffs
Tipos SBL
Youth Basketball Champions League, Group A
Youth Basketball Champions League, Group C
Youth Basketball Champions League, Semi-Finals 9-12
Zain Basketball League, Playoffs
Champions League, Playoffs
I A MCKL
Liga Košarkaškog saveza Herceg Bosne - Playoff
Liga Ouro, Classificação
NBA G League, Playoffs
Slovenian Second Basketball League, Finale
Superliga e Femrave
Youth Basketball Champions League, Classification 9-12
Serie A2, Women, Group B
Serie A2, Women, Playout
1 Liga Kobiet, O 3 miejsce
1 Liga Kobiet, Półfinały
1. Division, Women, Relegation Playoffs
A1 Women, Relegation Playoffs
Africa League, Group Sahara
Azerbaijan Basketball League, Final
Azerbaijan Basketball League, PLAY OFF 1/2
Azerbaijan Basketball League, Playoff
Basketligaen, Playoffs
CBI U19 Feminino, Grupo E
CibaCup, Playoffs
EGBL U19, Superfinal
Eurocup, Playoffs
Euroleague, Play-in
France Women's Basketball Cup
French Basketball Cup
Georgian cup , Playoff
Golden Square, Golden Square
Hungary NB 1.A, Placement Matches
Iceland Women's Basketball Premier League, Rel/Prom Playoff
Kvindebasketligaen , Bronzekamp
Kvindebasketligaen , Semifinaler
LNBF , Fase Regular
LSB - Liga B, Grupo C
NBL
NBL, Qualifying Round
NBL, Relegation/Promotion Playoff
Premier League, Women, Placement matches 5 to 8
Princ Caffe Superliga, Playoff
Qatar Cup , Knock-out Phase
Qatar Cup , Playoffs
Superleague Women, Плей-офф
The League, Финал
Top Division Women, Play Out
Youth Basketball Champions League, Group D
Youth Basketball Champions League, Semi-Finals 1-4
Youth Basketball Champions League, Semi-Finals 5-8 2
ZBL, Playout
Zweite Liga, Relegation Round
ABA Liga, Play-in
Kvalifikacije za prvu ligu
NBA, Play-IN Tournament
Saku I liiga
"""


def _norm(name: str) -> str:
    return " ".join(str(name or "").strip().casefold().split())


class V63LeagueBlacklist:
    """Exact-match league blacklist for V6.3 raw predictions.

    Uses casefold + whitespace normalization so minor formatting differences
    in live data don't bypass the filter.
    """

    def __init__(self) -> None:
        self._blocked: frozenset[str] = frozenset(
            _norm(x) for x in _RAW.splitlines() if _norm(x)
        )

    def is_blocked(self, league_name: str) -> tuple[bool, str | None]:
        """Return (blocked, reason) for a league name.

        Parameters
        ----------
        league_name:
            Raw league string from features or live match data.

        Returns
        -------
        (True, "v6_3_manual_blacklist") if blocked, (False, None) otherwise.
        """
        if _norm(league_name) in self._blocked:
            return True, "v6_3_manual_blacklist"
        return False, None

    def filter_rows(self, rows, league_fn=None):
        """Return (kept_rows, excluded_rows) from an iterable.

        Parameters
        ----------
        rows:
            Iterable of sample objects or dicts.
        league_fn:
            Callable(row) -> str.  If None, tries row["league"] or
            row.features_q4.get("league", "").
        """
        kept, excluded = [], []
        for r in rows:
            if league_fn is not None:
                league = league_fn(r)
            elif isinstance(r, dict):
                league = str(r.get("league", ""))
            else:
                league = str(
                    getattr(r, "features_q4", {}).get("league", "")
                    or getattr(r, "league", "")
                    or ""
                )
            blocked, _ = self.is_blocked(league)
            (excluded if blocked else kept).append(r)
        return kept, excluded


# Singleton for import convenience
_default: V63LeagueBlacklist | None = None


def get_blacklist() -> V63LeagueBlacklist:
    """Return a shared singleton instance."""
    global _default
    if _default is None:
        _default = V63LeagueBlacklist()
    return _default
