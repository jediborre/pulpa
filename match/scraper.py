"""
SofaScore basketball match data fetcher.

Approach: headless Chromium via Playwright.  The browser navigates
to the match page (to establish a valid session/cookies),
then ctx.request.get() fetches the JSON API endpoints automatically
handed the browser's session cookies.

SofaScore blocks direct requests (403) but handles requests from
within a browser context normally.

Data sources fetched for each match:
    - /event/{id}           -> metadata + final/quarter scores + team ratings
    - /event/{id}/incidents -> ALL incidents (scoring plays only: twoPoints, threePoints, freeThrow, plus period markers)
    - /event/{id}/graph     -> pressure/momentum curve points
    - /event/{id}/h2h       -> aggregate win counts (FALLBACK: only used when team discovery fails)
    - /team/{id}/events/last/{n} -> actual H2H matchups with PER-QUARTER scores (preferred)
    - /event/{id}/statistics -> per-team aggregated stats (field goals %, rebounds, etc.)
    - /event/{id}/lineups   -> starting lineups + per-player SofaScore rating, fouls, mins, +/-
    - /event/{id}/odds/1/all -> pre-match betting odds (1x2, spread, over/under)
    - incidents → computed   -> per-period (Q1-Q4) team statistics

Output dict keys:
    - match         : metadata (teams, slugs, IDs, ratings, date, venue, league)
    - score         : final + quarter scores
    - play_by_play  : scoring-only incidents (backward compatible)
    - events        : ALL incidents with incident_type + subtype + player_id
    - h2h           : head-to-head matches with per-quarter scores (from team events)
    - graph_points  : pressure/momentum curve
    - team_statistics : per-team aggregated stats (game totals)
    - period_stats  : per-period team stats computed from incidents (Q1-Q4)
    - lineups       : starting lineup info
    - player_stats  : per-player SofaScore ratings, fouls, mins, +/-
    - odds          : pre-match betting odds (1x2, spread, over/under)
"""

import re
import os
import time
from datetime import datetime, timezone
from contextlib import contextmanager

try:
    import anti_block
except ImportError:
    # Fallback if anti_block not available
    anti_block = None

_node_options = os.environ.get("NODE_OPTIONS", "").strip()
if "--no-deprecation" not in _node_options:
    os.environ["NODE_OPTIONS"] = (f"{_node_options} --no-deprecation").strip()


STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _normalize_backend(backend: str | None = None) -> str:
    value = (backend or os.getenv("SOFASCORE_SCRAPER_BACKEND", "chrome")).strip().lower()
    if value in {"obscura", "cdp"}:
        return "obscura"
    if value in {"chrome", "system_chrome"}:
        return "chrome"
    return "traditional"


def _obscura_cdp_url() -> str:
    return os.getenv("OBSCURA_CDP_URL", "http://127.0.0.1:9222").strip()


def _match_warmup_url(match_id: str) -> str:
    return f"https://www.sofascore.com/basketball/match/unknown/unknown#id:{match_id}"


@contextmanager
def _browser_context(warmup_url: str, backend: str | None = None):
    from playwright.sync_api import sync_playwright

    engine = _normalize_backend(backend)
    with sync_playwright() as p:
        if engine == "obscura":
            browser = p.chromium.connect_over_cdp(_obscura_cdp_url())
        elif engine == "chrome":
            # Use the real system-installed Google Chrome (harder to fingerprint as bot)
            browser = p.chromium.launch(channel="chrome", headless=True)
        else:
            # traditional: Playwright-bundled Chromium headless
            browser = p.chromium.launch(headless=True)

        ctx = browser.new_context(user_agent=STANDARD_UA)
        page = ctx.new_page()
        try:
            page.goto(warmup_url, wait_until="networkidle", timeout=45_000)
        except Exception:
            pass

        try:
            yield browser, ctx, page
        finally:
            try:
                ctx.close()
            except Exception:
                pass
            if engine in {"traditional", "chrome"}:
                try:
                    browser.close()
                except Exception:
                    pass

# Basketball: incidentClass / 'from' field values → point value
# SofaScore uses camelCase for incidentClass ("threePoints") and
# lowercase for the 'from' field ("threepoints").  Both are mapped here.
_CLASS_PTS: dict[str, int] = {
    "onepoint": 1,   "twopoints": 2,   "threepoints": 3,   # 'from' field
    "onePoint": 1,   "twoPoints": 2,   "threePoints": 3,   # incidentClass
    # Legacy soccer / generic sport scoring types kept for compatibility
    "points1": 1,    "points2": 2,     "points3": 3,
}

# Map incidentType values to a canonical type for DB storage
_INCIDENT_TYPE_MAP: dict[str, str] = {
    "goal": "goal",
    "score": "goal",
    "foul": "foul",
    "timeout": "timeout",
    "substitution": "substitution",
    "turnover": "turnover",
    "freethrow": "freethrow",
    "freePoint": "freethrow",
    "miss": "miss",
    "rebound": "rebound",
    "period": "period",
}

# Subtype categories for richer classification
_SHOT_SUBTYPES: set[str] = {
    "twoPoints", "threePoints", "onePoint",
    "twopoints", "threepoints", "onepoint",
    "freeThrow", "freethrow",
    "layup", "dunk", "jumpShot", "hookShot", "slamDunk",
}

_FOUL_SUBTYPES: set[str] = {
    "personalFoul", "offensiveFoul", "technicalFoul",
    "unsportsmanlikeFoul", "flagrantFoul", "charge",
    "personal", "offensive", "technical",
}

_TIMEOUT_SUBTYPES: set[str] = {
    "regular", "full", "short", "tv",
}

_PERIOD: dict[int | str, str] = {
    1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4",
    "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4",
}


# ── helpers ───────────────────────────────────────────────

def parse_match_id(url: str) -> str:
    """Extract the numeric match ID from any SofaScore match URL."""
    # Fragment form:  …#id:14442355
    m = re.search(r'[#&]id[:/](\d+)', url)
    if m:
        return m.group(1)
    # Path form: …/14442355  or …/14442355/…
    m = re.search(r'/(\d{6,})', url)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract match ID from URL: {url}")


def _quarter_label(period) -> str:
    # Handle SofaScore basketball format: "period1", "period2", … "period5" (OT)
    if isinstance(period, str) and period.startswith("period"):
        try:
            n = int(period[6:])
            return f"OT{n - 4}" if n > 4 else f"Q{n}"
        except ValueError:
            pass
    if period in _PERIOD:
        return _PERIOD[period]
    try:
        n = int(period)
        return f"OT{n - 4}" if n > 4 else f"Q{n}"
    except (TypeError, ValueError):
        return str(period)


def _period_from_game_seconds(time_secs: int) -> str:
    """
    Derive the quarter/OT label from cumulative game time in seconds.
    NBA: 4 quarters of 12 min (720 s) + OT periods of 5 min (300 s).
    Used as fallback when SofaScore incidents carry no explicit 'period' field.
    """
    if time_secs <= 720:
        return "Q1"
    if time_secs <= 1440:
        return "Q2"
    if time_secs <= 2160:
        return "Q3"
    if time_secs <= 2880:
        return "Q4"
    ot = (time_secs - 2881) // 300 + 1
    return f"OT{ot}"


def _time_str(inc: dict) -> str:
    """
    Return a human-readable game-clock string (time remaining in period).

    SofaScore basketball incidents carry:
      reversedPeriodTimeSeconds – TOTAL seconds remaining in the period (use this)
      reversedPeriodTime        – minutes portion only (less reliable, ignored)
      time                      – elapsed game minutes (fallback)
    """
    total = inc.get("reversedPeriodTimeSeconds")
    if total is not None:
        m, s = divmod(int(total), 60)
        return f"{m}:{s:02d}"
    t = inc.get("time", 0)
    return f"{t}:00"


# ── core parser ──────────────────────────────────────────

def _parse(event_json: dict, incidents: list, graph_points: list | None = None) -> dict:
    """
    Convert raw SofaScore API responses into our canonical match dict.

    Returns two event streams:
      - play_by_play : scoring incidents only (backward compatible)
      - events       : ALL incidents with classification (fouls, timeouts, subs, etc.)

    Parameters
    ----------
    event_json : response body from GET /event/{id}
    incidents  : list from GET /event/{id}/incidents  →  .incidents[]
    graph_points : list from GET /event/{id}/graph → .graphPoints[]
    """
    ev = event_json.get("event", event_json)

    home = ev["homeTeam"]["name"]
    away = ev["awayTeam"]["name"]
    home_slug = ev["homeTeam"].get("slug", "unknown")
    away_slug = ev["awayTeam"].get("slug", "unknown")
    event_slug = ev.get("slug", "unknown")
    custom_id = ev.get("customId", "")
    status = ev.get("status") or {}
    status_type = status.get("type", "")
    status_description = status.get("description", "")

    # Team IDs from SofaScore (useful for roster/player lookups)
    home_team_id = ev.get("homeTeam", {}).get("id")
    away_team_id = ev.get("awayTeam", {}).get("id")

    ts = ev.get("startTimestamp", 0)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)

    # Venue (stadium name preferred, city as fallback)
    v = ev.get("venue") or {}
    venue = (v.get("stadium") or {}).get("name", "") or (v.get("city") or {}).get("name", "")

    league = (ev.get("tournament") or {}).get("name", "")

    # Scores
    hs = ev.get("homeScore") or {}
    as_ = ev.get("awayScore") or {}
    home_total = hs.get("current", hs.get("normaltime", 0))
    away_total = as_.get("current", as_.get("normaltime", 0))

    quarters: dict[str, dict] = {}
    for i in range(1, 10):          # Q1–Q4 + up to 5 OTs
        h = hs.get(f"period{i}")
        a = as_.get(f"period{i}")
        if h is None or a is None:
            break
        quarters[_quarter_label(i)] = {"home": h, "away": a}

    # ── Parse ALL incidents ─────────────────────────────────
    pbp: dict[str, list] = {}       # scoring-only (backward compat)
    events: dict[str, list] = {}    # all incidents with classification

    for inc in incidents:
        raw_type = inc.get("incidentType", "")
        canon_type = _INCIDENT_TYPE_MAP.get(raw_type, raw_type)
        raw_class = inc.get("incidentClass") or inc.get("from") or inc.get("scoringType") or ""

        # Points value (for scoring plays)
        pts = (
            _CLASS_PTS.get(inc.get("incidentClass", ""))
            or _CLASS_PTS.get(inc.get("from", ""))
            or _CLASS_PTS.get(inc.get("scoringType", ""))
            or 0
        )

        # Determine subtype based on incident type and class
        subtype = ""
        if canon_type == "goal" or canon_type == "freethrow":
            if raw_class in _SHOT_SUBTYPES:
                subtype = raw_class
            elif pts == 1:
                subtype = "freeThrow"
            elif pts == 2:
                subtype = "twoPoints"
            elif pts == 3:
                subtype = "threePoints"
        elif canon_type == "foul":
            subtype = raw_class if raw_class in _FOUL_SUBTYPES else "personalFoul"
        elif canon_type == "timeout":
            subtype = raw_class if raw_class in _TIMEOUT_SUBTYPES else "regular"
        elif canon_type == "substitution":
            subtype = raw_class
        elif canon_type == "turnover":
            subtype = raw_class
        elif canon_type == "miss":
            subtype = raw_class

        # Period / quarter
        period_raw = inc.get("period") or inc.get("periodType")
        if period_raw is not None:
            q = _quarter_label(period_raw)
        else:
            q = _period_from_game_seconds(inc.get("timeSeconds", 0))

        # Player info
        player_obj = inc.get("player") or {}
        player = player_obj.get("shortName") or player_obj.get("name", "")
        player_id = player_obj.get("id")

        team = "home" if inc.get("isHome", True) else "away"

        event_row = {
            "time": _time_str(inc),
            "time_seconds": inc.get("timeSeconds"),
            "incident_type": canon_type,
            "subtype": subtype,
            "player": player,
            "player_id": player_id,
            "team": team,
            "points": pts if pts > 0 else None,
            "home_score": inc.get("homeScore"),
            "away_score": inc.get("awayScore"),
        }
        events.setdefault(q, []).append(event_row)

        # Keep legacy scoring-only PBP (backward compat)
        if canon_type == "goal" and pts > 0:
            pbp.setdefault(q, []).append({
                "time": _time_str(inc),
                "player": player,
                "points": pts,
                "team": team,
                "home_score": inc.get("homeScore"),
                "away_score": inc.get("awayScore"),
            })

    return {
        "match": {
            "home_team": home,
            "away_team": away,
            "home_slug": home_slug,
            "away_slug": away_slug,
            "event_slug": event_slug,
            "custom_id": custom_id,
            "status_type": status_type,
            "status_description": status_description,
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M"),
            "venue": venue,
            "league": league,
            "home_record": "",   # requires a separate standings endpoint
            "away_record": "",
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_rating": ev.get("homeTeam", {}).get("rating") or None,
            "away_rating": ev.get("awayTeam", {}).get("rating") or None,
        },
        "score": {
            "home": home_total,
            "away": away_total,
            "quarters": quarters,
        },
        "play_by_play": pbp,
        "events": events,
        "graph_points": graph_points or [],
    }


def _parse_h2h(match_id: str, h2h_data: list | dict | None) -> list[dict]:
    """
    Parse H2H data from multiple sources into a uniform list of match rows.

    Accepts:
      - list of match dicts (from team events discovery, or future /h2h list)
      - dict with 'teamDuel' key (from current /event/{id}/h2h aggregate endpoint)

    Returns list of dicts with per-match H2H data including quarter scores.
    """
    if not h2h_data:
        return []

    rows: list[dict] = []

    # Case 1: list of individual match dicts (from team events or future H2H endpoint)
    if isinstance(h2h_data, list):
        for entry in h2h_data:
            if not isinstance(entry, dict):
                continue
            mid = str(entry.get("id") or entry.get("h2h_match_id") or "")
            if not mid:
                continue
            hs = entry.get("homeScore") or {}
            as_ = entry.get("awayScore") or {}
            ts = entry.get("startTimestamp", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            rows.append({
                "match_id": match_id,
                "h2h_match_id": mid,
                "date": dt.strftime("%Y-%m-%d") if dt else "",
                "timestamp": ts or None,
                "home_team": (entry.get("homeTeam") or {}).get("name", ""),
                "away_team": (entry.get("awayTeam") or {}).get("name", ""),
                "home_score": hs.get("current", hs.get("normaltime")),
                "away_score": as_.get("current", as_.get("normaltime")),
                "q1_home": hs.get("period1"), "q1_away": as_.get("period1"),
                "q2_home": hs.get("period2"), "q2_away": as_.get("period2"),
                "q3_home": hs.get("period3"), "q3_away": as_.get("period3"),
                "q4_home": hs.get("period4"), "q4_away": as_.get("period4"),
                "tournament": (entry.get("tournament") or {}).get("name", ""),
            })

    # Case 2: aggregate dict with teamDuel (current /event/{id}/h2h response)
    elif isinstance(h2h_data, dict):
        duel = h2h_data.get("teamDuel") or {}
        if duel:
            rows.append({
                "match_id": match_id,
                "h2h_match_id": "H2H_SUMMARY",
                "date": "",
                "home_team": "",
                "away_team": "",
                "home_score": duel.get("homeWins"),
                "away_score": duel.get("awayWins"),
                "q1_home": duel.get("draws"),  # repurpose as draws
                "q1_away": None,
                "q2_home": None, "q2_away": None,
                "q3_home": None, "q3_away": None,
                "q4_home": None, "q4_away": None,
                "tournament": "",
            })

    return rows


def _parse_team_statistics(match_id: str, stats_data: dict | None) -> list[dict]:
    """
    Parse response from GET /event/{id}/statistics into team-level stat rows.

    Returns list of dicts: (match_id, period, group_name, stat_key,
                            stat_name, home_value, away_value, home_total, away_total)
    """
    if not stats_data:
        return []
    rows: list[dict] = []
    stats_list = stats_data.get("statistics") if isinstance(stats_data, dict) else None
    if not stats_list:
        return []
    for period_group in stats_list:
        period = period_group.get("period", "ALL")
        for group in period_group.get("groups", []):
            group_name = group.get("groupName", "")
            for item in group.get("statisticsItems", []):
                rows.append({
                    "match_id": match_id,
                    "period": period,
                    "group_name": group_name,
                    "stat_key": item.get("key", ""),
                    "stat_name": item.get("name", ""),
                    "home_value": item.get("homeValue"),
                    "away_value": item.get("awayValue"),
                    "home_total": item.get("homeTotal", item.get("homeValue")),
                    "away_total": item.get("awayTotal", item.get("awayValue")),
                    "home_display": item.get("home", ""),
                    "away_display": item.get("away", ""),
                })
    return rows


def _parse_lineups(match_id: str, lineup_data: dict | None) -> tuple[list[dict], list[dict]]:
    """
    Parse response from GET /event/{id}/lineups.

    Returns (lineup_rows, player_stat_rows):
      - lineup_rows: who played (match_id, team, player_name, player_id,
                     is_starter, shirt_number, position)
      - player_stat_rows: per-player game stats including SofaScore rating
    """
    lineup_rows: list[dict] = []
    player_stat_rows: list[dict] = []

    if not lineup_data:
        return lineup_rows, player_stat_rows

    for side in ("home", "away"):
        team_data = lineup_data.get(side)
        if not team_data:
            continue
        for p in team_data.get("players", []):
            pl = p.get("player", {})
            player_id = pl.get("id")
            player_name = pl.get("name", "")
            position = pl.get("position", "")
            shirt_number = p.get("shirtNumber") or p.get("jerseyNumber")
            is_starter = not p.get("substitute", True)
            team_id = p.get("teamId")

            lineup_rows.append({
                "match_id": match_id,
                "team": side,
                "player_name": player_name,
                "player_id": str(player_id) if player_id else None,
                "is_starter": int(is_starter),
                "shirt_number": int(shirt_number) if shirt_number else None,
                "position": position,
            })

            stats = p.get("statistics", {})
            if not isinstance(stats, dict):
                continue

            secs = stats.get("secondsPlayed")
            player_stat_rows.append({
                "match_id": match_id,
                "team": side,
                "player_name": player_name,
                "player_id": str(player_id) if player_id else None,
                "sofascore_rating": stats.get("rating"),
                "minutes_played": round(secs / 60, 1) if secs else None,
                "points": stats.get("points"),
                "fouls": stats.get("personalFouls"),
                "plus_minus": stats.get("plusMinus"),
                "field_goals_made": stats.get("fieldGoalsMade"),
                "field_goals_attempted": stats.get("fieldGoalAttempts"),
                "three_made": stats.get("threePointsMade"),
                "three_attempted": stats.get("threePointAttempts"),
                "free_throws_made": stats.get("freeThrowsMade"),
                "free_throws_attempted": stats.get("freeThrowAttempts"),
                "rebounds": stats.get("rebounds"),
                "assists": stats.get("assists"),
                "steals": stats.get("steals"),
                "turnovers": stats.get("turnovers"),
                "blocks": stats.get("blocks"),
            })

    return lineup_rows, player_stat_rows


def _parse_period_stats(
    match_id: str,
    incidents: list[dict],
    period_seconds: int = 600,
) -> list[dict]:
    """
    Compute per-quarter team statistics from incidents.

    Since the /event/{id}/statistics endpoint only returns game totals
    (period='ALL'), the per-quarter breakdown must be computed from
    the incident feed. Each incident has timeSeconds which determines
    which period it belongs to.

    Creates aggregated rows compatible with the team_statistics table,
    with period='Q1','Q2','Q3','Q4'.

    Returns list of dicts with keys: match_id, period, group_name,
    stat_key, stat_name, home_value, away_value, home_total, away_total.
    """
    if not incidents:
        return []

    # Determine period boundaries from periods / normal incidents
    # timeSeconds = seconds elapsed in the game
    # Basketball: 10 min periods => 600s each
    per = period_seconds

    # Filter out period markers
    scoring = [i for i in incidents if i.get("incidentType") != "period"]

    # Build period → {home: {points, ft_made, ...}, away: {...}}
    from collections import defaultdict

    period_data: dict[int, dict] = defaultdict(
        lambda: {
            "home_points": 0, "away_points": 0,
            "home_ft_made": 0, "away_ft_made": 0,
            "home_2pt_made": 0, "away_2pt_made": 0,
            "home_3pt_made": 0, "away_3pt_made": 0,
            "home_fouls": 0, "away_fouls": 0,
            "home_timeouts": 0, "away_timeouts": 0,
            "home_turnovers": 0, "away_turnovers": 0,
        }
    )

    for inc in scoring:
        ts = inc.get("timeSeconds") or 0
        p_num = (ts // per) + 1  # 1-indexed period

        ic = inc.get("incidentClass")
        inc_type = inc.get("incidentType")
        is_home = inc.get("isHome")

        pts = {"onePoint": 1, "twoPoints": 2, "threePoints": 3,
               "onepoint": 1, "twopoints": 2, "threepoints": 3}
        pt_val = pts.get(ic) or pts.get(inc.get("from", "")) or inc.get("pointValue") or 0

        side = "home" if is_home is True else ("away" if is_home is False else None)

        if pt_val > 0:
            if side == "home":
                period_data[p_num]["home_points"] += pt_val
                if pt_val == 1:
                    period_data[p_num]["home_ft_made"] += 1
                elif pt_val == 2:
                    period_data[p_num]["home_2pt_made"] += 1
                elif pt_val == 3:
                    period_data[p_num]["home_3pt_made"] += 1
            elif side == "away":
                period_data[p_num]["away_points"] += pt_val
                if pt_val == 1:
                    period_data[p_num]["away_ft_made"] += 1
                elif pt_val == 2:
                    period_data[p_num]["away_2pt_made"] += 1
                elif pt_val == 3:
                    period_data[p_num]["away_3pt_made"] += 1

            # If no team info (isHome=None), estimate from score changes
            if side is None:
                prev_h = inc.get("homeScore") or 0
                prev_a = inc.get("awayScore") or 0
                # TODO: determine team from score delta

        if inc_type == "foul":
            if side == "home":
                period_data[p_num]["home_fouls"] += 1
            elif side == "away":
                period_data[p_num]["away_fouls"] += 1
        elif inc_type == "timeout":
            if side == "home":
                period_data[p_num]["home_timeouts"] += 1
            elif side == "away":
                period_data[p_num]["away_timeouts"] += 1
        elif inc_type == "turnover":
            if side == "home":
                period_data[p_num]["home_turnovers"] += 1
            elif side == "away":
                period_data[p_num]["away_turnovers"] += 1

    # Build output rows (only for periods that had events)
    rows: list[dict] = []
    for p_num in sorted(period_data.keys()):
        pd = period_data[p_num]

        def _r(stat_key, stat_name, hv, av, group="Scoring"):
            return {
                "match_id": match_id,
                "period": f"Q{p_num}",
                "group_name": group,
                "stat_key": stat_key,
                "stat_name": stat_name,
                "home_value": hv,
                "away_value": av,
                "home_total": hv,
                "away_total": av,
                "home_display": str(hv),
                "away_display": str(av),
            }

        rows.append(_r("points", "Points", pd["home_points"], pd["away_points"]))
        rows.append(_r("freeThrowsMade", "Free throws made", pd["home_ft_made"], pd["away_ft_made"]))
        rows.append(_r("twoPointsMade", "2 pointers made", pd["home_2pt_made"], pd["away_2pt_made"]))
        rows.append(_r("threePointsMade", "3 pointers made", pd["home_3pt_made"], pd["away_3pt_made"]))
        rows.append(_r("fouls", "Fouls", pd["home_fouls"], pd["away_fouls"], "Other"))
        rows.append(_r("timeouts", "Timeouts", pd["home_timeouts"], pd["away_timeouts"], "Other"))
        rows.append(_r("turnovers", "Turnovers", pd["home_turnovers"], pd["away_turnovers"], "Other"))

    return rows


def _parse_odds(match_id: str, odds_data: dict | None) -> list[dict]:
    """
    Parse response from /event/{id}/odds/{providerId}/all.

    Odds are in fractional format (e.g., "53/100" = 1.53 decimal).
    Structure:
    {
      "markets": [
        {
          "marketId": 1,
          "marketName": "Full time",
          "marketPeriod": "Match",
          "marketGroup": "Home/Away",
          "structureType": 1,
          "choices": [
            {"name": "1", "fractionalValue": "53/100", "change": -1},
            {"name": "2", "fractionalValue": "8/5", "change": 1}
          ]
        }
      ]
    }

    Returns list of dicts with keys: match_id, odds_type, market_name,
    home_value, away_value, draw_value, timestamp, market_period
    """
    if not odds_data:
        return []

    import re

    def _frac_to_dec(frac: str | None) -> float | None:
        """Convert fractional odds like '53/100' or '8/5' to decimal."""
        if not frac or not isinstance(frac, str):
            return None
        frac = frac.strip()
        m = re.match(r"(\d+)/(\d+)", frac)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den > 0:
                return round(num / den + 1, 2)  # decimal = fractional + 1
        return None

    rows: list[dict] = []
    markets = odds_data.get("markets") or odds_data.get("odds") or []
    if not isinstance(markets, list):
        return []

    for market in markets:
        mname = market.get("marketName") or market.get("name", "")
        mgroup = market.get("marketGroup", "")
        mperiod = market.get("marketPeriod", "")
        choices = market.get("choices", [])
        home_val = None
        away_val = None
        draw_val = None
        for ch in choices:
            ch_name = ch.get("name", "").lower()
            frac = ch.get("fractionalValue") or ch.get("initialFractionalValue")
            dec = _frac_to_dec(frac)
            if ch_name in ("1", "home", "h"):
                home_val = dec
            elif ch_name in ("2", "away", "a"):
                away_val = dec
            elif ch_name in ("x", "draw"):
                draw_val = dec
            # Handle spread: name like "(-4.0) Detroit Pistons"
            if ch_name.startswith("(-") or ch_name.startswith("(+") or ch_name.startswith("(e"):
                if not home_val:
                    home_val = dec
                elif not away_val:
                    away_val = dec

        odds_type = mgroup or mname.replace(" ", "_").lower()
        rows.append({
            "match_id": match_id,
            "odds_type": odds_type,
            "market_name": mname,
            "market_period": mperiod,
            "home_value": home_val,
            "away_value": away_val,
            "draw_value": draw_val,
            "timestamp": "",
        })

    return rows


def fetch_team_recent_events(team_id: int | str, limit: int = 20) -> list[dict]:
    """
    Fetch recent events for a team via GET /team/{id}/events/last/{limit}.

    Returns list of match dicts with id, homeTeam, awayTeam, homeScore,
    awayScore, startTimestamp, tournament.
    """
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/{limit}"

    with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
        resp = ctx.request.get(url, headers=extra_headers, timeout=20_000)

        if not resp.ok:
            return []

        body = resp.json() or {}
        return body.get("events", [])


def fetch_h2h_via_teams(
    home_team_id: int | str,
    away_team_id: int | str,
    limit: int = 0,
) -> list[dict]:
    """
    Discover H2H matches with PER-QUARTER scores by fetching recent events
    for both teams from a single Playwright session.

    The /event/{id}/h2h endpoint only returns aggregate win counts, so we
    discover actual past matchups via /team/{id}/events/last/{n}.

    Returns list with per-match dicts:
        h2h_match_id, date, home_team, away_team,
        home_score, away_score,
        q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
    """
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
        def _req(team_id: int | str) -> list[dict]:
            url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/{limit}"
            resp = ctx.request.get(url, headers=extra_headers, timeout=20_000)
            if not resp.ok:
                return []
            return (resp.json() or {}).get("events", [])

        home_events = _req(home_team_id)
        away_events = _req(away_team_id)

    home_ids = {str(e.get("id")) for e in home_events if e.get("id")}
    away_ids = {str(e.get("id")) for e in away_events if e.get("id")}
    common_ids = home_ids & away_ids

    results: list[dict] = []
    for ev in home_events + away_events:
        mid = str(ev.get("id"))
        if mid not in common_ids:
            continue
        hs = ev.get("homeScore") or {}
        as_ = ev.get("awayScore") or {}
        ts = ev.get("startTimestamp", 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        results.append({
            "h2h_match_id": mid,
            "date": dt.strftime("%Y-%m-%d") if dt else "",
            "timestamp": ts or None,
            "home_team": (ev.get("homeTeam") or {}).get("name", ""),
            "away_team": (ev.get("awayTeam") or {}).get("name", ""),
            "home_score": hs.get("current", hs.get("normaltime")),
            "away_score": as_.get("current", as_.get("normaltime")),
            "q1_home": hs.get("period1"), "q1_away": as_.get("period1"),
            "q2_home": hs.get("period2"), "q2_away": as_.get("period2"),
            "q3_home": hs.get("period3"), "q3_away": as_.get("period3"),
            "q4_home": hs.get("period4"), "q4_away": as_.get("period4"),
            "tournament": (ev.get("tournament") or {}).get("name", ""),
        })

    # Deduplicate by match_id
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results:
        if r["h2h_match_id"] not in seen:
            seen.add(r["h2h_match_id"])
            unique.append(r)

    return unique


def fetch_team_strength(
    home_team_id: int | str,
    away_team_id: int | str,
    home_team_name: str = "",
    away_team_name: str = "",
) -> list[dict]:
    """
    Fetch team strength data for both teams: pregameForm (position, W-L, form)
    and performance points (per-match power rating).

    Calls /team/{id} and /team/{id}/performance for each team.
    Returns list with two dicts:
        team_id, team_name, position, wins, losses, form (JSON),
        perf_points (JSON dict of match_id → rating)
    """
    import json

    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
        results: list[dict] = []
        for team_id, team_name in (
            (home_team_id, home_team_name),
            (away_team_id, away_team_name),
        ):
            row: dict = {"team_id": int(team_id), "team_name": team_name}

            # /team/{id} → pregameForm
            resp = ctx.request.get(
                f"https://api.sofascore.com/api/v1/team/{team_id}",
                headers=extra_headers,
                timeout=20_000,
            )
            if resp.ok:
                body = resp.json()
                pf = body.get("pregameForm") or {}
                row["position"] = pf.get("position")
                val = pf.get("value", "") or ""
                if "-" in val:
                    parts = val.split("-")
                    row["wins"] = int(parts[0]) if parts[0].isdigit() else None
                    row["losses"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                else:
                    row["wins"] = row["losses"] = None
                row["form"] = json.dumps(pf.get("form", []))

            # /team/{id}/performance → points
            resp = ctx.request.get(
                f"https://api.sofascore.com/api/v1/team/{team_id}/performance",
                headers=extra_headers,
                timeout=20_000,
            )
            if resp.ok:
                body = resp.json()
                points = body.get("points") or {}
                row["perf_points"] = json.dumps(points)

            results.append(row)
            import time
            time.sleep(0.3)
    return results


# ── public fetch function ─────────────────────────────────────────────────────

def fetch_match(
    url: str,
    match_id: str,
    fetch_h2h: bool = True,
    fetch_statistics: bool = True,
    fetch_lineups: bool = True,
    fetch_team_data: bool = True,
    backend: str | None = None,
) -> dict:
    """
    Fetch match data using Playwright's BrowserContext.request API.

    Strategy:
      1. Launch headless Chromium and navigate to the SofaScore match page.
         This satisfies Cloudflare bot-detection and plants session cookies.
      2. Call these SofaScore JSON endpoints via ctx.request.get():
            - /event/{id}
            - /event/{id}/incidents
            - /event/{id}/graph
            - /event/{id}/h2h (optional)
            - /event/{id}/statistics (optional)
            - /event/{id}/lineups (optional)
         The Playwright HTTP client shares the browser context's cookie jar,
         so SofaScore sees these as legitimate browser requests.
      3. If fetch_h2h=True, also discover H2H matches with PER-QUARTER scores
         via /team/{id}/events/last/{n} (the /event/{id}/h2h endpoint only
         returns aggregate win counts).
      4. If fetch_team_data=True, also fetch pregameForm and performance
         points for both teams via /team/{id} and /team/{id}/performance.

    No direct requests (blocked with 403), no JS injection, no DOM scraping.
    """
    payloads = _fetch_match_payloads(
        warmup_url=url,
        match_id=match_id,
        fetch_h2h=fetch_h2h,
        fetch_statistics=fetch_statistics,
        fetch_lineups=fetch_lineups,
        backend=backend,
    )
    event_json, incidents_json, graph_json, h2h_json, statistics_json, lineups_json = payloads[:6]

    if not event_json or "event" not in event_json:
        raise RuntimeError(
            "Playwright could not retrieve event data — "
            "verify the match URL and try again."
        )

    parsed = _parse(
        event_json,
        incidents_json.get("incidents", []),
        graph_json.get("graphPoints", []),
    )

    # H2H: use aggregate from /event/{id}/h2h as fallback,
    # but prefer team-events discovery which has PER-QUARTER scores
    parsed["h2h"] = []
    if fetch_h2h:
        ev = event_json.get("event") or {}
        home_tid = (ev.get("homeTeam") or {}).get("id")
        away_tid = (ev.get("awayTeam") or {}).get("id")
        if home_tid and away_tid and home_tid != away_tid:
            try:
                team_h2h = fetch_h2h_via_teams(home_tid, away_tid)
                if team_h2h:
                    for r in team_h2h:
                        r["match_id"] = match_id
                    parsed["h2h"] = team_h2h
            except Exception:
                pass
        if not parsed["h2h"] and h2h_json:
            parsed["h2h"] = _parse_h2h(match_id, h2h_json)

    # Team statistics (game totals)
    if fetch_statistics and statistics_json:
        parsed["team_statistics"] = _parse_team_statistics(match_id, statistics_json)
    else:
        parsed["team_statistics"] = []

    # Per-period team statistics (computed from incidents)
    parsed["period_stats"] = []
    if incidents_json:
        ev = event_json.get("event") or {}
        period_len = (ev.get("time") or {}).get("periodLength", 600)
        try:
            parsed["period_stats"] = _parse_period_stats(
                match_id,
                incidents_json.get("incidents", []),
                period_seconds=period_len,
            )
        except Exception:
            pass

    # Lineups + player stats
    if fetch_lineups and lineups_json:
        lineup_rows, player_stat_rows = _parse_lineups(match_id, lineups_json)
        parsed["lineups"] = lineup_rows
        parsed["player_stats"] = player_stat_rows
    else:
        parsed["lineups"] = []
        parsed["player_stats"] = []

    # Odds
    odds_json = payloads[6] if len(payloads) > 6 else None
    if odds_json:
        parsed["odds"] = _parse_odds(match_id, odds_json)
    else:
        parsed["odds"] = []

    # Team strength (pregameForm + performance points)
    parsed["team_strength"] = []
    if fetch_team_data:
        ev = event_json.get("event") or {}
        home_tid = (ev.get("homeTeam") or {}).get("id")
        away_tid = (ev.get("awayTeam") or {}).get("id")
        home_name = (ev.get("homeTeam") or {}).get("name", "")
        away_name = (ev.get("awayTeam") or {}).get("name", "")
        if home_tid and away_tid:
            try:
                parsed["team_strength"] = fetch_team_strength(
                    home_tid, away_tid, home_name, away_name,
                )
            except Exception as exc:
                print(f"[scraper] fetch_team_strength error: {exc}")

    return parsed


def fetch_match_by_id(
    match_id: str,
    fetch_h2h: bool = True,
    fetch_statistics: bool = True,
    fetch_lineups: bool = True,
    fetch_team_data: bool = True,
    backend: str | None = None,
) -> dict:
    """Fetch match data by ID after warming session on the match route."""
    payloads = _fetch_match_payloads(
        warmup_url=_match_warmup_url(match_id),
        match_id=match_id,
        fetch_h2h=fetch_h2h,
        fetch_statistics=fetch_statistics,
        fetch_lineups=fetch_lineups,
        backend=backend,
    )
    event_json, incidents_json, graph_json, h2h_json, statistics_json, lineups_json = payloads[:6]

    parsed = _parse(
        event_json,
        incidents_json.get("incidents", []),
        graph_json.get("graphPoints", []),
    )

    # H2H: use aggregate from /event/{id}/h2h as fallback,
    # but prefer team-events discovery which has PER-QUARTER scores
    parsed["h2h"] = []
    if fetch_h2h:
        ev = event_json.get("event") or {}
        home_tid = (ev.get("homeTeam") or {}).get("id")
        away_tid = (ev.get("awayTeam") or {}).get("id")
        if home_tid and away_tid and home_tid != away_tid:
            try:
                team_h2h = fetch_h2h_via_teams(home_tid, away_tid)
                if team_h2h:
                    for r in team_h2h:
                        r["match_id"] = match_id
                    parsed["h2h"] = team_h2h
            except Exception:
                pass
        if not parsed["h2h"] and h2h_json:
            parsed["h2h"] = _parse_h2h(match_id, h2h_json)

    # Team statistics (game totals)
    if fetch_statistics and statistics_json:
        parsed["team_statistics"] = _parse_team_statistics(match_id, statistics_json)
    else:
        parsed["team_statistics"] = []

    # Per-period team statistics (computed from incidents)
    parsed["period_stats"] = []
    if incidents_json:
        ev = event_json.get("event") or {}
        period_len = (ev.get("time") or {}).get("periodLength", 600)
        try:
            parsed["period_stats"] = _parse_period_stats(
                match_id,
                incidents_json.get("incidents", []),
                period_seconds=period_len,
            )
        except Exception:
            pass

    # Lineups + player stats
    if fetch_lineups and lineups_json:
        lineup_rows, player_stat_rows = _parse_lineups(match_id, lineups_json)
        parsed["lineups"] = lineup_rows
        parsed["player_stats"] = player_stat_rows
    else:
        parsed["lineups"] = []
        parsed["player_stats"] = []

    # Odds
    odds_json = payloads[6] if len(payloads) > 6 else None
    if odds_json:
        parsed["odds"] = _parse_odds(match_id, odds_json)
    else:
        parsed["odds"] = []

    # Team strength (pregameForm + performance points)
    parsed["team_strength"] = []
    if fetch_team_data:
        ev = event_json.get("event") or {}
        home_tid = (ev.get("homeTeam") or {}).get("id")
        away_tid = (ev.get("awayTeam") or {}).get("id")
        home_name = (ev.get("homeTeam") or {}).get("name", "")
        away_name = (ev.get("awayTeam") or {}).get("name", "")
        if home_tid and away_tid:
            try:
                parsed["team_strength"] = fetch_team_strength(
                    home_tid, away_tid, home_name, away_name,
                )
            except Exception as exc:
                print(f"[scraper] fetch_team_strength error: {exc}")

    return parsed


def fetch_event_snapshot(match_id: str) -> dict:
    """Fetch lightweight event state for live/FT friendly reporting."""
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
        resp = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}",
            headers=extra_headers,
            timeout=20_000,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Event API returned HTTP {resp.status} for match {match_id}"
            )
        body = resp.json() or {}

    ev = body.get("event", body)
    status = ev.get("status") or {}
    hs = ev.get("homeScore") or {}
    as_ = ev.get("awayScore") or {}

    return {
        "status_type": status.get("type", ""),
        "status_description": status.get("description", ""),
        "status_code": status.get("code", None),
        "home_score": hs.get("current", hs.get("normaltime", 0)),
        "away_score": as_.get("current", as_.get("normaltime", 0)),
    }


def fetch_finished_match_ids_for_date(date_str: str, max_retries: int = 3) -> list[dict]:
    """Return finished basketball matches for a date (YYYY-MM-DD).
    
    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format.
    max_retries : int
        Number of times to retry on HTTP 403 (after respecting cooldown).
    
    Raises
    ------
    RuntimeError
        If API returns non-OK status after all retries.
    """
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    api_url = (
        "https://api.sofascore.com/api/v1/"
        f"sport/basketball/scheduled-events/{date_str}"
    )
    
    last_error = None
    # Retry loop with anti-block protection
    for attempt in range(max_retries + 1):
        # Check if we're in cooldown; wait if needed
        if anti_block and anti_block.get_global_fetch_cooldown_left() > 0:
            cooldown_left = anti_block.get_global_fetch_cooldown_left()
            if attempt == 0:
                raise RuntimeError(
                    f"Anti-block cooldown active ({cooldown_left:.0f}s remaining). "
                    f"Retry after approximately {int(cooldown_left) + 1} seconds."
                )
        
        # Apply global request spacing with jitter
        if anti_block:
            delay = anti_block.next_global_fetch_delay_secs()
            if delay > 0:
                time.sleep(delay)

        try:
            with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
                resp = ctx.request.get(api_url, headers=extra_headers, timeout=30_000)
                
                if not resp.ok:
                    err_msg = f"Daily events API returned HTTP {resp.status} for {date_str}"
                    if anti_block:
                        anti_block.note_fetch_error(err_msg, "fetch_finished_match_ids_for_date")
                    raise RuntimeError(err_msg)

                body = resp.json() or {}
                events = body.get("events", []) if isinstance(body, dict) else []
            
            # Success: reset error streak and return results
            if anti_block:
                anti_block.note_fetch_success()
            
            out = []
            for ev in events:
                status = (ev.get("status") or {}).get("type", "")
                if status != "finished":
                    continue

                hs = (ev.get("homeScore") or {}).get("current")
                as_ = (ev.get("awayScore") or {}).get("current")
                if hs is None or as_ is None:
                    continue

                out.append({
                    "match_id": str(ev.get("id", "")),
                    "event_date": date_str,
                    "status_type": status,
                    "home_team": (ev.get("homeTeam") or {}).get("name", ""),
                    "away_team": (ev.get("awayTeam") or {}).get("name", ""),
                    "league": ((ev.get("tournament") or {}).get("name", "")),
                })

            return [m for m in out if m["match_id"]]
        
        except RuntimeError as e:
            last_error = str(e)
            # On last attempt, re-raise with more context
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to fetch SofaScore data for {date_str} after {max_retries + 1} attempts. "
                    f"Final error: {last_error}"
                )
            
            # On 403, record error and potentially trigger cooldown
            err_text = str(e).lower()
            if "403" in err_text and anti_block:
                anti_block.note_fetch_error(str(e), "fetch_finished_match_ids_for_date")
                # If cooldown just triggered, raise immediately
                if anti_block.get_global_fetch_cooldown_left() > 0:
                    raise
            
            # Exponential backoff before retry
            backoff_secs = 0.5 * (1.5 ** attempt)
            time.sleep(backoff_secs)
        except Exception as e:
            last_error = str(e)
            # On last attempt, re-raise with context
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to fetch SofaScore data for {date_str} after {max_retries + 1} attempts. "
                    f"Final error: {last_error}"
                )
            # Other errors: exponential backoff and retry
            backoff_secs = 0.5 * (1.5 ** attempt)
            time.sleep(backoff_secs)


def fetch_live_match_ids(max_retries: int = 3) -> list[dict]:
    """Return currently live basketball matches from SofaScore.
    
    Parameters
    ----------
    max_retries : int
        Number of times to retry on HTTP 403.
    
    Raises
    ------
    RuntimeError
        If API returns non-OK status after all retries.
    """
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    api_url = "https://api.sofascore.com/api/v1/sport/basketball/events/live"
    
    last_error = None
    # Retry loop with anti-block protection
    for attempt in range(max_retries + 1):
        # Check if we're in cooldown
        if anti_block and anti_block.get_global_fetch_cooldown_left() > 0:
            cooldown_left = anti_block.get_global_fetch_cooldown_left()
            if attempt == 0:
                raise RuntimeError(
                    f"Anti-block cooldown active ({cooldown_left:.0f}s remaining). "
                    f"Retry after approximately {int(cooldown_left) + 1} seconds."
                )
        
        # Apply global request spacing with jitter
        if anti_block:
            delay = anti_block.next_global_fetch_delay_secs()
            if delay > 0:
                time.sleep(delay)

        try:
            with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
                resp = ctx.request.get(api_url, headers=extra_headers, timeout=30_000)
                if not resp.ok:
                    err_msg = f"Live events API returned HTTP {resp.status}"
                    if anti_block:
                        anti_block.note_fetch_error(err_msg, "fetch_live_match_ids")
                    raise RuntimeError(err_msg)

                body = resp.json() or {}
                events = body.get("events", []) if isinstance(body, dict) else []
            
            # Success
            if anti_block:
                anti_block.note_fetch_success()

            out = []
            for ev in events:
                match_id = str(ev.get("id", "") or "")
                if not match_id:
                    continue
                status = (ev.get("status") or {})
                time_info = ev.get("time") or {}
                hs = (ev.get("homeScore") or {}).get("current")
                as_ = (ev.get("awayScore") or {}).get("current")
                out.append({
                    "match_id": match_id,
                    "event_date": "",
                    "status_type": status.get("type", ""),
                    "status_description": status.get("description", ""),
                    "played_seconds": time_info.get("played"),
                    "home_team": (ev.get("homeTeam") or {}).get("name", ""),
                    "away_team": (ev.get("awayTeam") or {}).get("name", ""),
                    "league": ((ev.get("tournament") or {}).get("name", "")),
                    "home_score": hs,
                    "away_score": as_,
                })

            return out
        
        except RuntimeError as e:
            last_error = str(e)
            # On last attempt, re-raise with more context
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to fetch live SofaScore data after {max_retries + 1} attempts. "
                    f"Final error: {last_error}"
                )
            
            # On 403, record and check for cooldown trigger
            err_text = str(e).lower()
            if "403" in err_text and anti_block:
                anti_block.note_fetch_error(str(e), "fetch_live_match_ids")
                if anti_block.get_global_fetch_cooldown_left() > 0:
                    raise
            
            # Exponential backoff before retry
            backoff_secs = 0.5 * (1.5 ** attempt)
            time.sleep(backoff_secs)
        except Exception as e:
            last_error = str(e)
            # On last attempt, re-raise with context
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to fetch live SofaScore data after {max_retries + 1} attempts. "
                    f"Final error: {last_error}"
                )
            # Other errors: exponential backoff and retry
            backoff_secs = 0.5 * (1.5 ** attempt)
            time.sleep(backoff_secs)


def fetch_matches_by_ids(
    match_ids: list[str],
    fetch_h2h: bool = True,
) -> list[tuple[str, dict | None, str | None]]:
    """Fetch multiple matches reusing one browser session.

    Returns a list of tuples: (match_id, parsed_data_or_none, error_or_none).
    """
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    ids = [str(mid).strip() for mid in match_ids if str(mid).strip()]
    if not ids:
        return []

    out: list[tuple[str, dict | None, str | None]] = []

    with _browser_context("https://www.sofascore.com/basketball") as (_, ctx, _page):
        for match_id in ids:
            try:
                resp_event = ctx.request.get(
                    f"https://api.sofascore.com/api/v1/event/{match_id}",
                    headers=extra_headers,
                    timeout=15_000,
                )
                if not resp_event.ok:
                    raise RuntimeError(
                        f"Event API returned HTTP {resp_event.status} for match {match_id}"
                    )
                event_json: dict = resp_event.json()

                resp_inc = ctx.request.get(
                    f"https://api.sofascore.com/api/v1/event/{match_id}/incidents",
                    headers=extra_headers,
                    timeout=20_000,
                )
                if not resp_inc.ok:
                    raise RuntimeError(
                        f"Incidents API returned HTTP {resp_inc.status} for match {match_id}"
                    )
                incidents_json: dict = resp_inc.json()

                resp_graph = ctx.request.get(
                    f"https://api.sofascore.com/api/v1/event/{match_id}/graph",
                    headers=extra_headers,
                    timeout=20_000,
                )
                graph_json: dict = (
                    resp_graph.json() if resp_graph.ok else {"graphPoints": []}
                )

                # H2H (optional)
                h2h_list: list = []
                if fetch_h2h:
                    resp_h2h = ctx.request.get(
                        f"https://api.sofascore.com/api/v1/event/{match_id}/h2h",
                        headers=extra_headers,
                        timeout=20_000,
                    )
                    if resp_h2h.ok:
                        h2h_body = resp_h2h.json() or {}
                        h2h_list = h2h_body.get("h2h") or h2h_body.get("headToHead") or []

                parsed = _parse(
                    event_json,
                    incidents_json.get("incidents", []),
                    graph_json.get("graphPoints", []),
                )
                parsed["h2h"] = _parse_h2h(match_id, h2h_list)

                out.append((match_id, parsed, None))
            except Exception as exc:
                out.append((match_id, None, str(exc)))

    return out


def _fetch_match_payloads(
    warmup_url: str,
    match_id: str,
    fetch_h2h: bool = True,
    fetch_statistics: bool = True,
    fetch_lineups: bool = True,
    fetch_odds: bool = True,
    backend: str | None = None,
) -> tuple[dict, dict, dict, dict | None, dict | None, dict | None, dict | None]:
    """
    Fetch all SofaScore API payloads for a given match.

    Parameters
    ----------
    warmup_url : str
        URL to warm up the browser session (match page or basketball landing).
    match_id : str
        Numeric SofaScore match ID.
    fetch_h2h : bool
        Fetch /event/{id}/h2h (head-to-head history).
    fetch_statistics : bool
        Fetch /event/{id}/statistics.
    fetch_lineups : bool
        Fetch /event/{id}/lineups.
    fetch_odds : bool
        Fetch /event/{id}/odds/1/all (pre-match odds from provider 1).

    Returns
    -------
    Tuple of (event_json, incidents_json, graph_json, h2h_json,
              statistics_json, lineups_json, odds_json).
    """
    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with _browser_context(warmup_url, backend=backend) as (_, ctx, _page):

        # 1. Event metadata
        resp_event = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}",
            headers=extra_headers,
            timeout=15_000,
        )
        if not resp_event.ok:
            raise RuntimeError(
                f"Event API returned HTTP {resp_event.status} for match {match_id}"
            )
        event_json: dict = resp_event.json()

        # 2. Incidents (PBP + fouls + timeouts + substitutions + turnovers)
        resp_inc = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}/incidents",
            headers=extra_headers,
            timeout=30_000,
        )
        if not resp_inc.ok:
            raise RuntimeError(
                f"Incidents API returned HTTP {resp_inc.status} for match {match_id}"
            )
        incidents_json: dict = resp_inc.json()

        # 3. Graph points (pressure/momentum curve)
        resp_graph = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}/graph",
            headers=extra_headers,
            timeout=30_000,
        )
        graph_json: dict = (
            resp_graph.json() if resp_graph.ok else {"graphPoints": []}
        )

        # 4. H2H history (optional)
        h2h_json: dict | None = None
        if fetch_h2h:
            resp_h2h = ctx.request.get(
                f"https://api.sofascore.com/api/v1/event/{match_id}/h2h",
                headers=extra_headers,
                timeout=20_000,
            )
            if resp_h2h.ok:
                h2h_json = resp_h2h.json()

        # 5. Team statistics
        statistics_json: dict | None = None
        if fetch_statistics:
            resp_stats = ctx.request.get(
                f"https://api.sofascore.com/api/v1/event/{match_id}/statistics",
                headers=extra_headers,
                timeout=20_000,
            )
            if resp_stats.ok:
                statistics_json = resp_stats.json()

        # 6. Lineups + per-player statistics
        lineups_json: dict | None = None
        if fetch_lineups:
            resp_lu = ctx.request.get(
                f"https://api.sofascore.com/api/v1/event/{match_id}/lineups",
                headers=extra_headers,
                timeout=20_000,
            )
            if resp_lu.ok:
                lineups_json = resp_lu.json()

        # 7. Betting odds (pre-match from provider 1)
        odds_json: dict | None = None
        if fetch_odds:
            resp_odds = ctx.request.get(
                f"https://api.sofascore.com/api/v1/event/{match_id}/odds/1/all",
                headers=extra_headers,
                timeout=20_000,
            )
            if resp_odds.ok:
                odds_json = resp_odds.json()

    return event_json, incidents_json, graph_json, h2h_json, statistics_json, lineups_json, odds_json
