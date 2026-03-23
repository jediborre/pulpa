"""
SofaScore basketball match data fetcher.

Approach: headless Chromium via Playwright.  The browser navigates
to the match page (to establish a valid session/cookies),
then ctx.request.get() fetches the JSON API endpoints automatically
handed the browser's session cookies.

SofaScore blocks direct requests (403) but handles requests from
within a browser context normally.

Data sources fetched for each match:
    - /event/{id}           -> metadata + final/quarter scores
    - /event/{id}/incidents -> play-by-play scoring incidents
    - /event/{id}/graph     -> pressure/momentum curve points
"""

import re
from datetime import datetime, timezone


STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Basketball: incidentClass / 'from' field values → point value
# SofaScore uses camelCase for incidentClass ("threePoints") and
# lowercase for the 'from' field ("threepoints").  Both are mapped here.
_CLASS_PTS: dict[str, int] = {
    "onepoint": 1,   "twopoints": 2,   "threepoints": 3,   # 'from' field
    "onePoint": 1,   "twoPoints": 2,   "threePoints": 3,   # incidentClass
    # Legacy soccer / generic sport scoring types kept for compatibility
    "points1": 1,    "points2": 2,     "points3": 3,
}

_PERIOD: dict[int | str, str] = {
    1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4",
    "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4",
}


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── core parser ───────────────────────────────────────────────────────────────

def _parse(event_json: dict, incidents: list, graph_points: list | None = None) -> dict:
    """
    Convert raw SofaScore API responses into our canonical match dict.

    Parameters
    ----------
    event_json : response body from GET /event/{id}
    incidents  : list from GET /event/{id}/incidents  →  .incidents[]
    graph_points : list from GET /event/{id}/graph → .graphPoints[]
    """
    ev = event_json.get("event", event_json)

    home = ev["homeTeam"]["name"]
    away = ev["awayTeam"]["name"]

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

    # Play-by-play (scoring incidents only)
    # Basketball uses incidentType="goal"; soccer/generic uses "score".
    pbp: dict[str, list] = {}
    for inc in incidents:
        if inc.get("incidentType") not in ("goal", "score"):
            continue

        # Points: try incidentClass (basketball), then 'from', then scoringType
        pts = (
            _CLASS_PTS.get(inc.get("incidentClass", ""))
            or _CLASS_PTS.get(inc.get("from", ""))
            or _CLASS_PTS.get(inc.get("scoringType", ""))
            or 0
        )
        if pts == 0:
            continue

        # Period: basketball 'goal' incidents often lack a 'period' field.
        # Prefer explicit period field; fall back to deriving from timeSeconds.
        period_raw = inc.get("period") or inc.get("periodType")
        if period_raw is not None:
            q = _quarter_label(period_raw)
        else:
            q = _period_from_game_seconds(inc.get("timeSeconds", 0))

        player_obj = inc.get("player") or {}
        player = player_obj.get("shortName") or player_obj.get("name", "")

        pbp.setdefault(q, []).append({
            "time": _time_str(inc),
            "player": player,
            "points": pts,
            "team": "home" if inc.get("isHome", True) else "away",
            "home_score": inc.get("homeScore"),
            "away_score": inc.get("awayScore"),
        })

    return {
        "match": {
            "home_team": home,
            "away_team": away,
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M"),
            "venue": venue,
            "league": league,
            "home_record": "",   # requires a separate standings endpoint
            "away_record": "",
        },
        "score": {
            "home": home_total,
            "away": away_total,
            "quarters": quarters,
        },
        "play_by_play": pbp,
        "graph_points": graph_points or [],
    }


# ── public fetch function ─────────────────────────────────────────────────────

def fetch_match(url: str, match_id: str) -> dict:
    """
    Fetch match data using Playwright's BrowserContext.request API.

    Strategy:
      1. Launch headless Chromium and navigate to the SofaScore match page.
         This satisfies Cloudflare bot-detection and plants session cookies.
        2. Call these SofaScore JSON endpoints via ctx.request.get():
            - /event/{id}
            - /event/{id}/incidents
            - /event/{id}/graph
            The Playwright HTTP client shares the browser context's cookie jar,
            so SofaScore sees these as legitimate browser requests.

    No direct requests (blocked with 403), no JS injection, no DOM scraping.
    """
    event_json, incidents_json, graph_json = _fetch_match_payloads(
        warmup_url=url,
        match_id=match_id,
    )

    if not event_json or "event" not in event_json:
        raise RuntimeError(
            "Playwright could not retrieve event data — "
            "verify the match URL and try again."
        )

    return _parse(
        event_json,
        incidents_json.get("incidents", []),
        graph_json.get("graphPoints", []),
    )


def fetch_match_by_id(match_id: str) -> dict:
    """Fetch match data by ID after warming session on basketball landing page."""
    event_json, incidents_json, graph_json = _fetch_match_payloads(
        warmup_url="https://www.sofascore.com/basketball",
        match_id=match_id,
    )
    return _parse(
        event_json,
        incidents_json.get("incidents", []),
        graph_json.get("graphPoints", []),
    )


def fetch_event_snapshot(match_id: str) -> dict:
    """Fetch lightweight event state for live/FT friendly reporting."""
    from playwright.sync_api import sync_playwright

    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=STANDARD_UA)
        page = ctx.new_page()
        try:
            page.goto(
                "https://www.sofascore.com/basketball",
                wait_until="networkidle",
                timeout=45_000,
            )
        except Exception:
            pass

        resp = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}",
            headers=extra_headers,
            timeout=20_000,
        )
        if not resp.ok:
            browser.close()
            raise RuntimeError(
                f"Event API returned HTTP {resp.status} for match {match_id}"
            )
        body = resp.json() or {}
        browser.close()

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


def fetch_finished_match_ids_for_date(date_str: str) -> list[dict]:
    """Return finished basketball matches for a date (YYYY-MM-DD)."""
    from playwright.sync_api import sync_playwright

    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    api_url = (
        "https://api.sofascore.com/api/v1/"
        f"sport/basketball/scheduled-events/{date_str}"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=STANDARD_UA)
        page = ctx.new_page()

        try:
            page.goto(
                "https://www.sofascore.com/basketball",
                wait_until="networkidle",
                timeout=45_000,
            )
        except Exception:
            pass

        resp = ctx.request.get(api_url, headers=extra_headers, timeout=30_000)
        if not resp.ok:
            browser.close()
            raise RuntimeError(
                f"Daily events API returned HTTP {resp.status} for {date_str}"
            )

        body = resp.json() or {}
        events = body.get("events", []) if isinstance(body, dict) else []
        browser.close()

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


def _fetch_match_payloads(warmup_url: str, match_id: str) -> tuple[dict, dict, dict]:
    from playwright.sync_api import sync_playwright

    extra_headers = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=STANDARD_UA)
        page = ctx.new_page()

        try:
            page.goto(warmup_url, wait_until="networkidle", timeout=45_000)
        except Exception:
            pass

        resp_event = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}",
            headers=extra_headers,
            timeout=15_000,
        )
        if not resp_event.ok:
            browser.close()
            raise RuntimeError(
                f"Event API returned HTTP {resp_event.status} for match {match_id}"
            )
        event_json: dict = resp_event.json()

        resp_inc = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}/incidents",
            headers=extra_headers,
            timeout=30_000,
        )
        if not resp_inc.ok:
            browser.close()
            raise RuntimeError(
                f"Incidents API returned HTTP {resp_inc.status} for match {match_id}"
            )
        incidents_json: dict = resp_inc.json()

        resp_graph = ctx.request.get(
            f"https://api.sofascore.com/api/v1/event/{match_id}/graph",
            headers=extra_headers,
            timeout=30_000,
        )
        graph_json: dict = (
            resp_graph.json() if resp_graph.ok else {"graphPoints": []}
        )
        browser.close()

    return event_json, incidents_json, graph_json
