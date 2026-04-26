"""
Pulpa API - Replica la logica exacta del bot de Telegram.

Para CADA match del dia dado y modelo solicitado:
  - v12, v13, v15, v16, v17 -> _run_vXX_inference (mismo codigo del bot)
  - v2, v4, v6, v9     -> infer_match.run_inference con force_version

Ademas, si ya hay datos en bet_monitor_log (v12, v13) los usa directo
sin recomputar para ser mas rapido.
"""
import sys
import json
import importlib
import logging
import sqlite3
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
MATCH_DIR = ROOT / "match"
DB_PATH = MATCH_DIR / "matches.db"
CACHE_DIR = ROOT / "api_cache"   # JSON files: api_cache/v13_2026-04-17.json
CACHE_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(MATCH_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pulpa_api")

app = FastAPI(title="Pulpa API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Cache de engines pesados (cargados 1 vez)
ENGINES: Dict[str, Any] = {}

# Estado de progreso en tiempo real { "v4_2026-04-17": {done, total, bets, finished} }
PROGRESS: Dict[str, Dict] = {}

def _prog_key(version: str, date: str) -> str:
    return f"{version}_{date}"

def _prog_set(version: str, date: str, done: int, total: int, bets: int, finished: bool = False):
    PROGRESS[_prog_key(version, date)] = {
        "done": done, "total": total, "bets": bets,
        "pct": round(done / total * 100) if total else 0,
        "finished": finished,
    }

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Cache en disco ───────────────────────────────────────────────────────────
def _cache_path(version: str, date: str) -> Path:
    return CACHE_DIR / f"{version}_{date}.json"

def _cache_read(version: str, date: str) -> Optional[List[Dict]]:
    p = _cache_path(version, date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _cache_write(version: str, date: str, data: List[Dict]) -> None:
    p = _cache_path(version, date)
    try:
        p.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Cache guardado: %s", p.name)
    except Exception as e:
        logger.warning("No se pudo guardar cache: %s", e)


# ─── Cargar engines ───────────────────────────────────────────────────────────
def _get_v15_engine():
    if "v15" not in ENGINES:
        logger.info("Cargando V15 engine...")
        mod = importlib.import_module("training.v15.inference")
        ENGINES["v15"] = mod.V15Engine.load(mod.SUMMARY_PATH)
    return ENGINES["v15"]

def _get_v16_engine():
    if "v16" not in ENGINES:
        logger.info("Cargando V16 engine...")
        mod = importlib.import_module("training.v16.inference")
        ENGINES["v16"] = mod.V15Engine.load(mod.SUMMARY_PATH)
    return ENGINES["v16"]

def _get_v17_engine():
    if "v17" not in ENGINES:
        logger.info("Cargando V17 engine...")
        mod = importlib.import_module("training.v17.inference")
        ENGINES["v17"] = mod.V15Engine.load(mod.SUMMARY_PATH)
    return ENGINES["v17"]


# ─── Helper: determinar ganador ──────────────────────────────────────────────
def _winner(home: Optional[int], away: Optional[int]) -> Optional[str]:
    if home is None or away is None:
        return None
    if home > away: return "home"
    if away > home: return "away"
    return "push"


# ─── Obtener datos completos de un match desde la DB ─────────────────────────
def _get_match_data(match_id: str) -> Optional[dict]:
    """Fetches match data from DB tables (same structure as scraper)."""
    conn = _db()
    c = conn.cursor()
    c.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
    m = c.fetchone()
    if not m:
        conn.close()
        return None
    m = dict(m)

    c.execute("SELECT quarter, home, away FROM quarter_scores WHERE match_id = ?", (match_id,))
    quarters = {}
    for r in c.fetchall():
        quarters[r["quarter"]] = {"home": r["home"], "away": r["away"]}

    c.execute("SELECT seq, minute, value FROM graph_points WHERE match_id = ? ORDER BY seq", (match_id,))
    gp = [{"seq": r["seq"], "minute": r["minute"], "value": r["value"]} for r in c.fetchall()]

    c.execute("SELECT quarter, seq, time, player, points, team FROM play_by_play WHERE match_id = ? ORDER BY seq", (match_id,))
    pbp_raw = c.fetchall()
    pbp: Dict[str, list] = {}
    for r in pbp_raw:
        q = r["quarter"]
        if q not in pbp:
            pbp[q] = []
        pbp[q].append(dict(r))

    conn.close()

    total_home = sum(q.get("home", 0) or 0 for q in quarters.values())
    total_away = sum(q.get("away", 0) or 0 for q in quarters.values())

    return {
        "match": {
            "match_id": match_id,
            "home_team": m.get("home_team", ""),
            "away_team": m.get("away_team", ""),
            "league": m.get("league", ""),
            "date": m.get("date", ""),
            "status_type": m.get("status_type", "finished"),
            "gender": m.get("gender", "men"),
        },
        "score": {
            "home": total_home, "away": total_away,
            "quarters": quarters,
        },
        "graph_points": gp,
        "play_by_play": pbp,
    }


# ─── Helper: volatility & pace proxy ─────────────────────────────────────────
def _calc_volatility_and_pace(data: dict, target: str) -> tuple[float, str]:
    volatility = 0.0
    try:
        from training.v12 import infer_match_v12
        cutoff = 24 if target == "q3" else 36
        vol = infer_match_v12._compute_volatility(data, cutoff)
        if vol is not None:
            volatility = float(vol)
    except Exception:
        pass
    
    quarters = data.get("score", {}).get("quarters", {})
    def get_q(q): return (quarters.get(q) or {}).get("home", 0) + (quarters.get(q) or {}).get("away", 0)
    
    total_pts = get_q("Q1") + get_q("Q2")
    if target == "q4":
        total_pts += get_q("Q3")
        
    pace_bucket = "medium"
    if target == "q3":
        if total_pts > 0 and total_pts < 75: pace_bucket = "slow"
        elif total_pts > 86: pace_bucket = "fast"
    else:
        if total_pts > 0 and total_pts < 114: pace_bucket = "slow"
        elif total_pts > 130: pace_bucket = "fast"
        
    return volatility, pace_bucket


# ─── Inferencia por modelo (mismo que bot) ────────────────────────────────────
def _infer(match_id: str, target: str, version: str, data: dict) -> dict:
    """Ejecuta inferencia para un match/target/version."""
    try:
        if version == "v13":
            mod = importlib.import_module("training.v13.infer_match_v13")
            res = mod.run_inference(match_id=match_id, target=target)
            if not res.get("ok"):
                return {"ok": False}
            pred = res["prediction"]
            signal = getattr(pred, "winner_signal", "NO_BET") or "NO_BET"
            pick = getattr(pred, "winner_pick", None)
            confidence = float(getattr(pred, "winner_confidence", 0.5) or 0.5)

        elif version in ("v15", "v16", "v17"):
            engine = (
                _get_v15_engine() if version == "v15"
                else _get_v16_engine() if version == "v16"
                else _get_v17_engine()
            )
            m = data.get("match", {})
            s = data.get("score", {})
            quarters = s.get("quarters", {})
            gp = data.get("graph_points", [])
            pbp_dict = data.get("play_by_play", {})
            pbp = []
            for q_code, evts in (pbp_dict or {}).items():
                for e in (evts or []):
                    pbp.append({**e, "quarter": q_code})

            league = str(m.get("league", "Unknown"))

            def _qsc(q_code, side):
                return int((quarters.get(q_code) or {}).get(side) or 0)

            quarter_scores = {
                "q1_home": _qsc("Q1", "home"), "q1_away": _qsc("Q1", "away"),
                "q2_home": _qsc("Q2", "home"), "q2_away": _qsc("Q2", "away"),
            }
            if target == "q4":
                quarter_scores["q3_home"] = _qsc("Q3", "home")
                quarter_scores["q3_away"] = _qsc("Q3", "away")

            pred = engine.predict(
                match_id=match_id, target=target, league=league,
                quarter_scores=quarter_scores, graph_points=gp, pbp_events=pbp,
            )
            # --- V17 Multi-Snapshot Support ---
            if version == "v17" and pred.signal == "NO_BET":
                from training.v17 import config as v17_config
                snaps = v17_config.Q3_LIVE_SNAPSHOTS if target == "q3" else v17_config.Q4_LIVE_SNAPSHOTS
                for s_min in snaps:
                    if s_min == pred.debug.snapshot_minute: continue
                    p_alt = engine.predict(
                        match_id=match_id, target=target, league=league,
                        quarter_scores=quarter_scores, graph_points=gp, pbp_events=pbp,
                        snapshot_minute=s_min
                    )
                    if p_alt.signal != "NO_BET":
                        pred = p_alt
                        break

            signal = pred.signal
            pick = "home" if signal == "BET_HOME" else ("away" if signal == "BET_AWAY" else None)
            confidence = float(getattr(pred, "confidence", 0.5) or 0.5)

        elif version == "v12":
            mod = importlib.import_module("training.v12.infer_match_v12")
            res = mod.run_inference(match_id=match_id, target=target, fetch_missing=False)
            if isinstance(res, dict) and not res.get("ok", True):
                return {"ok": False}
            signal = getattr(res, "winner_signal", None) or (res.get("bet_signal") if isinstance(res, dict) else "NO_BET")
            pick = getattr(res, "winner_pick", None) or (res.get("predicted_winner") if isinstance(res, dict) else None)
            confidence = float(getattr(res, "winner_confidence", 0.5) or 0.5)

        else:
            # v2, v4, v6, v9
            infer_mod = importlib.import_module("training.infer_match")
            try:
                infer_mod.scraper_mod.fetch_event_snapshot = lambda _: None
            except Exception:
                pass
            res = infer_mod.run_inference(
                match_id=match_id,
                metric="f1",
                fetch_missing=False,
                force_version={"q3": version, "q4": version},
            )
            if not res.get("ok"):
                return {"ok": False}
            pred = res.get("predictions", {}).get(target, {})
            if not pred.get("available"):
                return {"ok": False}
            signal = str(pred.get("bet_signal") or pred.get("signal") or "NO_BET")
            pick = pred.get("predicted_winner")
            confidence = float(pred.get("confidence") or 0.5)

        NO_BET_SIGS = {"NO_BET", "NO BET", "UNAVAILABLE", "ERROR", "window_missed", "no_data", "no_graph"}
        if not signal or signal.upper().replace("_", " ") in {s.replace("_", " ") for s in NO_BET_SIGS}:
            return {"ok": True, "bet": False}

        if not pick:
            pick = "home" if "HOME" in str(signal).upper() else "away"

        vol, pace = _calc_volatility_and_pace(data, target)

        return {"ok": True, "bet": True, "signal": signal, "pick": pick, "confidence": confidence, "volatility": vol, "pace_bucket": pace}

    except Exception as exc:
        logger.warning("infer error match=%s target=%s version=%s: %s", match_id, target, version, exc)
        return {"ok": False}


# ─── Resolver hit/miss desde quarter_scores ───────────────────────────────────
def _resolve_hit(pick: str, match_id: str, target: str, conn) -> str:
    rows = conn.execute(
        "SELECT quarter, home, away FROM quarter_scores WHERE match_id = ?",
        (match_id,)
    ).fetchall()
    q_map = {r["quarter"]: (r["home"], r["away"]) for r in rows}
    q_key = "Q3" if target == "q3" else "Q4"
    if q_key not in q_map:
        return "pending"
    home, away = q_map[q_key]
    if home is None or away is None:
        return "pending"
    # Consider resolved if both scores exist and the quarter had actual activity
    # (home + away > 0 means the quarter was played, regardless of status_type).
    # This avoids dropping historical matches that were never marked 'finished' in the DB.
    try:
        h, a = int(home), int(away)
    except (TypeError, ValueError):
        return "pending"
    if h == 0 and a == 0:
        # Zero-zero could mean the quarter hasn't been played yet — fall back to status check
        m_row = conn.execute("SELECT status_type FROM matches WHERE match_id = ?", (match_id,)).fetchone()
        if not m_row or str(m_row["status_type"] or "").lower() != "finished":
            return "pending"
    actual = _winner(home, away)
    if actual == "push":
        return "push"
    return "hit" if actual == pick else "miss"


# ─── Endpoint principal ────────────────────────────────────────────────────────
@app.get("/api/compute/{version}/{date}")
def compute(version: str, date: str, force: bool = Query(False, description="Forzar recalculo ignorando cache")):
    """
    Computa stats para version+fecha.
    - ?force=false (default): usa cache en disco si existe, si no calcula y guarda
    - ?force=true: ignora cache, recalcula y sobreescribe
    """
    logger.info("compute version=%s date=%s force=%s", version, date, force)

    VALID_VERSIONS = {"v2", "v4", "v6", "v9", "v12", "v13", "v15", "v16", "v17"}
    if version not in VALID_VERSIONS:
        raise HTTPException(status_code=404, detail=f"Version desconocida: {version}")

    # ── Paso 0: Intentar servir desde cache ──────────────────────────────────
    if not force:
        cached = _cache_read(version, date)
        if cached is not None:
            logger.info("Cache HIT version=%s date=%s results=%d", version, date, len(cached))
            return cached

    try:
        conn = _db()
        results = []

        # ── Paso 1: eliminado ────────────────────────────────────────────────
        # Todos los modelos usan inferencia completa (Paso 2) para garantizar
        # consistencia con el bot de Telegram. El bet_monitor_log puede tener
        # bets incompletos si el daemon no estaba activo en todo momento.
        if False:  # deshabilitado — kept for reference only
            log_rows = conn.execute("""
                SELECT match_id, target, signal, pick, confidence, result, league
                FROM bet_monitor_log
                WHERE event_date = ? AND model = ?
                  AND signal NOT IN ('NO_BET','NO BET','UNAVAILABLE','ERROR','window_missed','no_data','no_graph')
                ORDER BY match_id, id ASC
            """, (date, version)).fetchall()

            seen_keys = set()
            for row in log_rows:
                key = f"{row['match_id']}_{row['target']}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                pick = str(row["pick"] or "").lower()
                sig = str(row["signal"] or "")
                if not pick:
                    pick = "home" if "HOME" in sig.upper() else "away"

                outcome = _resolve_hit(pick, row["match_id"], row["target"], conn)
                if outcome == "pending":
                    db_res = str(row["result"] or "pending").lower()
                    outcome = {"win": "hit", "loss": "miss", "push": "push"}.get(db_res, "pending")
                if outcome == "pending":
                    continue

                league = str(row["league"] or "Unknown")
                gender = "women" if any(w in league.lower() for w in ["women", "femenin", "wnba"]) else "men"
                hit = (outcome == "hit")
                results.append({
                    "match_id": row["match_id"], "target": row["target"],
                    "date": date, "league": league, "volatility": 0,
                    "gender": gender, "pace_bucket": "medium",
                    "confidence": float(row["confidence"] or 0.6),
                    "winner_pick": pick,
                    "target_winner": pick if hit else ("away" if pick == "home" else "home"),
                    "hit": hit, "outcome": outcome,
                })

            if results:
                conn.close()
                _cache_write(version, date, results)
                logger.info("done (monitor_log) version=%s date=%s results=%d", version, date, len(results))
                return results

        # ── Paso 2: Computar en tiempo real para cada match del dia ──────────
        # Usa el mismo offset UTC-6 que el bot para que los matches nocturnos
        # (guardados como 2026-04-10T00:xx UTC pero jugados el 9 en horario local)
        # queden incluidos correctamente.
        UTC_OFFSET_HOURS = -6
        match_rows2 = conn.execute(
            f"""
            SELECT match_id, league, status_type
            FROM matches
            WHERE date(datetime(date || ' ' || COALESCE(time,'00:00'), '{UTC_OFFSET_HOURS} hours')) = ?
            """,
            (date,)
        ).fetchall()
        conn.close()

        if not match_rows2:
            return []

        total_matches = len(match_rows2)
        bets_found = 0
        _prog_set(version, date, 0, total_matches, 0)

        def _progress_bar(done: int, total: int, bets: int, width: int = 28) -> str:
            pct = done / total if total else 0
            filled = int(pct * width)
            bar = "█" * filled + "░" * (width - filled)
            return f"[{version.upper()} {date}] [{bar}] {done}/{total} ({pct*100:.0f}%) | BETs: {bets}"

        logger.info("▶ Iniciando calculo | version=%s date=%s | %d partidos", version, date, total_matches)
        logger.info(_progress_bar(0, total_matches, 0))

        for idx, row in enumerate(match_rows2, 1):
            match_id = str(row["match_id"])
            league = str(row["league"] or "Unknown")
            gender = "women" if any(w in league.lower() for w in ["women", "femenin", "wnba"]) else "men"

            data = _get_match_data(match_id)
            if not data:
                if idx % 10 == 0 or idx == total_matches:
                    logger.info(_progress_bar(idx, total_matches, bets_found))
                continue

            for target in ("q3", "q4"):
                res = _infer(match_id, target, version, data)
                if not res.get("ok") or not res.get("bet"):
                    continue

                pick = str(res.get("pick") or "")
                if not pick:
                    continue

                conn2 = _db()
                outcome = _resolve_hit(pick, match_id, target, conn2)
                conn2.close()

                if outcome == "pending":
                    continue

                hit = (outcome == "hit")
                bets_found += 1
                results.append({
                    "match_id": match_id, "target": target,
                    "date": date, "league": league, 
                    "volatility": round(res.get("volatility", 0.0), 3),
                    "gender": gender, 
                    "pace_bucket": res.get("pace_bucket", "medium"),
                    "confidence": round(float(res.get("confidence", 0.5)), 4),
                    "winner_pick": pick,
                    "target_winner": pick if hit else ("away" if pick == "home" else "home"),
                    "hit": hit, "outcome": outcome,
                })

            # Log + progress update every 5 matches or at end
            if idx % 5 == 0 or idx == total_matches:
                logger.info(_progress_bar(idx, total_matches, bets_found))
                _prog_set(version, date, idx, total_matches, bets_found)

        _prog_set(version, date, total_matches, total_matches, bets_found, finished=True)
        logger.info("✔ Calculo completado | version=%s date=%s | BETs=%d/%d partidos",
                    version, date, bets_found, total_matches)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error version=%s date=%s", version, date)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("done version=%s date=%s results=%d", version, date, len(results))
    # ── Guardar en cache ──────────────────────────────────────────────────────
    _cache_write(version, date, results)
    return results


@app.get("/api/progress/{version}/{date}")
async def get_progress(version: str, date: str):
    """Retorna el progreso actual del calculo en tiempo real."""
    key = _prog_key(version, date)
    if key not in PROGRESS:
        return {"done": 0, "total": 0, "bets": 0, "pct": 0, "finished": False, "active": False}
    return {**PROGRESS[key], "active": True}


@app.get("/api/matches/count/{date}")
async def match_count(date: str):
    """Retorna el total de partidos de un dia."""
    try:
        conn = _db()
        c = conn.execute(
            "SELECT COUNT(*) as c FROM matches WHERE date(datetime(date || ' ' || COALESCE(time,'00:00'), '-6 hours')) = ?",
            (date,)
        )
        count = c.fetchone()["c"]
        conn.close()
        return {"date": date, "count": count}
    except Exception:
        return {"date": date, "count": 0}


@app.get("/api/health")
async def health():
    return {"status": "ok", "db": str(DB_PATH), "engines": list(ENGINES.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
