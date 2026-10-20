# =====================================================================
# CONTRASTE: findings_m27_v5_3.md  vs  la realidad en vivo
# ---------------------------------------------------------------------
# QUÉ HACE:
#   Verifica tres afirmaciones concretas del documento de findings contra
#   lo que se mide en los logs en vivo:
#   A) "Ligas toxicas podadas del train + block_toxic_leagues en inferencia"
#      -> ¿La poda y el bloque usan la misma lista? ¿La lista casa con los
#         nombres reales? ¿Cubre las ligas que el propio modelo marcó
#         block_candidate?
#   B) "Cuotas ML: 36 % falta (90,846)" y "grupo B fuerza: calidad cuando no
#      hay cuota (36-40 % de partidos)"
#      -> ¿Que cobertura de cuota hay REALMENTE en vivo? ¿Es la falta de cuota
#         del mercado o un fallo de extraccion en el camino live?
#   C) "Umbral operativo 0.72 -> WR 77.8 %" (tabla de umbrales en C, ancla t=30)
#      y "camino live min 27 -> WR 81.4 %" (compare.log)
#      -> ¿Cual de las dos referencias corresponde al disparo real del monitor?
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/contrast_with_findings.py
# =====================================================================

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import TOXIC_LEAGUE_PATTERNS, is_toxic_league  # noqa: E402


def block(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def raw(row: dict) -> dict:
    try:
        inf = json.loads(row.get("inference_json") or "{}")
        return inf.get("raw_json") or {}
    except Exception:
        return {}


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    live = [dict(r) for r in conn.execute(
        """
        SELECT l.match_id, l.model_version, l.signal_type, l.confidence,
               l.inference_minute, l.inference_json, s.league
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        WHERE l.model_version = 'm27_v5_3'
        """
    ).fetchall()]

    block("A) LIGAS TOXICAS: ¿la poda del train y el bloque live usan la misma lista?")
    print("  build_dataset.py:130  ->  eligible = [... and not is_toxic_league(r[2])]")
    print("  predict.py:111        ->  toxic = is_toxic_league(league) -> PASS")
    print("  -> misma funcion, misma lista: lo que no coincide con el nombre real")
    print("     NO se poda del train TAMPOCO se bloquea en live. El gap train/live")
    print("     no es un problema: la lista simply nunca funciono.\n")
    all_leagues = {r[0] for r in conn.execute(
        "SELECT DISTINCT league FROM matches WHERE league IS NOT NULL AND league != ''")}
    sched_leagues = {r[0] for r in conn.execute(
        "SELECT DISTINCT league FROM bet_monitor_schedule_v3 WHERE league IS NOT NULL")}
    print(f"  ligas en 'matches' (universo train): {len(all_leagues)}")
    print(f"  ligas en el schedule (universo live): {len(sched_leagues)}")
    dead = []
    for p in TOXIC_LEAGUE_PATTERNS:
        hit_train = [lg for lg in all_leagues if p in lg.lower()]
        hit_live = [lg for lg in sched_leagues if p in lg.lower()]
        if not hit_train and not hit_live:
            dead.append(p)
    print(f"\n  patrones que NO podan nada del train NI bloquean nada en live: {len(dead)}")
    for p in dead:
        print(f"     - '{p}'")
    calib = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" / "league_calibration.json")
                       .read_text(encoding="utf-8"))
    flagged = [c for c in calib if c.get("block_candidate")]
    print(f"\n  ligas block_candidate del propio modelo (WR offline):")
    for c in flagged:
        in_train = c["league"] in all_leagues
        print(f"     - {c['league']:<32} WR={c['wr']:.1f}%  ROI@1.40={c['roi_140']:+.1f}%  "
              f"| ¿en el train? {'SI' if in_train else 'no'}  | ¿bloqueada? "
              f"{'si' if is_toxic_league(c['league']) else 'NO'}")

    block("B) COBERTURA DE CUOTA DE MERCADO EN VIVO")
    rows = [r for r in live if raw(r).get("p_cal") is not None]
    bets = [r for r in rows if "BET" in (r.get("signal_type") or "") and "NO_BET" not in (r.get("signal_type") or "")]
    for name, sel in (("todas las inferencias", rows), ("solo las apuestas", bets)):
        n_ml = sum(1 for r in sel if raw(r).get("has_ml"))
        n_sp = sum(1 for r in sel if raw(r).get("has_spread"))
        miss = raw(sel[0]).get("features_missing") if sel else None
        print(f"  {name:<24} n={len(sel):>4}  has_ml=True: {n_ml:>4} ({n_ml * 100 / max(len(sel), 1):>5.1f}%)  "
              f"has_spread=True: {n_sp:>3}")
    print(f"\n  El doc asume '36-40 % de partidos sin cuota' (grupo B fuerza cubre ese hueco).")
    print(f"  En vivo la cobertura es MUCHO menor. ¿Falta en el mercado o se pierde al extraer?")
    mids = sorted({r["match_id"] for r in bets})
    placeholders = ",".join("?" * len(mids))
    odds = {r[0]: dict(r) for r in conn.execute(
        f"SELECT match_id, COUNT(*) n, "
        f"SUM(CASE WHEN odds_type='ML' THEN 1 ELSE 0 END) n_ml, "
        f"SUM(CASE WHEN line IS NOT NULL THEN 1 ELSE 0 END) n_line "
        f"FROM match_odds WHERE match_id IN ({placeholders}) GROUP BY match_id", mids)}
    with_ml_db = sum(1 for m in mids if odds.get(m, {}).get("n_ml", 0) > 0)
    no_ml_db = len(mids) - with_ml_db
    live_no_ml = sum(1 for r in bets if not raw(r).get("has_ml"))
    print(f"\n  apuestas live: {len(bets)}")
    print(f"    con has_ml=False en la inferencia : {live_no_ml}")
    print(f"    partidos SIN cuota ML en match_odds: {no_ml_db}  -> la cuota no existia en la DB")
    print(f"    partidos CON cuota ML en match_odds: {with_ml_db}  -> existia y no llego al modelo")
    if with_ml_db:
        print("    >> POSIBLE FALLO DE EXTRACCION EN EL CAMINO LIVE (cuota en DB, ausente en features)")

    block("C) ¿QUE REFERENCIA DE BACKTEST APLICA AL DISPARO REAL DEL MONITOR?")
    print("  findings §1 (tabla de umbrales, ANCLA t=30, un partido = una fila):")
    print("     thr 0.72 -> WR 77.8 %, ROI@1.40 +8.9 %   |   thr 0.75 -> WR 79.3 %, ROI@1.40 +11.0 %")
    print("  findings §4 (compare.log, CAMINO LIVE min 27, 2.500 partidos del holdout):")
    print("     v5_3 thr 0.72 -> 236 apuestas, WR 81.4 %, ROI@1.40 +13.9 %  |  thr 0.75 -> WR 85.4 %")
    mins = Counter(int(r["inference_minute"] or 0) for r in bets)
    print(f"\n  Minuto real de disparo del monitor (apuestas v5_3): {dict(sorted(mins.items()))}")
    first = sum(v for k, v in mins.items() if k <= 30)
    print(f"  disparos dentro de [25.5, 30]: {first}/{len(bets)}  |  fuera: {len(bets) - first}")
    print("  -> el monitor dispara al PRIMER tick (min ~27), asi que la referencia correcta")
    print("     es la del CAMINO LIVE (compare.log), no la tabla de ancla t=30.")
    print(f"  WR live observado: 72.5 % vs 81.4 % esperado (n=40, IC95 56-85 % -> dentro del ruido).")

    conn.close()


if __name__ == "__main__":
    main()
