# =====================================================================
# 1) ¿DONDE CAEN LOS 'CHAMPIONSHIP ROUND'?  2) ¿ALGUNA APUESTA LIVE EN 12 MIN?
# ---------------------------------------------------------------------
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/championship_and_12min.py
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    block("1. ETIQUETAS 'CHAMPIONSHIP' -> bucket de fase normalizado")
    rows = conn.execute(
        "SELECT league, stage, stage_detail, competition_type, quarter_duration_minutes, "
        "match_count FROM leagues_classification WHERE lower(stage) LIKE '%championship%' "
        "OR lower(league) LIKE '%championship%' ORDER BY match_count DESC"
    ).fetchall()
    print(f"  {'liga':<46}{'stage':<26}{'stage_detail':<24}{'comp':<12}{'m_cnt':>7}")
    for r in rows:
        print(f"  {r['league'][:44]:<46}{str(r['stage'])[:24]:<26}"
              f"{str(r['stage_detail'])[:22]:<24}{str(r['competition_type'])[:10]:<12}"
              f"{r['match_count'] or 0:>7,}")
    print(f"\n  total ligas con 'championship': {len(rows)}")

    block("2. APUESTAS EN VIVO: ¿alguna en liga de 12 min (o de alto Q4)?")
    bets = [dict(r) for r in conn.execute(
        """
        SELECT l.match_id, l.model_version, l.signal_type, l.picked_side, l.confidence,
               l.inference_minute, l.result, s.league,
               q.q4_home, q.q4_away, lc.stage_detail, lc.quarter_duration_minutes,
               lc.avg_q4_total_points, lc.competition_type, lc.is_youth, lc.is_college
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
        LEFT JOIN leagues_classification lc ON lc.league = s.league
        WHERE l.model_version IN ('m27_v5_2','m27_v5_3')
          AND l.signal_type LIKE '%BET%' AND l.signal_type NOT LIKE '%NO_BET%'
        ORDER BY l.created_at
        """
    ).fetchall()]
    print(f"  apuestas totales (v5_2 + v5_3): {len(bets)}")
    print(f"  {'liga':<44}{'q_dur':>6}{'Q4 medio':>10}{'fase':<22}{'comp':<10}{'Q4 real':>10}")
    for b in bets:
        q4r = "-"
        if b["q4_home"] is not None and b["q4_away"] is not None:
            q4r = f"{int(b['q4_home']) - int(b['q4_away']):+d}"
        print(f"  {str(b['league'])[:42]:<44}{str(b['quarter_duration_minutes']):>6}"
              f"{(b['avg_q4_total_points'] or 0):>10.1f}{str(b['stage_detail'])[:20]:<22}"
              f"{str(b['competition_type'])[:8]:<10}{q4r:>10}")

    block("3. PARTIDOS EN VIVO DONDE EL Q4 REAL SUPERA 45 (firma de 12 min)")
    altos = [b for b in bets if b["q4_home"] is not None and b["q4_away"] is not None
             and (int(b["q4_home"]) + int(b["q4_away"])) >= 45]
    if not altos:
        print("  ninguno: ninguna apuesta en vivo se hizo en un partido con Q4 de 45+ puntos")
    for b in altos:
        print(f"  {b['league']} | Q4 {b['q4_home']}-{b['q4_away']} | q_dur={b['quarter_duration_minutes']} "
              f"| {b['signal_type']}")

    block("4. LIGAS 12 MIN REALES PERO MARCADAS q_dur=10 (candidatas a corregir)")
    for r in conn.execute(
        "SELECT league, stage_detail, avg_total_points, avg_q4_total_points, match_count, "
        "quarter_duration_minutes FROM leagues_classification "
        "WHERE quarter_duration_minutes = 10 AND avg_q4_total_points >= 45 "
        "AND match_count >= 50 ORDER BY avg_q4_total_points DESC"
    ):
        print(f"  {r['league'][:56]:<58}{r['match_count'] or 0:>6} part.  "
              f"pts={r['avg_total_points']:>6.1f}  Q4={r['avg_q4_total_points']:>5.1f}")
    conn.close()


if __name__ == "__main__":
    main()
