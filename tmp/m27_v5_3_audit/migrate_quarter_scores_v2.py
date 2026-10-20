# =====================================================================
# MIGRACION: quarter_scores (legacy, formato largo) -> quarter_scores_v2
#            (canonica, formato ancho)
# ---------------------------------------------------------------------
# POR QUE:
#   - El ENTRENAMIENTO lee `quarter_scores` (1.006.077 filas, 252.956
#     partidos, 0 duplicados) y el MONITOR escribe en `quarter_scores_v2`
#     (2.916 partidos). Dos fuentes de verdad para los mismos datos.
#   - La v2 guarda lo que el monitor vio EN VIVO. En 54 partidos su Q4 no
#     coincide con el dato definitivo de la legacy y en 9 cambia el ganador
#     del cuarto. La liquidacion de apuestas (`reconcile_pending_results`)
#     lee la v2: 13 liquidaciones quedaron mal (ninguna de m27_v5_2/v5_3).
#   - NO HAY NADA QUE DESCARGAR: los 2.876 partidos con menos de 4 filas en
#     la legacy tampoco tienen 4 periodos en la API (25/25 en la muestra:
#     son partidos abandonados a los 2 cuartos).
#
# QUE HACE:
#   1. Backup de quarter_scores_v2.
#   2. Migra el pivot de la legacy a la v2 (la legacy gana: es el dato
#      definitivo al final del partido).
#   3. Verifica (filas, nulos, diferencias).
#   4. Re-liquida las apuestas cuyo resultado cambia con el dato definitivo.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/migrate_quarter_scores_v2.py            # dry-run
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/migrate_quarter_scores_v2.py --apply
# =====================================================================

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
BACKUP = "quarter_scores_v2_backup_20261020"

PIVOT = """
SELECT q.match_id,
  MAX(CASE WHEN q.quarter='Q1' THEN q.home END) q1_home, MAX(CASE WHEN q.quarter='Q1' THEN q.away END) q1_away,
  MAX(CASE WHEN q.quarter='Q2' THEN q.home END) q2_home, MAX(CASE WHEN q.quarter='Q2' THEN q.away END) q2_away,
  MAX(CASE WHEN q.quarter='Q3' THEN q.home END) q3_home, MAX(CASE WHEN q.quarter='Q3' THEN q.away END) q3_away,
  MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4_home, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4_away,
  MAX(CASE WHEN q.quarter='OT1' THEN q.home END) ot_home, MAX(CASE WHEN q.quarter='OT1' THEN q.away END) ot_away
FROM quarter_scores q GROUP BY q.match_id
"""


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="ejecuta la migracion (sin esto es dry-run)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    before_v2 = conn.execute("SELECT COUNT(*) FROM quarter_scores_v2").fetchone()[0]
    legacy = conn.execute("SELECT COUNT(DISTINCT match_id) FROM quarter_scores").fetchone()[0]
    conn.execute("DROP TABLE IF EXISTS temp._pivot")
    conn.execute(f"CREATE TEMP TABLE _pivot AS {PIVOT}")
    pivot_rows = conn.execute("SELECT COUNT(*) FROM _pivot").fetchone()[0]

    # Partidos con Q4 completo en la legacy (los que se pueden migrar con Q4)
    con_q4 = conn.execute(
        "SELECT COUNT(*) FROM _pivot WHERE q4_home IS NOT NULL AND q4_away IS NOT NULL").fetchone()[0]
    # Los que solo tienen Q1-Q3 (partidos abandonados) no se pueden migrar con Q4
    sin_q4 = pivot_rows - con_q4

    # Impacto en liquidaciones
    div = list(conn.execute(
        "SELECT p.match_id, p.q4_home lq4h, p.q4_away lq4a, v.q4_home vq4h, v.q4_away vq4a "
        "FROM _pivot p JOIN quarter_scores_v2 v ON v.match_id=p.match_id "
        "WHERE p.q4_home IS NOT NULL AND v.q4_home IS NOT NULL "
        "AND (p.q4_home<>v.q4_home OR p.q4_away<>v.q4_away)"))
    bets_afectadas = []
    for t in ("bet_monitor_log_v3", "bet_monitor_log_v2"):
        for b in conn.execute(
                f"SELECT match_id, model_version, signal_type, picked_side, result, id "
                f"FROM {t} WHERE signal_type LIKE '%BET%' AND signal_type NOT LIKE '%NO_BET%'"):
            for d in div:
                if d["match_id"] != b["match_id"]:
                    continue
                pick_home = b["picked_side"] == "HOME"
                if d["lq4h"] == d["lq4a"]:
                    esp = "loss"   # regla de negocio: empate = perdida
                else:
                    esp = "win" if pick_home == (d["lq4h"] > d["lq4a"]) else "loss"
                if b["result"] in ("win", "loss") and esp != b["result"]:
                    bets_afectadas.append((t, b["id"], b["match_id"], b["model_version"],
                                           b["result"], esp, d["lq4h"], d["lq4a"],
                                           d["vq4h"], d["vq4a"]))

    block("DRY-RUN: QUE HARIA LA MIGRACION")
    print(f"  quarter_scores_v2 ahora            : {before_v2:,} partidos")
    print(f"  quarter_scores (legacy)             : {legacy:,} partidos (0 duplicados)")
    print(f"  v2 despues                         : ~{pivot_rows:,} partidos")
    print(f"    de ellos, con Q4 completo        : {con_q4:,}")
    print(f"    sin Q4 (partidos abandonados)     : {sin_q4:,}  (no se pueden migrar)")
    print(f"  partidos cuyo Q4 cambia             : {len(div):,}")
    print(f"  liquidaciones que cambian           : {len(bets_afectadas):,}")
    for t, bid, mid, mv, guardado, esp, lqh, lqa, vqh, vqa in bets_afectadas:
        print(f"     {t[-2:]} id={bid:<6} {mid} {mv:<12} guardado={guardado:<5} -> {esp:<5} "
              f"(legacy Q4={lqh}-{lqa}, v2 Q4={vqh}-{vqa})")

    if not args.apply:
        print("\n  (dry-run: no se ha modificado nada. Usa --apply para migrar)")
        conn.close()
        return

    block("APLICANDO")
    t0 = time.time()
    with conn:
        if conn.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{BACKUP}'"
                        ).fetchone()[0] == 0:
            conn.execute(f"CREATE TABLE {BACKUP} AS SELECT * FROM quarter_scores_v2")
            print(f"  backup creado: {BACKUP} ({before_v2:,} filas)")
        else:
            print(f"  backup ya existente: {BACKUP} (se conserva)")

        conn.execute("""
            INSERT INTO quarter_scores_v2
              (match_id, q1_home, q1_away, q2_home, q2_away, q3_home, q3_away,
               q4_home, q4_away, ot_home, ot_away)
            SELECT match_id, q1_home, q1_away, q2_home, q2_away, q3_home, q3_away,
                   q4_home, q4_away, ot_home, ot_away
            FROM _pivot
            WHERE q4_home IS NOT NULL AND q4_away IS NOT NULL
            ON CONFLICT(match_id) DO UPDATE SET
              q1_home=excluded.q1_home, q1_away=excluded.q1_away,
              q2_home=excluded.q2_home, q2_away=excluded.q2_away,
              q3_home=excluded.q3_home, q3_away=excluded.q3_away,
              q4_home=excluded.q4_home, q4_away=excluded.q4_away,
              ot_home=excluded.ot_home, ot_away=excluded.ot_away
        """)
        print(f"  pivot migrado en {time.time() - t0:.1f}s")

        # Re-liquidar las apuestas afectadas con el dato definitivo
        for t, bid, mid, mv, guardado, esp, lqh, lqa, vqh, vqa in bets_afectadas:
            conn.execute(f"UPDATE {t} SET result=? WHERE id=?", (esp, bid))
        if bets_afectadas:
            print(f"  re-liquidadas {len(bets_afectadas)} apuestas con el dato definitivo")

    block("VERIFICACION")
    after = conn.execute("SELECT COUNT(*) FROM quarter_scores_v2").fetchone()[0]
    print(f"  quarter_scores_v2: {before_v2:,} -> {after:,} partidos")
    print(f"  con Q4 completo   : {conn.execute('SELECT COUNT(*) FROM quarter_scores_v2 WHERE q4_home IS NOT NULL').fetchone()[0]:,}")
    rest = conn.execute(
        "SELECT COUNT(*) FROM _pivot p JOIN quarter_scores_v2 v ON v.match_id=p.match_id "
        "WHERE p.q4_home IS NOT NULL AND (p.q4_home<>v.q4_home OR p.q4_away<>v.q4_away)").fetchone()[0]
    print(f"  discrepancias restantes con la legacy: {rest} (debe ser 0)")
    print(f"  backup intacto en {BACKUP}: "
          f"{conn.execute(f'SELECT COUNT(*) FROM {BACKUP}').fetchone()[0]:,} filas")
    conn.close()


if __name__ == "__main__":
    main()
