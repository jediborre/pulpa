# =====================================================================
# RECLASIFICAR LOS 'push' HISTÓRICOS COMO 'loss'
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   Criterio de negocio: una apuesta al cuarto que termina EMPATADO no gana,
#   por lo que debe contarse como PERDIDA. Antes se guardaba como 'push'
#   (anulado), lo que excluía esas apuestas del % de acierto.
#
# QUÉ HACE:
#   Convierte a 'loss' los logs de apuestas REALES (BET, no NO_BET) cuyo
#   result sea 'push'. Los NO_BET también usan 'push' como "sin apuesta" y
#   NO se tocan.
#
# USO:
#   .venv\Scripts\python.exe tmp\push_reclass\reclassify_push.py --dry-run
#   .venv\Scripts\python.exe tmp\push_reclass\reclassify_push.py
# =====================================================================

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.database.connection import get_db_connection  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="solo contar, no modificar")
    args = ap.parse_args()

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, match_id, model_version, signal_type, result
            FROM bet_monitor_log_v3
            WHERE result = 'push'
              AND signal_type LIKE '%BET%'
              AND signal_type NOT LIKE '%NO_BET%'
            """
        ).fetchall()

        print(f"Apuestas reales con result='push': {len(rows)}")
        for r in rows[:10]:
            print(f"  {r['match_id']} {r['model_version']} {r['signal_type']}")

        if args.dry_run:
            print("(dry-run: no se modificó nada)")
            return

        if not rows:
            print("nada que reclasificar")
            return

        with conn:
            cur = conn.execute(
                """
                UPDATE bet_monitor_log_v3
                SET result = 'loss'
                WHERE result = 'push'
                  AND signal_type LIKE '%BET%'
                  AND signal_type NOT LIKE '%NO_BET%'
                """
            )
            print(f"reclasificadas a 'loss': {cur.rowcount}")

        # Verificación
        push = conn.execute("SELECT COUNT(*) FROM bet_monitor_log_v3 WHERE result = 'push'").fetchone()[0]
        loss = conn.execute("SELECT COUNT(*) FROM bet_monitor_log_v3 WHERE result = 'loss'").fetchone()[0]
        print(f"resultado final → push restantes: {push} | loss totales: {loss}")


if __name__ == "__main__":
    main()
