# =====================================================================
# VERIFICACION: ¿el modelo limita el minuto internamente (snapshot_minute_used)?
# ---------------------------------------------------------------------
#   Para que la recomendacion de "tope de minuto 30" sea correcta hay que
#   confirmar que el vector de features se construye con el minuto real
#   (31-38) y no con un 30 interno (cierre de Q3).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/check_minute_clamp.py
# =====================================================================

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"


def raw(row) -> dict:
    try:
        return (json.loads(row["inference_json"] or "{}").get("raw_json") or {})
    except Exception:
        return {}


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        """
        SELECT l.match_id, l.inference_minute, l.signal_type, l.inference_json,
               l.confidence, l.picked_side
        FROM bet_monitor_log_v3 l
        WHERE l.model_version = 'm27_v5_3'
        ORDER BY l.inference_minute
        """
    ).fetchall()]
    conn.close()

    print(f"Filas v5_3: {len(rows)}")
    pairs = Counter()
    ood = []
    for r in rows:
        rw = raw(r)
        log_min = int(r["inference_minute"] or 0)
        snap = rw.get("snapshot_minute_used")
        pairs[(log_min, snap)] += 1
        if log_min > 30:
            ood.append((r, rw, log_min, snap))

    print("\nRelacion (inference_minute del log -> snapshot_minute_used por el modelo):")
    for (log_min, snap), n in sorted(pairs.items(), key=lambda kv: (kv[0][0] is None, kv[0][0])):
        flag = ""
        if log_min is not None and snap is not None and snap > 30:
            flag = "  <-- OUT OF RANGE (el modelo USO ese minuto)"
        elif log_min is not None and snap is not None and snap != log_min:
            flag = "  <-- el modelo clampeo"
        print(f"  log={str(log_min):>4} -> modelo={str(snap):>6}   n={n:>4}{flag}")

    print(f"\nFilas con inference_minute > 30: {len(ood)}")
    for r, rw, log_min, snap in ood[:15]:
        print(f"  match={r['match_id']} log_min={log_min} snapshot_usado={snap} "
              f"minuto_feature={rw.get('features', {}).get('minute')} "
              f"senal={r['signal_type']} conf={r['confidence']}")

    ood_bets = [x for x in ood if "BET" in (x[0]["signal_type"] or "") and "NO_BET" not in (x[0]["signal_type"] or "")]
    print(f"\nApuestas reales con minuto > 30: {len(ood_bets)}")
    for r, rw, log_min, snap in ood_bets:
        print(f"  match={r['match_id']} minuto={log_min} snapshot_usado={snap} "
              f"minute_feature={rw.get('features', {}).get('minute')} conf={r['confidence']} pick={r['picked_side']}")


if __name__ == "__main__":
    main()
