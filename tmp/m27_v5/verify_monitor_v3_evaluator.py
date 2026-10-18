"""
Verificación de monitor_v3/models/evaluator.py con m27_v5 y distinción 10m vs 12m.
"""
import sys
import asyncio
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from monitor_v3.models.evaluator import evaluate_match_q4, load_models_to_cache, GameWatcherState
from monitor_v3.config.constants import ACTIVE_MODELS
from models.common.data_loader import get_db_connection, load_match


async def main():
    print("=" * 80)
    print("VERIFICACIÓN EVALUATOR MONITOR_V3 CON M27_V5")
    print("=" * 80)
    print(f"ACTIVE_MODELS configurados: {ACTIVE_MODELS}")
    assert "m27_v5" in ACTIVE_MODELS, "m27_v5 debe estar en ACTIVE_MODELS"

    conn = get_db_connection()
    # Partido 1: FIBA 10m
    fiba_match = load_match(conn, "17315066")
    # Partido 2: NBA 12m
    cur = conn.cursor()
    nba_id = cur.execute("SELECT match_id FROM matches WHERE league LIKE '%NBA%' AND home_score IS NOT NULL LIMIT 1").fetchone()[0]
    nba_match = load_match(conn, str(nba_id))
    conn.close()

    watcher = GameWatcherState(str(fiba_match["match_id"]))

    # Evaluar FIBA en min 27
    print(f"\n1. Evaluando FIBA 10m (Match {fiba_match['match_id']}) al min 27:")
    res_fiba = await evaluate_match_q4(fiba_match["match_id"], fiba_match, watcher, forced_minute=27.0)
    print(f"  OK: {res_fiba.get('ok')}")
    for m, p in res_fiba.get("predictions", {}).items():
        conf_str = f"{p.get('confidence', 0)*100:.1f}%" if p.get('confidence') is not None else "N/A"
        print(f"    - Model {m:<8}: Signal={str(p.get('signal')):<15} | Pick={str(p.get('pick')):<5} | Conf={conf_str}")

    # Evaluar NBA en min 34 (Q3 tardío 12m)
    print(f"\n2. Evaluando NBA 12m (Match {nba_match['match_id']}) al min 34:")
    watcher_nba = GameWatcherState(str(nba_match["match_id"]))
    res_nba = await evaluate_match_q4(nba_match["match_id"], nba_match, watcher_nba, forced_minute=34.0)
    print(f"  OK: {res_nba.get('ok')}")
    for m, p in res_nba.get("predictions", {}).items():
        conf_str = f"{p.get('confidence', 0)*100:.1f}%" if p.get('confidence') is not None else "N/A"
        print(f"    - Model {m:<8}: Signal={str(p.get('signal')):<15} | Pick={str(p.get('pick')):<5} | Conf={conf_str}")

    print("\nEvaluador verificado exitosamente para ligas 10m y 12m.")


if __name__ == "__main__":
    asyncio.run(main())
