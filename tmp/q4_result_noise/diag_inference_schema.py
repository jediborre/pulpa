# Diagnóstico temporal: tipos/índices de match_id en model_inference_v3 y
# quarter_scores_v2 para eliminar el CAST que hace lento el /resumen.
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
conn = sqlite3.connect(Path(__file__).resolve().parents[2] / "matches.db")

for t in ("model_inference_v3", "quarter_scores_v2"):
    print(f"--- {t} ---")
    for r in conn.execute(f"PRAGMA table_info({t})"):
        if r[1] in ("match_id", "game_minute", "pick", "confidence", "q4_home", "q4_away"):
            print(f"  {r[1]}: {r[2]}")
    print("  índices:", [tuple(x) for x in conn.execute(f"PRAGMA index_list({t})")])
    vals = conn.execute(f"SELECT match_id, typeof(match_id) FROM {t} LIMIT 3").fetchall()
    print("  muestra match_id:", vals)
conn.close()
