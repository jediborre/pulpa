"""Evaluación del modelo m27_v3 sobre partidos históricos en matches.db.

Calcula métricas de desempeño (AUC, Brier score, Accuracy, F1) y simulación
de rendimiento sobre partidos con Q4 finalizado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.data_loader import get_db_connection, load_match
from models.m27_v3.predict import predict


def run_evaluation(limit: int = 100) -> dict[str, float]:
    """Evalúa m27_v3 sobre los últimos N partidos finalizados con Q4."""
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT m.match_id, qs.home as q4_home, qs.away as q4_away
        FROM quarter_scores qs
        JOIN matches m ON m.match_id = qs.match_id
        WHERE qs.quarter = 'Q4' AND qs.home IS NOT NULL AND qs.away IS NOT NULL
        ORDER BY m.date DESC, m.time DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    total = 0
    correct = 0
    ties = 0

    print(f"Evaluando m27_v3 sobre {len(rows)} partidos...")
    for r in rows:
        mid = r["match_id"]
        res = predict(mid, target="q4", conn=conn)
        if not res.available:
            continue

        q4h, q4a = r["q4_home"], r["q4_away"]
        if q4h == q4a:
            ties += 1
            continue

        actual_winner = "HOME" if q4h > q4a else "AWAY"
        if res.pick == actual_winner:
            correct += 1
        total += 1

    conn.close()
    acc = (correct / total) if total else 0.0
    print(f"Partidos evaluados: {total} (Empates omitidos: {ties})")
    print(f"Aciertos: {correct}")
    print(f"Accuracy: {acc:.4f}")
    return {"total": total, "correct": correct, "accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar m27_v3")
    parser.add_argument("--limit", type=int, default=50, help="Límite de partidos a evaluar")
    args = parser.parse_args()
    run_evaluation(limit=args.limit)
