"""Evaluador de rendimiento histórico para el modelo v6_2.

Prueba inferencias en Q3 y Q4 contra partidos con marcadores oficiales en matches.db.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.data_loader import get_db_connection
from models.v6_2.predict import predict


def run_evaluation(target: str = "q4", limit: int = 50) -> dict[str, float]:
    """Evalúa v6_2 sobre los últimos N partidos finalizados."""
    t = target.lower()
    conn = get_db_connection()
    q_col = "Q4" if t == "q4" else "Q3"
    rows = conn.execute(
        f"""
        SELECT m.match_id, qs.home as q_home, qs.away as q_away
        FROM quarter_scores qs
        JOIN matches m ON m.match_id = qs.match_id
        WHERE qs.quarter = '{q_col}' AND qs.home IS NOT NULL AND qs.away IS NOT NULL
        ORDER BY m.date DESC, m.time DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    total = 0
    correct = 0
    ties = 0

    print(f"Evaluando v6_2 para target={t.upper()} sobre {len(rows)} partidos...")
    for r in rows:
        mid = r["match_id"]
        res = predict(mid, target=t, conn=conn)
        if not res.available:
            continue

        qh, qa = r["q_home"], r["q_away"]
        if qh == qa:
            ties += 1
            continue

        actual_winner = "HOME" if qh > qa else "AWAY"
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
    parser = argparse.ArgumentParser(description="Evaluar v6_2")
    parser.add_argument("--target", choices=["q3", "q4"], default="q4", help="Cuarto objetivo")
    parser.add_argument("--limit", type=int, default=50, help="Límite de partidos")
    args = parser.parse_args()
    run_evaluation(target=args.target, limit=args.limit)
