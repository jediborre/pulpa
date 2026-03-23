"""Compare V1, V2 and V4 model metrics for Q3/Q4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_DIR = ROOT / "training" / "model_outputs"
V2_DIR = ROOT / "training" / "model_outputs_v2"
V4_DIR = ROOT / "training" / "model_outputs_v4"
OUT_DIR = ROOT / "training" / "model_comparison"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _best_model(
    rows: list[dict],
    metric: str,
    higher_is_better: bool,
) -> dict | None:
    if not rows:
        return None

    def sort_key(row: dict):
        value = _to_float(row.get(metric))
        if value is None:
            return float("-inf") if higher_is_better else float("inf")
        return value

    if higher_is_better:
        return max(rows, key=sort_key)
    return min(rows, key=sort_key)


def _annotate_version(rows: list[dict], version: str) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        r["version"] = version
        out.append(r)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    summary = {}

    for target in ("q3", "q4"):
        v1_rows = _annotate_version(
            _read_csv(V1_DIR / f"{target}_metrics.csv"),
            "v1",
        )
        v2_rows = _annotate_version(
            _read_csv(V2_DIR / f"{target}_metrics.csv"),
            "v2",
        )
        v4_rows = _annotate_version(
            _read_csv(V4_DIR / f"{target}_metrics.csv"),
            "v4",
        )

        all_rows = v1_rows + v2_rows + v4_rows
        comparison_rows.extend(all_rows)

        best_acc = _best_model(all_rows, "accuracy", higher_is_better=True)
        best_f1 = _best_model(all_rows, "f1", higher_is_better=True)
        best_ll = _best_model(all_rows, "log_loss", higher_is_better=False)

        summary[target] = {
            "best_accuracy": best_acc,
            "best_f1": best_f1,
            "best_log_loss": best_ll,
            "rows_compared": len(all_rows),
        }

    if comparison_rows:
        fields = sorted({k for row in comparison_rows for k in row.keys()})
        with (OUT_DIR / "version_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparison_rows)

    with (OUT_DIR / "version_comparison.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[compare] done")
    print(f"[compare] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
