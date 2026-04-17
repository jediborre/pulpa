"""

_auto_overrides.py - Genera league_overrides.py desde training_summary_v16.json.



Politica:

  - val_roi < 0 en un target -> force_nobet_<target>

  - val_hit_rate < MIN_ACCEPTABLE (72%) -> force_nobet_<target>

  - train_val_gap > 0.30 (leak severo) -> force_nobet_<target>

  - Si ambos targets cumplen alguna regla -> force_nobet full



Uso:

    python -m training.v16.reports._auto_overrides



Genera:

    training/v16/league_overrides.py (reemplaza el existente)

"""

from __future__ import annotations



import json

import sys

from pathlib import Path



HERE = Path(__file__).resolve().parent

V16 = HERE.parent

SUMMARY_PATH = V16 / "model_outputs" / "training_summary_v16.json"

OVERRIDES_PATH = V16 / "league_overrides.py"



MIN_ACCEPTABLE_HIT = 0.72

MAX_ALLOWED_GAP = 0.30

HEAVY_GAP = 0.25   # si gap > heavy gap AND val_hit < 0.75, tambien bloquear





def _should_block(model: dict) -> tuple[bool, str]:

    val_roi = (model.get("threshold") or {}).get("roi")

    val_hit = (model.get("threshold") or {}).get("hit_rate")

    # Preferir el campo explicito guardado por train.py (f1_train - f1_val).
    # Si no existe (corrida vieja) caer a accuracy_train - accuracy_val.
    gap = model.get("train_val_gap")
    if gap is None:
        train_acc = (model.get("train_metrics") or {}).get("accuracy") or 0.0
        val_acc = (model.get("val_metrics") or {}).get("accuracy") or 0.0
        gap = train_acc - val_acc



    reasons = []

    if val_roi is not None and val_roi < 0:

        reasons.append(f"val_roi={val_roi:+.3f}")

    if val_hit is not None and val_hit < MIN_ACCEPTABLE_HIT:

        reasons.append(f"val_hit={val_hit:.3f}<{MIN_ACCEPTABLE_HIT}")

    if gap > MAX_ALLOWED_GAP:

        reasons.append(f"gap={gap:+.3f}>{MAX_ALLOWED_GAP}")

    elif gap > HEAVY_GAP and (val_hit or 0) < 0.75:

        reasons.append(f"gap={gap:+.3f} and val_hit<0.75")

    if reasons:

        return True, "; ".join(reasons)

    return False, f"ok val_roi={val_roi:+.3f} hit={val_hit:.3f} gap={gap:+.3f}"





def generate() -> dict:

    if not SUMMARY_PATH.exists():

        print(f"[error] No existe {SUMMARY_PATH}. Corre `cli train` primero.")

        sys.exit(1)



    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    models = summary.get("models", [])



    by_league: dict[str, dict[str, dict]] = {}

    for m in models:

        if m.get("skipped"):

            continue

        lg = m["league"]

        tg = m["target"]

        block, reason = _should_block(m)

        by_league.setdefault(lg, {})[tg] = {

            "block": block,

            "reason": reason,

            "val_roi": (m.get("threshold") or {}).get("roi"),

            "val_hit": (m.get("threshold") or {}).get("hit_rate"),

        }



    overrides: dict[str, dict] = {}

    for lg, targets in by_league.items():

        q3 = targets.get("q3", {})

        q4 = targets.get("q4", {})

        block_q3 = q3.get("block", True)   # si no existe modelo, tambien bloqueamos

        block_q4 = q4.get("block", True)

        if block_q3 and block_q4:

            overrides[lg] = {

                "force_nobet": True,

                "notes": (f"q3: {q3.get('reason','(no model)')}; "

                          f"q4: {q4.get('reason','(no model)')}"),

            }

        elif block_q3 and not block_q4:

            overrides[lg] = {

                "force_nobet_q3": True,

                "notes": (f"q3 blocked: {q3.get('reason')}; "

                          f"q4 ok ROI={q4.get('val_roi')}"),

            }

        elif block_q4 and not block_q3:

            overrides[lg] = {

                "force_nobet_q4": True,

                "notes": (f"q4 blocked: {q4.get('reason')}; "

                          f"q3 ok ROI={q3.get('val_roi')}"),

            }



    return overrides





def write_overrides_file(overrides: dict) -> None:

    lines = [

        '"""',

        "league_overrides.py - Parametros tuneables POR LIGA (autogenerado).",

        "",

        "Generado desde training_summary_v16.json por _auto_overrides.py.",

        "Editar manualmente si hace falta; la proxima corrida lo sobreescribira.",

        "",

        "Regla de bloqueo automatico:",

        f"  - val_roi < 0  OR  val_hit < {MIN_ACCEPTABLE_HIT}  OR  gap > {MAX_ALLOWED_GAP}",

        f"  - (gap > {HEAVY_GAP} AND val_hit < 0.75)",

        '"""',

        "from __future__ import annotations",

        "from typing import TypedDict",

        "",

        "",

        "class LeagueOverride(TypedDict, total=False):",

        "    min_confidence_q3: float",

        "    min_confidence_q4: float",

        "    min_samples_train: int",

        "    min_gp_q3: int",

        "    min_gp_q4: int",

        "    min_pbp_q3: int",

        "    min_pbp_q4: int",

        "    enable_regression_filter: bool",

        "    force_nobet: bool",

        "    force_nobet_q3: bool",

        "    force_nobet_q4: bool",

        "    notes: str",

        "",

        "",

        "LEAGUE_OVERRIDES: dict[str, LeagueOverride] = {",

    ]

    for lg in sorted(overrides.keys()):

        data = overrides[lg]

        lines.append(f"    {json.dumps(lg)}: {{")

        for k, v in data.items():

            lines.append(f"        {json.dumps(k)}: {v!r},")

        lines.append("    },")

    lines.extend([

        "}",

        "",

        "",

        "def get_league_override(league: str) -> LeagueOverride:",

        "    return LEAGUE_OVERRIDES.get(league, {})",

        "",

        "",

        "def list_tuned_leagues() -> list[str]:",

        "    return sorted(LEAGUE_OVERRIDES.keys())",

        "",

    ])

    OVERRIDES_PATH.write_text("\n".join(lines), encoding="utf-8")





if __name__ == "__main__":

    overrides = generate()

    write_overrides_file(overrides)

    full = sum(1 for v in overrides.values() if v.get("force_nobet"))

    q3 = sum(1 for v in overrides.values() if v.get("force_nobet_q3"))

    q4 = sum(1 for v in overrides.values() if v.get("force_nobet_q4"))

    print(f"[auto-overrides] generados {len(overrides)} overrides:")

    print(f"  force_nobet (full): {full}")

    print(f"  force_nobet_q3    : {q3}")

    print(f"  force_nobet_q4    : {q4}")

    print(f"[auto-overrides] escrito -> {OVERRIDES_PATH}")

