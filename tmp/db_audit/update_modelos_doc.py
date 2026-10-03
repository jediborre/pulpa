"""
Ubicación original: tmp/db_audit/update_modelos_doc.py
Propósito / Qué hace:
Actualiza `modelos.md` y `docs/modelos.md` mapeando de forma exhaustiva los scripts de entrenamiento,
evaluación y ROI para cada una de las versiones de modelos (V1 a V17, m27_v1 a m27_v3, m30_v1)
y agregando a m27_v3 en la tabla comparativa rápida final.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

model_mappings = [
    # (old_script_line, new_lines)
    (
        "| **Script** | `train_q3_q4_models.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py`, `report_db_results.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v2.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v2.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v3.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v3.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v4.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v4.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v5.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v5.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v6.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v6.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py`, `report_db_results.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v6_1.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v6_1.py` |\n| **Script evaluación / ROI** | `test_v6_1_league_filter.py`, `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v6_2.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v6_2.py` |\n| **Script evaluación / ROI** | `report_v62_q4_roi.py`, `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v6_2b.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v6_2b.py` |\n| **Script evaluación / ROI** | `report_v62_q4_roi.py` |"
    ),
    (
        "| **Script** | `train_q4_models_v6_3.py` |",
        "| **Script entrenamiento** | `train_q4_models_v6_3.py` |\n| **Script evaluación / ROI** | `report_v63_q4_roi.py`, `compare_model_versions.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v7.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v7.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v8.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v8.py` |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_models_v9.py` |",
        "| **Script entrenamiento** | `train_q3_q4_models_v9.py` (`train_v9_fast.py`, `populate_v9.py`) |\n| **Script evaluación / ROI** | `report_model_comparison.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_regression_v10.py` |",
        "| **Script entrenamiento** | `train_q3_q4_regression_v10.py` (`train_v10_fast.py`, `train_v10_simple.py`) |\n| **Script evaluación / ROI** | `batch_populate_evals.py` |"
    ),
    (
        "| **Script** | `train_q3_q4_regression_v11.py` |",
        "| **Script entrenamiento** | `train_q3_q4_regression_v11.py` |\n| **Script evaluación / ROI** | `predict_v11.py` |"
    ),
    (
        "| **Script** | `v12/train_v12.py` |",
        "| **Script entrenamiento** | `v12/train_v12.py` |\n| **Script evaluación / ROI** | `v12/eval_v12.py`, `v12/validate_v12.py`, `report_v12_v13.py` |"
    ),
    (
        "| **Script** | `v13/train_v13.py` |",
        "| **Script entrenamiento** | `v13/train_v13.py` (`v13/train_clf.py`, `v13/train_reg.py`) |\n| **Script evaluación / ROI** | `v13/eval_v13.py`, `v13/walk_forward.py`, `report_v12_v13.py` |"
    ),
    (
        "| **Script** | — |",
        "| **Script entrenamiento** | — (Solo planificación en `PLAN_V14.md`) |\n| **Script evaluación / ROI** | — |"
    ),
    (
        "| **Script** | `v15/train.py` |",
        "| **Script entrenamiento** | `v15/train.py` |\n| **Script evaluación / ROI** | `v15/evaluate.py`, `v15/test_roi.py` |"
    ),
    (
        "| **Script** | `v16/train.py` |",
        "| **Script entrenamiento** | `v16/train.py` |\n| **Script evaluación / ROI** | `v16/evaluate.py`, `v16/test_roi.py`, `v16/test_roi_cli.py`, `v16/ab_compare.py` |"
    ),
    (
        "| **Script** | `v17/train.py` |",
        "| **Script entrenamiento** | `v17/train.py` |\n| **Script evaluación / ROI** | `v17/evaluate.py`, `v17/test_roi.py`, `v17/test_roi_cli.py`, `v17/ab_compare.py` |"
    ),
    (
        "| **Script** | `train_q4_m27_v1.py` |",
        "| **Script entrenamiento** | `train_q4_m27_v1.py` |\n| **Script evaluación / ROI** | `report_m_v1_roi.py`, `m27_v1_league_policy.py`, `compare_model_versions.py` |"
    ),
    (
        "| **Script** | `train_q4_m27_v2.py` |",
        "| **Script entrenamiento** | `train_q4_m27_v2.py` |\n| **Script evaluación / ROI** | `report_m_v1_roi.py`, `compare_model_versions.py` |"
    ),
    (
        "| **Script** | `train_q4_m27_v3.py` |",
        "| **Script entrenamiento** | `train_q4_m27_v3.py` |\n| **Script evaluación / ROI** | `report_m_v1_roi.py`, `compare_model_versions.py` |\n| **Script inferencia** | `infer_match.py` (`score_m27_v3`) |"
    ),
    (
        "| **Script** | `train_q4_m30_v1.py` |",
        "| **Script entrenamiento** | `train_q4_m30_v1.py` |\n| **Script evaluación / ROI** | `compare_q4_min30_models.py`, `report_m_v1_roi.py` |"
    ),
]

rapid_table_target = "| **m27_v2** | **27** | **0.668** / 0.671* | 0.623 / 0.626 | 4,132 | *10m filter. 86 feat (7 podadas). Match-level. | Idéntico a v1. Recent windows dominan. Features nuevas no suman |"
rapid_table_replacement = (
    "| **m27_v2** | **27** | **0.668** / 0.671* | 0.623 / 0.626 | 4,132 | *10m filter. 86 feat (7 podadas). Match-level. | Idéntico a v1. Recent windows dominan. Features nuevas no suman |\n"
    "| **m27_v3** | **27** | **0.789** | **0.705** | 4,132 | Snapshot 27. **+H2H features (28.5% imp)**. 99% test cov. Match-level. | **Campeón actual (+13% a +29% Yield)** |"
)

files_to_update = [
    ROOT / "modelos.md",
    ROOT / "docs" / "modelos.md"
]

for file_path in files_to_update:
    if not file_path.exists():
        print(f"No existe: {file_path}")
        continue
    content = file_path.read_text(encoding="utf-8")
    for old_s, new_s in model_mappings:
        if old_s in content:
            content = content.replace(old_s, new_s)
        else:
            print(f"ADVERTENCIA: no se encontro '{old_s[:30]}...' en {file_path.name}")
    
    # Rapid table replacement (handle escaped asterisk if needed)
    if rapid_table_target in content:
        content = content.replace(rapid_table_target, rapid_table_replacement)
        print(f"Tabla comparativa rapida actualizada en {file_path.name}")
    else:
        # Check if asterisk was escaped with backslash
        alt_target = rapid_table_target.replace("*", r"\*")
        alt_replacement = rapid_table_replacement.replace("*", r"\*")
        if alt_target in content:
            content = content.replace(alt_target, alt_replacement)
            print(f"Tabla comparativa rapida actualizada (con escape) en {file_path.name}")
        else:
            print(f"ADVERTENCIA: No se encontro rapid_table_target en {file_path.name}")

    file_path.write_text(content, encoding="utf-8")
    print(f"Archivo actualizado exitosamente: {file_path}")
