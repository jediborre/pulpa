"""Script oficial de entrenamiento y exportación del estado para M27 V4 (Tríada Canónica Multi-Head)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    """Ejecuta el pipeline iterativo V4.4 y exporta los artefactos a models/m27_v4/model_outputs/."""
    py_exe = sys.executable
    pipeline_script = ROOT_DIR / "tmp" / "eda_nuevo_modelo" / "iterative_training_pipeline.py"
    export_script = ROOT_DIR / "tmp" / "eda_nuevo_modelo" / "export_v4_production_state.py"

    print("1. Ejecutando entrenamiento iterativo y validación multi-corte (V4.4)...")
    subprocess.run([py_exe, str(pipeline_script)], check=True)

    print("2. Exportando estado O(1) y bundles de producción a models/m27_v4/model_outputs/...")
    subprocess.run([py_exe, str(export_script)], check=True)


if __name__ == "__main__":
    main()
