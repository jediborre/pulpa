"""
Ubicación original: tmp/db_audit/check_and_fix_training_db.py
Propósito / Qué hace:
Analiza todos los archivos en match/training y sus subdirectorios (v12, v13, v15, v16, v17, etc.)
que definen rutas a matches.db, y los reapunta a la raíz del repositorio (/matches.db),
asegurando que .exists() retorne True en cada caso.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "match" / "training"

matches_db_real = ROOT / "matches.db"
assert matches_db_real.exists(), f"Error: no existe {matches_db_real}"

print(f"Base de datos real en raíz: {matches_db_real} (Tamaño: {matches_db_real.stat().st_size} bytes)")

# 1. Archivos directos en match/training/
direct_files = [
    "calibrate_gate.py",
    "eda.py",
    "eval_report_by_model.py",
    "infer_match.py",
    "predict_v11.py",
    "report_db_results.py",
    "report_model_comparison.py",
    "report_v12_v13.py",
    "summarize_roi.py",
    "summarize_roi2.py",
    "train_q3_q4_models.py",
    "train_q3_q4_models_v2.py",
    "train_q3_q4_models_v3.py",
    "train_q3_q4_models_v4.py",
    "train_q3_q4_models_v5.py",
    "train_q3_q4_models_v6.bk.py",
    "train_q3_q4_models_v6.py",
    "train_q3_q4_models_v7.py",
    "train_q3_q4_models_v8.py",
    "train_q3_q4_models_v9.py",
    "train_q3_q4_regression_v10.py",
    "train_q3_q4_regression_v11.py",
    "train_v10_simple.py",
    "batch_populate_evals.py",
    "populate_v9.py",
    "populate_v9_only.py",
]

for fname in direct_files:
    fpath = TRAINING_DIR / fname
    if not fpath.exists():
        continue
    content = fpath.read_text(encoding="utf-8")
    modified = False

    # In direct files, ROOT is match/ (parents[1]).
    # We want DB_PATH = ROOT.parent / "matches.db" or PROJECT_ROOT / "matches.db"
    if 'DB_PATH = ROOT / "matches.db"' in content:
        content = content.replace('DB_PATH = ROOT / "matches.db"', 'DB_PATH = ROOT.parent / "matches.db"')
        modified = True
    if 'ROOT / "matches.db"' in content:
        content = content.replace('ROOT / "matches.db"', 'ROOT.parent / "matches.db"')
        modified = True
    if 'DB_PATH = Path(__file__).resolve().parents[1] / "matches.db"' in content:
        content = content.replace('DB_PATH = Path(__file__).resolve().parents[1] / "matches.db"', 'DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"')
        modified = True

    if modified:
        fpath.write_text(content, encoding="utf-8")
        print(f"Actualizado: match/training/{fname}")

# 2. Subdirectorios
# eda_overview
eda_file = TRAINING_DIR / "eda_overview" / "eda_analysis.py"
if eda_file.exists():
    c = eda_file.read_text(encoding="utf-8")
    if 'DB_PATH = ROOT / "matches.db"' in c:
        c = c.replace('DB_PATH = ROOT / "matches.db"', 'DB_PATH = ROOT.parent.parent / "matches.db"')
        eda_file.write_text(c, encoding="utf-8")
        print("Actualizado: eda_overview/eda_analysis.py")

# v12 files
v12_dir = TRAINING_DIR / "v12"
for fname in ["eval_v12.py", "train_v12.py", "validate_v12.py"]:
    fpath = v12_dir / fname
    if fpath.exists():
        c = fpath.read_text(encoding="utf-8")
        if 'DB_PATH = ROOT / "matches.db"' in c:
            c = c.replace('DB_PATH = ROOT / "matches.db"', 'DB_PATH = ROOT.parent / "matches.db"')
            fpath.write_text(c, encoding="utf-8")
            print(f"Actualizado: v12/{fname}")

wf_file = v12_dir / "fixed_validation" / "walk_forward.py"
if wf_file.exists():
    c = wf_file.read_text(encoding="utf-8")
    if 'DB_PATH = PROJECT_ROOT / "matches.db"' in c:
        c = c.replace('DB_PATH = PROJECT_ROOT / "matches.db"', 'DB_PATH = PROJECT_ROOT.parent / "matches.db"')
        wf_file.write_text(c, encoding="utf-8")
        print("Actualizado: v12/fixed_validation/walk_forward.py")

for le_name in ["live_betting.py", "virtual_bookmaker.py"]:
    le_file = v12_dir / "live_engine" / le_name
    if le_file.exists():
        c = le_file.read_text(encoding="utf-8")
        if 'DB_PATH = PROJECT_ROOT / "matches.db"' in c:
            c = c.replace('DB_PATH = PROJECT_ROOT / "matches.db"', 'DB_PATH = PROJECT_ROOT.parent / "matches.db"')
            le_file.write_text(c, encoding="utf-8")
            print(f"Actualizado: v12/live_engine/{le_name}")

# v13, v15, v16, v17 dataset.py
for v in ["v13", "v15", "v16", "v17"]:
    ds_file = TRAINING_DIR / v / "dataset.py"
    if ds_file.exists():
        c = ds_file.read_text(encoding="utf-8")
        if 'DB_PATH = Path(__file__).parents[2] / "matches.db"' in c:
            c = c.replace('DB_PATH = Path(__file__).parents[2] / "matches.db"', 'DB_PATH = Path(__file__).parents[3] / "matches.db"')
            ds_file.write_text(c, encoding="utf-8")
            print(f"Actualizado: {v}/dataset.py")

# v16, v17 dataset_analyzer.py
for v in ["v16", "v17"]:
    da_file = TRAINING_DIR / v / "dataset_analyzer.py"
    if da_file.exists():
        c = da_file.read_text(encoding="utf-8")
        old_da = 'DB_PATH = Path(__file__).resolve().parent.parent.parent / "matches.db"'
        new_da = 'DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "matches.db"'
        if old_da in c:
            c = c.replace(old_da, new_da)
            da_file.write_text(c, encoding="utf-8")
            print(f"Actualizado: {v}/dataset_analyzer.py")

# reports _scan_db.py
for v in ["v15", "v16"]:
    scan_file = TRAINING_DIR / v / "reports" / "_scan_db.py"
    if scan_file.exists():
        c = scan_file.read_text(encoding="utf-8")
        old_scan = 'DB = Path(__file__).parent.parent.parent.parent / "matches.db"'
        new_scan = 'DB = Path(__file__).parent.parent.parent.parent.parent / "matches.db"'
        if old_scan in c:
            c = c.replace(old_scan, new_scan)
            scan_file.write_text(c, encoding="utf-8")
            print(f"Actualizado: {v}/reports/_scan_db.py")

print("\nVerificando resolucion en todos los modulos...")
