"""
cli_menu.py - CLI interactivo con menus comprensibles para V15.

Ejecucion:

    python -m training.v15.cli_menu

Funciona con stdlib unicamente (sin dependencias externas). Usa colores ANSI
y flechas ASCII para que sea legible en cualquier terminal.

Si preferis una invocacion directa por argumentos, segui usando `cli.py`:

    python -m training.v15.cli train --train-days 72 ...
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Compatibilidad Windows (colores ANSI)
# ---------------------------------------------------------------------------

if os.name == "nt":
    try:
        os.system("")  # habilita ANSI en cmd.exe
    except Exception:
        pass

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers de estilo
# ---------------------------------------------------------------------------

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_BLUE = "\033[34m"


def _clear():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def _banner(title: str, subtitle: str = ""):
    line = "=" * max(60, len(title) + 6)
    print(f"{C_CYAN}{line}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  {title}{C_RESET}")
    if subtitle:
        print(f"{C_DIM}  {subtitle}{C_RESET}")
    print(f"{C_CYAN}{line}{C_RESET}")


def _section(title: str):
    print()
    print(f"{C_BOLD}{C_YELLOW}  {title}{C_RESET}")
    print(f"{C_DIM}  {'-' * max(50, len(title))}{C_RESET}")


def _prompt(msg: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"  {C_GREEN}>{C_RESET} {msg}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    return raw


def _prompt_int(msg: str, default: int | None = None) -> int:
    while True:
        raw = _prompt(msg, str(default) if default is not None else None)
        try:
            return int(raw)
        except ValueError:
            print(f"    {C_RED}valor invalido, probar de nuevo{C_RESET}")


def _pause():
    print()
    input(f"  {C_DIM}[enter para volver al menu]{C_RESET} ")


# ---------------------------------------------------------------------------
# Modelo de menu
# ---------------------------------------------------------------------------

@dataclass
class MenuItem:
    key: str
    label: str
    action: Callable[[], None]
    hint: str = ""


def _render_menu(title: str, items: list[MenuItem], extra_info: str = ""):
    _clear()
    _banner("V15 - Modelo live de apuestas NBA/basket", title)
    if extra_info:
        print()
        print(extra_info)
    print()
    for it in items:
        key = f"{C_BOLD}{C_MAGENTA}[{it.key}]{C_RESET}"
        hint = f"  {C_DIM}{it.hint}{C_RESET}" if it.hint else ""
        print(f"  {key}  {it.label}{hint}")
    print()
    print(f"  {C_BOLD}{C_MAGENTA}[q]{C_RESET}  salir")
    print()


def _run_menu(title: str, items: list[MenuItem], extra_info: str = ""):
    while True:
        _render_menu(title, items, extra_info)
        choice = input(f"  {C_GREEN}?{C_RESET} opcion: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return
        for it in items:
            if it.key.lower() == choice:
                try:
                    it.action()
                except KeyboardInterrupt:
                    print(f"\n  {C_YELLOW}[cancelado]{C_RESET}")
                    _pause()
                except Exception as e:
                    print(f"\n  {C_RED}[error]{C_RESET} {e}")
                    import traceback
                    traceback.print_exc()
                    _pause()
                break
        else:
            print(f"  {C_RED}opcion desconocida{C_RESET}")
            _pause()


# ---------------------------------------------------------------------------
# Wrapper de subprocess (ejecuta CLI como si fuera shell)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent  # .../pulpa/match


def _run_cli(args: list[str]):
    print(f"  {C_DIM}$ python -m training.v15.cli {' '.join(args)}{C_RESET}")
    print()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        subprocess.run(
            [sys.executable, "-u", "-m", "training.v15.cli", *args],
            cwd=str(ROOT),
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print(f"\n  {C_YELLOW}[interrumpido por usuario]{C_RESET}")


# ---------------------------------------------------------------------------
# Acciones - entrenamiento
# ---------------------------------------------------------------------------

def action_train_prod():
    """Entrenamiento modo produccion (config ganadora)."""
    _section("Entrenamiento modo PRODUCCION")
    print(f"  {C_DIM}Config ganadora del barrido:{C_RESET}")
    print("    train=72d  val=15d  cal=3d  holdout=0d")
    print("    min_samples_train=200  active_days=14")
    print()
    confirm = _prompt("Confirmar entrenamiento (s/n)", "s")
    if confirm.lower() not in ("s", "si", "y", "yes"):
        print(f"  {C_YELLOW}cancelado{C_RESET}")
        _pause()
        return
    _run_cli([
        "train",
        "--train-days", "72",
        "--val-days", "15",
        "--cal-days", "3",
        "--holdout-days", "0",
        "--min-samples-train", "200",
        "--active-days", "14",
    ])
    _pause()


def action_train_custom():
    """Entrenamiento con parametros custom."""
    _section("Entrenamiento CUSTOM")
    print(f"  {C_DIM}Ingresar los parametros (Enter para default).{C_RESET}")
    print()
    train_days = _prompt_int("train_days", 72)
    val_days = _prompt_int("val_days", 15)
    cal_days = _prompt_int("cal_days", 3)
    holdout_days = _prompt_int("holdout_days (0=sin holdout)", 0)
    min_train = _prompt_int("min_samples_train", 200)
    active = _prompt_int("active_days (0=desactivar filtro)", 14)
    no_cache = _prompt("Rebuilds samples desde cero? (s/n)", "n").lower() in ("s", "si", "y", "yes")

    args = [
        "train",
        "--train-days", str(train_days),
        "--val-days", str(val_days),
        "--cal-days", str(cal_days),
        "--holdout-days", str(holdout_days),
        "--min-samples-train", str(min_train),
    ]
    if active > 0:
        args += ["--active-days", str(active)]
    if no_cache:
        args += ["--no-cache"]
    _run_cli(args)
    _pause()


def action_train_baseline():
    """Entrenamiento con config baseline (para comparaciones)."""
    _section("Entrenamiento BASELINE (referencia)")
    print(f"  {C_DIM}Config original de v15:{C_RESET}")
    print("    train=50d  val=20d  cal=13d  holdout=7d")
    print("    min_samples_train=300  active_days=0 (sin filtro)")
    print()
    confirm = _prompt("Confirmar (s/n)", "s")
    if confirm.lower() not in ("s", "si", "y", "yes"):
        return
    _run_cli([
        "train",
        "--train-days", "50",
        "--val-days", "20",
        "--cal-days", "13",
        "--holdout-days", "7",
        "--min-samples-train", "300",
    ])
    _pause()


# ---------------------------------------------------------------------------
# Acciones - evaluacion
# ---------------------------------------------------------------------------

def action_test_roi():
    _section("Test de ROI rapido (resumen legible)")
    odds = _prompt("Odds", "1.40")
    min_bets = _prompt_int("Min apuestas por liga para incluir en portfolio", 5)
    top = _prompt_int("Cantidad de ligas a mostrar", 30)
    _run_cli([
        "test-roi",
        "--odds", odds,
        "--min-bets", str(min_bets),
        "--top", str(top),
    ])
    _pause()


def action_full_eval():
    _section("Backtest completo (eval.py)")
    print(f"  {C_DIM}Genera reportes detallados por match.{C_RESET}")
    odds = _prompt("Odds", "1.40")
    use_full = _prompt("Evaluar TODO el dataset? (s=full, n=solo holdout)", "n")
    args = ["eval", "--odds", odds]
    if use_full.lower() in ("s", "si", "y", "yes"):
        args.append("--full")
    _run_cli(args)
    _pause()


def action_generate_plots():
    _section("Generar 9 graficas de diagnostico")
    print("  Las graficas se guardan en model_outputs/plots/")
    _run_cli(["plots"])
    plots_dir = ROOT / "training" / "v15" / "model_outputs" / "plots"
    if plots_dir.exists():
        print()
        print(f"  {C_GREEN}graficas disponibles:{C_RESET}")
        for p in sorted(plots_dir.glob("*.png")):
            print(f"    - {p.name}")
    _pause()


def action_list_leagues():
    _section("Ligas entrenadas (tabla de ROI por liga)")
    _run_cli(["leagues"])
    _pause()


def action_summarize_prod():
    _section("Resumen del modelo PROD actual")
    script = ROOT / "training" / "v15" / "reports" / "_summarize_prod.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    else:
        print(f"  {C_RED}no existe {script}{C_RESET}")
    _pause()


# ---------------------------------------------------------------------------
# Acciones - inferencia
# ---------------------------------------------------------------------------

def action_infer_sample():
    _section("Inferencia - JSON de ejemplo")
    sample = {
        "match_id": "demo_001",
        "target": "q3",
        "league": "NBA",
        "quarter_scores": {
            "q1_home": 28, "q1_away": 24,
            "q2_home": 26, "q2_away": 29,
        },
        "graph_points": [],
        "pbp_events": [],
    }
    print("  JSON de prueba:")
    print(textwrap.indent(json.dumps(sample, indent=2), "    "))
    print()
    confirm = _prompt("Correr inferencia con este payload? (s/n)", "s")
    if confirm.lower() not in ("s", "si", "y", "yes"):
        return
    print()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "training.v15.cli", "infer"],
        input=json.dumps(sample),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        env=env,
    )
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(proc.stdout)
    else:
        print(f"  {C_RED}error:{C_RESET}")
        print(proc.stderr)
    _pause()


def action_infer_file():
    _section("Inferencia - JSON desde archivo")
    path = _prompt("Ruta al JSON")
    if not path:
        return
    p = Path(path)
    if not p.exists():
        print(f"  {C_RED}no existe{C_RESET}")
        _pause()
        return
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "training.v15.cli", "infer"],
        input=p.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
        env=env,
    )
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(proc.stdout)
    else:
        print(f"  {C_RED}error:{C_RESET}")
        print(proc.stderr)
    _pause()


# ---------------------------------------------------------------------------
# Acciones - config
# ---------------------------------------------------------------------------

def action_show_config():
    _section("Config activa y overrides por liga")
    _run_cli(["config"])
    _pause()


def action_show_prod_summary():
    _section("Config de entrenamiento PROD guardada")
    path = ROOT / "training" / "v15" / "model_outputs" / "training_summary_v15.json"
    if not path.exists():
        print(f"  {C_RED}no existe (falta entrenar){C_RESET}")
        _pause()
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  {C_BOLD}run_params:{C_RESET}")
    print(textwrap.indent(json.dumps(data.get("run_params", {}), indent=2), "    "))
    print(f"  {C_BOLD}splits:{C_RESET} {data.get('splits')}")
    print(f"  {C_BOLD}ligas entrenadas:{C_RESET} {data.get('n_leagues_trained')}")
    print(f"  {C_BOLD}targets skipped:{C_RESET} {data.get('n_leagues_skipped')}")
    print(f"  {C_BOLD}version:{C_RESET} {data.get('version')}")
    _pause()


# ---------------------------------------------------------------------------
# Acciones - utilidades
# ---------------------------------------------------------------------------

def action_scan_db():
    _section("Escanear base de datos")
    script = ROOT / "training" / "v15" / "reports" / "_scan_db.py"
    if script.exists():
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env)
    else:
        print(f"  {C_RED}no existe {script}{C_RESET}")
    _pause()


def action_run_sweep():
    _section("Barrido de configuraciones (A/B/C)")
    print(f"  {C_YELLOW}[advertencia]{C_RESET} esto toma aprox 5-8 minutos.")
    confirm = _prompt("Continuar? (s/n)", "n")
    if confirm.lower() not in ("s", "si", "y", "yes"):
        return
    script = ROOT / "training" / "v15" / "reports" / "_sweep.ps1"
    if not script.exists():
        print(f"  {C_RED}no existe {script}{C_RESET}")
        _pause()
        return
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=str(ROOT),
    )
    _pause()


def action_open_docs():
    _section("Documentos disponibles")
    docs = {
        "readme": ROOT / "training" / "v15" / "README.md",
        "recomendaciones": ROOT / "training" / "v15" / "RECOMENDACIONES.md",
        "roadmap": ROOT / "training" / "v15" / "ROADMAP.md",
    }
    for key, p in docs.items():
        exists = p.exists()
        icon = f"{C_GREEN}OK{C_RESET}" if exists else f"{C_RED}NO{C_RESET}"
        print(f"  [{icon}] {key}: {p}")
    print()
    choice = _prompt("Cual abrir? (readme/recomendaciones/roadmap/enter=cancel)", "")
    if not choice:
        return
    p = docs.get(choice.lower())
    if p and p.exists():
        if os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(p)])
    else:
        print(f"  {C_RED}no encontrado{C_RESET}")
    _pause()


# ---------------------------------------------------------------------------
# Submenus
# ---------------------------------------------------------------------------

def submenu_train():
    items = [
        MenuItem("1", "Entrenar PRODUCCION (config ganadora)",
                 action_train_prod,
                 "train=72d val=15d sin holdout, min=200, solo ligas activas"),
        MenuItem("2", "Entrenar CUSTOM (elegir parametros)",
                 action_train_custom,
                 "para experimentos"),
        MenuItem("3", "Entrenar BASELINE (referencia)",
                 action_train_baseline,
                 "config original v15 con holdout, para comparar"),
        MenuItem("4", "Barrido de 3 configuraciones (A/B/C)",
                 action_run_sweep,
                 "corre sweep.ps1 - demora ~5-8 minutos"),
    ]
    _run_menu("Menu: Entrenamiento", items)


def submenu_eval():
    items = [
        MenuItem("1", "Test ROI rapido (resumen legible)",
                 action_test_roi,
                 "test-roi: global, portfolio, leak check, por liga"),
        MenuItem("2", "Backtest completo (eval.py)",
                 action_full_eval,
                 "bets partido-a-partido, odds configurable"),
        MenuItem("3", "Listar ligas entrenadas",
                 action_list_leagues,
                 "tabla con val_roi y holdout_roi por liga"),
        MenuItem("4", "Resumen del modelo PROD actual",
                 action_summarize_prod,
                 "tabla ordenada por ROI de validacion"),
        MenuItem("5", "Generar 9 graficas de diagnostico",
                 action_generate_plots,
                 "gap train-val, calibration, ROI, etc"),
    ]
    _run_menu("Menu: Evaluacion", items)


def submenu_infer():
    items = [
        MenuItem("1", "Inferencia con payload de ejemplo",
                 action_infer_sample,
                 "usa un JSON hardcodeado de NBA"),
        MenuItem("2", "Inferencia desde archivo JSON",
                 action_infer_file,
                 "pasa el path absoluto al JSON"),
    ]
    _run_menu("Menu: Inferencia", items)


def submenu_config():
    items = [
        MenuItem("1", "Mostrar config global + overrides",
                 action_show_config,
                 "todo lo de config.py y league_overrides.py"),
        MenuItem("2", "Ver run_params del PROD guardado",
                 action_show_prod_summary,
                 "que ventana y min_samples uso el ultimo entrenamiento"),
        MenuItem("3", "Escanear base de datos",
                 action_scan_db,
                 "volumen por semana, ligas activas, top 25"),
        MenuItem("4", "Abrir documentacion",
                 action_open_docs,
                 "README, RECOMENDACIONES, ROADMAP"),
    ]
    _run_menu("Menu: Config y utilidades", items)


# ---------------------------------------------------------------------------
# Menu principal
# ---------------------------------------------------------------------------

def build_status_banner() -> str:
    """Muestra estado actual del modelo debajo del banner principal."""
    path = ROOT / "training" / "v15" / "model_outputs" / "training_summary_v15.json"
    if not path.exists():
        return f"  {C_YELLOW}[sin modelo entrenado]{C_RESET} Correr primero [1] Entrenamiento."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return f"  {C_YELLOW}[summary corrupto]{C_RESET}"
    rp = data.get("run_params") or {}
    n_train = data.get("n_leagues_trained", 0)
    n_skip = data.get("n_leagues_skipped", 0)
    return (
        f"  {C_GREEN}[modelo activo]{C_RESET} "
        f"train={rp.get('train_days', '?')}d val={rp.get('val_days', '?')}d "
        f"cal={rp.get('cal_days', '?')}d holdout={rp.get('holdout_days', '?')}d "
        f"min={rp.get('min_samples_train', '?')}  |  "
        f"{n_train} targets activos / {n_skip} skipped"
    )


def main():
    items = [
        MenuItem("1", "Entrenamiento",
                 submenu_train,
                 "entrenar modelos, barrido de configs"),
        MenuItem("2", "Evaluacion y backtesting",
                 submenu_eval,
                 "test-roi, eval completo, plots"),
        MenuItem("3", "Inferencia live",
                 submenu_infer,
                 "predicciones sobre un partido"),
        MenuItem("4", "Config y utilidades",
                 submenu_config,
                 "ver config, scanear DB, abrir docs"),
    ]

    while True:
        status = build_status_banner()
        _render_menu("Menu principal", items, extra_info=status)
        choice = input(f"  {C_GREEN}?{C_RESET} opcion: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print(f"  {C_CYAN}hasta pronto.{C_RESET}")
            return
        for it in items:
            if it.key.lower() == choice:
                try:
                    it.action()
                except KeyboardInterrupt:
                    print(f"\n  {C_YELLOW}[cancelado]{C_RESET}")
                    _pause()
                except Exception as e:
                    print(f"\n  {C_RED}[error]{C_RESET} {e}")
                    import traceback
                    traceback.print_exc()
                    _pause()
                break
        else:
            print(f"  {C_RED}opcion desconocida{C_RESET}")
            _pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(f"  {C_CYAN}hasta pronto.{C_RESET}")
