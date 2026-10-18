"""
Herramienta de deteccion y terminacion limpia de procesos de monitor
para menu.bat y control de ciclo de vida del sistema Pulpa.
"""
import os
import sys
import psutil

def kill_existing_monitors(include_v1_v2: bool = True, verbose: bool = True) -> int:
    """
    Busca y termina cualquier proceso activo de monitoreo (monitor_v3, monitor_v2, monitor_v1)
    incluyendo sus terminales cmd.exe lanzadas con start /k.
    """
    current_pid = os.getpid()
    targets = [
        "monitor_v3/main.py", "monitor_v3\\main.py", "monitor_v3.main",
    ]
    if include_v1_v2:
        targets.extend([
            "monitor_v2/main.py", "monitor_v2\\main.py", "monitor_v2.main",
            "monitor_v1/main.py", "monitor_v1\\main.py", "monitor_v1.main",
        ])

    to_kill = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline_list = proc.info.get('cmdline') or []
            cmdline = " ".join(cmdline_list)
            
            # Evitar matarse a si mismo o a este script
            if "kill_monitor" in cmdline:
                continue

            name = (proc.info.get('name') or "").lower()
            if not any(name.startswith(p) for p in ("python", "py", "cmd")):
                continue

            if any(t in cmdline for t in targets):
                to_kill.append((proc, proc.pid, name, cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    killed = []
    # Primero matar los procesos de python, luego las ventanas cmd contenedor
    for proc, pid, name, cmd in sorted(to_kill, key=lambda x: 0 if "py" in x[2] else 1):
        try:
            proc.kill()
            killed.append((pid, name, cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed:
        if verbose:
            print(f"[!] Se detectaron {len(killed)} proceso(s) de monitor en ejecucion:")
            for pid, name, cmd in killed:
                short_cmd = (cmd[:70] + "...") if len(cmd) > 70 else cmd
                print(f"    - PID {pid} [{name}]: {short_cmd}")
            print("[+] Monitor anterior terminado exitosamente.")
    else:
        if verbose:
            print("[OK] No se detectaron instancias previas del monitor en ejecucion.")

    return len(killed)

if __name__ == "__main__":
    kill_existing_monitors()
