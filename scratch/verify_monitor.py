import os
import sys
from pathlib import Path

# Configurar codificación UTF-8 para consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=== VERIFICACIÓN DE IMPORTACIÓN DE PAQUETE bet_monitor_v2 ===")

try:
    print("1. Cargando constants.py...")
    import bet_monitor_v2.config.constants as constants
    print("   [OK] constants.py importado correctamente.")
    print(f"   Modelos Activos: {constants.ACTIVE_MODELS}")
    
    print("2. Cargando leagues.yaml...")
    from bet_monitor_v2.main import get_leagues_config, get_league_mode
    cfg = get_leagues_config()
    print("   [OK] leagues.yaml cargado correctamente.")
    print(f"   Ligas excluidas: {cfg.get('excluded_leagues', {}).get('patterns', [])}")
    print(f"   Ligas FT-Only: {cfg.get('ft_only_leagues', {}).get('patterns', [])}")
    
    print("3. Probando conexión de base de datos...")
    from bet_monitor_v2.database.connection import get_db_connection, get_real_db_path
    print(f"   Ruta real DB resuelta: {get_real_db_path()}")
    conn = get_db_connection()
    print("   [OK] Conexión abierta con WAL y timeout habilitados.")
    
    print("4. Inicializando tablas y reconciliando resultados...")
    from bet_monitor_v2.database.repository import init_tables, reconcile_pending_results
    init_tables()
    print("   [OK] Tablas '_v2' creadas o verificadas.")
    reconcile_pending_results()
    print("   [OK] Reconciliación síncrona inicial completada.")
    
    # Buscar algún match_id real en matches.db
    print("5. Buscando un Match ID real para pruebas...")
    cursor = conn.execute("SELECT match_id FROM matches LIMIT 1;")
    row = cursor.fetchone()
    match_id = row[0] if row else None
    conn.close()
    
    if match_id:
        print(f"   [OK] Match ID encontrado en base de datos: {match_id}")
    else:
        print("   [WARNING] matches.db no contiene partidos para pruebas ligeras.")
        
    print("6. Cargando orquestador de modelos...")
    import bet_monitor_v2.models.evaluator as evaluator
    print("   [OK] evaluator.py cargado.")
    evaluator.load_models_to_cache()
    print("   [OK] Carga de modelos singleton ejecutada.")
    
    print("7. Verificando notificaciones de Telegram...")
    import bet_monitor_v2.notifications.telegram_bot as telegram_bot
    
    # Pruebas de formato de mensaje con diferentes confianzas
    tests = [
        ("v6_2", "Huddinge Basketklubb", True, "BET_HOME_LATE", 0.29, "🟡⚪️ APUESTA TARDIA Q4 [v6_2 29%] → 🏠 Huddinge Basketklubb"),
        ("v6_2", "Huddinge Basketklubb", True, "BET_HOME_LATE", 29.0, "🟡⚪️ APUESTA TARDIA Q4 [v6_2 29%] → 🏠 Huddinge Basketklubb"),
        ("v6_2", "Huddinge Basketklubb", True, "BET_HOME", 0.29, "🟡 APUESTA Q4 [v6_2 29%] → 🏠 Huddinge Basketklubb"),
        ("v6_2", "Huddinge Basketklubb", True, "BET_HOME_LATE", 0.35, "⚪️ APUESTA TARDIA Q4 [v6_2 35%] → 🏠 Huddinge Basketklubb"),
        ("v6_2", "Huddinge Basketklubb", True, "BET_HOME", 0.35, "🟢 APUESTA Q4 [v6_2 35%] → 🏠 Huddinge Basketklubb"),
        ("v6_2", "Huddinge Basketklubb", True, "BET_HOME", None, "🟢 APUESTA Q4 [v6_2] → 🏠 Huddinge Basketklubb")
    ]
    
    for model, team, is_home, signal, conf, expected in tests:
        res = telegram_bot.format_bet_message(model, team, is_home, signal, confidence=conf)
        print(f"   [TEST] Model={model}, Conf={conf}, Signal={signal} -> {res}")
        assert res == expected, f"ERROR: Esperaba '{expected}' pero obtuve '{res}'"
        
    print("   [OK] Todas las pruebas de formato de telegram_bot pasaron correctamente.")

    
    print("8. Verificando utilidades...")
    import bet_monitor_v2.utils.logger as logger
    import bet_monitor_v2.utils.helpers as helpers
    print(f"   Tiempo humano (7300s): {helpers.format_human_time(7300)}")
    print(f"   Jitter sleep (30s, q4_window): {helpers.calculate_jitter_sleep_secs(30, 'q4_window'):.2f}s")
    print("   [OK] logger.py y helpers.py verificados.")
    
    print("\n=== ¡TODOS LOS MÓDULOS DE bet_monitor_v2 CARGAN PERFECTAMENTE Y SIN ERRORES! ===")
    sys.exit(0)

except Exception as e:
    print(f"\n   [ERROR] Fallo en la verificación: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
