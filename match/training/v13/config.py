"""
config.py — Fuente única de verdad para umbrales de V13.

Importado por train_v13.py, infer_match_v13.py, y referencia para bet_monitor.py.
Nunca duplicar estos valores en otros archivos.

Al entrenar, los valores de PACE_* se recalculan con los percentiles reales del
DB y se persisten en model_outputs/training_summary.json. En inferencia,
infer_match_v13.py los lee del training_summary para garantizar consistencia.
"""

# ── Ventanas de features (minuto máximo de graph_points incluidos) ─────────────
Q3_GRAPH_CUTOFF = 22      # antes: 24 en V12. Permite operar antes del fin Q2.
Q4_GRAPH_CUTOFF = 31      # antes: 36 en V12. Crítico: Q3 puede estar en curso.

# ── Gates mínimos de datos (bet_monitor Y modelo deben usar los mismos) ────────
MIN_GP_Q3 = 14            # graph_points con minuto ≤ Q3_GRAPH_CUTOFF
MIN_GP_Q4 = 16            # graph_points con minuto ≤ Q4_GRAPH_CUTOFF
MIN_PBP_Q3 = 12           # play-by-play eventos Q1+Q2
MIN_PBP_Q4 = 14           # play-by-play eventos Q1+Q2+Q3

# ── Umbrales de confianza (separados por cuarto) ───────────────────────────────
MIN_CONFIDENCE_Q3 = 0.62  # Q3 tiene mejor calidad de snapshot → se exige más
MIN_CONFIDENCE_Q4 = 0.55  # Q4 llega con snapshot parcial → umbral menor

# ── Gate de volatilidad ────────────────────────────────────────────────────────
MAX_VOLATILITY = 0.70

# ── Gate de liga (modo degradado en vez de bloqueo duro) ──────────────────────
LEAGUE_MIN_SAMPLES_BLOCK = 15    # < 15 muestras → bloqueo total
LEAGUE_MIN_SAMPLES_PENALIZE = 30 # < 30 muestras → reducir confianza máxima al 50%
LEAGUE_MIN_SAMPLES_FULL = 60     # ≥ 60 muestras → sin penalización

# ── Segmentación por ritmo de anotación ──────────────────────────────────────
# Umbrales REALES calculados del DB (2026-04-15):
#   Q3 (halftime total): p33=72, p66=85 → low ≤72, medium 72-85, high ≥85
#   Q4 (3 cuartos total): p33=108, p66=126 → low ≤108, medium 108-126, high ≥126
# Estos valores se recalculan al entrenar y se persisten en training_summary.json.
PACE_Q3_LOW_UPPER    = 72   # pts totales Q1+Q2 (ambos equipos): bucket LOW  si ≤ 72
PACE_Q3_HIGH_LOWER   = 85   # pts totales Q1+Q2: bucket HIGH si ≥ 85 (else MEDIUM)
PACE_Q4_LOW_UPPER    = 108  # pts totales Q1+Q2+Q3: bucket LOW si ≤ 108
PACE_Q4_HIGH_LOWER   = 126  # pts totales Q1+Q2+Q3: bucket HIGH si ≥ 126
PACE_MIN_SAMPLES     = 100  # muestras mínimas por bucket; si hay menos → fallback a "medium"

# ── Selección de algoritmos por tamaño de subset ─────────────────────────────
# Cada combinación (target × gender × pace) puede tener distinto nº de muestras.
ALGO_FULL_THRESHOLD    = 500  # ≥ 500: ensemble completo [logreg, gb, xgb, lgbm, catboost]
ALGO_MEDIUM_THRESHOLD  = 200  # 200-499: [logreg, xgb, catboost]
# < 200: [logreg, xgb] + advertencia en training_summary

# ── Parámetros de modelo ───────────────────────────────────────────────────────
MIN_EDGE_FOR_OVER_UNDER = 4.0    # puntos de ventaja mínima para señal O/U
DEFAULT_ODDS = 1.91
LOSS_PENALTY = 2.0               # penalización asimétrica

# ── Parámetros de timing para bet_monitor ────────────────────────────────────
# (Estos son una guía; bet_monitor.py los debe adoptar explícitamente para V13)
MONITOR_Q3_MINUTE = 22           # minuto objetivo para evaluar Q3
MONITOR_Q4_MINUTE = 31           # minuto objetivo para evaluar Q4
MONITOR_WAKE_BEFORE = 2          # minutos antes de MONITOR_Qx_MINUTE para despertar
MONITOR_CONFIRM_TICKS_Q3 = 1     # ticks de confirmación antes de enviar NO_BET Q3
MONITOR_CONFIRM_TICKS_Q4 = 0     # sin espera en Q4: primer resultado es final
