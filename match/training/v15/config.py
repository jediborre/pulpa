"""
config.py - Fuente unica de verdad para umbrales de V15.

Reglas:
- Ningun otro archivo hardcodea estos valores.
- Overrides por liga en league_overrides.py.
- Umbrales de confianza minimos pueden ser sustituidos por el threshold
  optimo por liga aprendido en backtest (ver inference.gates).
"""

from __future__ import annotations

# ==============================================================================
# VERSION / PATH
# ==============================================================================
VERSION = "v15"

# ==============================================================================
# OBJETIVO DE NEGOCIO
# ==============================================================================
# Odds pesimista: todo el pipeline se optimiza para ROI con este momio.
DEFAULT_ODDS = 1.40
# Break-even matematico con odds 1.40 = 1 / 1.40 = 71.43%
BREAK_EVEN_PCT = 1.0 / DEFAULT_ODDS
# Target operativo (margen de seguridad sobre el break-even)
TARGET_HIT_RATE = 0.75
# Win rate aceptable que indica ROI positivo (con cierto margen).
MIN_ACCEPTABLE_HIT_RATE = 0.72

# ==============================================================================
# FILTROS DE DATASET
# ==============================================================================
# Genero permitido (por ahora solo hombres).
ALLOWED_GENDERS = ("men",)
# Keywords que identifican ligas femeninas (bloqueo a nivel de dataset).
WOMEN_KEYWORDS = (
    "women", "femenin", "feminin", "feminino", "femenine",
    "damen", "kobiet", "donna", "lbf", "wnba", "wcba",
    "women's", "divisione femminile", "liga femenina",
)

# ==============================================================================
# CUTOFFS DE TIEMPO (entrenamos antes de que termine el cuarto previo)
# ==============================================================================
Q3_GRAPH_CUTOFF = 22   # minuto de snapshot para Q3 (Q2 aun activo)
Q4_GRAPH_CUTOFF = 31   # minuto de snapshot para Q4 (Q3 aun activo)

# Snapshots multiples que se usan durante training (cutoff dinamico).
# En inferencia SIEMPRE se usa el cutoff "productivo" (22 / 31).
Q3_TRAIN_SNAPSHOTS = (18, 20, 21, 22, 23)
Q4_TRAIN_SNAPSHOTS = (28, 29, 30, 31, 32)

# ==============================================================================
# GATES MINIMOS DE DATOS (calidad de input del partido)
# ==============================================================================
MIN_GP_Q3 = 14         # graph_points con minuto <= Q3_GRAPH_CUTOFF
MIN_GP_Q4 = 16
MIN_PBP_Q3 = 12        # eventos de play-by-play Q1+Q2
MIN_PBP_Q4 = 14

# ==============================================================================
# GATES DE LIGA - exigentes, sin fallback global
# ==============================================================================
# Una liga debe tener AL MENOS estas muestras de training para generar modelo.
# Si no, directamente NO_BET en esa liga (el usuario pidio: sin fallback global).
LEAGUE_MIN_SAMPLES_TRAIN = 300
# Muestras minimas en validacion para aprender threshold optimo por liga.
LEAGUE_MIN_SAMPLES_VAL = 60
# Historico en inferencia (si una liga tiene <N partidos historicos en DB
# aun si el modelo se entreno, bloqueamos por seguridad).
LEAGUE_MIN_HISTORY_FOR_INFERENCE = 15

# ==============================================================================
# UMBRAL DE CONFIANZA BASE
# ==============================================================================
# Es un piso: cada liga tiene su propio threshold aprendido en backtest.
# Si el threshold aprendido es menor que este, se usa este piso.
MIN_CONFIDENCE_BASE = 0.75
# Si TARGET_HIT_RATE aun no se alcanza en backtest, se sube el threshold
# hasta este techo.
MAX_CONFIDENCE_CAP = 0.92

# ==============================================================================
# SPLIT TEMPORAL DEL DATASET
# ==============================================================================
# En dias desde el partido mas antiguo en DB.
# Train amplio (v13 usaba solo 10 dias -> underfitting).
TRAIN_DAYS = 50
VAL_DAYS = 20
# El resto va a calibration/holdout.
# Se calculan fechas concretas en dataset.split_temporal().

# ==============================================================================
# SELECCION DE ALGORITMOS POR TAMANO DEL SUBSET
# ==============================================================================
ALGO_FULL_THRESHOLD = 800      # >= full stack
ALGO_MEDIUM_THRESHOLD = 300    # >= xgb + catboost + logreg
# < MEDIUM -> solo [logreg, xgb] (con warning)

# ==============================================================================
# CALIBRACION
# ==============================================================================
CALIBRATION_METHOD = "isotonic"   # "isotonic" | "sigmoid" (Platt)
CALIBRATION_CV = 3

# ==============================================================================
# GATES ADICIONALES PARA INFERENCIA LIVE
# ==============================================================================
# Si hay muchos swings en el graph -> partido volatil, baja la fiabilidad.
MAX_VOLATILITY_SWINGS = 8
# Si la racha actual (current_run) es enorme, el modelo puede estar capturando
# un estado transitorio; cap por seguridad.
MAX_CURRENT_RUN_PTS = 14

# ==============================================================================
# REGRESION (auxiliar, se usa como CONFIRMACION, no como trigger primario)
# ==============================================================================
ENABLE_REGRESSION_CONFIRMATION = True
# Si el clasificador dice "home" pero la regresion predice away con spread
# mayor a este valor, se bloquea la apuesta.
REG_DISAGREEMENT_BLOCK_PTS = 4.0
# MAE maximo aceptable para que la regresion se use como filtro.
REG_MAX_MAE_ACCEPTABLE = 8.0

# ==============================================================================
# DEBUG / LOGGING
# ==============================================================================
# En inferencia se genera un payload completo con toda esta info para que
# el usuario pueda explorar en UI grafica o alimentar un dashboard.
INCLUDE_TOP_FEATURES_IN_DEBUG = 15
