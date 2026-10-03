"""
config.py - Fuente unica de verdad para umbrales de V17.

Novedades respecto a V15:
- LIVE_SNAPSHOTS por cuarto (retomado de v13): predicciones en varios minutos
  durante el partido, no solo en un unico cutoff pre-cuarto.
- TOTAL_REG_AS_FEATURE: la regresion de totales se usa como feature
  adicional del clasificador.
- LEAGUE_ACTIVATION_*: umbrales para selector inteligente de ligas.
- VERSIONING + LOGGING: paths y flags para versionado y log de predicciones.
- TELEGRAM: flags y placeholders para el bot.
"""
from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# VERSION / PATH
# ==============================================================================
VERSION = "v17"
V17_DIR = Path(__file__).resolve().parent

# ==============================================================================
# OBJETIVO DE NEGOCIO
# ==============================================================================
DEFAULT_ODDS = 1.40
BREAK_EVEN_PCT = 1.0 / DEFAULT_ODDS   # 71.43%
TARGET_HIT_RATE = 0.75
MIN_ACCEPTABLE_HIT_RATE = 0.72

# ==============================================================================
# FILTROS DE DATASET
# ==============================================================================
ALLOWED_GENDERS = ("men",)
WOMEN_KEYWORDS = (
    "women", "femenin", "feminin", "feminino", "femenine",
    "damen", "kobiet", "donna", "lbf", "wnba", "wcba",
    "women's", "divisione femminile", "liga femenina",
)

# ==============================================================================
# CUTOFFS DE TIEMPO - LIVE MULTI-SNAPSHOT (nuevo en v17)
# ==============================================================================
# En v15 las predicciones solo ocurrian en el pre-cuarto (22 y 31).
# En v17 el engine live predice EN MULTIPLES MINUTOS durante el partido
# para que el bot/telegram pueda disparar apuestas cuando la confianza
# cruza un umbral.
#
# Cada snapshot es un "momento de decision". El modelo que se carga en el
# motor de inferencia elige el modelo entrenado con snapshot mas cercano.
#
# Criterio: el primer snapshot de Q3 (minuto 17) equivale a "despues del
# tercer play del Q2". El ultimo (minuto 23) esta ya arrancando Q3.

# Cutoffs principales (compatibilidad con v15)
Q3_GRAPH_CUTOFF = 22
Q4_GRAPH_CUTOFF = 31

# Snapshots en TRAINING (dinamico, data augmentation)
Q3_TRAIN_SNAPSHOTS = (17, 19, 21, 22, 23)
Q4_TRAIN_SNAPSHOTS = (26, 28, 30, 31, 32, 33)

# Snapshots en INFERENCIA LIVE (cuando el bot decide si apostar)
# Cada snapshot es "minuto real del partido al momento de decidir".
Q3_LIVE_SNAPSHOTS = (17, 19, 21, 22, 23)
Q4_LIVE_SNAPSHOTS = (26, 28, 30, 31, 32)

# ==============================================================================
# GATES MINIMOS DE DATOS (calidad de input del partido)
# ==============================================================================
# Se escalan con el snapshot minute: mas minutos = mas data esperada.
MIN_GP_Q3 = 12         # minimo absoluto; aplica al snapshot mas temprano
MIN_GP_Q4 = 14
MIN_PBP_Q3 = 10
MIN_PBP_Q4 = 12

# ==============================================================================
# GATES DE LIGA
# ==============================================================================
LEAGUE_MIN_SAMPLES_TRAIN = 200
LEAGUE_MIN_SAMPLES_VAL = 60
LEAGUE_MIN_HISTORY_FOR_INFERENCE = 15

# ==============================================================================
# UMBRAL DE CONFIANZA BASE
# ==============================================================================
MIN_CONFIDENCE_BASE = 0.70
MAX_CONFIDENCE_CAP = 0.89

# ==============================================================================
# SPLIT TEMPORAL DEL DATASET
# ==============================================================================
# Defaults = MODO PRODUCCION. Para barridos pasar overrides por CLI:
#   --train-days 65 --val-days 15 --cal-days 3 --holdout-days 7
TRAIN_DAYS = 72
VAL_DAYS = 15
CAL_DAYS = 3
HOLDOUT_DAYS = 0
# Holdout para barridos comparativos (no se aplica automaticamente).
HOLDOUT_DAYS_SWEEP = 7

# ==============================================================================
# SELECCION DE ALGORITMOS POR TAMANO DEL SUBSET
# ==============================================================================
ALGO_FULL_THRESHOLD = 800      # >= full stack
ALGO_MEDIUM_THRESHOLD = 300    # >= xgb + catboost + logreg

# ==============================================================================
# CALIBRACION
# ==============================================================================
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 3
# Fallback a temperature scaling cuando hay poca data
CALIBRATION_MIN_CAL_FOR_ISOTONIC = 100

# ==============================================================================
# GATES ADICIONALES PARA INFERENCIA LIVE
# ==============================================================================
MAX_VOLATILITY_SWINGS = 8
MAX_CURRENT_RUN_PTS = 14

# ==============================================================================
# REGRESION (nuevo en v17)
# ==============================================================================
ENABLE_REGRESSION_CONFIRMATION = True
REG_DISAGREEMENT_BLOCK_PTS = 5.0
REG_MAX_MAE_ACCEPTABLE = 8.5

# Nuevo en v17: el total predicho se usa como FEATURE del clasificador.
# Esto permite al classifier ver "este partido va a ser alto-scoring" y ajustarse.
TOTAL_REG_AS_FEATURE = True

# Nuevo en v17: quantile regression para obtener intervalos de confianza
# en la prediccion del total. Si el spread (p90 - p10) supera el umbral,
# la regresion no se usa como filtro (muy incierta).
REG_QUANTILE_UNCERTAINTY_GATE = 20.0

# ==============================================================================
# SELECTOR INTELIGENTE DE LIGAS (nuevo en v17)
# ==============================================================================
# Score minimo (0-100) para que una liga entre al catalogo de training.
# Ver dataset_analyzer.compute_league_score.
USE_LEAGUE_ACTIVATION = True
LEAGUE_ACTIVATION_MIN_SCORE = 32

# Dias sin partidos despues de los cuales una liga se considera inactiva.
LEAGUE_ACTIVATION_INACTIVE_DAYS = 14

# Minimo absoluto de partidos en los ultimos 30 dias para considerar
# la liga "viva" (mas permisivo que solo tener data historica).
LEAGUE_ACTIVATION_MIN_RECENT = 10

# ==============================================================================
# VERSIONADO DE MODELOS (nuevo en v17)
# ==============================================================================
MODEL_VERSIONING_ENABLED = True
WEEKLY_MODELS_DIR = V17_DIR / "model_outputs" / "weekly"
LATEST_LINK_NAME = "latest"  # symlink o carpeta "latest" con la ultima version

# ==============================================================================
# LOGGING ESTRUCTURADO (nuevo en v17)
# ==============================================================================
PREDICTION_LOG_DB = V17_DIR / "logs" / "predictions.db"
PREDICTION_LOG_ENABLED = True

# ==============================================================================
# DRIFT DETECTION (nuevo en v17)
# ==============================================================================
# Si el hit_rate rolling (ventana N) cae bajo este valor, alerta drift.
DRIFT_HIT_RATE_ALERT = 0.70
DRIFT_ROLLING_WINDOW = 50   # ultimas 50 apuestas evaluadas
# Train-val gap alert por liga
DRIFT_TRAIN_VAL_GAP_ALERT = 0.15

# ==============================================================================
# TELEGRAM BOT (nuevo en v17)
# ==============================================================================
TELEGRAM_ENABLED = bool(os.environ.get("V17_TELEGRAM_TOKEN"))
TELEGRAM_TOKEN = os.environ.get("V17_TELEGRAM_TOKEN", "")
TELEGRAM_ALLOWED_CHAT_IDS: tuple[int, ...] = tuple(
    int(x) for x in os.environ.get("V17_TELEGRAM_CHAT_IDS", "").split(",") if x.strip()
)

# ==============================================================================
# FORECASTING DE LA SERIE diff (home - away) - Fase 3 de v14
# ==============================================================================
# Backend principal: TimesFM (Google Research). Si el paquete `timesfm` esta
# instalado y disponible se usa el modelo pre-entrenado.
#
# Backends alternativos (preparados pero no activos por defecto):
#   - "chronos": Amazon Chronos (chronos-forecasting). Soporta Python 3.13.
#                Activar SOLO si TimesFM no esta disponible o no convence en
#                validacion. Requiere `pip install chronos-forecasting`.
#   - "holt":    Fallback siempre disponible, sin dependencias.
#
# Cuando se pide TimesFM y el paquete no esta instalado, se cae automaticamente
# al fallback Holt (contract-compatible). Las features tfm_* son las mismas
# en cualquiera de los 3 backends, asi que los modelos entrenados son
# intercambiables.
FORECAST_BACKEND = "holt"                # "timesfm" | "chronos" | "holt"

TIMESFM_ENABLED = False                  # v17 nace con forecast opcional, no central
TIMESFM_FORECAST_HORIZON = 20            # eventos a proyectar hacia adelante
TIMESFM_CHECKPOINT: str | None = None    # None = HF repo default google/timesfm-1.0-200m-pytorch

# Gate: si la incertidumbre (p90 - p10) supera este umbral, las features
# tfm_* no se consideran confiables para decision (pero siguen pasando al modelo).
TIMESFM_UNCERTAINTY_GATE = 15.0

# Chronos (si se activa via FORECAST_BACKEND="chronos")
CHRONOS_MODEL_NAME = "amazon/chronos-bolt-tiny"  # bolt: ~10x mas rapido que t5 en CPU

# ==============================================================================
# HIBRIDACION LEGACY (nuevo en v17)
# ==============================================================================
ENABLE_LEGACY_PRESSURE_FEATURES = True
ENABLE_LEGACY_MONTE_CARLO_FEATURES = True
LEGACY_MONTE_CARLO_SIMS = 750
ENABLE_LEGACY_CLUTCH_FEATURES = True

# ==============================================================================
# FEATURE PRUNING (mejora #1 del ROADMAP, 18-abr-2026)
# ==============================================================================
# Lista negra de features que el `feature_audit.py` identifico como
# redundantes (|Pearson| >= 0.90 con otra feature mas util) o con varianza
# despreciable. Se aplica en `features.py` al final de build_features_for_sample.
#
# Se curo manualmente desde la union del audit (20 candidatos) para evitar
# falsos positivos:
# - score_q3_* NO se drop globalmente: son 0 en q3 por diseno, pero criticas
#   en q4. Solo se droppea `score_q3_vs_ht_momentum` (derivada redundante).
# - meta_snapshot_minute se MANTIENE; se droppea su complemento
#   meta_minutes_to_quarter_end (deterministica de snapshot_minute + target).
# - gp_latest_diff se MANTIENE (|r_y|=0.072 en q3, aporta).
# - traj_score_diff_end se MANTIENE (aparece 21x en top_features de modelos).
# - score_halftime_diff y score_halftime_total se MANTIENEN (base,
#   ampliamente usadas). Se droppean sus derivadas redundantes:
#   score_halftime_diff_ratio, score_cumulative_total, score_cumulative_diff.
FEATURE_BLACKLIST: tuple[str, ...] = (
    # Varianza nula por target (constante dentro del modelo)
    "meta_target_is_q4",
    # Redundantes con `meta_snapshot_minute`
    "meta_minutes_to_quarter_end",
    # Redundantes con `score_halftime_total` y `score_q3_total`
    "pace_total_prior",
    "pace_ratio_vs_median",
    "score_cumulative_total",
    "score_cumulative_diff",
    "score_halftime_diff_ratio",
    "score_q3_vs_ht_momentum",
    # Redundantes con `league_ht_total_mean` (agregados globales de liga)
    "league_q3_total_mean",
    "league_q4_total_mean",
    # Redundantes con `gp_stddev` y `traj_largest_lead_away`
    "gp_amplitude",
    "gp_valley",
)

FEATURE_PRUNING_ENABLED = True

# ==============================================================================
# DEBUG / LOGGING
# ==============================================================================
INCLUDE_TOP_FEATURES_IN_DEBUG = 15


IGNORE_KILL_SWITCHES = True
