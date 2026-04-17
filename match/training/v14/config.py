"""
config.py — Fuente única de verdad para umbrales de V14.

REGLA: Ningún otro archivo debe hardcodear estos valores.
       Si necesitás un umbral en infer_match_v14.py, telegram_bot.py, etc.
       → importar de aquí.

Cambios respecto a V13:
- train_days: 45 días (vs. 10 en V13) → ver dataset.py
- MIN_CONFIDENCE_Q4: 0.52 (bajado de 0.55 — Q4 live tiene menos datos disponibles)
- TIMESFM_ENABLED: flag para activar/desactivar TimesFM sin cambiar código
- TIMESFM_UNCERTAINTY_GATE: si tfm_uncertainty > este valor, ignorar señal TimesFM
"""

# ── Ventanas de features (minuto máximo incluido en snapshot) ─────────────────
Q3_GRAPH_CUTOFF = 22      # igual que V13
Q4_GRAPH_CUTOFF = 31      # igual que V13

# ── Gates mínimos de datos ────────────────────────────────────────────────────
MIN_GP_Q3 = 14
MIN_GP_Q4 = 16
MIN_PBP_Q3 = 12
MIN_PBP_Q4 = 14

# ── Umbrales de confianza ─────────────────────────────────────────────────────
MIN_CONFIDENCE_Q3 = 0.62
MIN_CONFIDENCE_Q4 = 0.52   # bajado de 0.55 — Q4 live llega con snapshot parcial

# ── Gate de volatilidad ───────────────────────────────────────────────────────
MAX_VOLATILITY = 0.70

# ── Gate de liga ──────────────────────────────────────────────────────────────
LEAGUE_MIN_SAMPLES_BLOCK    = 15
LEAGUE_MIN_SAMPLES_PENALIZE = 30
LEAGUE_MIN_SAMPLES_FULL     = 60

# ── Segmentación por ritmo de anotación ──────────────────────────────────────
# Recalcular con percentiles reales al entrenar V14 y actualizar estos valores.
# Los valores actuales son los de V13 como punto de partida.
PACE_Q3_LOW_UPPER  = 72
PACE_Q3_HIGH_LOWER = 85
PACE_Q4_LOW_UPPER  = 108
PACE_Q4_HIGH_LOWER = 126
PACE_MIN_SAMPLES   = 100

# ── Selección de algoritmos por tamaño de subset ─────────────────────────────
ALGO_FULL_THRESHOLD   = 500
ALGO_MEDIUM_THRESHOLD = 200

# ── Dataset split (nuevo en V14) ──────────────────────────────────────────────
# Días de entrenamiento contados desde la fecha más antigua del dataset.
# V13 usaba 10 días → overfitting severo en mujeres.
# V14 usa 45 días → ~4-5x más muestras de entrenamiento.
TRAIN_DAYS = 45    # días para train desde oldest_date
VAL_DAYS   = 25    # días para val (inmediatamente después de train)
# El resto del dataset va a calibración.

# ── TimesFM integration (Fase 3) ─────────────────────────────────────────────
# Poner TIMESFM_ENABLED = False hasta completar Fase 1 y Fase 2.
TIMESFM_ENABLED = False  # TODO: activar en Fase 3

# Umbral de incertidumbre de TimesFM. Si quantile spread > este valor,
# las features tfm_* no se pasan al clasificador (se usan 0.0 como fallback).
TIMESFM_UNCERTAINTY_GATE = 15.0  # puntos de spread (p90 - p10)

# Ruta al checkpoint fine-tuneado. None = usar checkpoint base de HuggingFace.
# Actualizar después de completar finetune_timesfm.py
TIMESFM_CHECKPOINT: str | None = None  # "training/v14/timesfm_finetuned/"

# ── Parámetros de modelo ───────────────────────────────────────────────────────
MIN_EDGE_FOR_OVER_UNDER = 4.0
DEFAULT_ODDS = 1.91
LOSS_PENALTY = 2.0

# ── Parámetros de timing ──────────────────────────────────────────────────────
MONITOR_Q3_MINUTE        = 22
MONITOR_Q4_MINUTE        = 31
MONITOR_WAKE_BEFORE      = 2
MONITOR_CONFIRM_TICKS_Q3 = 1
MONITOR_CONFIRM_TICKS_Q4 = 0   # Q4: primer resultado es final (sin espera)
