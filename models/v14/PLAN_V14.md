# V14 — Plan de diseño completo
_Creado: 2026-04-15 | Base: análisis de V13 en producción_

---

## 0. Contexto de este documento

Este documento captura todas las ideas, hallazgos y decisiones de diseño para
V14 antes de que se empiecen a olvidar. Incluye:
- Las debilidades concretas y medidas de V13 en producción
- La propuesta de integración de TimesFM como regresor complementario
- El plan por fases con criterios de éxito

**Nota de lectura**: No implementar todo junto. Las fases están ordenadas por
impacto/esfuerzo. Fase 1 da el mayor retorno con el menor riesgo.

---

## 1. Diagnóstico de V13 — qué funciona y qué no

### 1.1 Lo que funciona bien en V13

- **Segmentación por pace**: separar low/medium/high mejoró F1 en hombres
- **Cutoff dinámico**: entrenar con múltiples snapshots (18-23 min) eliminó el
  problema de Q4 que siempre daba NO_BET en V12
- **Config centralizado**: un solo `config.py` elimina desajustes entre capas
- **Walk-forward validation**: sin data leakage comprobado (avg gap 0.143 todos
  los modelos)
- **Fallback automático a medium bucket**: evita crashes cuando pace es inusual

### 1.2 Debilidades críticas medidas (datos reales de training_summary.json)

#### A. Ventana de entrenamiento demasiado pequeña

| Métrica | Valor actual | Objetivo V14 |
|---|---|---|
| Días de train | **10 días** (22-31 ene) | ≥ 45 días |
| Muestras train total | **13,200** de 164,240 (8%) | ≥ 30% del dataset |
| Muestras disponibles ignoradas | **~151,000** | < 20,000 |

El dataset tiene 164K muestras (Jan 22 – Apr 15) pero solo 13K se usan para
entrenar. El 92% del dataset se usa solo para validación/calibración. Esto es
el cuello de botella principal de toda la arquitectura.

**Causa raíz**: El split temporal fue diseñado para ser "conservador" y evitar
data snooping. Fue demasiado conservative — 10 días no alcanza.

#### B. Modelos de mujeres — overfitting severo y MAE inútil

| Modelo | Train F1 | Val F1 | Gap | MAE total | Estado |
|---|---|---|---|---|---|
| q4_high_women | 0.929 | 0.608 | **0.322** | 12.5 pts | 🔴 Bloquear |
| q3_high_women | 0.832 | 0.648 | **0.183** | 16.9 pts | 🔴 Solo clasificación |
| q3_medium_women | 0.804 | 0.618 | **0.186** | 15.1 pts | 🔴 Solo clasificación |
| q4_medium_women | 0.812 | 0.654 | **0.158** | 8.2 pts | ⚠️ Usar con precaución |
| q3_low_women | 0.784 | 0.582 | **0.202** | 18.5 pts | 🔴 Solo clasificación |
| q4_low_women | 0.752 | 0.582 | **0.170** | 21.5 pts | 🔴 Bloquear regresor |

Causa: pocos datos de entrenamiento (85-570 muestras) vs. 1,000+ necesarios.
MAE de 15-21 pts en regresión hace que todos los "predicted_total" para mujeres
sean básicamente ruido.

#### C. Modelos de hombres — regresión mediocre

| Modelo | MAE home | MAE away | MAE total |
|---|---|---|---|
| q3_high_men | 8.4 pts | 7.6 pts | 13.5 pts |
| q4_high_men | 9.3 pts | 8.7 pts | **16.3 pts** |
| q4_low_men | 8.9 pts | 7.4 pts | 13.9 pts |

Un MAE total de 13-16 pts significa que la predicción de spread (total O/U)
tiene un error de ±16 puntos — demasiado para apostar en mercados O/U cerrados.
Solo `q3_low_men` (MAE 7.1) y `q3_medium_men` (MAE 10.6) son usables.

#### D. Features de series temporales descartadas

La pipeline actual usa solo **features resumen** de la serie histórica del partido:
`gp_diff`, `gp_slope_3m`, `gp_slope_5m`, `gp_acceleration`, `gp_amplitude`, `gp_swings`.

Esto descarta información de **trayectoria**: un equipo que va perdiendo -10 y
remonte a 0 tiene el mismo `gp_slope_3m` que uno que va ganando +10 y baje a 0,
pero su significado predictivo es completamente distinto (comeback vs. colapso).

La tabla `play_by_play` guarda `home_score` y `away_score` en cada evento —
la serie temporal real del score está disponible pero nunca se usa como secuencia.

#### E. Ligas con datos flojos en producción

Ligas que aparecen seguido en producción pero tienen señal débil porque:

| Liga / Problema | Causa |
|---|---|
| NCAA Women's Division I | 11,440 muestras totales pero solo ~700 en train (10 días) |
| Ligas latinoamericanas (CIBACOPA, Liga Argentina, Brazil NBB) | Muchos partidos de mujeres → MAE alto |
| Ligas europeas pequeñas (B2 League, Élite 2, Serie B) | Sub-representadas en train, estilos de juego distintos |
| Ligas que aparecen solo en val/cal | Cero historial → `league_quality: weak` → bloqueo total |
| Ligas con `⚠️ no en entrenamiento` | Modelo extrapola fuera de distribución sin saberlo |

El gate de liga bloquea todo con < 15 muestras históricas. Es correcto para
seguridad pero genera muchos NO_BET en ligas nuevas que pueden ser apostables.

#### F. `graph_points.value` es momentum, no score real

El dato de `graph_points` es la **curva de presión/momentum de Sofascore**
(un índice normalizado), no el marcador real. Para predicción de puntos esto es
un proxy indirecto potencialmente ruidoso.

La serie de score real existe en `play_by_play.home_score` y
`play_by_play.away_score` evento a evento pero nunca se usa como secuencia.

---

## 2. Ideas y hallazgos de esta sesión (2026-04-15)

### 2.1 TimesFM como regresor complementario

**Idea**: usar TimesFM 2.5 (Google Research, 200M params) no como predictor final
sino como generador de features adicionales para el clasificador CatBoost.

**Flujo propuesto**:
```
PBP home_score series  ──┐
PBP away_score series  ──┼──► TimesFM forecast ──► [tfm_winner_pick,
                          │                           tfm_margin,
                          │                           tfm_uncertainty]
                          │                               │
Halftime diff, pace ─────┘                               ▼
+ features de V13 existentes ──────────────────► CatBoost classifier
                                                (aprende a ponderar TimesFM
                                                 según contexto de liga/pace)
```

**Por qué es complementario y no reemplazante**:
- V13 ya clasifica bien cuando la ventaja es clara y el pace es consistente
- TimesFM aportaría señal en casos difíciles: partidos ajustados donde la
  **trayectoria importa** más que los números actuales
- CatBoost aprende cuándo confiar en cada fuente
- Si TimesFM tiene alta incertidumbre → `tfm_uncertainty` alto → CatBoost
  aprende a ignorar la señal en esos casos

**Feasibilidad confirmada**:
- RTX 3050 (4GB VRAM): TimesFM 2.5 200M en fp16 ≈ 400MB — carga sin problema
- Fine-tuning con LoRA en 164K muestras: factible en RTX 3050
- Repositorio: https://github.com/google-research/timesfm (activo, actualizado Apr 2026)
- XReg disponible: permite pasar covariables externas (odds, league) junto a la serie

**Las 3 features que genera TimesFM**:

| Feature | Descripción | Tipo |
|---|---|---|
| `tfm_winner_pick` | 1 si TimesFM predice que home gana al final del cuarto | int (0/1) |
| `tfm_margin` | Diferencia predicha home-away al final del cuarto | float |
| `tfm_uncertainty` | Spread del intervalo de cuantiles (p90-p10) de TimesFM | float |

**Ventaja sobre regresores actuales**:
Los regressors de V13 toman features resumen de la serie. TimesFM ve la forma
completa del momentum — "comeback vs. colapso" tiene patrones de serie distintos
aunque tengan el mismo slope_3m.

**Nota de implementación**:
- TimesFM debe cargarse UNA sola vez al inicio del bot (no por partido)
- En inferencia: extraer serie de `play_by_play` hasta cutoff minute
- Resampling a intervalo fijo (e.g. 1 evento cada 30 segundos) para longitud
  consistente
- Horizon: ~15-20 eventos restantes en el cuarto (no en minutos reales)
- Usar `use_continuous_quantile_head=True` para obtener intervalos

### 2.2 Split de entrenamiento ampliado

El insight más directo: cambiar el split de 10 días train a 45+ días usando
**walk-forward rolling window** en vez de un split fijo.

Ejemplo de split propuesto para V14:
```
Train:    Jan 22 - Mar 07  (45 días, ~60,000 muestras)
Val:      Mar 08 - Apr 01  (25 días, ~80,000 muestras)
Cal:      Apr 02 - Apr 15  (14 días, ~24,000 muestras)
```

Esto multiplica los modelos de mujeres por ~4.5x en muestras de entrenamiento.

### 2.3 Features nuevas de PBP como series temporales (sin TimesFM)

Incluso sin TimesFM, hay features de trayectoria que se pueden calcular
directamente del PBP y que V13 ignora:

```python
# Ejemplo de features de trayectoria que se pierden en V13:
'score_diff_at_q1_end'           # diff al final Q1 (exacto, no slope)
'score_diff_trend_last_10_events' # promedio diff últimos 10 eventos
'largest_lead_home'              # mayor ventaja que tuvo home en el partido
'largest_lead_away'              # mayor ventaja que tuvo away
'lead_changes'                   # número de veces que cambió el lider
'times_tied'                     # veces que el marcador estuvo empatado
'current_run_home'               # racha actual de puntos consecutivos home
'current_run_away'               # racha actual de puntos consecutivos away
'last_5_events_home_pts'         # puntos de home en últimos 5 eventos
'last_5_events_away_pts'         # puntos de away en últimos 5 eventos
'q3_momentum_shift'              # si el lider cambió en últimos 3 minutos
```

Estas features son de mayor señal que `gp_slope_3m` porque usan el **marcador
real** (no el índice de momentum de Sofascore) y capturan eventos discretos.

### 2.4 Covariables externas (future work)

Si en el futuro se tienen odds de bookmaker disponibles en tiempo real:
- `pregame_home_odds`: odds implican la probabilidad pregame
- `pregame_away_odds`: complemento
- `live_total_line`: línea O/U live del bookmaker

TimesFM XReg puede pasar estas como covariables externas.
Sin odds, la alternativa es head-to-head histórico entre equipos.

---

## 3. Plan de implementación por fases

### Fase 1 — Datos y split (MAYOR impacto, MENOR riesgo) — ~3 días

**Objetivo**: multiplicar datos de entrenamiento x4-5 sin cambiar la arquitectura.

Tareas:
1. Modificar `dataset.py` para aceptar parámetro `train_days=45` (vs. 10 hardcoded)
2. Implementar sliding window: train=[oldest, split1], val=[split1, split2], cal=[split2, newest]
3. Re-entrenar todos los modelos con la nueva ventana
4. Comparar `training_summary.json` de V13 vs V14: verificar que gap baja de 0.143 a < 0.10

Criterio de éxito:
- `q4_high_women.samples_train` ≥ 400 (vs. 85 en V13)
- `q3_low_women.train_val_gap` < 0.12 (vs. 0.202 en V13)
- Ningún modelo con < 200 muestras train

### Fase 2 — Features de trayectoria PBP (ALTO impacto, BAJO riesgo) — ~2 días

**Objetivo**: reemplazar features resumen de graph_points con features de
trayectoria directas del PBP (score real).

Tareas:
1. En `features.py`, agregar función `_trajectory_features(pbp, cutoff)`:
   - `score_diff_at_q1_end`, `score_diff_at_q2_end`
   - `lead_changes`, `times_tied`
   - `largest_lead_home`, `largest_lead_away`
   - `current_run_home`, `current_run_away` (racha activa)
   - `last_5_events_diff` (últimos 5 eventos: quién anotó más)
2. Mantener features de graph_points existentes (backward compatible)
3. Re-entrenar con ambos sets y comparar

**Criterio de éxito**:
- Val F1 para hombres sube de 0.61-0.67 a ≥ 0.68
- MAE total para `q3_low_men` baja de 7.1 a < 6.0

### Fase 3 — TimesFM como regresor (MEDIO impacto, MEDIO riesgo) — ~1 semana

**Objetivo**: integrar TimesFM como generador de 3 features adicionales
(`tfm_winner_pick`, `tfm_margin`, `tfm_uncertainty`) para el clasificador.

Tareas:
1. Fine-tuning de TimesFM 2.5 con LoRA sobre dataset de scores del PBP:
   - Input: serie `home_score - away_score` por evento hasta cutoff
   - Output: predicción del score diff al final del cuarto
   - Fine-tuning script: `training/v14/finetune_timesfm.py`
2. Evaluación offline: ¿TimesFM predice mejor el resultado que `gp_diff` solo?
3. Generar las 3 features en `features.py` como `_timesfm_features(pbp, target, cutoff)`
4. Re-entrenar clasificador CatBoost con features extendidas
5. Integrar en `infer_match_v14.py`:
   - Cargar modelo TimesFM una vez al inicio (model = TimesFM.load())
   - Pasar a `_timesfm_features()` en cada inferencia

**Dependencias**:
- `pip install timesfm[torch]` (o `[flax]`)
- PyTorch con CUDA para RTX 3050: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Repositorio: `git clone https://github.com/google-research/timesfm.git`
- Fine-tuning example: `timesfm/timesfm-forecasting/examples/finetuning/`

**Criterio de éxito/fracaso**:
- Si `tfm_winner_pick` tiene accuracy > 0.60 offline → incluir como feature
- Si `tfm_uncertainty` correlaciona con errores del clasificador → usar como gate
- Si Val F1 con TimesFM features ≤ Val F1 sin ellas → descartar TimesFM

### Fase 4 — Regresor de puntos mejorado (BAJO impacto hoy, MEDIO futuro) — ~3 días

**Objetivo**: reemplazar los regressors de CatBoost con TimesFM directamente
para predicción de puntos (home_pts, away_pts, total).

Solo avanzar si Fase 3 muestra que TimesFM tiene MAE < 8 pts en regresión.

Tareas:
1. Comparar MAE de TimesFM fine-tuneado vs CatBoost regresor actual
2. Si mejor en ≥ 6 de 12 modelos → reemplazar
3. Usar intervalos de cuantiles de TimesFM como `mae_live` (más informativo
   que el MAE promedio del training set que se usa hoy)

---

## 4. Guía de estilo de código para V14

Estas convenciones surgen de problemas reales en V13:

### 4.1 Fuente única de verdad para umbrales

```python
# config.py — ÚNICO lugar donde van todos los umbrales
# NUNCA hardcodear en infer_match_v14.py, telegram_bot.py, etc.
# Si necesitás el mismo valor en otro archivo: importar de config

from training.v14 import config
threshold = config.MIN_CONFIDENCE_Q3  # ✅
threshold = 0.62                       # ❌ — si cambia, hay que buscar en todo el código
```

### 4.2 Features deben ser deterministas y libre de leakage

```python
# ✅ CORRECTO: solo datos disponibles ANTES del cutoff
def _trajectory_features(pbp: list[dict], cutoff_minute: float) -> dict:
    events_before_cutoff = [e for e in pbp if e['minute'] <= cutoff_minute]
    ...

# ❌ MAL: usa datos del cuarto que se está prediciendo
def _q3_features(pbp: list[dict]) -> dict:
    q3_events = [e for e in pbp if e['quarter'] == 'Q3']  # LEAKAGE si predecimos Q3
    ...
```

### 4.3 Adaptador para inferencia live vs. entrenamiento

V13 tuvo el bug `'dict' object has no attribute 'match_id'` porque
`build_features_for_sample()` esperaba `TrainingSample` pero recibía `dict`.

En V14: definir un protocolo explícito:

```python
# En dataset.py — define el protocolo como clase abstracta o TypedDict
from typing import Protocol

class SampleLike(Protocol):
    match_id: str
    target: str          # 'q3' | 'q4'
    snapshot_minute: int
    features: dict       # features pre-calculadas (halftime_diff, etc.)

# features.py solo acepta SampleLike — nunca dict raw
def build_features_for_sample(sample: SampleLike, ...) -> dict:
    ...

# infer_match_v14.py crea un objeto que cumple el protocolo
@dataclass
class LiveSample:
    match_id: str
    target: str
    snapshot_minute: int
    features: dict
```

### 4.4 Separar carga de modelo de inferencia

```python
# ✅ CORRECTO: cargar una vez, inferir muchas veces
class V14Engine:
    def __init__(self):
        self.models = {}         # {model_key: clf_ensemble}
        self.timesfm = None      # TimesFM instance, cargado lazy
        self._load_all_models()

    def predict(self, match_id: str, target: str) -> V14Prediction:
        ...

# ❌ MAL: cargar modelo dentro de la función de predicción
def infer_match(match_id: str, target: str) -> dict:
    model = joblib.load(f"{model_key}_clf.joblib")  # I/O en cada predict
    ...
```

### 4.5 Logging estructurado

```python
# ✅ Logging con contexto suficiente para debugging post-mortem
import logging
logger = logging.getLogger(__name__)

logger.info(
    "v14.predict",
    extra={
        "match_id": match_id,
        "target": target,
        "pace_bucket": pace_bucket,
        "model_key": model_key,
        "fallback_used": fallback_used,
        "tfm_uncertainty": tfm_uncertainty,
        "final_signal": signal,
    }
)
```

### 4.6 Validar umbrales al cargar modelos

```python
# Al cargar training_summary.json, verificar que los umbrales del modelo
# coinciden con los de config.py activo. Si no coinciden → warning explícito.

def _validate_thresholds(summary: dict, current_config) -> None:
    saved = summary.get("pace_thresholds", {})
    if saved.get("q3_low_upper") != current_config.PACE_Q3_LOW_UPPER:
        logger.warning(
            "Pace threshold mismatch: model was trained with q3_low_upper=%s "
            "but config says %s — using model's saved thresholds",
            saved["q3_low_upper"], current_config.PACE_Q3_LOW_UPPER
        )
```

---

## 5. Estructura de archivos propuesta para V14

```
training/v14/
├── PLAN_V14.md                 ← este archivo
├── config.py                   ← umbrales centralizados (igual que V13)
├── dataset.py                  ← igual que V13 pero con train_days param
├── features.py                 ← features V13 + trajectory + timesfm_features()
├── dataset_protocol.py         ← SampleLike Protocol, LiveSample dataclass
├── train_clf.py                ← igual que V13
├── train_reg.py                ← igual que V13
├── train_v14.py                ← orchestrator
├── finetune_timesfm.py         ← NUEVO: fine-tuning TimesFM sobre PBP series
├── eval_timesfm.py             ← NUEVO: comparar TimesFM MAE vs CatBoost MAE
├── infer_match_v14.py          ← inferencia live, carga TimesFM lazy
├── walk_forward.py             ← (reusar de V13)
├── eval_v14.py                 ← evaluación
└── model_outputs/              ← modelos entrenados + training_summary.json
```

---

## 6. Métricas objetivo V14

| Métrica | V13 actual | Objetivo V14 | Método |
|---|---|---|---|
| Avg train-val gap | 0.143 | < 0.08 | Más datos train |
| Max gap (q4_high_women) | 0.322 | < 0.15 | Más datos train |
| Val F1 hombres promedio | 0.625 | ≥ 0.68 | Features trayectoria |
| Val F1 mujeres promedio | 0.615 | ≥ 0.64 | Más datos + features |
| MAE total hombres (low) | 7.1 pts | < 5.5 pts | Features trayectoria |
| MAE total mujeres (low) | 21.5 pts | < 10 pts | Más datos train |
| Modelos bloqueados por datos | 2 (high_women) | 0 | Más datos train |
| Ligas cubiertas sin bloqueo | ~70% | ≥ 85% | Split más amplio |

---

## 7. Orden de prioridades recomendado

```
1. ── FASE 1 ──────────────── Ampliar ventana de entrenamiento (45 días)
   Estimado: 3 días | Impacto: ALTO | Riesgo: BAJO
   Desbloquea: modelos de mujeres, MAE general, ligas raras

2. ── FASE 2 ──────────────── Features de trayectoria PBP
   Estimado: 2 días | Impacto: ALTO | Riesgo: BAJO
   Depende de: nada (puede hacerse en paralelo con Fase 1)

3. ── FASE 3 ──────────────── TimesFM como features
   Estimado: 1 semana | Impacto: MEDIO | Riesgo: MEDIO
   Depende de: Fase 1 (necesita datos suficientes para fine-tuning)
   Gate: solo avanzar si offline accuracy tfm_winner_pick > 0.60

4. ── FASE 4 ──────────────── TimesFM como regresor directo
   Estimado: 3 días | Impacto: variable | Riesgo: MEDIO
   Depende de: Fase 3 (necesita TimesFM fine-tuneado)
   Gate: solo reemplazar si MAE TimesFM < MAE CatBoost en ≥ 6/12 modelos
```

---

## 8. Notas de instalación TimesFM (para cuando llegue Fase 3)

```bash
# En el venv del proyecto
pip install timesfm[torch]

# PyTorch con CUDA (RTX 3050 → CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Fine-tuning con LoRA (ejemplo ya disponible en el repo de TimesFM)
# Ver: timesfm/timesfm-forecasting/examples/finetuning/

# Checkpoint base (se descarga automáticamente):
# google/timesfm-2.5-200m-pytorch en HuggingFace
```

**Verificar que la GPU se usa:**
```python
import torch
print(torch.cuda.is_available())   # debe ser True
print(torch.cuda.get_device_name(0))  # debe ser RTX 3050
```

---

## 9. Bugs conocidos de V13 a no repetir en V14

| Bug | Causa | Fix en V14 |
|---|---|---|
| `'dict' object has no attribute 'match_id'` | `build_features_for_sample()` espera objeto pero recibe dict | Usar `SampleLike` Protocol + `LiveSample` dataclass desde el día 1 |
| `mae_home`/`mae_away` siempre null | `_predict_points()` solo leía MAE del regresor total | Calcular MAE de los 3 regressors desde el inicio |
| JSON no aparece en consola para V13 | `_get_or_compute_predictions()` usaba path del engine viejo | Siempre usar `_compute_and_store_predictions()` para modelos avanzados |
| Umbrales desajustados entre capas | `bet_monitor.py` tenía umbrales distintos a V12 | Un solo `config.py`, importado en todas las capas |
| Flip NO_BET→BET Q4 muy tarde | `CONFIRM_TICKS=2` × 100s = 3.3 min en un cuarto de 12 min | Para Q4: CONFIRM_TICKS=0 (primer resultado es final) |
| `q4_high_women` overfitting extremo | 85 muestras train, train_f1=0.929 | Ampliar ventana → ≥400 muestras antes de desplegar |

---
_Fin del documento. Actualizar cuando haya cambios de diseño o nuevos hallazgos._
