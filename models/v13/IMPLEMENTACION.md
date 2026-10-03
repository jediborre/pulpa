# V13 — Implementación Completada (2026-04-15)

## ✅ Lo Que Se Ha Implementado

### 1. Análisis de Datos Real
- **Script**: `temp_scripts/analyze_pace_distribution.py`
- **Resultados**: 
  - Q3 (halftime): low ≤72, medium 72-85, high ≥85 pts
  - Q4 (3 cuartos): low ≤108, medium 108-126, high ≥126 pts
  - Muestras por bucket: ~4,354 (men), ~1,111 (women) → **Suficiente para Optuna**

### 2. Estructura Modular Completa

```
training/v13/
├── __init__.py                    # Package marker
├── config.py                      # ✅ Umbrales centralizados (actualizados con datos reales)
├── dataset.py                     # ✅ Dataset builder + pace buckets + cutoff dinámico
├── features.py                    # ✅ Feature engineering (graph + PBP)
├── walk_forward.py                # ✅ Walk-forward validation para league stats
├── train_clf.py                   # ✅ Entrenamiento de clasificadores
├── train_reg.py                   # ✅ Entrenamiento de regresores
├── train_v13.py                   # ✅ Orquestador principal
├── infer_match_v13.py             # ✅ Inferencia live
├── PLAN_V13.md                    # Plan original
├── TIMING_LIVE_ANALYSIS.md        # Análisis de timing
├── ANALISIS_V12_V13.md            # Análisis comparativo
└── model_outputs/                 # Directorio para modelos
```

### 3. Walk-Forward Validation
- **Archivo**: `walk_forward.py`
- **Función**: `compute_league_stats_walkforward()`
- **Cómo funciona**:
  - Para cada partido, calcula stats de liga usando SOLO partidos anteriores
  - Evita data leakage de futuro
  - Guarda historial acumulativo por liga

### 4. Dataset Builder
- **Archivo**: `dataset.py`
- **Funciones clave**:
  - `build_samples()`: Construye samples desde DB
  - `classify_pace_bucket()`: Clasifica por ritmo de anotación
  - `split_temporal()`: Divide train/val/cal por fechas
  - Caching para evitar reconstruir cada vez

### 5. Feature Engineering
- **Archivo**: `features.py`
- **Features**:
  - Graph points: count, diff, slope 3m/5m, acceleration, peak, valley, amplitude, swings
  - PBP events: count, pts per event, home/away pts, 3PT rate
  - Score features: quarter diffs

### 6. Entrenamiento
- **Clasificadores**: LogReg, GB, XGBoost, CatBoost
- **Regresores**: Ridge, GB, XGBoost, CatBoost
- **Selección automática** por tamaño de subset:
  - ≥500 muestras: todos los algoritmos
  - 200-499: LogReg, XGB, CatBoost
  - <200: LogReg, XGB
- **Ensemble** ponderado por F1 (clf) o inverso de MAE (reg)

### 7. Inferencia Live
- **Archivo**: `infer_match_v13.py`
- **Funciones**:
  - `run_inference()`: Predicción completa
  - Carga modelos según (target, gender, pace)
  - Fallback a "medium" si no hay modelo específico
  - Aplica gates de confianza

### 8. Config Actualizado
- **Umbrales de pace**: Basados en percentiles reales del DB
- **Gates**: Centralizados en un solo lugar
- **Timing**: MONITOR_Q3_MINUTE=22, MONITOR_Q4_MINUTE=31

---

## 📊 Hallazgos Clave del Análisis

### Distribución de Puntos (REAL)
```
Q3 (halftime total ambos equipos):
  Media: 79.6 pts
  p33: 72 pts → low ≤72
  p66: 85 pts → high ≥85
  
Q4 (3 cuartos total ambos equipos):
  Media: 119.0 pts
  p33: 108 pts → low ≤108
  p66: 126 pts → high ≥126
```

### Muestras por Bucket
```
Men Q3: ~4,354 por bucket → ✅ Excelente
Men Q4: ~4,353 por bucket → ✅ Excelente
Women Q3: ~1,112 por bucket → ✅ Bueno
Women Q4: ~1,111 por bucket → ✅ Bueno
```

**Conclusión**: Hay datos más que suficientes para todos los submodelos, incluyendo Optuna tuning.

---

## 🚀 Cómo Ejecutar

### Entrenar
```bash
cd match
python training/v13/train_v13.py
```

### Entrenar sin tuning (más rápido)
```bash
python training/v13/train_v13.py --skip-tuning
```

### Entrenar solo un subset
```bash
python training/v13/train_v13.py --subset q3_high
```

### Inferencia
```bash
python training/v13/infer_match_v13.py <match_id>
```

---

## ⚠️ Lo Que Falta (Para Implementar Después)

### 1. Evaluación con Simulación Live (`eval_v13.py`)
- Simular llegada de datos en minutos {20,22,24} para Q3
- Simular llegada de datos en minutos {30,32,34} para Q4
- Medir "time to stable signal"
- Contar flips NO_BET→BET

### 2. Integración en Telegram/Monitor
- Añadir "v13" a AVAILABLE_MODELS
- Rama V13 en `_run_inference_sync`
- Actualizar defaults a v13

### 3. Optuna Tuning (Opcional)
- `hparam_search.py` con tuning por submodelo
- Guardar mejores params en config
- Flag `--skip-tuning` ya existe

### 4. Calibración Platt
- `CalibratedClassifierCV` post-entrenamiento
- Mejora confianza para que sea probabilidad real

### 5. Gráficas de Diagnóstico
- Learning curves
- Calibration curves
- Feature importance
- Walk-forward performance

---

## 🎯 Diferencias Clave V12 vs V13

| Aspecto | V12 | V13 |
|---------|-----|-----|
| **League stats** | Usa TODO (leakage) | Walk-forward (solo pasado) |
| **Snapshots** | Fijos (min 24/36) | Dinámicos (18-23, 28-32) |
| **Modelos** | 2 clf + 12 reg | 12 clf + 36 reg (por pace) |
| **Config** | Duplicada | Centralizada |
| **Confianza** | 0.65 (igual) | 0.62 Q3, 0.55 Q4 |
| **Q4 live** | Siempre NO_BET | Snapshot realista → BET posible |
| **Pace buckets** | No existe | low/medium/high por percentiles |

---

## 📝 Notas Importantes

1. **Data Leakage**: V13 lo resuelve con walk-forward validation
2. **Pace Thresholds**: Actualizados con datos reales (no adivinados)
3. **Muestras**: Suficientes para todos los buckets (≥1,100)
4. **Optuna**: Viable pero opcional (usar `--skip-tuning` si hay prisa)
5. **Fallback**: Si un bucket no tiene modelo, usa "medium"

---

## 🔄 Próximos Pasos Recomendados

1. **Ejecutar entrenamiento de prueba**:
   ```bash
   python training/v13/train_v13.py --subset q3_medium
   ```

2. **Verificar que funcione** con un match de prueba

3. **Entrenar completo** (puede tardar ~30 min):
   ```bash
   python training/v13/train_v13.py
   ```

4. **Evaluar con simulación live** (cuando se implemente eval_v13.py)

5. **Integrar en telegram_bot.py** y probar con monitor

---

**Estado**: ✅ **LISTO PARA ENTRENAR Y PROBAR**

**Documentación**: Ver `ANALISIS_V12_V13.md` para análisis completo, `PLAN_V13.md` para plan original.
