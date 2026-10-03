# V12 en Telegram Bot - Integración Completa

## ✅ Qué Se Implementó

### 1. V12 en el Selector de Modelos
- **AVAILABLE_MODELS** ahora incluye `"v12"`: `["v2", "v4", "v6", "v9", "v12"]`
- Puedes seleccionar V12 para Q3 y/o Q4 desde el menú de modelos
- Cuando V12 está activo, el bot usa `_run_v12_inference()` en lugar del infer_match estándar

### 2. Botón "V12 LIVE Bookmaker" en Match Detail
- Cada match ahora tiene un botón **"V12 LIVE Bookmaker"**
- Al presionarlo, analiza ambos quarters (Q3 y Q4) con líneas justas
- Muestra:
  - Proyecciones de puntos
  - Momentum del graph
  - Mercados con odds justas
  - Thresholds de VALUE (fair_odds × 1.15)

### 3. Funciones Agregadas al Bot

#### `_run_v12_inference(match_id, target)`
- Corre inferencia V12 para predicción de ganador
- Retorna estructura compatible con el formato de predicciones del bot
- Incluye: winner_pick, confidence, predicted_total, risk_level, etc.

#### `_run_v12_live_analysis(...)`
- Corre el Virtual Bookmaker para un quarter específico
- Genera líneas justas para 1X2, Handicap, Over/Under
- Calcula thresholds de VALUE

#### `_handle_v12_live_analysis(update, context, query, match_id, page)`
- Handler principal del botón V12 LIVE
- Analiza Q3 y Q4 del match
- Muestra resultado formateado con botón de refresh

### 4. Modificación en `_compute_and_store_predictions`
```python
# Detecta si V12 está seleccionado
using_v12 = "v12" in MODEL_CONFIG.values()

if using_v12:
    # Usa V12 inference para los quarters seleccionados
    for quarter in ["q3", "q4"]:
        if MODEL_CONFIG[quarter] == "v12":
            v12_result = _run_v12_inference(match_id, quarter)
            # ... store prediction
```

## 📋 Cómo Usar V12 en el Bot

### Predicción Pre-Bet con V12:
1. Ve al menú principal → "Modelos"
2. Selecciona V12 para Q3 y/o Q4
3. Ve a un match → "Refresh datos + pred"
4. El bot calculará predicciones usando V12

### Análisis LIVE con V12 Bookmaker:
1. Abre cualquier match (en vivo o terminado)
2. Presiona **"V12 LIVE Bookmaker"**
3. Verás:
   ```
   V12 LIVE BOOKMAKER - Match 15879747
   
   ========================================
   Q3 - Marcador: 8-3
   ========================================
   Proyección: 15-15 (diff -0)
   Momentum: -6.8 (away)
   Total proyectado: 30 pts
   
   Over 30 pts en quarter (actual: 11)
     Prob: 55.3% | Fair: 1.81
     → Si tu casa ofrece Over 30 a 2.08 o mayor → VALUE
   
   Under 30 pts en quarter (actual: 11)
     Prob: 44.7% | Fair: 2.24
     → Si tu casa ofrece Under 30 a 2.57 o mayor → VALUE
   ```
4. Compara con tu casa de apuestas
5. Si hay VALUE → apuesta

### Refresh del Análisis:
- Botón "Refresh análisis" para actualizar con datos más recientes

## 🔧 Archivos Modificados

### telegram_bot.py:
- Línea ~89: `AVAILABLE_MODELS` actualizado
- Línea ~87-92: Imports de V12 (MODEL_OUTPUTS_V12_DIR, etc.)
- Línea ~120-220: Funciones `_run_v12_inference` y `_run_v12_live_analysis`
- Línea ~220-358: Función `_handle_v12_live_analysis`
- Línea ~3320-3370: Modificación en `_compute_and_store_predictions`
- Línea ~4070-4073: Botón "V12 LIVE Bookmaker" en keyboard
- Línea ~5170-5195: Handlers de callback para V12 LIVE

## 📊 Dónde Se Guardan los Resultados

### Tabla: `eval_match_results`
- V12 guarda resultados en la MISMA tabla que los otros modelos
- Columnas dinámicas se crean automáticamente:
  - `q3_pick__bot_hybrid_f1`
  - `q3_signal__bot_hybrid_f1`
  - `q3_outcome__bot_hybrid_f1`
  - `q3_confidence__bot_hybrid_f1`
  - etc.

### Stats en el Bot:
- El menú de fechas muestra stats de TODOS los modelos incluyendo V12
- Hit rate, profit, etc. se calculan automáticamente

## 🎯 Flujo Completo de Uso

```
1. ANTES del partido:
   Bot → Modelos → Selecciona V12 para Q4
   Bot → Match → Refresh datos + pred
   → V12 predice ganador con 55% confidence

2. DURANTE el quarter (min 6):
   Bot → Match → V12 LIVE Bookmaker
   → Muestra líneas justas:
      Handicap: Home +7.5 (fair odds 1.95)
      Threshold: 2.24
   
3. Comparas con tu casa:
   Tu casa: Home +8.5 @ 2.40
   → 2.40 > 2.24 → VALUE ✅
   
4. Apuestas y trackeas resultado
```

## ⚠️ Notas Importantes

1. **V12 necesita estar entrenado**: Corre `python training/v12/train_v12.py` primero
2. **Solo funciona en matches completos**: Necesita 4 quarters con datos
3. **LIVE analysis simula min 6**: Usa la mitad del score del quarter como punto de análisis
4. **Resultados se guardan automáticamente**: Igual que otros modelos

## 🚀 Próximas Mejoras Posibles

1. **Input manual de odds del usuario**: Para cálculo automático de VALUE
2. **Alerts en vivo**: Cuando hay VALUE claro
3. **Tracking de apuestas**: Guardar qué apostaste y resultado
4. **Dashboard de stats V12**: Ver hit rate específico de V12

---

*Integración completada: 2026-04-13*
*V12 fully integrated into Telegram Bot*
