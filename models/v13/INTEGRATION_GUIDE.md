# V13 — Guía de Integración con Telegram Bot y Bet Monitor

## 📋 Resumen

V13 está diseñado para integrarse seamless con el sistema existente de `telegram_bot.py` y `bet_monitor.py`.

---

## 🔌 Integración con bet_monitor.py

### Paso 1: Importar módulo de integración

En `bet_monitor.py`, añadir al inicio:

```python
from training.v13 import integration as v13_integration
```

### Paso 2: Registrar V13 como modelo disponible

En la inicialización del monitor:

```python
# Al inicio de bet_monitor.py
AVAILABLE_MODELS = ["v2", "v4", "v6", "v9", "v12", "v13"]  # Añadir v13

# Registrar V13 y actualizar configuración
monitor_config = v13_integration.integrate_with_bet_monitor(monitor_config)
```

### Paso 3: Añadir rama V13 en _run_inference_sync

En la función `_run_inference_sync`:

```python
def _run_inference_sync(self, match_id: str, target: str) -> Dict:
    version = self._model_config.get(target, "v4")
    
    # ... existing code for other versions ...
    
    if version == "v13":
        result = v13_integration.run_v13_inference(match_id, target)
        return v13_integration.format_for_telegram(result)
    
    # ... rest of function ...
```

### Paso 4: Actualizar umbrales del monitor

V13 usa umbrales diferentes a V12:

```python
# ANTES (V12):
Q3_MINUTE = 24
Q4_MINUTE = 36
WAKE_BEFORE = 4
NO_BET_CONFIRM_TICKS = 2

# DESPUÉS (V13):
Q3_MINUTE = 22               # antes del fin de Q2
Q4_MINUTE = 31               # antes del fin de Q3
WAKE_BEFORE = 2              # menos margen muerto
NO_BET_CONFIRM_TICKS_Q3 = 1  # Q3: confirmación rápida
NO_BET_CONFIRM_TICKS_Q4 = 0  # Q4: sin espera (crítico)
```

---

## 📱 Integración con telegram_bot.py

### Paso 1: Añadir V13 a AVAILABLE_MODELS

En `telegram_bot.py` línea ~102:

```python
AVAILABLE_MODELS: list[str] = ["v2", "v4", "v6", "v9", "v12", "v13"]
```

### Paso 2: Registrar V13 al iniciar

En la función de inicialización del bot:

```python
from training.v13 import integration as v13_integration

def initialize_bot():
    # ... existing initialization ...
    
    # Register V13 if available
    MODEL_CONFIG = v13_integration.register_v13_models(MODEL_CONFIG)
    MONITOR_MODEL_CONFIG = v13_integration.register_v13_models(MONITOR_MODEL_CONFIG)
```

### Paso 3: Mostrar info de V13 en el panel

En el panel de debug o estado:

```python
def show_v13_status():
    info = v13_integration.get_v13_model_info()
    
    if info['available']:
        return f"""
🎯 V13 Model Status:
   Trained: {info['trained_at']}
   Models: {info['models_trained']}
   Leakage: {info['leakage_assessment']}
   Dataset: {info['dataset']['total_matches']} matches
"""
    else:
        return "🎯 V13: Not trained yet"
```

---

## 🔄 Flujo Completo de Inferencia

```
┌─────────────────────────────────────────────────────────────┐
│                     bet_monitor.py                          │
│                                                             │
│  1. Detecta partido en minuto objetivo (Q3=22, Q4=31)      │
│  2. Scrapea datos (graph points, PBP, scores)              │
│  3. _run_inference_sync(match_id, target)                  │
│         │                                                   │
│         ├─ Si version == "v13":                            │
│         │    run_v13_inference(match_id, target)            │
│         │         │                                         │
│         │         ├─ Carga modelo específico                │
│         │         ├─ Construye features                     │
│         │         ├─ Predice ganador                        │
│         │         ├─ Predice puntos                         │
│         │         └─ Aplica gates                           │
│         │                                                   │
│         └─ format_for_telegram(result)                     │
│              │                                              │
│              └─ Dict compatible con _render_inference_debug │
│                                                             │
│  4. Envía señal por Telegram                                │
└─────────────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────┐
│                    telegram_bot.py                          │
│                                                             │
│  5. _render_inference_debug(predictions)                    │
│         │                                                   │
│         ├─ Muestra winner_pick, confidence                  │
│         ├─ Muestra predicted_total ± MAE                    │
│         ├─ Muestra league_quality, volatility               │
│         └─ Muestra reasoning                                │
│                                                             │
│  6. Usuario ve señal completa                               │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuración Requerida

### En bet_monitor.py

```python
# V13 Configuration
V13_CONFIG = {
    # Timing
    'q3_minute': 22,
    'q4_minute': 31,
    'wake_before': 2,
    'confirm_ticks_q3': 1,
    'confirm_ticks_q4': 0,
    
    # Gates
    'min_gp_q3': 14,
    'min_gp_q4': 16,
    'min_pbp_q3': 12,
    'min_pbp_q4': 14,
    'min_confidence_q3': 0.62,
    'min_confidence_q4': 0.55,
    'max_volatility': 0.70,
}
```

### En telegram_bot.py

```python
# Available models
AVAILABLE_MODELS = ["v2", "v4", "v6", "v9", "v12", "v13"]

# Default models for monitor
MONITOR_MODEL_CONFIG = {"q3": "v13", "q4": "v13"}
```

---

## 🧪 Testing de Integración

### Paso 1: Verificar que V13 está disponible

```bash
cd match
python training/v13/cli.py status
```

Debe mostrar:
```
✅ Training summary found
   Version: v13
   Models trained: 12+
```

### Paso 2: Probar inferencia directa

```bash
python training/v13/cli.py infer <match_id> --target q3
```

### Paso 3: Probar desde bet_monitor

En el monitor de Telegram:
```
/monitor → ⚙️ Modelos → Q4: [v13]
```

Luego esperar siguiente partido y verificar que llega señal V13.

### Paso 4: Verificar formato en Telegram

La señal debe verse así:
```
🎯 Q3 Prediction (V13)
   Winner: Home (0.623)
   Signal: BET_HOME
   Pred Total: 85.2 ± 5.0
   League: strong ✅
   Volatility: 0.42
   Reasoning: Confidence 0.623 > 0.620
```

---

## ⚠️ Puntos Críticos

### 1. No Reiniciar para Cambiar Modelo

El selector de modelo funciona sin reiniciar:
```
Usuario: /monitor → ⚙️ Modelos → Q4: [v13]
→ callback: monmodel:set:q4:v13
→ set_model_config({"q4": "v13"})  ← efecto inmediato
→ se guarda en DB ← persiste
```

### 2. Fallback si V13 No Está Entrenado

Si `check_v13_available()` retorna False:
- V13 no aparece en AVAILABLE_MODELS
- El bot usa el modelo anterior (v12)
- No hay error, solo ignora V13

### 3. Umbrales Sincronizados

Todos los umbrales vienen de `config.py`:
- `bet_monitor.py` lee de `v13_integration.get_v13_config()`
- `infer_match_v13.py` lee de `config.py`
- **Un solo lugar para cambiar**

### 4. Formato de Output

`format_for_telegram()` retorna dict compatible con:
- `_render_inference_debug`
- `inference_debug_log.inference_json`
- Panel de Telegram

---

## 📁 Archivos de Integración

| Archivo | Propósito |
|---------|-----------|
| `integration.py` | Helpers para telegram_bot y bet_monitor |
| `infer_match_v13.py` | Motor de inferencia V13 |
| `config.py` | Umbrales centralizados |
| `cli.py` | CLI para testing y administración |

---

## 🚀 Checklist de Activación

Antes de poner V13 en producción:

- [ ] Entrenamiento completado exitosamente
- [ ] `leakage_detection.assessment` == "PASS"
- [ ] Prueba de inferencia funciona (`cli.py infer`)
- [ ] V13 añadido a AVAILABLE_MODELS en telegram_bot.py
- [ ] V13 añadido a AVAILABLE_MODELS en bet_monitor.py
- [ ] Rama V13 añadida en `_run_inference_sync`
- [ ] Umbrales actualizados en bet_monitor.py
- [ ] Prueba end-to-end con partido real
- [ ] Verificar que formato en Telegram es correcto

---

## 🔧 Troubleshooting

### "Model not found for q3_high_men"
- Ejecutar entrenamiento: `cli.py train`
- Verificar que todos los buckets tienen modelos

### "No V13 inference result"
- Verificar que match_id existe en DB
- Verificar que hay graph points y PBP
- Revisar logs de infer_match_v13.py

### "V13 not in available models"
- Verificar que entrenamiento se completó
- Reiniciar bot si es necesario (raro)

### "Confidence always NO_BET"
- Verificar que umbrales están sincronizados
- Revisar que datos live llegan completos
- Comparar con evaluación estática (`cli.py evaluate`)

---

**Estado**: ✅ **IMPLEMENTADO**  
**Archivos**: `integration.py`, `cli.py`, `eval_v13.py`  
**Documentación**: Esta guía + README actualizado
