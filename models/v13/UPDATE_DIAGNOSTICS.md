# V13 — Actualización: Dataset Metadata y Gráficas de Diagnóstico

## ✅ Lo Que Se Agregó

### 1. Gráficas de Diagnóstico Automáticas

**Archivo**: `plots.py`

Cada entrenamiento genera automáticamente las siguientes gráficas en `model_outputs/plots/`:

#### a) Learning Curves (`plots/{model_key}/learning_curve.png`)
- Muestra train vs validation score a diferentes tamaños de dataset
- **Detecta**: Overfitting, underfitting, data leakage
- Indicador automático:
  - Gap < 0.10 → ✅ HEALTHY
  - Gap 0.10-0.15 → ⚡ MODERATE
  - Gap > 0.15 → 🚨 POSSIBLE LEAKAGE

#### b) Feature Importance (`plots/{model_key}/feature_importance.png`)
- Top 20 features por importancia
- **Detecta**: Leakage directo (si `date`, `match_id`, `event` aparecen)
- Colores: azul (positivo), rojo (negativo)

#### c) Calibration Curves (`plots/{model_key}/calibration_curve.png`)
- Reliability diagram: predicted vs observed probability
- Histograma de distribuciónde predicciones
- **Detecta**: Sobre-confianza o sub-confianza
- Error de calibración < 0.05 → ✅ Bien calibrado

#### d) Leakage Detection Global (`plots/leakage_detection.png`)
- Compara TODOS los modelos en una sola gráfica
- Barras de train vs validation score
- Anota modelos con gap > 0.15
- **Assessment automático**: PASS / WARNING / FAIL

#### e) Dataset Summary (`plots/dataset_summary.png`)
4 paneles:
- Split temporal (train/val/cal)
- Buckets de pace por género
- Distribución Q3 vs Q4
- Top 10 ligas por muestras

---

### 2. Metadata Completa del Dataset

**Archivo**: `train_v13.py` (función `compute_dataset_metadata`)

Cada `training_summary.json` ahora incluye:

```json
{
  "dataset": {
    "total_samples": 82000,
    "total_matches": 16400,
    "date_range": {
      "oldest": "2026-01-22",
      "newest": "2026-04-15"
    },
    "splits": {
      "train": {
        "samples": 57400,
        "matches": 11480,
        "date_range": {"oldest": "...", "newest": "..."}
      },
      "validation": {
        "samples": 16400,
        "matches": 3280,
        "date_range": {"oldest": "...", "newest": "..."}
      },
      "calibration": {
        "samples": 8200,
        "matches": 1640,
        "date_range": {"oldest": "...", "newest": "..."}
      }
    },
    "by_target": {"q3": 41000, "q4": 41000},
    "by_gender": {"men": 65600, "women": 16400},
    "by_pace": {"low": 27333, "medium": 27333, "high": 27334},
    "by_model_key": {"q3_low_men": 6833, ...},
    "top_leagues": {"NCAA Women": 5645, ...},
    "pace_bucket_details": {
      "men_low": {"target": "q3", "samples": 6833, "positive_rate": 0.512},
      ...
    }
  }
}
```

### 3. Detección Automática de Leakage

**En `training_summary.json`**:

```json
{
  "leakage_detection": {
    "average_train_val_gap": 0.023,
    "max_gap": 0.045,
    "model_gaps": {
      "q3_high_men": {"train": 0.651, "val": 0.631, "gap": 0.020},
      "q3_medium_men": {"train": 0.671, "val": 0.659, "gap": 0.012},
      ...
    },
    "assessment": "PASS",
    "note": "Gap < 0.10 is healthy, 0.10-0.15 needs review, > 0.15 indicates possible leakage"
  }
}
```

**Criterios**:
- **PASS**: Gap promedio < 0.10 → Sin leakage sistemático
- **WARNING**: Gap 0.10-0.15 → Revisar modelos individuales
- **FAIL**: Gap > 0.15 → Leakage probable, NO usar modelo

**El script imprime advertencia al final si assessment es FAIL o WARNING**.

---

### 4. Información por Modelo

Cada modelo en `models_trained` ahora incluye:

```json
{
  "key": "q3_high_men",
  "samples_train": 6833,
  "samples_val": 1366,
  "samples_cal": 683,
  "val_accuracy": 0.643,
  "val_f1": 0.631,
  "train_f1": 0.651,
  "train_val_gap": 0.020,
  "algorithms": ["logreg", "gb", "xgb", "catboost"],
  "weights": {"logreg": 0.15, "gb": 0.25, "xgb": 0.35, "catboost": 0.25},
  "regression_mae": {"home": 3.2, "away": 3.1, "total": 5.0}
}
```

---

## 📊 Cómo Interpretar las Gráficas

### Learning Curve Saludable

```
Score
0.70 |              ┌───── Train
     |             /
0.65 |            /
     |           /
0.60 |          /    ┌─── Val
     |         /    /
0.55 |________/____/_________________
       10%   30%   50%   70%   100%
                  Samples
```

✅ Train y val convergen  
✅ Gap pequeño (< 0.05)  
✅ Ambas curvas se estabilizan  

### Learning Curve con Leakage

```
Score
0.80 |     ┌───── Train
     |    /
0.70 |   /
     |  /
0.60 | /          ┌─── Val
     |/          /
0.50 |__________/___________________
       10%   30%   50%   70%   100%
                  Samples
```

🚨 Gap grande (> 0.15) constante  
🚨 Train score muy alto  
🚨 Val score estancado  

### Feature Importance con Leakage

```
Feature           | Importance
------------------+------------
date              | ████████████ 0.85  ← 🚨 LEAKAGE!
match_id          | ██████ 0.42        ← 🚨 LEAKAGE!
gp_diff           | ████ 0.28
pbp_count         | ██ 0.15
```

🚨 Si `date`, `match_id`, `event`, `custom_id` aparecen con alta importancia → **LEAKAGE DIRECTO**

---

## 🚀 Uso

### Entrenamiento con todas las gráficas

```bash
cd match
python training/v13/train_v13.py
```

Al finalizar, revisar:

1. **`model_outputs/training_summary.json`**:
   - Ver `leakage_detection.assessment`
   - Si es "FAIL" → NO usar modelo
   - Si es "WARNING" → revisar modelos con mayor gap

2. **`model_outputs/plots/leakage_detection.png`**:
   - Verificar que no haya gaps grandes sistemáticos

3. **`model_outputs/plots/{model_key}/learning_curve.png`**:
   - Revisar curvas de cada modelo

4. **`model_outputs/plots/{model_key}/feature_importance.png`**:
   - Verificar que no haya features de leakage

---

## 📁 Estructura de Outputs

```
model_outputs/
├── training_summary.json          # Metadata completa + leakage detection
├── q3_*_clf_ensemble.joblib       # Modelos
├── q4_*_clf_ensemble.joblib
├── *_vectorizer.joblib
├── *_scaler.joblib
└── plots/
    ├── dataset_summary.png        # Resumen del dataset
    ├── leakage_detection.png      # Vista global de leakage
    ├── q3_high_men/
    │   ├── learning_curve.png
    │   ├── feature_importance.png
    │   └── calibration_curve.png
    ├── q3_medium_men/
    │   └── ...
    └── ...
```

---

## ✅ Checklist de Validación Post-Entrenamiento

Antes de usar el modelo en producción:

- [ ] `leakage_detection.assessment` == "PASS"
- [ ] Average gap < 0.10
- [ ] Ningún modelo con gap > 0.15
- [ ] Learning curves muestran convergencia
- [ ] Feature importance NO incluye `date`, `match_id`, `event`
- [ ] Calibration error < 0.10
- [ ] Val F1 > 0.55 (mejor que random)
- [ ] Dataset split tiene fechas correctas (train < val < cal)

---

**Estado**: ✅ **IMPLEMENTADO**  
**Archivos nuevos**: `plots.py`, `train_v13.py` actualizado  
**Documentación**: `README.md` actualizado con secciones de gráficas y metadata
