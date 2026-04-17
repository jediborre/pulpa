# Análisis de datos faltantes para entrenamiento v13
_Generado: 2026-04-15 — base: training_summary.json_

---

## 1. ¿Qué anomalías pueden aparecer en inferencia?

Cuando corrés `infer match` el sistema puede marcar estas observaciones:

| Observación | Causa |
|---|---|
| `🚨 Low samples` | El modelo de ese bucket tiene < 200 muestras de entrenamiento |
| `⚠️ Moderate samples` | El modelo tiene entre 200–499 muestras (ensemble reducido) |
| `Fallback usado` | No existía modelo para el pace bucket real → usó `medium` |
| `⚠️ MAE se superpone` | El error de regresión es mayor que la diferencia predicha de puntos |
| `⚠️ Alta volatilidad` | El partido tiene swings extremos en graph_points |
| `⚠️ no en entrenamiento` | La liga del partido nunca apareció en el training set |
| `NO_BET (confianza baja)` | Confianza < MIN_CONFIDENCE (0.62 q3 / 0.55 q4) |
| `League quality: weak` | La liga tiene < 15 muestras históricas → bloqueo total |

---

## 2. Datos que MÁS FALTAN para entrenar bien

### 2.1 Muestras de entrenamiento por modelo (solo ventana train Jan 22–31)

| Modelo | Muestras train | Estado | Gap |
|---|---|---|---|
| **q4_high_women** | 85 | 🔴 Crítico | 0.322 |
| **q3_high_women** | 95 | 🔴 Crítico | 0.183 |
| **q3_medium_women** | 345 | 🔴 Crítico | 0.186 |
| **q4_medium_women** | 350 | ⚠️ Insuficiente | 0.158 |
| **q3_low_women** | 570 | ⚠️ Insuficiente | 0.202 |
| **q4_low_women** | 575 | ⚠️ Insuficiente | 0.170 |
| q3_low_men | 1,205 | ✅ Aceptable | 0.125 |
| q4_low_men | 1,150 | ✅ Aceptable | 0.123 |
| q3_medium_men | 1,730 | ✅ Aceptable | 0.068 |
| q4_medium_men | 1,880 | ✅ Aceptable | 0.109 |
| q3_high_men | 2,655 | ✅ Bueno | 0.038 |
| q4_high_men | 2,560 | ✅ Bueno | 0.037 |

### 2.2 Causa raíz del problema

La ventana de entrenamiento es **solo 10 días (22–31 enero)**.
De esos 10 días, los partidos de mujeres con pace alto son extremadamente escasos
porque ese ritmo (high pace mujeres) casi no existe en enero.

### 2.3 MAE de regresión más problemáticos

Los modelos de regresión (predicción de puntos por equipo) tienen MAE inaceptable en mujeres:

| Modelo | MAE home | MAE away | MAE total |
|---|---|---|---|
| q3_low_women | **15.3 pts** | 6.4 pts | **18.5 pts** |
| q4_low_women | 10.3 pts | **13.4 pts** | **21.5 pts** |
| q3_medium_women | 12.7 pts | 5.5 pts | **15.1 pts** |
| q3_high_women | 12.0 pts | 7.4 pts | **16.9 pts** |

Un MAE total de 18–21 pts en predicción de cuartos hace que la señal O/U sea inútil para mujeres.

### 2.4 Ligas más valiosas que añadir (por tamaño y ausencia en train)

La liga con más partidos en val/cal pero **sub-representada en train** (10 días):

| Liga | Muestras totales | En train (est.) |
|---|---|---|
| NCAA Women's Division I | 11,440 | ~700 |
| NBA | 5,720 | ~350 |
| NBA G League | 3,410 | ~200 |
| Poland 2nd League | 3,340 | ~200 |
| Liga Argentina | 2,510 | ~150 |

---

## 3. Qué hacer para mejorar

### Prioridad ALTA
1. **Ampliar la ventana de entrenamiento** → al menos 30 días (vs 10 actuales)
   - Impacto directo: todos los modelos de mujeres multiplicarían sus muestras por ~3
   - Gap esperado post-ampliación: mujeres bajaría de ~0.20 a ~0.12

2. **Bloquear `q4_high_women` en inferencia** hasta tener ≥ 200 muestras
   - Actualmente: 85 muestras, train_f1=0.929 → puro overfitting
   - Workaround: el código ya hace fallback a `medium` → verificar que funcione

3. **MAE de regresión para mujeres** → no usar predicciones de score en ligades de mujeres hasta tener MAE < 8 pts

### Prioridad MEDIA
4. **Ligas no vistas en entrenamiento** → cuando aparezca `⚠️ no en entrenamiento`, el modelo está extrapolando completamente fuera de distribución

---

## 4. Criterio para producción

Ver sección siguiente.
