"""Script para actualizar modelos.md y docs/modelos/modelos.md con las métricas del modelo limpio podado."""

with open("modelos.md", "r", encoding="utf-8") as f:
    content = f.read()

pos_start = content.find("Para garantizar rigurosamente que el modelo no tiene sobreajuste")
pos_end = content.find("### 5. Resumen de Ligas")

new_block = """### 3. Validación Walk-Forward Multi-Corte en el Tiempo (Anti-Overfitting sobre Dataset Podado)

Para garantizar rigurosamente que el modelo no tiene sobreajuste ni depende de rachas pasajeras, se entrenó y probó en tres ventanas temporales mutuamente excluyentes hacia el futuro sobre los **197,364 partidos profesionales limpios**:

```
[2015 .................... 2022] ---> TEST 2023 (31,215 partidos) -> WR: 71.85% | AUC: 0.6433 | Brier: 0.2319
[2015 .............................. 2023] ---> TEST 2024 (37,704 partidos) -> WR: 73.57% | AUC: 0.6559 | Brier: 0.2285
[2015 ........................................ 2024] ---> TEST 2025-2026 (34,922 partidos) -> WR: 75.64% | AUC: 0.6695 | Brier: 0.2250
```

| Corte Temporal | Train Window | Test Window | N Test | Test AUC | Test Brier | Apuestas Filtro Verde | Win Rate Operativo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corte 1** | $\\le 2022$ | **2023** | 31,215 | **0.6433** | 0.2319 | 4,028 | **71.85%** |
| **Corte 2** | $\\le 2023$ | **2024** | 37,704 | **0.6559** | 0.2285 | 5,595 | **73.57%** |
| **Corte 3** | $\\le 2024$ | **2025-2026** | 34,922 | **0.6695** | 0.2250 | 5,785 | **75.64%** |

> **Diagnóstico:** La poda de ligas ruidosas elevó la precisión en **más de 4.5 puntos porcentuales de Win Rate** en todos los cortes temporales. El modelo alcanza un **75.64% de acierto operativo** en el período 2025-2026 con un AUC de **0.6695**, demostrando que la curación de datos eliminó el lastre de falsos favoritos.

---

### 4. Diagnóstico de Overfitting en el Split Reciente (162k Train vs 35k Test)

* **Train AUC:** `0.6672` | **Test AUC:** `0.6708` | **Brecha:** `-0.0036` *(Test supera a Train gracias a la fuerte regularización L2 / lambda=50)*.
* **Train Brier:** `0.2255` | **Test Brier:** `0.2246` *(Calibración óptima y menor error cuadrático)*.
* **Win Rate en Señales Verdes (Test):** `75.58%` (5,884 apuestas operables de alta convicción).
* **Ubicación del Dashboard Gráfico:** [`models/m27_v5/model_outputs/m27_v5_diagnostic_dashboard.png`](file:///C:/Users/App/Desktop/pulpa/models/m27_v5/model_outputs/m27_v5_diagnostic_dashboard.png). Contiene:
  1. *Top 12 Feature Importances* (XGBoost Gain).
  2. *Curva de Calibración de Probabilidades* contra la diagonal perfecta de 45°.
  3. *Estabilidad Temporal Walk-Forward* (AUC y WR año con año).
  4. *Dispersión del Regresor Dual-Head de Margen Q4* (MAE = 4.62 pts).

---

"""

if pos_start != -1 and pos_end != -1:
    content = content[:pos_start] + new_block + content[pos_end:]
    with open("modelos.md", "w", encoding="utf-8") as f:
        f.write(content)
    with open("docs/modelos/modelos.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("modelos.md y docs/modelos/modelos.md actualizados con éxito!")
else:
    print("Error buscando posiciones:", pos_start, pos_end)
