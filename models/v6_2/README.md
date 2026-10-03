# Modelo Campeón Q3/Q4: `v6_2`

## Descripción
`v6_2` es el modelo campeón consolidado para la predicción de ganadores de cuartos Q3 y Q4. Mantiene la ingeniería de características avanzada de la familia V6 (presión anotadora, ventanas dinámicas `clutch`, gráficas de momentum), e introduce **poda algorítmica de ligas** (*league pruning*):
- Ligas con menos de 30 muestras de entrenamiento o con diferencial de señal < 0.015 son colapsadas a `LEAGUE_OTHER_SIGNAL_WEAK`.
- Exclusión declarativa de patrones conflictivos vía `v6_2_league_name_exclusions.json`.

## Arquitectura
- **Target Q3:**
  - Snapshot: Minuto 24 (fin de Q2).
  - Estimador: `XGBClassifier` único (`q3_champion.joblib`).
  - Umbrales de señal: `LEAN >= 0.10`, `BET >= 0.18`.
- **Target Q4:**
  - Snapshot: Minuto 36 (fin de Q3).
  - Estimador: Ensamble lineal `0.6 * XGBoost + 0.4 * HistGradientBoosting` (`q4_champion.joblib`).
  - Umbrales de señal: `LEAN >= 0.08`, `BET >= 0.14`.

## Estructura de Archivos
- `train.py`: Pipeline de entrenamiento y exportación de modelos.
- `predict.py`: Motor de inferencia para Q3 y Q4 retornando `PredictionResult`.
- `features.py`: Extractor de features F0-F15 y presión de anotación.
- `evaluate.py`: Evaluación offline contra base de datos histórica.
- `model_outputs/`: Modelos `.joblib` y tablas de exclusión/métricas.
