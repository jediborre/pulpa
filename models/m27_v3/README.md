# Modelo Campeón Q4: `m27_v3`

## Descripción
`m27_v3` es el modelo campeón de producción para predicción del ganador del cuarto cuarto (Q4) en partidos de baloncesto. Utiliza una ventana de observación estricta fijada en el **minuto 27** de juego (durante el desarrollo de Q3), integrando estadísticas de momentum, rachas recientes y métricas históricas de enfrentamientos directos cara a cara (**Head-to-Head / H2H**).

## Arquitectura
- **Snapshot:** Minuto 27 exacto.
- **Features Clave:**
  - Diferencial y puntuación de medio tiempo (Q1 + Q2).
  - Parciales acumulados de Q3 y tasas anotadoras por minuto.
  - Rachas anotadoras inmediatas en ventanas de 2 y 3 minutos (`clutch_run_diff`).
  - Historial H2H previo: `h2h_avg_q1_diff`, `h2h_recent3_home_won`, `h2h_last_home_won`.
  - Prior win rates de equipos en ventana móvil de 12 partidos.
- **Modelos:** Ensamble ponderado de `XGBoost` y `HistGradientBoostingClassifier`.
- **Calibración:** `IsotonicRegression` aplicada sobre las probabilidades del ensamble.

## Rendimiento Registrado
- **ROC AUC:** 0.789 - 0.791 (Split de Test / Validación)
- **Accuracy Q4:** ~71.0%
- **Yield / ROI estimado:** +13.0% a +29.0%

## Estructura de Archivos
- `train.py`: Pipeline de entrenamiento completo y generación de artefactos.
- `predict.py`: Motor de inferencia en vivo y por ID de partido, retornando `PredictionResult`.
- `features.py`: Extractor estandarizado de features del minuto 27 y H2H.
- `evaluate.py`: Script de evaluación sobre partidos históricos en `matches.db`.
- `model_outputs/`: Modelos serializados (`.joblib`) y resumen JSON de entrenamiento.
