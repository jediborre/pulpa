# Modelo Q4: `v6_3`

## Descripción
`v6_3` es una variante especializada para Q4 que añade una **lista negra manual exhaustiva de ligas** tóxicas o impredecibles (NBA, ligas juveniles, ligas femeninas secundarias) y soporta snapshots temporales tempranos a los minutos 27 y 30.

## Arquitectura
- **Target:** Q4 exclusivamente.
- **Snapshots:** Minuto 27 y Minuto 30.
- **Estimador:** Ensamble 50/50 `XGBoost + HistGradientBoosting`.
- **Filtro:** Lista negra (`league_blacklist.py`) aplicada antes de ejecutar la inferencia.
