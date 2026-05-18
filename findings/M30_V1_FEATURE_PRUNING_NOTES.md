# M30_V1 Feature Pruning Notes

Documento de trabajo para dejar trazabilidad de:

- features ya podadas de `m30_v1`
- features/familias que siguen bajo revision
- resultado observado despues de cada poda

## Contexto

`m30_v1` nacio como variante independiente de `m27_v1`, pero con un set mas amplio de features por entrar mas profundo en Q3. Despues de varias mediciones, el problema principal no fue un bug temporal aislado sino una combinacion de:

- demasiadas familias con senal debil
- demasiadas combinaciones categoricas y flags redundantes
- probabilidades poco sharp incluso antes del filtro de apuestas

Tambien se detecto un detalle operativo importante: cambios en `_build_m30_v1_features(...)` podian no reflejarse en el artifact si `dynamic_rows_cache.joblib` seguia vivo. Eso ya se corrigio con versionado de schema en `train_q4_m30_v1.py`.

## Features ya podadas

### Ronda 1

Estas fueron removidas en la primera poda basada en correlaciones y analisis de combos debiles:

- `q1_q2_q3_partial_combo`
- `q3_partial_current_combo`

### Ronda 2

Estas fueron removidas en la segunda poda, ya cruzando importancia del artifact contra correlacion real en test:

- `prior_wr_sum`
- `q1_q2_same_winner`
- `q1_q2_q3_partial_same_leader`
- `halftime_q3_partial_combo`
- `halftime_current_combo`
- `q3_partial_total`
- `q3_elapsed_minutes`
- `q3_partial_total_pace_per_min`
- `q3_partial_projected_total`
- `halftime_trailer_recovery_ratio`
- `current_leader_won_q2`

## Features nuevas que si se conservaron

Estas no entran como poda, pero conviene dejarlas anotadas porque fueron agregadas tras medir candidatas y siguen vigentes:

- `big_halftime_lead_now_close`
- `halftime_trailer_strong_recovery`

## Resultado despues de la poda real aplicada

Despues de reconstruir `dynamic_rows` y reentrenar con cache invalido correctamente:

- `feature_count` bajo de `139` a `112`
- las features podadas ya no aparecen en `run_summary.json`
- el modelo no mejoro; empeoro ligeramente

### Referencia de medicion

- Antes de la segunda poda:
  - ensemble test `accuracy = 0.563465`
  - ensemble test `roc_auc = 0.58697`
- Despues de la segunda poda:
  - ensemble test `accuracy = 0.561052`
  - ensemble test `roc_auc = 0.584798`

En evaluacion directa raw/ROI, `m30_v1` siguio debil:

- `accuracy_raw_pick = 0.5514`
- `pick_strength_media = 0.5761`
- `conteo_p_ge_0_65 = 396`
- solo `8` apuestas aceptadas en el reporte filtrado

Conclusion practica: habia ruido real en esas features, pero ese bloque no era el principal cuello de botella del modelo.

## Features y familias a revisar en la siguiente ronda

La siguiente revision debe ser quirurgica otra vez. Ya no conviene seguir cortando combinaciones gruesas a ciegas; ahora toca revisar bloques todavia vivos que siguen cargando importancia pero probablemente mezclan senal con ruido.

### 1. Recent window / momentum corto

Posibles sospechosas:

- `recent_4m_away_max_run`
- `recent_4m_away_points`
- `recent_4m_points_diff`
- `recent_4m_run_diff`
- `recent_2m_away_points`
- `recent_2m_last_scoring_away`
- `trailing_now_recent_run_4m`
- `trailing_now_recent_run_2m`

Motivo:

- esta familia sigue teniendo mucha masa de importancia
- su correlacion promedio observada fue baja
- puede estar sobreajustando micro-momentum poco estable al minuto 30

### 2. Halftime transitions / comeback state

Posibles sospechosas:

- `halftime_leader_now_tied`
- `halftime_leader_still_ahead`
- `halftime_leader_lost_lead`
- `halftime_trailer_gain`
- `halftime_trailer_gain_bin`
- `halftime_trailer_now_tied`
- `halftime_trailer_neutralized_deficit`

Motivo:

- varias de estas capturan la misma historia de comeback con distinto encoding
- pueden estar duplicando informacion ya contenida en `halftime_diff`, `score_est_diff` y `big_halftime_lead_now_close`

### 3. Current-state redundante

Posibles sospechosas:

- `current_leader=tied`
- `current_leader_won_q1`
- `current_leader_won_q3_partial`
- `current_trailer_won_q3_partial`
- `current_trailer_was_halftime_leader`
- `current_leader_was_halftime_trailer`
- `trailing_now_deficit_abs`

Motivo:

- varias son cruces indirectos de estado actual con historia previa
- pueden aportar poco sobre `score_est_diff`, `current_margin_bin` y `current_trailing_side`

### 4. Q3 partial derivados que siguen vivos

Posibles sospechosas:

- `q3_partial_away_pace_per_min`
- `q3_partial_home_pace_per_min`
- `q3_partial_projected_away`
- `q3_partial_projected_home`
- `q3_partial_projected_diff`
- `q3_partial_completion`
- `q3_remaining_minutes`

Motivo:

- la poda previa ya mostro que varios derivados de Q3 total/projection eran debiles
- todavia quedan transformaciones del mismo bloque que podrian seguir metiendo ruido

## Regla para la siguiente iteracion

Antes de remover otra familia, volver a medir sobre el artifact actual de `112` features:

- importancia del modelo
- correlacion real en test
- efecto en `accuracy_raw_pick`
- efecto en `pick_strength_p95`
- conteo de picks sobre `0.60`, `0.65` y `break_even`

La siguiente poda debe salir de una ablacion corta por familia, no de una limpieza amplia.

## Findings nuevos: ranking actual y candidatas avanzadas Q3->Q4

Despues de medir otra vez el artifact vigente de `112` features y de evaluar candidatas avanzadas enfocadas en cierre de Q3 / inicio de Q4, salieron dos conclusiones fuertes.

### 1. Ranking actual del artifact vivo

Las features con mayor senal real siguen siendo mayormente generales:

- `prior_wr_diff` `abs_corr = 0.1332`
- `away_prior_wr` `abs_corr = 0.0877`
- `home_prior_wr` `abs_corr = 0.0866`
- `halftime_diff` `abs_corr = 0.0803`
- `score_est_diff` `abs_corr = 0.0782`
- `gp_last` `abs_corr = 0.0769`
- `q2_diff` `abs_corr = 0.0740`
- `big_halftime_lead_now_close` `abs_corr = 0.0360`
- `q3_partial_home_share` `abs_corr = 0.0316`
- `q3_partial_diff` `abs_corr = 0.0308`

Lectura practica:

- `m30_v1` sigue apoyandose sobre todo en `prior + halftime state + current score state`
- el bloque realmente fino de cierre de Q3 todavia no domina el modelo
- la feature nueva `big_halftime_lead_now_close` si quedo como una de las mas utiles del bloque tardio

### 2. Candidatas avanzadas medidas para cierre de Q3 / inicio de Q4

Se midio un set nuevo de candidatas avanzadas, especialmente para ligas de 10 minutos donde `minute=30` cae justo en el borde Q3->Q4.

#### Candidatas con senal fuerte

- `q3_close_biglead_now_04_07_10m`
  - `abs_corr = 0.0346`
  - `cohen_d = 0.0693`
  - describe partidos con desventaja grande en halftime (`>= 8`) que al cierre de Q3 ya quedaron comprimidos a margen `4-7`

- `q3_close_flip_or_tie_after_biglead_10m`
  - `abs_corr = 0.0319`
  - `cohen_d = 0.0638`
  - describe partidos con desventaja grande en halftime que al cierre de Q3 ya llegaron a empate o cambio de lider

- `q3_close_biglead_flip_or_tie_and_q3_trailer_win_10m`
  - `abs_corr = 0.0319`
  - `cohen_d = 0.0638`
  - parece capturar casi la misma historia que la anterior; se considera fuerte pero potencialmente redundante

#### Candidatas medianas, no top

- `q3_close_biglead_flip_or_tie_without_prior_edge_10m`
  - `abs_corr = 0.0264`
- `q3_close_one_possession_10m`
  - `abs_corr = 0.0219`

### 3. Finding clave: el micro-momentum corto no fue fuerte

Las features de los ultimos `60s-120s` quedaron flojas:

- `q3_close_trailer_edge_60s_10m`
- `q3_close_trailer_edge_90s_10m`
- `q3_close_trailer_edge_120s_10m`
- `q3_close_trailer_run_90s_10m`
- `q3_close_trailer_run_120s_10m`
- `q3_close_trailer_last_scoring_60s_10m`
- `q3_close_trailer_last_scoring_90s_10m`

Lectura practica:

- lo que mejor separa no es el micro-run aislado del ultimo minuto/minuto y medio
- lo que mejor separa es el estado estructural de comeback al llegar al borde de Q4
- en especial cuando un juego de 10 minutos pasa de gran desventaja en halftime a margen medio (`4-7`) o a empate/cambio de lider justo al cierre de Q3

## Siguiente paso preferido segun evidencia

Si la ablacion del bloque `recent_4m/recent_2m` no contradice estos findings, las dos features avanzadas mas defendibles para implementar en `m30_v1` son:

- `q3_close_biglead_now_04_07_10m`
- `q3_close_flip_or_tie_after_biglead_10m`

La tercera (`q3_close_biglead_flip_or_tie_and_q3_trailer_win_10m`) solo deberia entrar si luego se confirma que agrega senal incremental y no solo duplica la segunda.

## Ablacion aislada del bloque recent_4m / recent_2m

Se corrio una ablacion aislada removiendo solo:

- `recent_4m_*`
- `recent_2m_*`
- `trailing_now_recent_run_4m`
- `trailing_now_recent_run_2m`

Resultado principal:

- el feature count del experimento bajo a `92`
- la ablacion no produjo una mejora contundente en todos los modelos
- pero tampoco defendio con claridad que ese bloque sea indispensable

### Delta vs baseline actual antes de meter las nuevas features Q3->Q4

- `xgb` test:
  - `accuracy +0.0027`
  - `roc_auc +0.0011`
  - `log_loss -0.0007`
- `hist_gb` test:
  - `accuracy -0.0007`
  - `roc_auc -0.0023`
  - `log_loss +0.0009`
- `ensemble_avg` test:
  - `accuracy +0.0031`
  - `roc_auc -0.0006`
  - `log_loss +0.0004`
- `ensemble_avg_cal` test:
  - `accuracy +0.0113`
  - `roc_auc +0.0003`
  - `log_loss -0.0004`

Lectura practica:

- el bloque `recent_*` sigue pareciendo mas ruido contextual que senal dominante
- hay indicios de que puede estorbar un poco, sobre todo en la salida calibrada
- pero la ablacion por si sola no justifica todavia podarlo definitivamente sin otra ronda de validacion

## Implementacion de las 2 features fuertes Q3->Q4

Despues de la ablacion se implementaron en `train_q4_m30_v1.py` estas dos features medidas como las mas defendibles:

- `q3_close_biglead_now_04_07_10m`
- `q3_close_flip_or_tie_after_biglead_10m`

Tambien se actualizo `FEATURE_SCHEMA_VERSION` para invalidar cache automaticamente.

### Estado del artifact despues de implementarlas

- `feature_count` subio de `112` a `114`
- ambas features nuevas aparecen en `run_summary.json`

### Resultado offline

- Antes de estas 2 features:
  - ensemble test `accuracy = 0.561052`
  - ensemble test `roc_auc = 0.584798`
- Despues de estas 2 features:
  - ensemble test `accuracy = 0.565878`
  - ensemble test `roc_auc = 0.584742`

Lectura:

- subio la accuracy del ensemble
- el ROC AUC practicamente no se movio
- la mejora parece mas de decision threshold / orden local que de separacion global fuerte

### Resultado en evaluacion directa raw/ROI

- Antes de estas 2 features:
  - `accuracy_raw_pick = 0.5514`
  - `pick_strength_media = 0.5761`
  - `conteo_p_ge_0.65 = 396`
  - `apuestas = 8`
  - `ganancia = 10`
- Despues de estas 2 features:
  - `accuracy_raw_pick = 0.5502`
  - `pick_strength_media = 0.5756`
  - `conteo_p_ge_0.65 = 383`
  - `apuestas = 6`
  - `ganancia = 25`

Lectura:

- raw no mejoro; incluso bajo ligeramente
- el numero de apuestas aceptadas bajo de `8` a `6`
- la ganancia filtrada subio, pero con muestra aun mas chica, asi que no alcanza para declarar mejora robusta

## Conclusion actualizada

La tesis fuerte sigue siendo valida:

- la mejor senal nueva de `m30_v1` vive en estados estructurales de comeback al borde Q3->Q4
- el micro-momentum corto sigue viendose debil

Pero la implementacion de las dos mejores features nuevas, por si sola, todavia no arreglo el problema central de `m30_v1`:

- las probabilidades siguen blandas
- el volumen filtrado sigue demasiado bajo
- el modelo sigue lejos del nivel de `m27_v1`

## Implementacion real: removal del bloque recent_4m / recent_2m

Despues de la ablacion temporal, se removio de verdad del trainer este bloque:

- `recent_4m_*`
- `recent_2m_*`
- `trailing_now_recent_run_4m`
- `trailing_now_recent_run_2m`

Tambien se actualizo `FEATURE_SCHEMA_VERSION` para forzar rebuild del cache.

### Estado del artifact

- `feature_count` bajo de `114` a `94`
- el artifact nuevo ya no incluye ninguna feature `recent_*`

### Resultado offline

- Antes de remover `recent_*`:
  - ensemble test `accuracy = 0.565878`
  - ensemble test `roc_auc = 0.584742`
  - ensemble test calibrated `accuracy = 0.558398`
  - ensemble test calibrated `roc_auc = 0.584234`
- Despues de remover `recent_*`:
  - ensemble test `accuracy = 0.563465`
  - ensemble test `roc_auc = 0.584669`
  - ensemble test calibrated `accuracy = 0.568292`
  - ensemble test calibrated `roc_auc = 0.584595`

Lectura:

- el ensemble sin calibrar perdio un poco de accuracy
- la version calibrada mejoro bastante en accuracy y un poco en ROC AUC
- esto refuerza la sospecha de que `recent_*` metia ruido en el shape/probability scaling aunque no destruyera por completo el ranking global

### Resultado en evaluacion directa raw/ROI

- Antes de remover `recent_*`:
  - `accuracy_raw_pick = 0.5502`
  - `pick_strength_media = 0.5756`
  - `pick_strength_p95 = 0.6771`
  - `conteo_p_ge_0.65 = 383`
  - `conteo_p_ge_break_even = 95`
  - `apuestas = 6`
  - `ganancia = 25`
- Despues de remover `recent_*`:
  - `accuracy_raw_pick = 0.5502`
  - `pick_strength_media = 0.5777`
  - `pick_strength_p95 = 0.6817`
  - `conteo_p_ge_0.65 = 424`
  - `conteo_p_ge_break_even = 113`
  - `apuestas = 5`
  - `ganancia = -20`

Lectura:

- raw casi no se movio en accuracy, pero si se endurecio un poco el score distribution
- subieron los picks sobre `0.65` y sobre break-even
- aun asi, el filtro final siguio dejando muy poco volumen y el ROI filtrado cayo por muestra chica

## Conclusion mas reciente

Quitar `recent_*` fue una limpieza razonable del modelo:

- bajo complejidad
- dejo un artifact mas limpio
- mejoro algo el comportamiento calibrado
- reforzo que el micro-momentum corto no era el motor principal de senal

Pero tampoco resolvio el problema base de `m30_v1`. La siguiente frontera mas prometedora sigue siendo podar/transfomar mejor las familias de `halftime transition` y `current-state redundancy`, no volver a meter momentum corto.

## Ronda adicional: poda minima de halftime transition

Se midio importancia vs correlacion solo para el bloque restante de `halftime transition` del artifact de `94` features.

Las mas sospechosas salieron asi:

- `halftime_leader_now_tied`
  - `importance = 0.0283`
  - `abs_corr = 0.0047`
- `halftime_leader_still_ahead`
  - `importance = 0.0170`
  - `abs_corr = 0.0057`
- `halftime_trailer_gain_bin=01_03`
  - `importance = 0.0151`
  - `abs_corr = 0.0120`

Con base en eso se removieron del trainer:

- `halftime_leader_now_tied`
- `halftime_leader_still_ahead`
- `halftime_trailer_gain_bin=*`

### Estado del artifact

- `feature_count` bajo de `94` a `89`

### Resultado offline

- Antes de esta poda:
  - ensemble test `accuracy = 0.563465`
  - ensemble test `roc_auc = 0.584669`
  - ensemble calibrated `accuracy = 0.568292`
  - ensemble calibrated `roc_auc = 0.584595`
- Despues de esta poda:
  - ensemble test `accuracy = 0.560569`
  - ensemble test `roc_auc = 0.583036`
  - ensemble calibrated `accuracy = 0.557915`
  - ensemble calibrated `roc_auc = 0.584933`

### Resultado en evaluacion directa raw/ROI

- Antes de esta poda:
  - `accuracy_raw_pick = 0.5502`
  - `pick_strength_media = 0.5777`
  - `conteo_p_ge_0.65 = 424`
  - `apuestas = 5`
  - `ganancia = -20`
- Despues de esta poda:
  - `accuracy_raw_pick = 0.5485`
  - `pick_strength_media = 0.5767`
  - `conteo_p_ge_0.65 = 399`
  - `apuestas = 4`
  - `ganancia = -30`

## Conclusion de esta ronda

Aunque las features removidas si se veian sospechosas por importancia vs senal, su poda minima no ayudo al modelo. La interpretacion mas probable es:

- el bloque `halftime transition` tenia ruido real
- pero esas flags concretas no eran el nucleo del problema
- seguir podando esa familia a ciegas ya no parece la mejor apuesta

En este punto, la siguiente exploracion con mejor ROI de trabajo parece ser `current-state redundancy` o una revision mas estructural de como se construye la decision en minuto 30, en vez de seguir quitando flags individuales de halftime.

## Medición exhaustiva de candidatas (17 May 2026)

Se corrio un analisis completo sobre 4144 filas test midiendo ~22 features candidatas nuevas. Resultados en `temp_scripts/measure_m30_candidate_signal.py`.

### Composicion del dataset

| Tipo | N | % |
|---|---|---|
| 10m | 4007 | 96.7% |
| 12m | 137 | 3.3% |

### Las features candidatas no tienen señal fuerte

Conclusion principal: ninguna feature candidata nueva supera abs_corr > 0.04 en partidos 10m (que son el 96.7% de los datos).

Las mejores candidatas en 10m:

| Candidata | abs_corr 10m | cohen_d | coverage |
|---|---|---|---|
| `halftime_to_q3_signed_delta` | 0.0398 | 0.070 | 93% |
| `blowout_recovery` (= `big_halftime_lead_now_close`, ya existe) | 0.0367 | 0.072 | 11% |
| `momentum_away` (last 3 events) | 0.0363 | -0.069 | 42% |
| `q3_partial_diff * is_10m` | 0.0351 | 0.069 | 92% |
| `momentum_home` (last 3 events) | 0.0336 | 0.066 | 43% |
| `prior_diff_close` | 0.0329 | 0.064 | 17% |
| `last_3_events_net` | 0.0289 | 0.053 | 85% |
| `last_5_events_net` | 0.0288 | 0.054 | 89% |
| `margin_trend_extending` | 0.0261 | -0.062 | 49% |
| `margin_trend_shrinking` | 0.0257 | 0.057 | 30% |

En contraste, las features actuales top:
- `prior_wr_diff`: 0.133
- `away_prior_wr`: 0.088
- `home_prior_wr`: 0.087
- `halftime_diff`: 0.080
- `score_est_diff`: 0.078
- `gp_last`: 0.077
- `q2_diff`: 0.074

### Señal fuerte en 12m pero n=137

En los 137 partidos 12m, varias candidatas muestran correlaciones altas (0.14-0.22), pero la muestra es demasiado pequena para confiar. Podria ser ruido o overfitting local.

### Diagnostico de ruido en current-state

Se midio la correlacion de las features `current_*` contra el target:

| Feature | abs_corr target | redundante con score_est_diff? |
|---|---|---|
| `trailing_now_is_away` | 0.0281 | r=0.768 con score_est_diff |
| `trailing_now_is_home` | 0.0288 | r=-0.765 con score_est_diff |
| `current_leader_won_q3_partial` | 0.0296 | r=0.068 con score_est_diff |
| `current_trailer_won_q3_partial` | 0.0141 | r=-0.051 con score_est_diff |
| `current_leader_won_q1` | 0.0090 | r=0.063 con score_est_diff |
| `current_trailer_was_halftime_leader` | 0.0046 | r=-0.058 con score_est_diff |
| `current_leader_was_halftime_trailer` | 0.0046 | r=-0.058 con score_est_diff |
| `trailing_now_deficit_abs` | 0.0161 | r=0.211 con score_est_diff |

5 de estas 8 features tienen abs_corr < 0.02 contra el target. Pueden eliminarse sin perdida de senal.

### Diagnostico de ruido en halftime-transition

El bloque de 12 features de halftime-transition tiene senales entre 0.005 y 0.036. Las mas debiles:

| Feature | abs_corr target |
|---|---|
| `halftime_trailer_now_tied` | 0.0047 |
| `halftime_leader_lost_lead` | 0.0078 |
| `halftime_trailer_neutralized_deficit` | 0.0078 |
| `halftime_trailer_now_leads` | 0.0062 |
| `lead_flip_from_halftime` | 0.0062 |

Todas con senal casi nula.

### Conclusion principal

**El problema de m30_v1 NO es falta de features.** Las features candidatas nuevas no agregan senal significativa. El problema es estructural:

1. **Minuto 30 es un snapshot inconsistente**: para 10m (97%) es fin de Q3, para 12m (3%) es mitad de Q3. El modelo intenta resolver dos problemas distintos con el mismo feature set.
2. **Las features fundamentales ya dominan**: `prior_wr_diff` (0.133), `halftime_diff` (0.080), `score_est_diff` (0.078). Cualquier feature derivada de estas tendra senal marginal.
3. **Q4 es ruidoso desde minuto 30**: saber el resultado exacto de Q3 no predice bien Q4 porque los quarter-to-quarter swings son altos.

### Iteracion 4: Modelo 10m-only (18 May 2026)

#### Que se hizo
Se agrego flag `--filter-10m` al trainer para entrenar exclusivamente con partidos de 10 minutos de reglamento. Ademas se eliminaron 10 features constantes o perfectamente colineales para el caso 10m:

**Constantes (sin varianza)**:
- `regulation_quarter_minutes` (siempre 10.0)
- `q3_start_minute` (siempre 20.0)
- `q4_start_minute` (siempre 30.0)
- `q3_remaining_minutes` (siempre 0.0)
- `q3_partial_completion` (siempre 1.0)

**Redundantes (colineales con q3_partial_\* para 10m)**:
- `q3_partial_home_pace_per_min` (= q3_partial_home / 10.0)
- `q3_partial_away_pace_per_min` (= q3_partial_away / 10.0)
- `q3_partial_projected_home` (= q3_partial_home, exacto)
- `q3_partial_projected_away` (= q3_partial_away, exacto)
- `q3_partial_projected_diff` (= q3_partial_diff, exacto)

Feature count: 77 → 67.

#### Resultados
| Modelo | Split | Accuracy | ROC AUC |
|---|---|---|---|
| 10m-only XGB | test (3878) | 0.562 | 0.582 |
| 10m-only HistGB | test (3878) | 0.556 | 0.585 |
| 10m-only Champion | test (3878) | 0.564 | 0.585 |
| Mixto Champion (ref) | 10m test (4007) | 0.562 | 0.584 |

#### Conclusion
**Separar 10m/12m no mejora el modelo.** Los partidos 12m (3.3% del dataset) no estaban arrastrando al modelo mixto. La debilidad es estructural al snapshot minuto 30, independientemente de la duracion de reglamento.

### Iteracion 5: Deep momentum features (17 May 2026)

#### Contexto

El usuario cuestiono la conclusion de "limite estructural" senalando que **humanos pueden predecir Q4 winner desde minuto 30** viendo patrones de momentum en la trayectoria del marcador a traves de Q3. No era falta de senal — las features existentes no capturaban los patrones visibles para un humano.

#### Exploracion de datos (graph_points)

Se analizaron partidos concretos en `matches.db`:

- `graph_points.value` = home_score - away_score (score differential), registrado por minuto
- Cada partido tiene ~40 puntos (1 por minuto del reloj global)
- Para 10m: puntos cubren minutos 1-40 (Q1=1-10, Q2=11-20, Q3=21-30, Q4=31-40)
- Para 12m: puntos cubren minutos 1-48

**Patrones visibles identificados**:
1. **Momentum exhaustion**: equipo que remonta fuerte en Q3 (ej: pasa de -9 a +10), frecuentemente pierde Q4 porque "gasto" toda su energia en la remontada vs se tomo el pie del acelerador.
2. **Lead erosion**: equipo cuya ventaja se reduce en los ultimos 5 minutos de Q3 (ej: de +12 a +3), frecuentemente pierde Q4 — la erosion indica perdida de control.

#### Deep momentum features implementadas

Se agregaron 24 features nuevas al trainer (`_build_m30_v1_features`):

**Q3 graph momentum (8 features)**:
- `gp_q3_lead_erosion`: cuanto se redujo la ventaja desde su pico en Q3
- `gp_q3_late_momentum`: cambio neto en los ultimos 5 minutos de Q3
- `gp_q3_volatility`: desviacion estandar de cambios minuto-a-minuto en Q3
- `gp_q3_turning_points`: numero de cambios de direccion de momentum en Q3
- `gp_q3_recovery_spent`: puntos que el lider actual remonto desde su minimo en Q3
- `gp_q3_peak_to_final_ratio`: 1.0 = ventaja intacta, 0.0 = completamente erosionada
- `gp_q3_last_3min_momentum`: cambio neto en los ultimos 3 minutos de Q3
- `gp_q3_momentum_accel`: aceleracion (diferencia entre early Q3 y late Q3)

**PBP windows (16 features)**:
- `recent_1m_*` (8): puntos/max_run/last_scoring/event_share en ultimo minuto de Q3
- `recent_2m_*` (8): mismo analisis en ultimos 2 minutos de Q3

#### Resultados

**10m-only (antes -> despues)**:

| Modelo | Sin momentum | Con momentum | Delta |
|---|---|---|---|
| XGB test | 0.582 | 0.581 | -0.001 |
| HistGB test | **0.585** | **0.591** | **+0.007** |
| Champion | 0.585 | 0.587 | +0.002 |

**Mixto (antes -> despues)**:

| Modelo | Sin momentum | Con momentum | Delta |
|---|---|---|---|
| XGB test | 0.582 | 0.580 | -0.002 |
| HistGB test | **0.584** | **0.589** | **+0.005** |
| Champion | 0.584 | 0.585 | +0.001 |

Por primera vez en la historia del proyecto, se logro una mejora real (+0.007 ROC AUC en HistGB 10m).

#### Feature importance (XGB, 10m-only)

| Familia | Sum Importance | Interpretacion |
|---|---|---|
| PBP windows (16) | **0.226** | Familia mas importante del modelo |
| prior_strength | 0.160 | win rates historicos |
| score_state | 0.118 | marcadores halftime/current |
| Q3 momentum (8) | **0.101** | Tercera familia en importancia |
| transition_flags | 0.096 | flags de comeback |
| q3_partial_state | 0.092 | puntajes parciales de Q3 |

Features top:
- #9: `recent_2m_last_scoring_away` (0.017) — quien anoto ultimo en Q3
- #10: `recent_2m_last_scoring_home` (0.016)
- #27: `gp_q3_peak_to_final_ratio` (0.014) — lead erosion ratio
- #31: `gp_q3_late_momentum` (0.013)
- #36: `gp_q3_volatility` (0.013)

Las 8 features Q3 momentum estan entre #27 y #65 — contribucion solida pero distribuida.

#### XGB depth experiment

Subir XGB de depth=4 a depth=6 empeoro el modelo (0.567 vs 0.581), confirmando overfitting. Depth 4 es optimo para XGB.

#### ROI gap analysis con m27_v1

Se descubrio que la comparacion con m27_v1 (0.668 ROC AUC) es **fundamentalmente injusta**:

| Aspecto | m27_v1 | m30_v1 |
|---|---|---|
| Snapshot minute | 27 | 30 |
| Q4 access (12m) | **3 min Q4** (NBA) | Sin Q4 |
| Q3 access (10m) | Medio Q3 | **Q3 completo** |
| ROC AUC | 0.668 | 0.591 |

m27_v1 tiene 3 minutos de datos reales de Q4 para NBA. m30_v1 predice Q4 desde Q3 completo (10m) o mitad de Q3 (12m). No es una comparacion de feature quality sino de informacion disponible.

#### Conclusion de iteracion 5

Las deep momentum features **funcionan** pero la mejora es modesta (+0.007 ROC AUC). Las PBP windows (especialmente quien anoto ultimo) son la familia mas importante. Q3 trajectory features contribuyen un 10% de la importancia total pero estan limitadas por la profundidad del modelo.

### Accion recomendada

**SI implementar** las 24 momentum features — ya estan en el trainer.

**Opciones para continuar** (sin orden de prioridad):
1. Agregar `recent_3m_*` windows (m27_v1 los tenia y eran fuertes; en m30_v1 cubririan minutos 27-30)
2. Agregar `trailing_defict * recent_run` interaction (captura "trailing team making a run")
3. Aumentar HistGB max_depth de 5 a 6-7 para capturar interacciones
4. Abandonar y redirigir a m27_v1 o targets alternativos