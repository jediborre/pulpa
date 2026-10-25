# Justificacion de features de m27_v5_3_1

Correlacion (Pearson) y AUC univariado sobre filas de entrenamiento no-NaN; cobertura en train/test; ganancia XGBoost; ablacion = caida de AUC en C al quitar la feature (HGB, mismo split). Negativo = la feature estorba.

| # | Feature | Grupo | corr(y) | AUC uni | cob. train % | cob. test % | ganancia XGB % | ablacion dAUC | Para que sirve / que arregla |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `minute` | A estado | +0.003 | 0.502 | 100 | 100 | 0.0 | +0.0000 | Minuto real del snapshot (25.5-30). Hace al modelo consciente de cuanta informacion tiene; antes el 'm27' era en realidad m30. |
| 1 | `cur_score_diff` | A estado | +0.088 | 0.536 | 100 | 100 | 1.7 | -0.0002 | Marcador local-visitante al minuto t. Base del estado; en v5_2 se entrenaba con Q1+Q2+Q3 completos y en vivo llegaba parcial. |
| 2 | `abs_cur_score_diff` | A estado | -0.005 | 0.493 | 100 | 100 | 0.8 | -0.0003 | Magnitud de la ventaja; separa blowouts (basura) de juegos cerrados. |
| 3 | `ht_diff` | A estado | +0.089 | 0.541 | 100 | 100 | 0.8 | -0.0002 | Diferencial al descanso (Q1+Q2). |
| 4 | `q1_diff` | A estado | +0.074 | 0.538 | 100 | 100 | 0.5 | -0.0002 | Diferencial Q1. |
| 5 | `q2_diff` | A estado | +0.059 | 0.529 | 100 | 100 | 0.6 | -0.0001 | Diferencial Q2. |
| 6 | `q3_partial_diff` | A estado | +0.038 | 0.517 | 100 | 100 | 1.5 | -0.0005 | Diferencial del Q3 jugado hasta t (cur - ht). Reemplaza q3_diff completo para ser valido a cualquier minuto. |
| 7 | `q3_pace_diff` | A estado | +0.038 | 0.517 | 100 | 100 | 2.1 | -0.0005 | q3_partial_diff reescalado a 10 min: momentum del Q3 comparable entre t=26 y t=30 (reemplaza q3_eff_diff). |
| 8 | `q3_vs_ht_momentum` | A estado | -0.022 | 0.487 | 100 | 100 | 0.9 | -0.0000 | Q3 reescalado menos media del descanso: cambio de inercia 2a mitad vs 1a. |
| 9 | `recent_pts_diff_2m` | A estado | +0.018 | 0.510 | 49 | 96 | 0.6 | +0.0000 | Puntos local-visitante en los ultimos 2 min reales (match_events). En v5_2 era siempre 0 en vivo y en train un proxy q3*0.2. |
| 10 | `recent_pts_diff_5m` | A estado | +0.032 | 0.516 | 49 | 96 | 0.4 | -0.0002 | Igual con ventana de 5 min: tendencia reciente menos ruidosa. |
| 11 | `recent_pace_ratio` | A estado | -0.008 | 0.495 | 49 | 96 | 0.6 | -0.0002 | Puntos totales de los ultimos 4 min / ritmo esperado: detecta ritmo reciente alto/bajo (varianza del Q4). |
| 12 | `pace_pts_per_min` | A estado | +0.004 | 0.502 | 100 | 100 | 0.5 | -0.0001 | Puntos totales por minuto jugado al minuto t. |
| 13 | `pace_vs_league_prior` | A estado | +0.005 | 0.503 | 100 | 100 | 0.5 | -0.0001 | Ritmo menos el esperado de la liga (priors de leagues_classification). |
| 14 | `gp_val` | A estado | +0.085 | 0.535 | 64 | 96 | 0.5 | +0.0005 | Ultimo valor de la grafica de presion de SofaScore hasta t (NaN si no hay grafica, no 0). |
| 15 | `gp_slope_3m` | A estado | +0.021 | 0.511 | 64 | 96 | 0.4 | -0.0000 | Pendiente de la grafica en 3 puntos: momentum reciente segun SofaScore (se habia podado por entrenarse con ceros). |
| 16 | `lead_z` | A estado | +0.090 | 0.536 | 100 | 100 | 4.0 | +0.0000 | cur_diff/sqrt(puntos totales): ventaja normalizada por el volumen de anotacion. |
| 17 | `lead_vs_lg_vol` | A estado | +0.082 | 0.535 | 100 | 100 | 3.0 | +0.0000 | cur_diff / volatilidad Q4 de la liga: cuantas 'desviaciones de Q4' vale la ventaja en esa liga. |
| 18 | `d_elo_ft` | B fuerza | +0.172 | 0.599 | 100 | 100 | 3.7 | -0.0001 | Diferencial ELO de partido (K=20, decaimiento por inactividad), as-of via PriorState. |
| 19 | `d_elo_q4` | B fuerza | +0.131 | 0.573 | 100 | 100 | 1.1 | -0.0002 | Diferencial ELO especifico de Q4 (K=15): equipos que cierran mejor. |
| 20 | `d_wr15` | B fuerza | +0.164 | 0.592 | 100 | 100 | 0.6 | -0.0004 | Win-rate bayesiano ultimos 15 juegos (misma definicion en train y live; v5_2 usaba ELO/180 en vivo). |
| 21 | `d_q4_bayes` | B fuerza | +0.135 | 0.573 | 100 | 100 | 0.5 | -0.0003 | Margen medio Q4 ultimos 15 juegos (bayesiano). v5_2 lo aproximaba con ELO/35 en vivo. |
| 22 | `d_ft_margin15` | B fuerza | +0.192 | 0.605 | 100 | 100 | 11.7 | +0.0000 | NUEVA: margen medio de partido ultimos 15 juegos; mas informativo que el win-rate binario. |
| 23 | `d_q4_net12` | B fuerza | +0.131 | 0.571 | 100 | 100 | 0.7 | -0.0003 | Perfil 'closer' (margen neto Q4 ultimos 12). En v5_2 era siempre 0 en vivo (q_profile inexistente). |
| 24 | `d_q3_net12` | B fuerza | +0.142 | 0.576 | 100 | 100 | 0.6 | -0.0002 | NUEVA: margen neto Q3 historico; permite medir cuanto el Q3 actual excede lo habitual (fatiga/regresion). |
| 25 | `d_season_wr` | B fuerza | +0.169 | 0.595 | 100 | 100 | 1.6 | -0.0000 | NUEVA: record de la temporada as-of (reinicia tras 150 dias) calculado desde matches.db; reemplaza ts_delta_wp_bayes cuyo standings se bajo en oct-2026 para todo el historico (look-ahead). |
| 26 | `h2h_ft_margin` | B fuerza | +0.139 | 0.571 | 100 | 100 | 1.5 | -0.0005 | Margen H2H (ultimos 8, encogido) calculado del historico propio. En v5_2 el live pasaba una lista y el extractor esperaba dict => 0. |
| 27 | `h2h_q4_margin` | B fuerza | +0.090 | 0.545 | 100 | 100 | 0.8 | -0.0002 | Margen H2H en Q4 (ultimos 8, encogido). |
| 28 | `elo_reliability` | B fuerza | +0.008 | 0.505 | 100 | 100 | 1.6 | +0.0003 | min(juegos previos)/12: confianza de los priors; el modelo la usa para descontarlos en equipos nuevos. |
| 29 | `has_ml` | C mercado | -0.000 | 0.500 | 100 | 100 | 1.7 | +0.0000 | 1 si hay cuota ML pre-partido (SofaScore no la publica en ~40% de partidos). Antes se confundia con 0. |
| 30 | `odds_log_ratio` | C mercado | +0.186 | 0.606 | 64 | 60 | 4.0 | +0.0000 | ln(cuota visita/cuota local), ML PRE-PARTIDO (NaN si no hay). El parser ahora descarta mercados in-play que contaminaban el valor. |
| 31 | `mkt_p_home` | C mercado | +0.185 | 0.606 | 64 | 60 | 4.7 | -0.0006 | Probabilidad implicita local sin vig; escala natural para el modelo logistico. |
| 32 | `expected_ft_margin` | C mercado | +0.180 | 0.606 | 100 | 100 | 33.4 | -0.0007 | Margen esperado = 6.5*olr + 0.5*d_elo/28 (con ML) o d_elo/28; misma formula train y live. |
| 33 | `mkt_margin_spread` | C mercado | +nan | nan | 0 | 5 | 0.0 | +0.0000 | NUEVA: -linea de handicap pre-partido (margen esperado del mercado). Solo existe en partidos con backfill de lineas / capturados pre-tip. |
| 34 | `total_line` | C mercado | +nan | nan | 0 | 5 | 0.0 | +0.0000 | NUEVA: linea de total pre-partido (ritmo esperado). Mismo origen que el spread. |
| 35 | `cur_vs_expected` | C mercado | +0.030 | 0.497 | 100 | 100 | 4.0 | +0.0004 | cur_diff - 0.75*expected_ft_margin: ventaja real vs esperada (rubber_band_gap verdadero; en v5_2 live era otra formula). |
| 36 | `lg_avg_ft` | D liga | -0.001 | 0.499 | 100 | 100 | 1.2 | -0.0002 | Puntos totales promedio de la liga (leagues_classification). |
| 37 | `lg_avg_q4` | D liga | -0.002 | 0.498 | 100 | 100 | 1.0 | -0.0005 | Puntos promedio de Q4 en la liga. |
| 38 | `lg_q4_vol` | D liga | -0.009 | 0.494 | 100 | 100 | 0.9 | -0.0000 | NUEVA: volatilidad |margen Q4| de la liga, expanding (as-of). Normaliza ventajas y detecta ligas de varianza alta. |
| 39 | `lg_home_q4_rate` | D liga | +0.012 | 0.507 | 100 | 100 | 0.7 | +0.0000 | NUEVA: tasa as-of de victoria local en Q4 por liga (sesgo local por liga). |
| 40 | `is_women` | D liga | -0.000 | 0.500 | 100 | 100 | 1.1 | -0.0003 | Liga femenina (distinta dinamica y volatilidad). |
| 41 | `is_home_view` | E orientacion | +nan | nan | 100 | 100 | 3.4 |  | 1 en la vista original, 0 en la espejo: permite la aumentacion local<->visitante conservando la ventaja de local. |

## Ablacion por grupo

| Grupo | dAUC al quitarlo |
|---|---|
| A estado | +0.0008 |
| B fuerza | +0.0026 |
| C mercado | +0.0134 |
| D liga | +0.0011 |