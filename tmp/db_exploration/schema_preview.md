
### Tabla `matches` (48,623 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `match_id` | `TEXT` | Sí | Sí | `15757331` |
| `home_team` | `TEXT` | No | No | `Putinų Grizliai` |
| `away_team` | `TEXT` | No | No | `Putinų Pelkės` |
| `date` | `TEXT` | No | No | `2026-03-20` |
| `time` | `TEXT` | No | No | `13:15` |
| `venue` | `TEXT` | Sí | No | `Putinų gimnazijos sporto salė` |
| `league` | `TEXT` | Sí | No | `Putinų Gimnazijos Krepšinio Lyga, Atk...` |
| `home_record` | `TEXT` | Sí | No | `` |
| `away_record` | `TEXT` | Sí | No | `` |
| `home_score` | `INTEGER` | Sí | No | `62` |
| `away_score` | `INTEGER` | Sí | No | `87` |
| `home_slug` | `TEXT` | Sí | No | `None` |
| `away_slug` | `TEXT` | Sí | No | `None` |
| `event_slug` | `TEXT` | Sí | No | `None` |
| `custom_id` | `TEXT` | Sí | No | `None` |
| `status_type` | `TEXT` | Sí | No | `None` |
| `status_description` | `TEXT` | Sí | No | `None` |
| `home_team_id` | `INTEGER` | Sí | No | `None` |
| `away_team_id` | `INTEGER` | Sí | No | `None` |
| `home_rating` | `REAL` | Sí | No | `None` |
| `away_rating` | `REAL` | Sí | No | `None` |
| `details_checked_at` | `TEXT` | Sí | No | `None` |

### Tabla `quarter_scores` (188,689 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `match_id` | `TEXT` | No | Sí | `15757331` |
| `quarter` | `TEXT` | No | Sí | `Q1` |
| `home` | `INTEGER` | Sí | No | `16` |
| `away` | `INTEGER` | Sí | No | `22` |

### Tabla `quarter_scores_v2` (1,475 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `match_id` | `TEXT` | Sí | Sí | `16213184` |
| `q1_home` | `INTEGER` | Sí | No | `26` |
| `q1_away` | `INTEGER` | Sí | No | `21` |
| `q2_home` | `INTEGER` | Sí | No | `17` |
| `q2_away` | `INTEGER` | Sí | No | `28` |
| `q3_home` | `INTEGER` | Sí | No | `17` |
| `q3_away` | `INTEGER` | Sí | No | `18` |
| `q4_home` | `INTEGER` | Sí | No | `19` |
| `q4_away` | `INTEGER` | Sí | No | `12` |
| `ot_home` | `INTEGER` | Sí | No | `None` |
| `ot_away` | `INTEGER` | Sí | No | `None` |

### Tabla `play_by_play` (3,709,807 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `2609` |
| `match_id` | `TEXT` | No | No | `15757331` |
| `quarter` | `TEXT` | No | No | `Q4` |
| `seq` | `INTEGER` | No | No | `0` |
| `time` | `TEXT` | Sí | No | `-1:00` |
| `player` | `TEXT` | Sí | No | `` |
| `points` | `INTEGER` | Sí | No | `2` |
| `team` | `TEXT` | Sí | No | `home` |
| `home_score` | `INTEGER` | Sí | No | `62` |
| `away_score` | `INTEGER` | Sí | No | `87` |

### Tabla `match_events` (3,897,995 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `1823` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `quarter` | `TEXT` | No | No | `Q4` |
| `seq` | `INTEGER` | No | No | `0` |
| `time` | `TEXT` | Sí | No | `0:00` |
| `time_seconds` | `INTEGER` | Sí | No | `2400` |
| `incident_type` | `TEXT` | No | No | `period` |
| `subtype` | `TEXT` | Sí | No | `` |
| `player` | `TEXT` | Sí | No | `` |
| `player_id` | `TEXT` | Sí | No | `None` |
| `team` | `TEXT` | Sí | No | `home` |
| `points` | `INTEGER` | Sí | No | `None` |
| `home_score` | `INTEGER` | Sí | No | `97` |
| `away_score` | `INTEGER` | Sí | No | `102` |

### Tabla `graph_points` (1,719,271 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `match_id` | `TEXT` | No | Sí | `15742497` |
| `seq` | `INTEGER` | No | Sí | `0` |
| `minute` | `INTEGER` | No | No | `1` |
| `value` | `INTEGER` | No | No | `-1` |

### Tabla `match_h2h` (69,070 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `32` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `h2h_match_id` | `TEXT` | No | No | `14491673` |
| `date` | `TEXT` | No | No | `2026-05-17` |
| `timestamp` | `INTEGER` | Sí | No | `1779040800` |
| `home_team` | `TEXT` | Sí | No | `La Laguna Tenerife` |
| `away_team` | `TEXT` | Sí | No | `Barça Basket` |
| `home_score` | `INTEGER` | Sí | No | `97` |
| `away_score` | `INTEGER` | Sí | No | `102` |
| `q1_home` | `INTEGER` | Sí | No | `16` |
| `q1_away` | `INTEGER` | Sí | No | `29` |
| `q2_home` | `INTEGER` | Sí | No | `27` |
| `q2_away` | `INTEGER` | Sí | No | `29` |
| `q3_home` | `INTEGER` | Sí | No | `28` |
| `q3_away` | `INTEGER` | Sí | No | `17` |
| `q4_home` | `INTEGER` | Sí | No | `26` |
| `q4_away` | `INTEGER` | Sí | No | `27` |
| `tournament` | `TEXT` | Sí | No | `Liga ACB` |

### Tabla `team_statistics` (1,204,175 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `1075` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `period` | `TEXT` | No | No | `Q1` |
| `group_name` | `TEXT` | Sí | No | `Scoring` |
| `stat_key` | `TEXT` | No | No | `points` |
| `stat_name` | `TEXT` | Sí | No | `Points` |
| `home_value` | `REAL` | Sí | No | `16.0` |
| `away_value` | `REAL` | Sí | No | `32.0` |
| `home_total` | `REAL` | Sí | No | `16.0` |
| `away_total` | `REAL` | Sí | No | `32.0` |
| `home_display` | `TEXT` | Sí | No | `16` |
| `away_display` | `TEXT` | Sí | No | `32` |

### Tabla `player_stats` (952,182 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `568` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `team` | `TEXT` | No | No | `home` |
| `player_name` | `TEXT` | No | No | `Wesley Van Beck` |
| `player_id` | `TEXT` | Sí | No | `1549847` |
| `sofascore_rating` | `REAL` | Sí | No | `6.5` |
| `minutes_played` | `INTEGER` | Sí | No | `15.2` |
| `points` | `INTEGER` | Sí | No | `6` |
| `fouls` | `INTEGER` | Sí | No | `3` |
| `plus_minus` | `INTEGER` | Sí | No | `8` |
| `field_goals_made` | `INTEGER` | Sí | No | `2` |
| `field_goals_attempted` | `INTEGER` | Sí | No | `5` |
| `three_made` | `INTEGER` | Sí | No | `0` |
| `three_attempted` | `INTEGER` | Sí | No | `3` |
| `free_throws_made` | `INTEGER` | Sí | No | `2` |
| `free_throws_attempted` | `INTEGER` | Sí | No | `2` |
| `rebounds` | `INTEGER` | Sí | No | `1` |
| `assists` | `INTEGER` | Sí | No | `0` |
| `steals` | `INTEGER` | Sí | No | `0` |
| `turnovers` | `INTEGER` | Sí | No | `0` |
| `blocks` | `INTEGER` | Sí | No | `1` |

### Tabla `lineups` (952,182 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `568` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `team` | `TEXT` | No | No | `home` |
| `player_name` | `TEXT` | No | No | `Wesley Van Beck` |
| `player_id` | `TEXT` | Sí | No | `1549847` |
| `is_starter` | `INTEGER` | No | No | `0` |
| `shirt_number` | `INTEGER` | Sí | No | `4` |
| `position` | `TEXT` | Sí | No | `G` |

### Tabla `match_odds` (64,034 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `1` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `odds_type` | `TEXT` | No | No | `Home/Away` |
| `market_name` | `TEXT` | Sí | No | `Full time` |
| `market_period` | `TEXT` | Sí | No | `Match` |
| `home_value` | `REAL` | Sí | No | `2.35` |
| `away_value` | `REAL` | Sí | No | `1.51` |
| `draw_value` | `REAL` | Sí | No | `None` |
| `timestamp` | `TEXT` | Sí | No | `` |

### Tabla `team_strength` (33,766 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `49` |
| `team_id` | `INTEGER` | No | No | `78043` |
| `team_name` | `TEXT` | No | No | `La Laguna Tenerife` |
| `match_id` | `TEXT` | No | No | `14491673` |
| `position` | `INTEGER` | Sí | No | `8` |
| `wins` | `INTEGER` | Sí | No | `17` |
| `losses` | `INTEGER` | Sí | No | `14` |
| `form` | `TEXT` | Sí | No | `["W", "L", "L", "L", "L"]` |
| `perf_points` | `TEXT` | Sí | No | `{"14491607": 1.28, "15980519": 1.4, "...` |
| `fetched_at` | `TEXT` | No | No | `2026-05-18 23:10:55` |

### Tabla `eval_match_results` (11,125 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `event_date` | `TEXT` | No | Sí | `2026-02-23` |
| `match_id` | `TEXT` | No | Sí | `15548171` |
| `home_team` | `TEXT` | Sí | No | `UGMK Ekaterinburg` |
| `away_team` | `TEXT` | Sí | No | `Nadezhda Orenburg Oblast` |
| `q3_home_score` | `INTEGER` | Sí | No | `17` |
| `q3_away_score` | `INTEGER` | Sí | No | `12` |
| `q3_winner` | `TEXT` | Sí | No | `home` |
| `q4_home_score` | `INTEGER` | Sí | No | `20` |
| `q4_away_score` | `INTEGER` | Sí | No | `21` |
| `q4_winner` | `TEXT` | Sí | No | `away` |
| `created_at` | `TEXT` | No | No | `2026-03-29 21:27:06` |
| `updated_at` | `TEXT` | No | No | `2026-03-29 22:25:09` |
| `q3_pick__hybrid_f1_v1` | `TEXT` | Sí | No | `None` |
| `q3_signal__hybrid_f1_v1` | `TEXT` | Sí | No | `None` |
| `q3_outcome__hybrid_f1_v1` | `TEXT` | Sí | No | `None` |
| `q3_available__hybrid_f1_v1` | `INTEGER` | Sí | No | `None` |
| `q4_pick__hybrid_f1_v1` | `TEXT` | Sí | No | `None` |
| `q4_signal__hybrid_f1_v1` | `TEXT` | Sí | No | `None` |
| `q4_outcome__hybrid_f1_v1` | `TEXT` | Sí | No | `None` |
| `q4_available__hybrid_f1_v1` | `INTEGER` | Sí | No | `None` |
| `q3_pick__hybrid_f1_v2` | `TEXT` | Sí | No | `None` |
| `q3_signal__hybrid_f1_v2` | `TEXT` | Sí | No | `None` |
| `q3_outcome__hybrid_f1_v2` | `TEXT` | Sí | No | `None` |
| `q3_available__hybrid_f1_v2` | `INTEGER` | Sí | No | `None` |
| `q4_pick__hybrid_f1_v2` | `TEXT` | Sí | No | `None` |
| `q4_signal__hybrid_f1_v2` | `TEXT` | Sí | No | `None` |
| `q4_outcome__hybrid_f1_v2` | `TEXT` | Sí | No | `None` |
| `q4_available__hybrid_f1_v2` | `INTEGER` | Sí | No | `None` |
| `q3_pick__hybrid_f1_v3` | `TEXT` | Sí | No | `None` |
| `q3_signal__hybrid_f1_v3` | `TEXT` | Sí | No | `None` |
| `q3_outcome__hybrid_f1_v3` | `TEXT` | Sí | No | `None` |
| `q3_available__hybrid_f1_v3` | `INTEGER` | Sí | No | `None` |
| `q4_pick__hybrid_f1_v3` | `TEXT` | Sí | No | `None` |
| `q4_signal__hybrid_f1_v3` | `TEXT` | Sí | No | `None` |
| `q4_outcome__hybrid_f1_v3` | `TEXT` | Sí | No | `None` |
| `q4_available__hybrid_f1_v3` | `INTEGER` | Sí | No | `None` |
| `q3_pick__auto_f1` | `TEXT` | Sí | No | `None` |
| `q3_signal__auto_f1` | `TEXT` | Sí | No | `None` |
| `q3_outcome__auto_f1` | `TEXT` | Sí | No | `None` |
| `q3_available__auto_f1` | `INTEGER` | Sí | No | `None` |
| `q4_pick__auto_f1` | `TEXT` | Sí | No | `None` |
| `q4_signal__auto_f1` | `TEXT` | Sí | No | `None` |
| `q4_outcome__auto_f1` | `TEXT` | Sí | No | `None` |
| `q4_available__auto_f1` | `INTEGER` | Sí | No | `None` |
| `q3_pick__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q3_signal__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q3_outcome__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q3_available__bot_hybrid_f1` | `INTEGER` | Sí | No | `None` |
| `q4_pick__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q4_signal__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q4_outcome__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q4_available__bot_hybrid_f1` | `INTEGER` | Sí | No | `None` |
| `q3_confidence__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_threshold_lean__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_threshold_bet__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_confidence__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_threshold_lean__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_threshold_bet__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_pick__v4` | `TEXT` | Sí | No | `home` |
| `q3_signal__v4` | `TEXT` | Sí | No | `LEAN` |
| `q3_outcome__v4` | `TEXT` | Sí | No | `hit` |
| `q3_available__v4` | `INTEGER` | Sí | No | `1` |
| `q3_confidence__v4` | `REAL` | Sí | No | `0.5843096606850623` |
| `q3_threshold_lean__v4` | `REAL` | Sí | No | `0.55` |
| `q3_threshold_bet__v4` | `REAL` | Sí | No | `0.65` |
| `q4_pick__v4` | `TEXT` | Sí | No | `away` |
| `q4_signal__v4` | `TEXT` | Sí | No | `LEAN` |
| `q4_outcome__v4` | `TEXT` | Sí | No | `hit` |
| `q4_available__v4` | `INTEGER` | Sí | No | `1` |
| `q4_confidence__v4` | `REAL` | Sí | No | `0.5304235890734172` |
| `q4_threshold_lean__v4` | `REAL` | Sí | No | `0.55` |
| `q4_threshold_bet__v4` | `REAL` | Sí | No | `0.65` |
| `q3_pick__v5` | `TEXT` | Sí | No | `home` |
| `q3_signal__v5` | `TEXT` | Sí | No | `LEAN` |
| `q3_outcome__v5` | `TEXT` | Sí | No | `hit` |
| `q3_available__v5` | `INTEGER` | Sí | No | `1` |
| `q3_confidence__v5` | `REAL` | Sí | No | `0.5161301064703295` |
| `q3_threshold_lean__v5` | `REAL` | Sí | No | `0.55` |
| `q3_threshold_bet__v5` | `REAL` | Sí | No | `0.65` |
| `q4_pick__v5` | `TEXT` | Sí | No | `away` |
| `q4_signal__v5` | `TEXT` | Sí | No | `BET` |
| `q4_outcome__v5` | `TEXT` | Sí | No | `hit` |
| `q4_available__v5` | `INTEGER` | Sí | No | `1` |
| `q4_confidence__v5` | `REAL` | Sí | No | `0.7455893599076839` |
| `q4_threshold_lean__v5` | `REAL` | Sí | No | `0.55` |
| `q4_threshold_bet__v5` | `REAL` | Sí | No | `0.65` |
| `q3_pick__v6` | `TEXT` | Sí | No | `home` |
| `q3_signal__v6` | `TEXT` | Sí | No | `BET` |
| `q3_outcome__v6` | `TEXT` | Sí | No | `hit` |
| `q3_available__v6` | `INTEGER` | Sí | No | `1` |
| `q3_confidence__v6` | `REAL` | Sí | No | `0.6721261153926789` |
| `q3_threshold_lean__v6` | `REAL` | Sí | No | `0.55` |
| `q3_threshold_bet__v6` | `REAL` | Sí | No | `0.65` |
| `q4_pick__v6` | `TEXT` | Sí | No | `away` |
| `q4_signal__v6` | `TEXT` | Sí | No | `BET` |
| `q4_outcome__v6` | `TEXT` | Sí | No | `hit` |
| `q4_available__v6` | `INTEGER` | Sí | No | `1` |
| `q4_confidence__v6` | `REAL` | Sí | No | `0.7694445882840482` |
| `q4_threshold_lean__v6` | `REAL` | Sí | No | `0.55` |
| `q4_threshold_bet__v6` | `REAL` | Sí | No | `0.65` |
| `q3_pick__v7` | `TEXT` | Sí | No | `home` |
| `q3_signal__v7` | `TEXT` | Sí | No | `LEAN` |
| `q3_outcome__v7` | `TEXT` | Sí | No | `hit` |
| `q3_available__v7` | `INTEGER` | Sí | No | `1` |
| `q3_confidence__v7` | `REAL` | Sí | No | `0.5742811203359796` |
| `q3_threshold_lean__v7` | `REAL` | Sí | No | `0.55` |
| `q3_threshold_bet__v7` | `REAL` | Sí | No | `0.65` |
| `q4_pick__v7` | `TEXT` | Sí | No | `away` |
| `q4_signal__v7` | `TEXT` | Sí | No | `BET` |
| `q4_outcome__v7` | `TEXT` | Sí | No | `hit` |
| `q4_available__v7` | `INTEGER` | Sí | No | `1` |
| `q4_confidence__v7` | `REAL` | Sí | No | `0.801549420027609` |
| `q4_threshold_lean__v7` | `REAL` | Sí | No | `0.55` |
| `q4_threshold_bet__v7` | `REAL` | Sí | No | `0.65` |
| `q3_reasoning__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q3_predicted_home__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_predicted_away__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_predicted_total__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_mae__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_mae_home__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q3_mae_away__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_reasoning__bot_hybrid_f1` | `TEXT` | Sí | No | `None` |
| `q4_predicted_home__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_predicted_away__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_predicted_total__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_mae__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_mae_home__bot_hybrid_f1` | `REAL` | Sí | No | `None` |
| `q4_mae_away__bot_hybrid_f1` | `REAL` | Sí | No | `None` |

### Tabla `eval_match_results_v2` (474 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `match_id` | `TEXT` | Sí | Sí | `16225942` |
| `available` | `INTEGER` | Sí | No | `1` |
| `q4_signal__v6_2` | `TEXT` | Sí | No | `BET_HOME` |
| `q4_pick__v6_2` | `TEXT` | Sí | No | `HOME` |
| `q4_confidence__v6_2` | `TEXT` | Sí | No | `0.154887` |
| `q4_outcome__v6_2` | `TEXT` | Sí | No | `loss` |
| `q4_signal__m27_v3` | `TEXT` | Sí | No | `None` |
| `q4_pick__m27_v3` | `TEXT` | Sí | No | `None` |
| `q4_confidence__m27_v3` | `TEXT` | Sí | No | `None` |
| `q4_outcome__m27_v3` | `TEXT` | Sí | No | `None` |

### Tabla `bet_monitor_log_v2` (1,001 filas)
| Columna | Tipo | Nulo | PK | Ejemplo / Descripción |
| :--- | :--- | :---: | :---: | :--- |
| `id` | `INTEGER` | Sí | Sí | `1` |
| `match_id` | `TEXT` | Sí | No | `16225941` |
| `model_version` | `TEXT` | Sí | No | `v6_2` |
| `target_quarter` | `INTEGER` | Sí | No | `4` |
| `inference_minute` | `INTEGER` | Sí | No | `31` |
| `graph_points_count` | `INTEGER` | Sí | No | `26` |
| `raw_json` | `TEXT` | Sí | No | `{"match": {"home_team": "Al Difaa Al ...` |
| `signal_type` | `TEXT` | Sí | No | `BET_HOME` |
| `picked_side` | `TEXT` | Sí | No | `HOME` |
| `confidence` | `REAL` | Sí | No | `0.481309` |
| `actual_home_score` | `INTEGER` | Sí | No | `57` |
| `actual_away_score` | `INTEGER` | Sí | No | `38` |
| `result` | `TEXT` | Sí | No | `win` |
| `created_at` | `TEXT` | Sí | No | `2026-05-24T06:50:38.790047` |
| `inference_json` | `TEXT` | Sí | No | `None` |