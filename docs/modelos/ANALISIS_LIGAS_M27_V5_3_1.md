# 📊 Análisis por liga y por umbral — modelo m27_v5_3_1 (clon)

> Clon de `m27_v5_3` con la corrección del clasificador: `Runda Zasadnicza` (temporada regular polaca) dejó de clasificarse como `SEGUNDA_FASE`. Ligas añadidas al entrenamiento: `Poland 2nd Basketball League, Runda Zasadnicza` (852) y `1 Liga Kobiet, Runda Zasadnicza` (270).

> Holdout **C** (último 18 %, anclas t=30). Umbral operativo: **P ≥ 0.75**. Equilibrio a cuota 1.40: **71.4 %**.

## 1. Resumen ejecutivo

- **Ligas analizadas:** 251 (≥ 30 anclas en holdout C).
- **Partidos (anclas) en holdout C:** 24,563 · en entrenamiento (A): 73,577.
- **Apuestas totales a 0.75 en C:** 2,070.
- **Ligas en arranque en frío:** 108 (43.0 %).
- **Ligas candidatas a bloqueo (IC95 sup < 71.4 % a 0.75):** 0.
- **Ligas con ROI@1.40 > 20 % a 0.75:** 14.

**Comparación con m27_v5_3 (original):** eligible matches 184,709 vs 182,803 (+1,906 por Runda Zasadnicza). AUC test 0.6544 vs 0.6545. Apuestas @0.75 (ancla): 2,771 vs 2,538.

## 2. Tabla global de umbrales (anclas t=30, bloque C)

| P ≥  | Apuestas | WR (IC95)          | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---                | ---:      | ---:      |
| 0.65 | 9,807    | 71.6 % (70.7–72.5) | +0.20 %   | +30.98 %  |
| 0.68 | 7,060    | 73.8 % (72.7–74.8) | +3.29 %   | +35.02 %  |
| 0.70 | 5,510    | 76.0 % (74.8–77.1) | +6.33 %   | +38.99 %  |
| 0.72 | 4,238    | 78.0 % (76.7–79.2) | +9.21 %   | +42.76 %  |
| 0.75 | 2,771    | 81.4 % (79.9–82.8) | +13.93 %  | +48.92 %  |
| 0.80 | 1,326    | 86.3 % (84.3–88.0) | +20.78 %  | +57.88 %  |

## 3. Tabla resumen por liga (ordenada por WR descendente a 0.75)

| País          | Liga                                                        | Entrenado (A) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---           | ---                                                         | ---:          | ---:        | ---:           | ---:    | ---:     | ---:     | ---       | ---:      | ---:      | ---     |
| USA           | CBI U19, Sede Rio                                           | 0             | 37          | 21             | 20      | 1        | 95.2 %   | 77.3–99.2 | +33.33 %  | +74.29 %  | 🆕      |
| Other         | Saudi Basketball League, Premier League                     | 0             | 45          | 20             | 19      | 1        | 95.0 %   | 76.4–99.1 | +33.00 %  | +73.85 %  | 🆕      |
| International | BNXT League                                                 | 222           | 224         | 18             | 17      | 1        | 94.4 %   | 74.2–99.0 | +32.22 %  | +72.83 %  |         |
| USA           | CBI U15 , Classificatória A                                 | 0             | 76          | 27             | 25      | 2        | 92.6 %   | 76.6–97.9 | +29.63 %  | +69.44 %  | 🆕      |
| France        | LNB, Women                                                  | 184           | 60          | 39             | 36      | 3        | 92.3 %   | 79.7–97.3 | +29.23 %  | +68.92 %  |         |
| Other         | ZBL, Women                                                  | 694           | 74          | 36             | 32      | 4        | 88.9 %   | 74.7–95.6 | +24.44 %  | +62.67 %  |         |
| Other         | Prva Liga                                                   | 0             | 135         | 27             | 24      | 3        | 88.9 %   | 71.9–96.1 | +24.44 %  | +62.67 %  | 🆕      |
| Other         | Top Division Women, Regular Season                          | 0             | 88          | 35             | 31      | 4        | 88.6 %   | 74.0–95.5 | +24.00 %  | +62.09 %  | 🆕      |
| Other         | Finland Division 1A Women                                   | 307           | 87          | 17             | 15      | 2        | 88.2 %   | 65.7–96.7 | +23.53 %  | +61.47 %  |         |
| Other         | LDB, Primeira fase                                          | 0             | 128         | 16             | 14      | 2        | 87.5 %   | 64.0–96.5 | +22.50 %  | +60.13 %  | 🆕      |
| Australia     | NBL, East                                                   | 78            | 87          | 31             | 27      | 4        | 87.1 %   | 71.1–94.9 | +21.94 %  | +59.39 %  |         |
| Poland        | MPBL                                                        | 319           | 264         | 31             | 27      | 4        | 87.1 %   | 71.1–94.9 | +21.94 %  | +59.39 %  |         |
| USA           | CBI U17, Sede Curitiba                                      | 0             | 67          | 15             | 13      | 2        | 86.7 %   | 62.1–96.3 | +21.33 %  | +58.60 %  | 🆕      |
| Australia     | NBL1 East, Women                                            | 153           | 118         | 36             | 31      | 5        | 86.1 %   | 71.3–93.9 | +20.56 %  | +57.58 %  |         |
| Other         | Estonian-Latvian Basketball League                          | 610           | 159         | 25             | 21      | 4        | 84.0 %   | 65.3–93.6 | +17.60 %  | +53.72 %  |         |
| Other         | 1 Division, Women                                           | 17            | 89          | 18             | 15      | 3        | 83.3 %   | 60.8–94.2 | +16.67 %  | +52.50 %  |         |
| Other         | 1st Division Women                                          | 0             | 52          | 23             | 19      | 4        | 82.6 %   | 62.9–93.0 | +15.65 %  | +51.17 %  | 🆕      |
| Australia     | NBL Women , Regular Season                                  | 0             | 106         | 17             | 14      | 3        | 82.3 %   | 59.0–93.8 | +15.29 %  | +50.71 %  | 🆕      |
| Other         | Big V, Women                                                | 108           | 82          | 28             | 23      | 5        | 82.1 %   | 64.4–92.1 | +15.00 %  | +50.32 %  |         |
| Other         | FSHB Superliga, K.K.M 2025-26                               | 0             | 84          | 38             | 31      | 7        | 81.6 %   | 66.6–90.8 | +14.21 %  | +49.29 %  | 🆕      |
| Brazil        | Brazil NBB                                                  | 1,444         | 272         | 16             | 13      | 3        | 81.2 %   | 57.0–93.4 | +13.75 %  | +48.69 %  |         |
| USA           | NCAA Women's Division I, Regular season                     | 0             | 1,566       | 261            | 212     | 49       | 81.2 %   | 76.0–85.5 | +13.72 %  | +48.64 %  | 🆕      |
| Other         | Basketligaen                                                | 651           | 108         | 30             | 24      | 6        | 80.0 %   | 62.7–90.5 | +12.00 %  | +46.40 %  |         |
| Other         | Swiss Basketball League Women                               | 147           | 67          | 15             | 12      | 3        | 80.0 %   | 54.8–93.0 | +12.00 %  | +46.40 %  |         |
| France        | LNB, Clausura                                               | 0             | 49          | 15             | 12      | 3        | 80.0 %   | 54.8–93.0 | +12.00 %  | +46.40 %  | 🆕      |
| Poland        | Poland 2nd Basketball League, Runda Zasadnicza              | 0             | 771         | 164            | 131     | 33       | 79.9 %   | 73.1–85.3 | +11.83 %  | +46.18 %  | 🆕      |
| Other         | Tipos Extraliga, Women                                      | 424           | 63          | 40             | 31      | 9        | 77.5 %   | 62.5–87.7 | +8.50 %   | +41.83 %  |         |
| Other         | 2nd League                                                  | 271           | 131         | 22             | 17      | 5        | 77.3 %   | 56.6–89.9 | +8.18 %   | +41.41 %  |         |
| Other         | Hungary Divison A Women                                     | 284           | 107         | 35             | 27      | 8        | 77.1 %   | 61.0–87.9 | +8.00 %   | +41.17 %  |         |
| Other         | 1 Liga Kobiet, Runda Zasadnicza                             | 0             | 239         | 17             | 13      | 4        | 76.5 %   | 52.7–90.4 | +7.06 %   | +39.94 %  | 🆕      |
| Other         | Switzerland LNA                                             | 695           | 91          | 21             | 16      | 5        | 76.2 %   | 54.9–89.4 | +6.67 %   | +39.43 %  |         |
| Other         | United League                                               | 1,143         | 169         | 37             | 28      | 9        | 75.7 %   | 59.9–86.6 | +5.95 %   | +38.49 %  |         |
| USA           | CBI U15 , Classificatória B                                 | 0             | 78          | 20             | 15      | 5        | 75.0 %   | 53.1–88.8 | +5.00 %   | +37.25 %  | 🆕      |
| Other         | Russian Women's Basketball Premier League                   | 726           | 102         | 27             | 20      | 7        | 74.1 %   | 55.3–86.8 | +3.70 %   | +35.56 %  |         |
| Lithuania     | Nostra.lt-RKL Division B, Reguliarusis sezonas              | 0             | 164         | 22             | 16      | 6        | 72.7 %   | 51.8–86.8 | +1.82 %   | +33.09 %  | 🆕      |
| Other         | Swedish Basketball League Women                             | 239           | 112         | 32             | 23      | 9        | 71.9 %   | 54.6–84.4 | +0.62 %   | +31.53 %  |         |
| Australia     | NBL, West                                                   | 75            | 84          | 24             | 17      | 7        | 70.8 %   | 50.8–85.1 | -0.83 %   | +29.62 %  | ⚠️     |
| Other         | SLB, Women                                                  | 0             | 87          | 24             | 17      | 7        | 70.8 %   | 50.8–85.1 | -0.83 %   | +29.62 %  | ⚠️ 🆕  |
| Other         | Premijer ženska liga                                        | 305           | 96          | 32             | 22      | 10       | 68.8 %   | 51.4–82.0 | -3.75 %   | +25.81 %  | ⚠️     |
| Other         | Liga 1 Masculin, Faza II                                    | 21            | 119         | 17             | 11      | 6        | 64.7 %   | 41.3–82.7 | -9.41 %   | +18.41 %  | ⚠️     |
| Other         | Sparkasse Prva Liga Zeni                                    | 0             | 63          | 17             | 9       | 8        | 52.9 %   | 31.0–73.8 | -25.88 %  | -3.12 %   | ⚠️ 🆕  |
| Other         | 1. SKL, Women                                               | 44            | 48          | 14             | 0       | 14       | —        | —         | —         | —         |         |
| Other         | A1 Women                                                    | 821           | 78          | 14             | 0       | 14       | —        | —         | —         | —         |         |
| Other         | Rwanda Basketball League, Regular Season                    | 0             | 65          | 14             | 0       | 14       | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U13, Group D                                           | 0             | 34          | 12             | 0       | 12       | —        | —         | —         | —         | 🆕      |
| Other         | 1. Division B                                               | 100           | 99          | 12             | 0       | 12       | —        | —         | —         | —         |         |
| Other         | EYBL U14, Group E                                           | 0             | 52          | 12             | 0       | 12       | —        | —         | —         | —         | 🆕      |
| Other         | Jedinstvena kadetska liga                                   | 0             | 64          | 12             | 0       | 12       | —        | —         | —         | —         | 🆕      |
| Other         | Liga Nacional Femenina Chile, Fase regular                  | 0             | 34          | 12             | 0       | 12       | —        | —         | —         | —         | 🆕      |
| Other         | Liga Femenina, Women, Clausura                              | 0             | 115         | 11             | 0       | 11       | —        | —         | —         | —         | 🆕      |
| Japan         | B League Premier                                            | 3,136         | 568         | 10             | 0       | 10       | —        | —         | —         | —         |         |
| Japan         | B League One                                                | 3,097         | 369         | 10             | 0       | 10       | —        | —         | —         | —         |         |
| Other         | Basket Liga, Women                                          | 785           | 110         | 10             | 0       | 10       | —        | —         | —         | —         |         |
| Other         | Rwanda Basketball League Women, Regular Season              | 0             | 42          | 10             | 0       | 10       | —        | —         | —         | —         | 🆕      |
| Other         | 1. ZLS, Women                                               | 53            | 59          | 10             | 0       | 10       | —        | —         | —         | —         |         |
| Australia     | NBL1 West, Women                                            | 127           | 95          | 10             | 0       | 10       | —        | —         | —         | —         |         |
| Other         | VBA                                                         | 119           | 57          | 10             | 0       | 10       | —        | —         | —         | —         |         |
| Other         | Estonian National Women's Basketball League                 | 57            | 49          | 9              | 0       | 9        | —        | —         | —         | —         |         |
| Other         | Austrian Women's Basketball Bundesliga                      | 144           | 48          | 9              | 0       | 9        | —        | —         | —         | —         |         |
| Other         | LF Challenge, Regular Season                                | 0             | 145         | 9              | 0       | 9        | —        | —         | —         | —         | 🆕      |
| Other         | Latvian Basketball Cup                                      | 0             | 31          | 9              | 0       | 9        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U15, Group D                                           | 0             | 41          | 9              | 0       | 9        | —        | —         | —         | —         | 🆕      |
| Australia     | NBL Men , Regular Season                                    | 0             | 125         | 9              | 0       | 9        | —        | —         | —         | —         | 🆕      |
| Brazil        | LBF, Women                                                  | 184           | 61          | 9              | 0       | 9        | —        | —         | —         | —         |         |
| Australia     | NBL1 Central, Women                                         | 84            | 58          | 9              | 0       | 9        | —        | —         | —         | —         |         |
| Other         | LMB, Clausura                                               | 120           | 67          | 9              | 0       | 9        | —        | —         | —         | —         |         |
| Other         | Swedish Basketball League                                   | 917           | 124         | 8              | 0       | 8        | —        | —         | —         | —         |         |
| Other         | Liga Femenina, Women                                        | 1,185         | 190         | 8              | 0       | 8        | —        | —         | —         | —         |         |
| Other         | EYBL U17, Group E                                           | 0             | 49          | 8              | 0       | 8        | —        | —         | —         | —         | 🆕      |
| Other         | Indonesian Basketball League                                | 768           | 91          | 8              | 0       | 8        | —        | —         | —         | —         |         |
| Other         | EYBL U17, Group G                                           | 0             | 37          | 8              | 0       | 8        | —        | —         | —         | —         | 🆕      |
| Other         | The Basketball League, Regular Season                       | 63            | 126         | 8              | 0       | 8        | —        | —         | —         | —         |         |
| Australia     | NBL, Playoffs                                               | 71            | 41          | 8              | 0       | 8        | —        | —         | —         | —         |         |
| Other         | Liga e Parë                                                 | 277           | 70          | 7              | 0       | 7        | —        | —         | —         | —         |         |
| Australia     | NBL                                                         | 1,965         | 301         | 7              | 0       | 7        | —        | —         | —         | —         |         |
| France        | France Pro A                                                | 1,740         | 184         | 7              | 0       | 7        | —        | —         | —         | —         |         |
| Other         | Superliga, Women                                            | 0             | 36          | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| Other         | Superliga e Femrave                                         | 0             | 41          | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| Other         | Campeonato Nacional da 1ª Divisao Feminina, 1ª Fase - Sul   | 0             | 50          | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| Other         | Liga 1 Masculin, Faza I                                     | 101           | 47          | 7              | 0       | 7        | —        | —         | —         | —         |         |
| Other         | 1. liga, Women                                              | 0             | 75          | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| Other         | U21 Espoirs Elite                                           | 0             | 142         | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| Other         | Portugal LBP                                                | 772           | 101         | 7              | 0       | 7        | —        | —         | —         | —         |         |
| USA           | NCAA Women's March Madness                                  | 0             | 59          | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| USA           | CBI U19, Sede Curitiba                                      | 0             | 44          | 7              | 0       | 7        | —        | —         | —         | —         | 🆕      |
| Spain         | Liga ACB                                                    | 1,770         | 250         | 6              | 0       | 6        | —        | —         | —         | —         |         |
| Other         | Campeonato Nacional da 1ª Divisao Feminina, 1ª Fase - Norte | 0             | 35          | 6              | 0       | 6        | —        | —         | —         | —         | 🆕      |
| Other         | Kvindebasketligaen , Grundspil                              | 0             | 43          | 6              | 0       | 6        | —        | —         | —         | —         | 🆕      |
| International | Euroleague: regular season                                  | 223           | 88          | 6              | 0       | 6        | —        | —         | —         | —         |         |
| Australia     | NBL1, East                                                  | 148           | 118         | 6              | 0       | 6        | —        | —         | —         | —         |         |
| Other         | FBU Superleague                                             | 739           | 79          | 5              | 0       | 5        | —        | —         | —         | —         |         |
| Other         | Liga Nationala                                              | 752           | 152         | 5              | 0       | 5        | —        | —         | —         | —         |         |
| Other         | I B MCKL                                                    | 0             | 36          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| Spain         | Primera FEB                                                 | 1,670         | 205         | 5              | 0       | 5        | —        | —         | —         | —         |         |
| Other         | EYBL U20, Group E                                           | 0             | 37          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U15, Group F                                           | 0             | 31          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| USA           | CBI U15 Fem, CBI U15 Fem                                    | 0             | 34          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U14, Group G                                           | 0             | 42          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U13, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 0             | 30          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U16, CHALLENGE CUP                                     | 0             | 37          | 5              | 0       | 5        | —        | —         | —         | —         | 🆕      |
| Other         | Uruguay LUB                                                 | 250           | 83          | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Australia     | WNBL                                                        | 553           | 67          | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Italy         | Serie A2, Women, Group A                                    | 146           | 118         | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Other         | Turkish Basketball Super League                             | 1,387         | 164         | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Greece        | Stoiximan GBL                                               | 1,024         | 127         | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Other         | Lebanese Basketball League                                  | 94            | 85          | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Other         | Liga Košarkaškog saveza Herceg Bosne                        | 0             | 40          | 4              | 0       | 4        | —        | —         | —         | —         | 🆕      |
| Other         | 1. Division                                                 | 264           | 97          | 4              | 0       | 4        | —        | —         | —         | —         |         |
| International | Eurocup, Group B                                            | 279           | 64          | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Germany       | Germany BBL                                                 | 1,750         | 240         | 4              | 0       | 4        | —        | —         | —         | —         |         |
| France        | France LF2, Women, Regular Season                           | 104           | 103         | 4              | 0       | 4        | —        | —         | —         | —         |         |
| France        | Mexico LNBP Women                                           | 93            | 52          | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Australia     | NBL1 South, Women                                           | 198           | 153         | 4              | 0       | 4        | —        | —         | —         | —         |         |
| Other         | Iraqi Professional League, Iraqi Professional league        | 0             | 53          | 4              | 0       | 4        | —        | —         | —         | —         | 🆕      |
| France        | LNB Clausura                                                | 0             | 67          | 4              | 0       | 4        | —        | —         | —         | —         | 🆕      |
| Other         | 1st Division Men                                            | 0             | 91          | 3              | 0       | 3        | —        | —         | —         | —         | 🆕      |
| Argentina     | Argentina Liga Nacional                                     | 2,654         | 251         | 3              | 0       | 3        | —        | —         | —         | —         |         |
| Other         | Cyprus Basketball Division A                                | 354           | 66          | 3              | 0       | 3        | —        | —         | —         | —         |         |
| Other         | Super League                                                | 650           | 242         | 3              | 0       | 3        | —        | —         | —         | —         |         |
| Other         | A1                                                          | 0             | 69          | 3              | 0       | 3        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U15, Group C                                           | 0             | 32          | 3              | 0       | 3        | —        | —         | —         | —         | 🆕      |
| Other         | Swedish Basketball Superettan                               | 357           | 135         | 3              | 0       | 3        | —        | —         | —         | —         |         |
| Other         | Kızlar Basketbol Gençler Ligi - A Grubu                     | 0             | 34          | 3              | 0       | 3        | —        | —         | —         | —         | 🆕      |
| Australia     | NBL1, Central                                               | 88            | 51          | 3              | 0       | 3        | —        | —         | —         | —         |         |
| USA           | WNBA                                                        | 1,206         | 292         | 3              | 0       | 3        | —        | —         | —         | —         |         |
| Other         | Liga de Ascenso                                             | 360           | 88          | 3              | 0       | 3        | —        | —         | —         | —         |         |
| USA           | CBI U17, Fase Final                                         | 0             | 49          | 3              | 0       | 3        | —        | —         | —         | —         | 🆕      |
| Other         | I A MCKL                                                    | 122           | 117         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Israel        | Israeli National League Basketball                          | 450           | 176         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Saku I liiga                                                | 125           | 95          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Azerbaijan Basketball League, B Group                       | 0             | 38          | 2              | 0       | 2        | —        | —         | —         | —         | 🆕      |
| Italy         | Serie A2                                                    | 652           | 246         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Korisliiga                                                  | 1,116         | 105         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Hungary NB 1.A                                              | 779           | 141         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | SuperSport Premijer Liga                                    | 902           | 151         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | 1st Division                                                | 337           | 122         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | A1 Liga                                                     | 0             | 105         | 2              | 0       | 2        | —        | —         | —         | —         | 🆕      |
| Other         | Azerbaijan Basketball League, A Group                       | 0             | 37          | 2              | 0       | 2        | —        | —         | —         | —         | 🆕      |
| International | Eurocup, Group A                                            | 270           | 70          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | 1. A SKL                                                    | 646           | 108         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Nationale 1, Group A                                        | 0             | 120         | 2              | 0       | 2        | —        | —         | —         | —         | 🆕      |
| Other         | Iceland Women's Basketball Premier League                   | 203           | 69          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Spain         | Segunda FEB, Group West                                     | 137           | 112         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Portugal LFB Women                                          | 185           | 92          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Israel        | Israeli Women Basketball Premier League                     | 300           | 70          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| France        | France LFB Women                                            | 873           | 107         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | 2. Muška Liga                                               | 288           | 43          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Prva RL Istok                                               | 314           | 31          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | 2. ZLS, Women                                               | 0             | 37          | 2              | 0       | 2        | —        | —         | —         | —         | 🆕      |
| Australia     | NBL1, West                                                  | 141           | 108         | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Other         | Colombia LPB                                                | 206           | 46          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| Brazil        | Brazil NBB, Playoffs                                        | 246           | 30          | 2              | 0       | 2        | —        | —         | —         | —         |         |
| USA           | CBI U19, Fase Final                                         | 0             | 32          | 2              | 0       | 2        | —        | —         | —         | —         | 🆕      |
| Other         | Belgian Basketball 2nd Division                             | 164           | 161         | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Princ Caffe Superliga                                       | 591           | 77          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | 1. Liga, Group East                                         | 338           | 66          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Austrian Basketball Superliga                               | 919           | 75          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Poland        | Poland 1st Division Basketball                              | 782           | 202         | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Iranian Basketball Super League                             | 366           | 38          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Liga de Baloncesto Puertorriquena , Seccion Este            | 90            | 65          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| International | Euroleague                                                  | 1,211         | 300         | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Iceland Basketball Premier League                           | 467           | 88          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | EYBL U16, Group B                                           | 0             | 31          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U17, Group A                                           | 0             | 30          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | Zain Basketball League, Premier League                      | 0             | 41          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | Qatari Basketball League, Regular Season                    | 49            | 48          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | EYBL U20, Group B                                           | 0             | 30          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U14, Group D                                           | 0             | 34          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | Elite League                                                | 338           | 57          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Turkey        | Türkiye Kadınlar Basketbol Ligi                             | 0             | 36          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | Divisione Regionale 1 Toscana, Girone A                     | 0             | 66          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | CIBACOPA , Primera Vuelta                                   | 27            | 96          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | A1 ESKAH                                                    | 291           | 32          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | A2 ESKAH                                                    | 284           | 35          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Superliga                                                   | 210           | 119         | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | EYBL U15, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 0             | 32          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| France        | Chile LNB                                                   | 701           | 47          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Australia     | NBL1, South                                                 | 194           | 135         | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Big V                                                       | 594           | 82          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Argentina     | Liga Argentina, Playoffs                                    | 74            | 56          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Australia     | NBL1, North                                                 | 113           | 71          | 1              | 0       | 1        | —        | —         | —         | —         |         |
| France        | Mexico LNBP                                                 | 1,265         | 162         | 1              | 0       | 1        | —        | —         | —         | —         |         |
| Other         | Saku II Liiga                                               | 0             | 35          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | 1. liga, Women, základní část                               | 0             | 31          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Historical    | Nostra.lt-RKL Division A, Reguliarus sezonas                | 0             | 38          | 1              | 0       | 1        | —        | —         | —         | —         | 🆕      |
| Other         | Liga Femenina                                               | 64            | 36          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Liga de Baloncesto Puertorriquena , Seccion Oeste           | 85            | 64          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Liga de Baloncesto Puertorriquena , Seccion Sur             | 86            | 67          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| International | AdmiralBet ABA League, Group B                              | 0             | 59          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Historical    | PGE Basket Liga                                             | 1,352         | 165         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Élite 2                                                     | 1,887         | 261         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Italy         | Lega A Basket                                               | 1,418         | 153         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Italy         | Serie B, Group A                                            | 583           | 204         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Lithuania     | NKL                                                         | 700           | 165         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Lithuania     | RKL Division A                                              | 0             | 185         | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Superleague                                                 | 407           | 82          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Liga de Baloncesto Puertorriquena , Seccion Norte           | 85            | 68          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | 1. Liga, Group West                                         | 309           | 61          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | SLB                                                         | 90            | 115         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Enovos League, Regular Season                               | 0             | 85          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| International | AdmiralBet ABA League, Group A                              | 0             | 57          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Meridianbet KLS                                             | 1,266         | 182         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| International | ABA League 2                                                | 251           | 39          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Nationale 1, Group B                                        | 0             | 132         | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | The League                                                  | 135           | 84          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Spain         | Segunda FEB, Group East                                     | 140           | 110         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Slovenian Second Basketball League, Regular Season          | 0             | 70          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| International | U19 ABA League                                              | 0             | 33          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | French Basketball Cup                                       | 221           | 40          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| International | EuroCup, Women, Playoffs                                    | 287           | 55          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | EYBL U20, Group D                                           | 0             | 30          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Australia     | Bulgaria NBL                                                | 684           | 85          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Germany       | Germany Pro A                                               | 1,534         | 100         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| France        | Pro B, South                                                | 0             | 59          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Druga muška liga - Centar                                   | 113           | 34          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Italy         | Serie B Interregionale, Girone C                            | 0             | 40          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Italy         | Serie B Interregionale, Girone E                            | 0             | 33          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Erkekler Basketbol Gençler Ligi - A Grubu                   | 0             | 35          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Italy         | Serie B Interregionale, Girone D                            | 0             | 46          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Divisione Regionale 2 - Lazio, Girone C                     | 0             | 30          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Serie C Toscana                                             | 0             | 49          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Italy         | Serie B Interregionale, Girone A                            | 0             | 45          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Italy         | Serie B Interregionale, Girone B                            | 0             | 39          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Serie C Girone H sud                                        | 0             | 30          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | B Ethniki - Group 1                                         | 0             | 32          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | B Ethniki - Group 2                                         | 0             | 30          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | B Ethniki - Group 3                                         | 0             | 33          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | TBL                                                         | 723           | 56          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Divisione Regionale 1 Toscana, Girone B                     | 0             | 60          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Basket World League                                         | 0             | 31          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| France        | Pro B, North                                                | 0             | 57          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Italy         | Serie B Interregionale, Girone F                            | 0             | 42          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Belize Elite Basketball League, Regular Season              | 0             | 31          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Egyptian Women's Basketball Super League, Regular Season    | 0             | 55          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | EYBL U16, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 0             | 32          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Lithuania     | NKL, Playoffs                                               | 89            | 32          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Puerto Rico BSN                                             | 247           | 156         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| France        | LNB 2                                                       | 0             | 44          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | CIBACOPA , Segunda Vuelta                                   | 0             | 88          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Australia     | New Zealand NBL                                             | 184           | 73          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Rapid League, Regular Season                                | 5             | 83          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Australia     | NBL1 North, Women                                           | 113           | 74          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Canadian Elite Basketball League                            | 95            | 103         | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Italy         | Serie B, Playoffs                                           | 61            | 43          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| USA           | CBI U17, Sede Vila Velha                                    | 0             | 57          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Puerto Rico BSN, Playoffs                                   | 43            | 32          | 0              | 0       | 0        | —        | —         | —         | —         |         |
| Other         | Copa Espana                                                 | 0             | 41          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | LDB, Fase Final                                             | 0             | 39          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | BSNF, Women                                                 | 0             | 62          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | Divisional Tercera de Ascenso, Fase Regular DTA 2026        | 0             | 44          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |
| Other         | 2. MLS                                                      | 0             | 30          | 0              | 0       | 0        | —        | —         | —         | —         | 🆕      |

## 4. Detalle por liga y por umbral — Top 30 por volumen

### USA — NCAA Women's Division I, Regular season

- Entrenado (A): **0** · Holdout (C): **1,566** · Sesgo @0.75: +0.0129

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 572      | 70.5 % | 66.6–74.0 | -1.36 %   | +28.93 %  |
| 0.68 | 468      | 72.4 % | 68.2–76.3 | +1.41 %   | +32.56 %  |
| 0.70 | 403      | 74.7 % | 70.2–78.7 | +4.57 %   | +36.68 %  |
| 0.72 | 347      | 76.7 % | 71.9–80.8 | +7.32 %   | +40.28 %  |
| 0.75 | 261      | 81.2 % | 76.0–85.5 | +13.72 %  | +48.64 %  |
| 0.80 | 164      | 88.4 % | 82.6–92.5 | +23.78 %  | +61.80 %  |

### Poland — Poland 2nd Basketball League, Runda Zasadnicza

- Entrenado (A): **0** · Holdout (C): **771** · Sesgo @0.75: +0.0172

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 372      | 69.1 % | 64.2–73.6 | -3.28 %   | +26.43 %  |
| 0.68 | 291      | 72.2 % | 66.8–77.0 | +1.03 %   | +32.06 %  |
| 0.70 | 254      | 73.2 % | 67.5–78.3 | +2.52 %   | +34.01 %  |
| 0.72 | 224      | 75.0 % | 68.9–80.2 | +5.00 %   | +37.25 %  |
| 0.75 | 164      | 79.9 % | 73.1–85.3 | +11.83 %  | +46.18 %  |
| 0.80 | 93       | 85.0 % | 76.3–90.8 | +18.92 %  | +55.45 %  |

### Other — Tipos Extraliga, Women

- Entrenado (A): **424** · Holdout (C): **63** · Sesgo @0.75: +0.0881

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 53       | 69.8 % | 56.5–80.5 | -2.26 %   | +27.75 %  |
| 0.68 | 48       | 72.9 % | 59.0–83.4 | +2.08 %   | +33.44 %  |
| 0.70 | 46       | 73.9 % | 59.7–84.4 | +3.48 %   | +35.26 %  |
| 0.72 | 45       | 73.3 % | 59.0–84.0 | +2.67 %   | +34.20 %  |
| 0.75 | 40       | 77.5 % | 62.5–87.7 | +8.50 %   | +41.83 %  |
| 0.80 | 31       | 83.9 % | 67.4–92.9 | +17.42 %  | +53.48 %  |

### France — LNB, Women

- Entrenado (A): **184** · Holdout (C): **60** · Sesgo @0.75: -0.0224

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 49       | 91.8 % | 80.8–96.8 | +28.57 %  | +68.06 %  |
| 0.68 | 47       | 91.5 % | 80.1–96.6 | +28.09 %  | +67.43 %  |
| 0.70 | 43       | 90.7 % | 78.4–96.3 | +26.98 %  | +65.98 %  |
| 0.72 | 41       | 90.2 % | 77.5–96.1 | +26.34 %  | +65.15 %  |
| 0.75 | 39       | 92.3 % | 79.7–97.3 | +29.23 %  | +68.92 %  |
| 0.80 | 37       | 91.9 % | 78.7–97.2 | +28.65 %  | +68.16 %  |

### Other — FSHB Superliga, K.K.M 2025-26

- Entrenado (A): **0** · Holdout (C): **84** · Sesgo @0.75: +0.0047

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 55       | 76.4 % | 63.7–85.6 | +6.91 %   | +39.75 %  |
| 0.68 | 50       | 78.0 % | 64.8–87.2 | +9.20 %   | +42.74 %  |
| 0.70 | 47       | 78.7 % | 65.1–88.0 | +10.21 %  | +44.06 %  |
| 0.72 | 46       | 78.3 % | 64.4–87.7 | +9.57 %   | +43.22 %  |
| 0.75 | 38       | 81.6 % | 66.6–90.8 | +14.21 %  | +49.29 %  |
| 0.80 | 23       | 91.3 % | 73.2–97.6 | +27.83 %  | +67.09 %  |

### Other — United League

- Entrenado (A): **1,143** · Holdout (C): **169** · Sesgo @0.75: +0.0379

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 92       | 75.0 % | 65.3–82.7 | +5.00 %   | +37.25 %  |
| 0.68 | 75       | 76.0 % | 65.2–84.2 | +6.40 %   | +39.08 %  |
| 0.70 | 63       | 77.8 % | 66.1–86.3 | +8.89 %   | +42.33 %  |
| 0.72 | 50       | 76.0 % | 62.6–85.7 | +6.40 %   | +39.08 %  |
| 0.75 | 37       | 75.7 % | 59.9–86.6 | +5.95 %   | +38.49 %  |
| 0.80 | 16       | 75.0 % | 50.5–89.8 | +5.00 %   | +37.25 %  |

### Other — ZBL, Women

- Entrenado (A): **694** · Holdout (C): **74** · Sesgo @0.75: -0.0485

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 49       | 83.7 % | 71.0–91.5 | +17.14 %  | +53.12 %  |
| 0.68 | 45       | 84.4 % | 71.2–92.3 | +18.22 %  | +54.53 %  |
| 0.70 | 44       | 86.4 % | 73.3–93.6 | +20.91 %  | +58.05 %  |
| 0.72 | 42       | 85.7 % | 72.2–93.3 | +20.00 %  | +56.86 %  |
| 0.75 | 36       | 88.9 % | 74.7–95.6 | +24.44 %  | +62.67 %  |
| 0.80 | 24       | 87.5 % | 69.0–95.7 | +22.50 %  | +60.12 %  |

### Australia — NBL1 East, Women

- Entrenado (A): **153** · Holdout (C): **118** · Sesgo @0.75: -0.0445

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 78       | 75.6 % | 65.1–83.8 | +5.90 %   | +38.42 %  |
| 0.68 | 62       | 79.0 % | 67.4–87.3 | +10.65 %  | +44.63 %  |
| 0.70 | 58       | 77.6 % | 65.3–86.4 | +8.62 %   | +41.98 %  |
| 0.72 | 48       | 83.3 % | 70.4–91.3 | +16.67 %  | +52.50 %  |
| 0.75 | 36       | 86.1 % | 71.3–93.9 | +20.56 %  | +57.58 %  |
| 0.80 | 20       | 90.0 % | 69.9–97.2 | +26.00 %  | +64.70 %  |

### Other — Top Division Women, Regular Season

- Entrenado (A): **0** · Holdout (C): **88** · Sesgo @0.75: -0.0493

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 61       | 83.6 % | 72.4–90.8 | +17.05 %  | +53.00 %  |
| 0.68 | 56       | 82.1 % | 70.2–90.0 | +15.00 %  | +50.32 %  |
| 0.70 | 49       | 83.7 % | 71.0–91.5 | +17.14 %  | +53.12 %  |
| 0.72 | 43       | 86.0 % | 72.7–93.4 | +20.47 %  | +57.47 %  |
| 0.75 | 35       | 88.6 % | 74.0–95.5 | +24.00 %  | +62.09 %  |
| 0.80 | 25       | 88.0 % | 70.0–95.8 | +23.20 %  | +61.04 %  |

### Other — Hungary Divison A Women

- Entrenado (A): **284** · Holdout (C): **107** · Sesgo @0.75: +0.0656

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 71       | 71.8 % | 60.5–81.0 | +0.56 %   | +31.45 %  |
| 0.68 | 59       | 76.3 % | 64.0–85.3 | +6.78 %   | +39.58 %  |
| 0.70 | 52       | 75.0 % | 61.8–84.8 | +5.00 %   | +37.25 %  |
| 0.72 | 46       | 78.3 % | 64.4–87.7 | +9.57 %   | +43.22 %  |
| 0.75 | 35       | 77.1 % | 61.0–87.9 | +8.00 %   | +41.17 %  |
| 0.80 | 27       | 74.1 % | 55.3–86.8 | +3.70 %   | +35.56 %  |

### Other — Swedish Basketball League Women

- Entrenado (A): **239** · Holdout (C): **112** · Sesgo @0.75: +0.1073

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 61       | 70.5 % | 58.1–80.4 | -1.31 %   | +29.00 %  |
| 0.68 | 51       | 74.5 % | 61.1–84.5 | +4.31 %   | +36.35 %  |
| 0.70 | 47       | 72.3 % | 58.2–83.1 | +1.28 %   | +32.38 %  |
| 0.72 | 40       | 75.0 % | 59.8–85.8 | +5.00 %   | +37.25 %  |
| 0.75 | 32       | 71.9 % | 54.6–84.4 | +0.62 %   | +31.53 %  |
| 0.80 | 19       | 68.4 % | 46.0–84.6 | -4.21 %   | +25.21 %  |

### Other — Premijer ženska liga

- Entrenado (A): **305** · Holdout (C): **96** · Sesgo @0.75: +0.1371

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 58       | 60.3 % | 47.5–71.9 | -15.52 %  | +10.43 %  |
| 0.68 | 52       | 59.6 % | 46.1–71.8 | -16.54 %  | +9.10 %   |
| 0.70 | 48       | 62.5 % | 48.4–74.8 | -12.50 %  | +14.38 %  |
| 0.72 | 42       | 66.7 % | 51.6–79.0 | -6.67 %   | +22.00 %  |
| 0.75 | 32       | 68.8 % | 51.4–82.0 | -3.75 %   | +25.81 %  |
| 0.80 | 20       | 65.0 % | 43.3–81.9 | -9.00 %   | +18.95 %  |

### Australia — NBL, East

- Entrenado (A): **78** · Holdout (C): **87** · Sesgo @0.75: -0.0561

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 54       | 81.5 % | 69.2–89.6 | +14.07 %  | +49.11 %  |
| 0.68 | 48       | 81.2 % | 68.1–89.8 | +13.75 %  | +48.69 %  |
| 0.70 | 39       | 87.2 % | 73.3–94.4 | +22.05 %  | +59.54 %  |
| 0.72 | 34       | 88.2 % | 73.4–95.3 | +23.53 %  | +61.47 %  |
| 0.75 | 31       | 87.1 % | 71.1–94.9 | +21.94 %  | +59.39 %  |
| 0.80 | 15       | 86.7 % | 62.1–96.3 | +21.33 %  | +58.60 %  |

### Poland — MPBL

- Entrenado (A): **319** · Holdout (C): **264** · Sesgo @0.75: -0.0851

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 103      | 68.0 % | 58.4–76.2 | -4.85 %   | +24.37 %  |
| 0.68 | 80       | 72.5 % | 61.9–81.1 | +1.50 %   | +32.67 %  |
| 0.70 | 64       | 81.2 % | 70.0–88.9 | +13.75 %  | +48.69 %  |
| 0.72 | 53       | 79.2 % | 66.5–88.0 | +10.94 %  | +45.02 %  |
| 0.75 | 31       | 87.1 % | 71.1–94.9 | +21.94 %  | +59.39 %  |
| 0.80 | 7        | —      | —         | —         | —         |

### Other — Basketligaen

- Entrenado (A): **651** · Holdout (C): **108** · Sesgo @0.75: +0.0024

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 57       | 71.9 % | 59.2–81.9 | +0.70 %   | +31.63 %  |
| 0.68 | 49       | 73.5 % | 59.7–83.8 | +2.86 %   | +34.45 %  |
| 0.70 | 46       | 73.9 % | 59.7–84.4 | +3.48 %   | +35.26 %  |
| 0.72 | 39       | 76.9 % | 61.7–87.4 | +7.69 %   | +40.77 %  |
| 0.75 | 30       | 80.0 % | 62.7–90.5 | +12.00 %  | +46.40 %  |
| 0.80 | 15       | 86.7 % | 62.1–96.3 | +21.33 %  | +58.60 %  |

### Other — Big V, Women

- Entrenado (A): **108** · Holdout (C): **82** · Sesgo @0.75: +0.0013

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 51       | 70.6 % | 57.0–81.3 | -1.18 %   | +29.18 %  |
| 0.68 | 44       | 77.3 % | 63.0–87.2 | +8.18 %   | +41.41 %  |
| 0.70 | 36       | 77.8 % | 61.9–88.3 | +8.89 %   | +42.33 %  |
| 0.72 | 33       | 78.8 % | 62.2–89.3 | +10.30 %  | +44.18 %  |
| 0.75 | 28       | 82.1 % | 64.4–92.1 | +15.00 %  | +50.32 %  |
| 0.80 | 14       | —      | —         | —         | —         |

### USA — CBI U15 , Classificatória A

- Entrenado (A): **0** · Holdout (C): **76** · Sesgo @0.75: -0.0879

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 39       | 84.6 % | 70.3–92.8 | +18.46 %  | +54.85 %  |
| 0.68 | 34       | 94.1 % | 80.9–98.4 | +31.76 %  | +72.24 %  |
| 0.70 | 30       | 93.3 % | 78.7–98.2 | +30.67 %  | +70.80 %  |
| 0.72 | 28       | 92.9 % | 77.4–98.0 | +30.00 %  | +69.93 %  |
| 0.75 | 27       | 92.6 % | 76.6–97.9 | +29.63 %  | +69.44 %  |
| 0.80 | 21       | 95.2 % | 77.3–99.2 | +33.33 %  | +74.29 %  |

### Other — Prva Liga

- Entrenado (A): **0** · Holdout (C): **135** · Sesgo @0.75: -0.0713

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 42       | 78.6 % | 64.1–88.3 | +10.00 %  | +43.79 %  |
| 0.68 | 37       | 81.1 % | 65.8–90.5 | +13.51 %  | +48.38 %  |
| 0.70 | 32       | 84.4 % | 68.2–93.1 | +18.12 %  | +54.41 %  |
| 0.72 | 27       | 88.9 % | 71.9–96.1 | +24.44 %  | +62.67 %  |
| 0.75 | 27       | 88.9 % | 71.9–96.1 | +24.44 %  | +62.67 %  |
| 0.80 | 16       | 93.8 % | 71.7–98.9 | +31.25 %  | +71.56 %  |

### Other — Russian Women's Basketball Premier League

- Entrenado (A): **726** · Holdout (C): **102** · Sesgo @0.75: +0.0829

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 59       | 79.7 % | 67.7–88.0 | +11.53 %  | +45.78 %  |
| 0.68 | 44       | 77.3 % | 63.0–87.2 | +8.18 %   | +41.41 %  |
| 0.70 | 40       | 75.0 % | 59.8–85.8 | +5.00 %   | +37.25 %  |
| 0.72 | 37       | 73.0 % | 57.0–84.6 | +2.16 %   | +33.54 %  |
| 0.75 | 27       | 74.1 % | 55.3–86.8 | +3.70 %   | +35.56 %  |
| 0.80 | 20       | 70.0 % | 48.1–85.5 | -2.00 %   | +28.10 %  |

### Other — Estonian-Latvian Basketball League

- Entrenado (A): **610** · Holdout (C): **159** · Sesgo @0.75: -0.0701

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 80       | 81.2 % | 71.3–88.3 | +13.75 %  | +48.69 %  |
| 0.68 | 65       | 84.6 % | 73.9–91.4 | +18.46 %  | +54.85 %  |
| 0.70 | 50       | 88.0 % | 76.2–94.4 | +23.20 %  | +61.04 %  |
| 0.72 | 39       | 89.7 % | 76.4–95.9 | +25.64 %  | +64.23 %  |
| 0.75 | 25       | 84.0 % | 65.3–93.6 | +17.60 %  | +53.72 %  |
| 0.80 | 3        | —      | —         | —         | —         |

### Australia — NBL, West

- Entrenado (A): **75** · Holdout (C): **84** · Sesgo @0.75: +0.0979

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 50       | 76.0 % | 62.6–85.7 | +6.40 %   | +39.08 %  |
| 0.68 | 40       | 72.5 % | 57.2–83.9 | +1.50 %   | +32.67 %  |
| 0.70 | 30       | 73.3 % | 55.6–85.8 | +2.67 %   | +34.20 %  |
| 0.72 | 28       | 71.4 % | 52.9–84.7 | +0.00 %   | +30.71 %  |
| 0.75 | 24       | 70.8 % | 50.8–85.1 | -0.83 %   | +29.62 %  |
| 0.80 | 12       | —      | —         | —         | —         |

### Other — SLB, Women

- Entrenado (A): **0** · Holdout (C): **87** · Sesgo @0.75: +0.1151

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 54       | 66.7 % | 53.4–77.8 | -6.67 %   | +22.00 %  |
| 0.68 | 45       | 66.7 % | 52.1–78.6 | -6.67 %   | +22.00 %  |
| 0.70 | 40       | 67.5 % | 52.0–79.9 | -5.50 %   | +23.53 %  |
| 0.72 | 29       | 65.5 % | 47.3–80.1 | -8.28 %   | +19.90 %  |
| 0.75 | 24       | 70.8 % | 50.8–85.1 | -0.83 %   | +29.62 %  |
| 0.80 | 16       | 75.0 % | 50.5–89.8 | +5.00 %   | +37.25 %  |

### Other — 1st Division Women

- Entrenado (A): **0** · Holdout (C): **52** · Sesgo @0.75: +0.0080

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 36       | 72.2 % | 56.0–84.2 | +1.11 %   | +32.17 %  |
| 0.68 | 28       | 82.1 % | 64.4–92.1 | +15.00 %  | +50.32 %  |
| 0.70 | 26       | 80.8 % | 62.1–91.5 | +13.08 %  | +47.81 %  |
| 0.72 | 24       | 83.3 % | 64.1–93.3 | +16.67 %  | +52.50 %  |
| 0.75 | 23       | 82.6 % | 62.9–93.0 | +15.65 %  | +51.17 %  |
| 0.80 | 16       | 87.5 % | 64.0–96.5 | +22.50 %  | +60.13 %  |

### Other — 2nd League

- Entrenado (A): **271** · Holdout (C): **131** · Sesgo @0.75: +0.0226

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 67       | 70.2 % | 58.3–79.8 | -1.79 %   | +28.37 %  |
| 0.68 | 48       | 75.0 % | 61.2–85.1 | +5.00 %   | +37.25 %  |
| 0.70 | 38       | 73.7 % | 58.0–85.0 | +3.16 %   | +34.84 %  |
| 0.72 | 33       | 72.7 % | 55.8–84.9 | +1.82 %   | +33.09 %  |
| 0.75 | 22       | 77.3 % | 56.6–89.9 | +8.18 %   | +41.41 %  |
| 0.80 | 8        | —      | —         | —         | —         |

### Lithuania — Nostra.lt-RKL Division B, Reguliarusis sezonas

- Entrenado (A): **0** · Holdout (C): **164** · Sesgo @0.75: +0.0664

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 61       | 73.8 % | 61.6–83.2 | +3.28 %   | +35.00 %  |
| 0.68 | 47       | 72.3 % | 58.2–83.1 | +1.28 %   | +32.38 %  |
| 0.70 | 40       | 77.5 % | 62.5–87.7 | +8.50 %   | +41.83 %  |
| 0.72 | 34       | 76.5 % | 60.0–87.6 | +7.06 %   | +39.94 %  |
| 0.75 | 22       | 72.7 % | 51.8–86.8 | +1.82 %   | +33.09 %  |
| 0.80 | 8        | —      | —         | —         | —         |

### USA — CBI U19, Sede Rio

- Entrenado (A): **0** · Holdout (C): **37** · Sesgo @0.75: -0.1058

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 32       | 78.1 % | 61.2–89.0 | +9.38 %   | +42.97 %  |
| 0.68 | 29       | 82.8 % | 65.5–92.4 | +15.86 %  | +51.45 %  |
| 0.70 | 29       | 82.8 % | 65.5–92.4 | +15.86 %  | +51.45 %  |
| 0.72 | 29       | 82.8 % | 65.5–92.4 | +15.86 %  | +51.45 %  |
| 0.75 | 21       | 95.2 % | 77.3–99.2 | +33.33 %  | +74.29 %  |
| 0.80 | 13       | —      | —         | —         | —         |

### Other — Switzerland LNA

- Entrenado (A): **695** · Holdout (C): **91** · Sesgo @0.75: +0.0325

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 43       | 74.4 % | 59.8–85.1 | +4.19 %   | +36.19 %  |
| 0.68 | 35       | 74.3 % | 57.9–85.8 | +4.00 %   | +35.94 %  |
| 0.70 | 32       | 71.9 % | 54.6–84.4 | +0.62 %   | +31.53 %  |
| 0.72 | 23       | 73.9 % | 53.5–87.5 | +3.48 %   | +35.26 %  |
| 0.75 | 21       | 76.2 % | 54.9–89.4 | +6.67 %   | +39.43 %  |
| 0.80 | 8        | —      | —         | —         | —         |

### Other — Saudi Basketball League, Premier League

- Entrenado (A): **0** · Holdout (C): **45** · Sesgo @0.75: -0.1263

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 32       | 81.2 % | 64.7–91.1 | +13.75 %  | +48.69 %  |
| 0.68 | 28       | 85.7 % | 68.5–94.3 | +20.00 %  | +56.86 %  |
| 0.70 | 26       | 88.5 % | 71.0–96.0 | +23.85 %  | +61.88 %  |
| 0.72 | 24       | 87.5 % | 69.0–95.7 | +22.50 %  | +60.12 %  |
| 0.75 | 20       | 95.0 % | 76.4–99.1 | +33.00 %  | +73.85 %  |
| 0.80 | 13       | —      | —         | —         | —         |

### USA — CBI U15 , Classificatória B

- Entrenado (A): **0** · Holdout (C): **78** · Sesgo @0.75: +0.0748

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 42       | 76.2 % | 61.5–86.5 | +6.67 %   | +39.43 %  |
| 0.68 | 35       | 77.1 % | 61.0–87.9 | +8.00 %   | +41.17 %  |
| 0.70 | 31       | 74.2 % | 56.8–86.3 | +3.87 %   | +35.77 %  |
| 0.72 | 30       | 76.7 % | 59.1–88.2 | +7.33 %   | +40.30 %  |
| 0.75 | 20       | 75.0 % | 53.1–88.8 | +5.00 %   | +37.25 %  |
| 0.80 | 12       | —      | —         | —         | —         |

### International — BNXT League

- Entrenado (A): **222** · Holdout (C): **224** · Sesgo @0.75: -0.1677

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 76       | 75.0 % | 64.2–83.4 | +5.00 %   | +37.25 %  |
| 0.68 | 54       | 74.1 % | 61.1–83.9 | +3.70 %   | +35.56 %  |
| 0.70 | 36       | 69.4 % | 53.1–82.0 | -2.78 %   | +27.08 %  |
| 0.72 | 26       | 80.8 % | 62.1–91.5 | +13.08 %  | +47.81 %  |
| 0.75 | 18       | 94.4 % | 74.2–99.0 | +32.22 %  | +72.83 %  |
| 0.80 | 2        | —      | —         | —         | —         |

## 5. Ligas destacadas

### 5.1 Mejores ROI @1.40 a 0.75 (mín. 30 apuestas)

| País      | Liga                                           | Entrenado | Apuestas @0.75 | WR     | ROI @1.40 |
| ---       | ---                                            | ---:      | ---:           | ---:   | ---:      |
| France    | LNB, Women                                     | 184       | 39             | 92.3 % | +29.23 %  |
| Other     | ZBL, Women                                     | 694       | 36             | 88.9 % | +24.44 %  |
| Other     | Top Division Women, Regular Season             | 0         | 35             | 88.6 % | +24.00 %  |
| Australia | NBL, East                                      | 78        | 31             | 87.1 % | +21.94 %  |
| Poland    | MPBL                                           | 319       | 31             | 87.1 % | +21.94 %  |
| Australia | NBL1 East, Women                               | 153       | 36             | 86.1 % | +20.56 %  |
| Other     | FSHB Superliga, K.K.M 2025-26                  | 0         | 38             | 81.6 % | +14.21 %  |
| USA       | NCAA Women's Division I, Regular season        | 0         | 261            | 81.2 % | +13.72 %  |
| Other     | Basketligaen                                   | 651       | 30             | 80.0 % | +12.00 %  |
| Poland    | Poland 2nd Basketball League, Runda Zasadnicza | 0         | 164            | 79.9 % | +11.83 %  |

### 5.2 Peores ROI @1.40 a 0.75 (mín. 30 apuestas)

| País      | Liga                                           | Entrenado | Apuestas @0.75 | WR     | ROI @1.40 |
| ---       | ---                                            | ---:      | ---:           | ---:   | ---:      |
| Other     | Premijer ženska liga                           | 305       | 32             | 68.8 % | -3.75 %   |
| Other     | Swedish Basketball League Women                | 239       | 32             | 71.9 % | +0.62 %   |
| Other     | United League                                  | 1,143     | 37             | 75.7 % | +5.95 %   |
| Other     | Hungary Divison A Women                        | 284       | 35             | 77.1 % | +8.00 %   |
| Other     | Tipos Extraliga, Women                         | 424       | 40             | 77.5 % | +8.50 %   |
| Poland    | Poland 2nd Basketball League, Runda Zasadnicza | 0         | 164            | 79.9 % | +11.83 %  |
| Other     | Basketligaen                                   | 651       | 30             | 80.0 % | +12.00 %  |
| USA       | NCAA Women's Division I, Regular season        | 0         | 261            | 81.2 % | +13.72 %  |
| Other     | FSHB Superliga, K.K.M 2025-26                  | 0         | 38             | 81.6 % | +14.21 %  |
| Australia | NBL1 East, Women                               | 153       | 36             | 86.1 % | +20.56 %  |

## 5.3 Marcadores de la columna "Bloqueo"

- **🆕 (arranque en frío):** la liga no tuvo ningún partido en el bloque de entrenamiento (A). El modelo no la vio; sus priors son as-of. Hay 108 ligas así.

- **⚠️ (ROI negativo a 0.75):** la liga tiene ROI@1.40 < 0 % a umbral 0.75. Candidata a bloqueo por medición. Hay 5 ligas así.

- **(vacío):** liga normal, sin alertas.

### 5.4 Ligas en arranque en frío (Entrenado = 0)

| País          | Liga                                               | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 |
| ---           | ---                                                | ---:        | ---:           | ---:    | ---:     | ---:     |
| USA           | NCAA Women's Division I, Regular season            | 1,566       | 261            | 212     | 49       | 81.2 %   |
| Poland        | Poland 2nd Basketball League, Runda Zasadnicza     | 771         | 164            | 131     | 33       | 79.9 %   |
| Other         | FSHB Superliga, K.K.M 2025-26                      | 84          | 38             | 31      | 7        | 81.6 %   |
| Other         | Top Division Women, Regular Season                 | 88          | 35             | 31      | 4        | 88.6 %   |
| USA           | CBI U15 , Classificatória A                        | 76          | 27             | 25      | 2        | 92.6 %   |
| Other         | Prva Liga                                          | 135         | 27             | 24      | 3        | 88.9 %   |
| Other         | SLB, Women                                         | 87          | 24             | 17      | 7        | 70.8 %   |
| Other         | 1st Division Women                                 | 52          | 23             | 19      | 4        | 82.6 %   |
| Lithuania     | Nostra.lt-RKL Division B, Reguliarusis sezonas     | 164         | 22             | 16      | 6        | 72.7 %   |
| USA           | CBI U19, Sede Rio                                  | 37          | 21             | 20      | 1        | 95.2 %   |
| Other         | Saudi Basketball League, Premier League            | 45          | 20             | 19      | 1        | 95.0 %   |
| USA           | CBI U15 , Classificatória B                        | 78          | 20             | 15      | 5        | 75.0 %   |
| Australia     | NBL Women , Regular Season                         | 106         | 17             | 14      | 3        | 82.3 %   |
| Other         | 1 Liga Kobiet, Runda Zasadnicza                    | 239         | 17             | 13      | 4        | 76.5 %   |
| Other         | Sparkasse Prva Liga Zeni                           | 63          | 17             | 9       | 8        | 52.9 %   |
| Other         | LDB, Primeira fase                                 | 128         | 16             | 14      | 2        | 87.5 %   |
| USA           | CBI U17, Sede Curitiba                             | 67          | 15             | 13      | 2        | 86.7 %   |
| France        | LNB, Clausura                                      | 49          | 15             | 12      | 3        | 80.0 %   |
| International | AdmiralBet ABA League, Group B                     | 59          | 0              | 0       | 0        | —        |
| Lithuania     | RKL Division A                                     | 185         | 0              | 0       | 0        | —        |
| Other         | Enovos League, Regular Season                      | 85          | 0              | 0       | 0        | —        |
| International | AdmiralBet ABA League, Group A                     | 57          | 0              | 0       | 0        | —        |
| Other         | Nationale 1, Group B                               | 132         | 0              | 0       | 0        | —        |
| Other         | Slovenian Second Basketball League, Regular Season | 70          | 0              | 0       | 0        | —        |
| International | U19 ABA League                                     | 33          | 0              | 0       | 0        | —        |
| Other         | EYBL U20, Group D                                  | 30          | 0              | 0       | 0        | —        |
| France        | Pro B, South                                       | 59          | 0              | 0       | 0        | —        |
| Italy         | Serie B Interregionale, Girone C                   | 40          | 0              | 0       | 0        | —        |
| Italy         | Serie B Interregionale, Girone E                   | 33          | 0              | 0       | 0        | —        |
| Other         | Erkekler Basketbol Gençler Ligi - A Grubu          | 35          | 0              | 0       | 0        | —        |
| Italy         | Serie B Interregionale, Girone D                   | 46          | 0              | 0       | 0        | —        |
| Other         | Divisione Regionale 2 - Lazio, Girone C            | 30          | 0              | 0       | 0        | —        |
| Other         | Serie C Toscana                                    | 49          | 0              | 0       | 0        | —        |
| Italy         | Serie B Interregionale, Girone A                   | 45          | 0              | 0       | 0        | —        |
| Italy         | Serie B Interregionale, Girone B                   | 39          | 0              | 0       | 0        | —        |
| Other         | Serie C Girone H sud                               | 30          | 0              | 0       | 0        | —        |
| Other         | B Ethniki - Group 1                                | 32          | 0              | 0       | 0        | —        |
| Other         | B Ethniki - Group 2                                | 30          | 0              | 0       | 0        | —        |
| Other         | B Ethniki - Group 3                                | 33          | 0              | 0       | 0        | —        |
| Other         | Divisione Regionale 1 Toscana, Girone B            | 60          | 0              | 0       | 0        | —        |

_...y 14 ligas más._


### 5.6 Ligas en arranque en frío con **todas** las apuestas perdidas (0 % WR)

| País       | Liga                                                        | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos |
| ---        | ---                                                         | ---:        | ---:           | ---:    | ---:     |
| Other      | Rwanda Basketball League, Regular Season                    | 65          | 14             | 0       | 14       |
| Other      | EYBL U13, Group D                                           | 34          | 12             | 0       | 12       |
| Other      | EYBL U14, Group E                                           | 52          | 12             | 0       | 12       |
| Other      | Jedinstvena kadetska liga                                   | 64          | 12             | 0       | 12       |
| Other      | Liga Nacional Femenina Chile, Fase regular                  | 34          | 12             | 0       | 12       |
| Other      | Liga Femenina, Women, Clausura                              | 115         | 11             | 0       | 11       |
| Other      | Rwanda Basketball League Women, Regular Season              | 42          | 10             | 0       | 10       |
| Other      | LF Challenge, Regular Season                                | 145         | 9              | 0       | 9        |
| Other      | Latvian Basketball Cup                                      | 31          | 9              | 0       | 9        |
| Other      | EYBL U15, Group D                                           | 41          | 9              | 0       | 9        |
| Australia  | NBL Men , Regular Season                                    | 125         | 9              | 0       | 9        |
| Other      | EYBL U17, Group E                                           | 49          | 8              | 0       | 8        |
| Other      | EYBL U17, Group G                                           | 37          | 8              | 0       | 8        |
| Other      | Superliga, Women                                            | 36          | 7              | 0       | 7        |
| Other      | Superliga e Femrave                                         | 41          | 7              | 0       | 7        |
| Other      | Campeonato Nacional da 1ª Divisao Feminina, 1ª Fase - Sul   | 50          | 7              | 0       | 7        |
| Other      | 1. liga, Women                                              | 75          | 7              | 0       | 7        |
| Other      | U21 Espoirs Elite                                           | 142         | 7              | 0       | 7        |
| USA        | NCAA Women's March Madness                                  | 59          | 7              | 0       | 7        |
| USA        | CBI U19, Sede Curitiba                                      | 44          | 7              | 0       | 7        |
| Other      | Campeonato Nacional da 1ª Divisao Feminina, 1ª Fase - Norte | 35          | 6              | 0       | 6        |
| Other      | Kvindebasketligaen , Grundspil                              | 43          | 6              | 0       | 6        |
| Other      | I B MCKL                                                    | 36          | 5              | 0       | 5        |
| Other      | EYBL U20, Group E                                           | 37          | 5              | 0       | 5        |
| Other      | EYBL U15, Group F                                           | 31          | 5              | 0       | 5        |
| USA        | CBI U15 Fem, CBI U15 Fem                                    | 34          | 5              | 0       | 5        |
| Other      | EYBL U14, Group G                                           | 42          | 5              | 0       | 5        |
| Other      | EYBL U13, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 30          | 5              | 0       | 5        |
| Other      | EYBL U16, CHALLENGE CUP                                     | 37          | 5              | 0       | 5        |
| Other      | Liga Košarkaškog saveza Herceg Bosne                        | 40          | 4              | 0       | 4        |
| Other      | Iraqi Professional League, Iraqi Professional league        | 53          | 4              | 0       | 4        |
| France     | LNB Clausura                                                | 67          | 4              | 0       | 4        |
| Other      | 1st Division Men                                            | 91          | 3              | 0       | 3        |
| Other      | A1                                                          | 69          | 3              | 0       | 3        |
| Other      | EYBL U15, Group C                                           | 32          | 3              | 0       | 3        |
| Other      | Kızlar Basketbol Gençler Ligi - A Grubu                     | 34          | 3              | 0       | 3        |
| USA        | CBI U17, Fase Final                                         | 49          | 3              | 0       | 3        |
| Other      | Azerbaijan Basketball League, B Group                       | 38          | 2              | 0       | 2        |
| Other      | A1 Liga                                                     | 105         | 2              | 0       | 2        |
| Other      | Azerbaijan Basketball League, A Group                       | 37          | 2              | 0       | 2        |
| Other      | Nationale 1, Group A                                        | 120         | 2              | 0       | 2        |
| Other      | 2. ZLS, Women                                               | 37          | 2              | 0       | 2        |
| USA        | CBI U19, Fase Final                                         | 32          | 2              | 0       | 2        |
| Other      | EYBL U16, Group B                                           | 31          | 1              | 0       | 1        |
| Other      | EYBL U17, Group A                                           | 30          | 1              | 0       | 1        |
| Other      | Zain Basketball League, Premier League                      | 41          | 1              | 0       | 1        |
| Other      | EYBL U20, Group B                                           | 30          | 1              | 0       | 1        |
| Other      | EYBL U14, Group D                                           | 34          | 1              | 0       | 1        |
| Turkey     | Türkiye Kadınlar Basketbol Ligi                             | 36          | 1              | 0       | 1        |
| Other      | Divisione Regionale 1 Toscana, Girone A                     | 66          | 1              | 0       | 1        |
| Other      | EYBL U15, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 32          | 1              | 0       | 1        |
| Other      | Saku II Liiga                                               | 35          | 1              | 0       | 1        |
| Other      | 1. liga, Women, základní část                               | 31          | 1              | 0       | 1        |
| Historical | Nostra.lt-RKL Division A, Reguliarus sezonas                | 38          | 1              | 0       | 1        |

### 5.5 Ligas con ROI negativo a 0.75 (⚠️)

| País      | Liga                     | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |
| ---       | ---                      | ---:      | ---:           | ---:    | ---:     | ---:     | ---:      |
| Other     | Sparkasse Prva Liga Zeni | 0         | 17             | 9       | 8        | 52.9 %   | -25.88 %  |
| Other     | Liga 1 Masculin, Faza II | 21        | 17             | 11      | 6        | 64.7 %   | -9.41 %   |
| Other     | Premijer ženska liga     | 305       | 32             | 22      | 10       | 68.8 %   | -3.75 %   |
| Australia | NBL, West                | 75        | 24             | 17      | 7        | 70.8 %   | -0.83 %   |
| Other     | SLB, Women               | 0         | 24             | 17      | 7        | 70.8 %   | -0.83 %   |

## 6. Caso específico — `Poland 2nd Basketball League, Runda Zasadnicza`

> Esta liga (852 partidos históricos, cobertura 100 %) fue **excluida** en `m27_v5_3` por el bug del clasificador (`Runda Zasadnicza` → `SEGUNDA_FASE`). En `m27_v5_3_1` se incluye.

- **Entrenado (A):** 0 · **Holdout (C):** 771

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 372      | 69.1 % | 64.2–73.6 | -3.28 %   | +26.43 %  |
| 0.68 | 291      | 72.2 % | 66.8–77.0 | +1.03 %   | +32.06 %  |
| 0.70 | 254      | 73.2 % | 67.5–78.3 | +2.52 %   | +34.01 %  |
| 0.72 | 224      | 75.0 % | 68.9–80.2 | +5.00 %   | +37.25 %  |
| 0.75 | 164      | 79.9 % | 73.1–85.3 | +11.83 %  | +46.18 %  |
| 0.80 | 93       | 85.0 % | 76.3–90.8 | +18.92 %  | +55.45 %  |

A 0.75: **164 apuestas, WR 79.88 %, ROI@1.40 +11.83 %** (en vivo: 4 partidos, v5_3 2W/2L, v5_3_1 2W/0L).

## 7. Prueba contra los partidos en vivo de `m27_v5_3`

- **m27_v5_3 (original):** 85 apuestas, 67 ganadas, 18 perdidas → WR 78.8 %.
- **m27_v5_3_1 (clon):** 59 apuestas, 47 ganadas, 12 perdidas → WR 79.7 %.

### 7.1 Resumen por liga — apuestas en vivo (original vs clon)

| Dif. WR   | WR v5_3 | WR v5_3_1 | V5_3 Wins | V5_3_1 Wins | País          | Liga                                                                        | En vivo (partidos) | v5_3: apuestas | v5_3: perdidas | v5_3_1: apuestas | v5_3_1: perdidas |      |
| ---       | ---:    | ---:      | ---:      | ---:        | ---           | ---:                                                                        | ---:               | ---:           | ---:           | ---:             | ---:             | ---: |
| ⚪        | 80 %    | 80 %      | 4         | 4           | Other         | LBP, Group B                                                                | 5                  | 5              | 1              | 5                | 1                |      |
| 🟢        | 80 %    | 100 %     | 4         | 4           | France        | LNB, Women                                                                  | 5                  | 5              | 1              | 4                | 0                |      |
| ⚪        | 75 %    | 75 %      | 3         | 3           | Other         | 1. ZLS, Women                                                               | 4                  | 4              | 1              | 4                | 1                |      |
| ⚪        | 100 %   | 100 %     | 3         | 3           | Other         | Road to Basketball Africa League, Group D                                   | 3                  | 3              | 0              | 3                | 0                |      |
| ⚪        | 33 %    | 33 %      | 1         | 1           | Australia     | NBL, West                                                                   | 3                  | 3              | 2              | 3                | 2                |      |
| 🟢        | 50 %    | 100 %     | 2         | 2           | Poland        | Poland 2nd Basketball League, Runda Zasadnicza                              | 4                  | 4              | 2              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 3         | 2           | Other         | Swedish Basketball League Women                                             | 3                  | 3              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | United League                                                               | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Historical    | L1BF BT, Faza I                                                             | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | Estonian-Latvian Basketball League                                          | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | Tipos Extraliga, Women                                                      | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 4         | 1           | Other         | Basketligaen                                                                | 4                  | 4              | 0              | 1                | 0                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Other         | 2nd League                                                                  | 2                  | 2              | 1              | 1                | 1                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | International | FIBA Women Basketball League Africa - Qualifers, Group A                    | 2                  | 2              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 2         | 1           | Other         | LMB, Clausura                                                               | 2                  | 2              | 0              | 1                | 0                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Japan         | B League One                                                                | 2                  | 2              | 1              | 1                | 1                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Other         | A1 Women                                                                    | 2                  | 2              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 13 | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Brazil        | Brazil NBB                                                                  | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Greece        | Stoiximan GBL                                                               | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Turkish Basketball Super League                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Switzerland LNA                                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Nostra.lt-RKL Division A, Reguliarus sezonas                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Other         | Women's Basketball Division A                                               | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Russian Women's Basketball Premier League                                   | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Kvindebasketligaen , Grundspil                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 14 | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Super League                                                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | International | Eurocup, Group B                                                            | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Liga Nationala                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Liga Nacional Femenina Chile, Fase zonal                                    | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Semi-Finals 19    | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Other         | 2. ZLS, Women                                                               | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Premijer ženska liga                                                        | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | FSHB Superliga                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | 1 Liga Kobiet, Runda Zasadnicza                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Liga 1 Masculin, Faza I                                                     | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Top Division Women, Regular Season                                          | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Catarinense, Women, Playoffs                                                | 1                  | 1              | 0              | 1                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Japan         | B League Premier                                                            | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | FIBA Women Basketball League Africa - Qualifers, Group B                    | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | EuroCup, Women, Group B                                                     | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | Swedish Basketball Superettan                                               | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | France        | Mexico LNBP, Playoffs                                                       | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Other         | Hungary NB 1.A                                                              | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | AdmiralBet ABA League, Group B                                              | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | Cyprus Basketball Division A                                                | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | 1. Division                                                                 | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | ?             | Nostra.lt-RKL Division B, Reguliarus sezonas                                | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Israel        | Israeli Women Basketball Premier League                                     | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | France        | France Pro A                                                                | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Other         | 1st Division Women, Deildarkeppni                                           | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Other         | Hungary Divison A Women                                                     | 1                  | 1              | 0              | 0                | 0                |      |
| **TOTAL** | **85**  | **85**    | **67**    | **18**      | **79 %**      | **59**                                                                      | **47**             | **12**         | **80 %**       | **🟢**           |                  |      |

> Nota: la simulación secuencial con $100 por apuesta se integra en el barrido de umbrales de la sección 8 (columnas de bankroll final). El umbral operativo es 0.75.


## 8. Barrido de umbrales sobre las apuestas en vivo (m27_v5_3 vs clon)

> Simula qué habría pasado si se hubiera operado con distintos umbrales de confianza sobre los mismos partidos en vivo. Para el clon se aplica también el guard de margen (margin ≥ 0). Bank inicial $1,000, $100 por apuesta.

### Momio 1.40

| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR   | O:ROI    | C:apuestas | C:WR    | C:ROI    | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |
| ---    | ---:       | ---:      | ---:      | ---:       | ---:       | ---:   | ---:     | ---:       | ---:    | ---:     | ---:       | ---:       | ---:         | ---:         | ---:          |
| 0.60   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +10.35 % | 85         | 55.3 %  | -22.59 % | $+880.00   | $-920.00 * | $1,880.00    | $80.00       | 🔴            |
| 0.62   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +10.35 % | 85         | 55.3 %  | -22.59 % | $+880.00   | $-920.00 * | $1,880.00    | $80.00       | 🔴            |
| 0.65   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +10.35 % | 85         | 55.3 %  | -22.59 % | $+880.00   | $-920.00 * | $1,880.00    | $80.00       | 🔴            |
| 0.68   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +10.35 % | 85         | 55.3 %  | -22.59 % | $+880.00   | $-920.00 * | $1,880.00    | $80.00       | 🔴            |
| 0.70   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +10.35 % | 85         | 55.3 %  | -22.59 % | $+880.00   | $-920.00 * | $1,880.00    | $80.00       | 🔴            |
| 0.72   | 85         | 67        | 47        | 18         | 31         | 78.8 % | +10.35 % | 78         | 60.3 %  | -15.64 % | $+880.00   | $-940.00 * | $1,880.00    | $60.00       | 🔴            |
| 0.75   | 73         | 60        | 47        | 13         | 12         | 82.2 % | +15.07 % | 59         | 79.7 %  | +11.53 % | $+1,100.00 | $+680.00   | $2,100.00    | $1,680.00    | 🟢            |
| 0.78   | 36         | 29        | 31        | 7          | 6          | 80.6 % | +12.78 % | 37         | 83.8 %  | +17.30 % | $+460.00   | $+640.00   | $1,460.00    | $1,640.00    | 🟢            |
| 0.80   | 26         | 22        | 23        | 4          | 3          | 84.6 % | +18.46 % | 26         | 88.5 %  | +23.85 % | $+480.00   | $+620.00   | $1,480.00    | $1,620.00    | 🟢            |
| 0.85   | 12         | 11        | 10        | 1          | 0          | 91.7 % | +28.33 % | 10         | 100.0 % | +40.00 % | $+340.00   | $+400.00   | $1,340.00    | $1,400.00    | 🟢            |
> * = se detuvo por falta de fondos.


### Momio 1.83

| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR   | O:ROI    | C:apuestas | C:WR    | C:ROI    | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |
| ---    | ---:       | ---:      | ---:      | ---:       | ---:       | ---:   | ---:     | ---:       | ---:    | ---:     | ---:       | ---:       | ---:         | ---:         | ---:          |
| 0.60   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +44.25 % | 85         | 55.3 %  | +1.19 %  | $+3,761.00 | $-919.00 * | $4,761.00    | $81.00       | 🔴            |
| 0.62   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +44.25 % | 85         | 55.3 %  | +1.19 %  | $+3,761.00 | $-919.00 * | $4,761.00    | $81.00       | 🔴            |
| 0.65   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +44.25 % | 85         | 55.3 %  | +1.19 %  | $+3,761.00 | $-919.00 * | $4,761.00    | $81.00       | 🔴            |
| 0.68   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +44.25 % | 85         | 55.3 %  | +1.19 %  | $+3,761.00 | $-919.00 * | $4,761.00    | $81.00       | 🔴            |
| 0.70   | 85         | 67        | 47        | 18         | 38         | 78.8 % | +44.25 % | 85         | 55.3 %  | +1.19 %  | $+3,761.00 | $-919.00 * | $4,761.00    | $81.00       | 🔴            |
| 0.72   | 85         | 67        | 47        | 18         | 31         | 78.8 % | +44.25 % | 78         | 60.3 %  | +10.27 % | $+3,761.00 | $-904.00 * | $4,761.00    | $96.00       | 🔴            |
| 0.75   | 73         | 60        | 47        | 13         | 12         | 82.2 % | +50.41 % | 59         | 79.7 %  | +45.78 % | $+3,680.00 | $+2,701.00 | $4,680.00    | $3,701.00    | 🟢            |
| 0.78   | 36         | 29        | 31        | 7          | 6          | 80.6 % | +47.42 % | 37         | 83.8 %  | +53.32 % | $+1,707.00 | $+1,973.00 | $2,707.00    | $2,973.00    | 🟢            |
| 0.80   | 26         | 22        | 23        | 4          | 3          | 84.6 % | +54.85 % | 26         | 88.5 %  | +61.88 % | $+1,426.00 | $+1,609.00 | $2,426.00    | $2,609.00    | 🟢            |
| 0.85   | 12         | 11        | 10        | 1          | 0          | 91.7 % | +67.75 % | 10         | 100.0 % | +83.00 % | $+813.00   | $+830.00   | $1,813.00    | $1,830.00    | 🟢            |
> * = se detuvo por falta de fondos.


**Lectura:** el clon es más conservador (59 vs 85 apuestas) con acierto ligeramente superior (79.7 % vs 78.8 %). Diferencia pequeña, muestra diminuta. No se justifica cambio de modelo solo con esta evidencia.

## 9. Conclusiones

1. La inclusión de las 2 ligas polacas (Runda Zasadnicza) **no cambia materialmente** las métricas globales (AUC 0.6544 vs 0.6545, +1906 partidos elegibles).
2. En vivo, el clon es más conservador y ligeramente más preciso, pero la muestra es diminuta (94 partidos).
3. El caso polaco es prometedor: evitó las 2 apuestas perdedoras (2/2 = 100 % en 4 partidos).
4. **Recomendación:** mantener `m27_v5_3` en producción (modo sombra) y seguir acumulando evidencia. El clon puede re-evaluarse cuando haya ≥ 200 apuestas liquidadas.

## 10. Notas metodológicas

- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis_clone.py`.
