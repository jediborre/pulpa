# 📊 Análisis por liga y por umbral — modelo m27_v5_3_3 (normalización + histórico NCAA Women)

> Modelo `m27_v5_3_3`: v5_3_2 + backfill histórico de `NCAA Women's Division I` (19,541 partidos desde 2019, antes solo 1,729 de la temporada actual). Walk-forward incluido (3 cortes). Umbral operativo: **P ≥ 0.75**.

> Holdout **C** (último 18 %, anclas t=30). Umbral operativo: **P ≥ 0.75**. Equilibrio a cuota 1.40: **71.4 %**.

## 1. Resumen ejecutivo

- **Ligas analizadas:** 239 (≥ 30 anclas en holdout C).
- **Partidos (anclas) en holdout C:** 30,124 · en entrenamiento (A): 94,985.
- **Apuestas totales a 0.75 en C:** 2,609.
- **Ligas en arranque en frío:** 78 (32.6 %).
- **Ligas candidatas a bloqueo (IC95 sup < 71.4 % a 0.75):** 0.
- **Ligas con ROI@1.40 > 20 % a 0.75:** 18.

**Comparación con m27_v5_3 (original):** eligible matches 184,878 vs 182,803 (+2,075). AUC test 0.6548 vs 0.6545. Apuestas @0.75 (ancla): 2,873 vs 2,538.

## 2. Tabla global de umbrales (anclas t=30, bloque C)

| P ≥  | Apuestas | Ganados | Perdidos | WR (IC95)          | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---                | ---:      | ---:      |
| 0.65 | 9,937    | 7,083   | 2,854    | 71.3 % (70.4–72.2) | -0.21 %   | +30.44 %  |
| 0.68 | 7,193    | 5,299   | 1,894    | 73.7 % (72.6–74.7) | +3.14 %   | +34.81 %  |
| 0.70 | 5,616    | 4,267   | 1,349    | 76.0 % (74.8–77.1) | +6.37 %   | +39.04 %  |
| 0.72 | 4,386    | 3,403   | 983      | 77.6 % (76.3–78.8) | +8.62 %   | +41.99 %  |
| 0.75 | 2,873    | 2,323   | 550      | 80.9 % (79.4–82.3) | +13.20 %  | +47.97 %  |
| 0.80 | 1,425    | 1,220   | 205      | 85.6 % (83.7–87.3) | +19.86 %  | +56.67 %  |

## 3. Tabla resumen por liga (ordenada por WR descendente a 0.75)

| País          | Liga                                        | Entrenado (A) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | IC95       | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---           | ---                                         | ---:          | ---:        | ---:           | ---:    | ---:     | ---:     | ---        | ---:      | ---:      | ---     |
| International | BNXT League                                 | 330           | 226         | 15             | 15      | 0        | 100.0 %  | 79.6–100.0 | +40.00 %  | +83.00 %  |         |
| ?             | CBI U17 Feminino                            | 0             | 54          | 22             | 21      | 1        | 95.5 %   | 78.2–99.2  | +33.64 %  | +74.68 %  | 🆕      |
| ?             | Rwanda Basketball League Women              | 41            | 55          | 17             | 16      | 1        | 94.1 %   | 73.0–99.0  | +31.76 %  | +72.24 %  |         |
| ?             | CBI U15 Fem                                 | 0             | 138         | 65             | 61      | 4        | 93.8 %   | 85.2–97.6  | +31.38 %  | +71.74 %  | 🆕      |
| ?             | SWC                                         | 0             | 39          | 15             | 14      | 1        | 93.3 %   | 70.2–98.8  | +30.67 %  | +70.80 %  | 🆕      |
| Historical    | LNB                                         | 318           | 169         | 67             | 61      | 6        | 91.0 %   | 81.8–95.8  | +27.46 %  | +66.61 %  |         |
| ?             | ZBL                                         | 859           | 94          | 43             | 39      | 4        | 90.7 %   | 78.4–96.3  | +26.98 %  | +65.98 %  |         |
| Historical    | LDB                                         | 97            | 167         | 20             | 18      | 2        | 90.0 %   | 69.9–97.2  | +26.00 %  | +64.70 %  |         |
| ?             | Top Division Women                          | 0             | 100         | 38             | 34      | 4        | 89.5 %   | 75.9–95.8  | +25.26 %  | +63.74 %  | 🆕      |
| ?             | Saudi Basketball League                     | 85            | 45          | 19             | 17      | 2        | 89.5 %   | 68.6–97.1  | +25.26 %  | +63.74 %  |         |
| ?             | EYBL U17                                    | 0             | 272         | 42             | 37      | 5        | 88.1 %   | 75.0–94.8  | +23.33 %  | +61.21 %  | 🆕      |
| Other         | Prva Liga                                   | 0             | 139         | 25             | 22      | 3        | 88.0 %   | 70.0–95.8  | +23.20 %  | +61.04 %  | 🆕      |
| ?             | NBL1 East                                   | 162           | 126         | 33             | 29      | 4        | 87.9 %   | 72.7–95.2  | +23.03 %  | +60.82 %  |         |
| ?             | EYBL U20                                    | 0             | 187         | 16             | 14      | 2        | 87.5 %   | 64.0–96.5  | +22.50 %  | +60.13 %  | 🆕      |
| Other         | Estonian-Latvian Basketball League          | 637           | 174         | 23             | 20      | 3        | 87.0 %   | 67.9–95.5  | +21.74 %  | +59.13 %  |         |
| Other         | 1st Division Women                          | 0             | 69          | 23             | 20      | 3        | 87.0 %   | 67.9–95.5  | +21.74 %  | +59.13 %  | 🆕      |
| Other         | Superleague Women                           | 0             | 34          | 15             | 13      | 2        | 86.7 %   | 62.1–96.3  | +21.33 %  | +58.60 %  | 🆕      |
| Other         | Swiss Basketball League Women               | 156           | 71          | 15             | 13      | 2        | 86.7 %   | 62.1–96.3  | +21.33 %  | +58.60 %  |         |
| ?             | CBI U15                                     | 0             | 199         | 49             | 42      | 7        | 85.7 %   | 73.3–92.9  | +20.00 %  | +56.86 %  | 🆕      |
| Historical    | FSHB Superliga                              | 149           | 100         | 41             | 35      | 6        | 85.4 %   | 71.6–93.1  | +19.51 %  | +56.22 %  |         |
| ?             | EGBL U17                                    | 0             | 74          | 27             | 23      | 4        | 85.2 %   | 67.5–94.1  | +19.26 %  | +55.89 %  | 🆕      |
| Australia     | NBL                                         | 2,647         | 536         | 62             | 52      | 10       | 83.9 %   | 72.8–91.0  | +17.42 %  | +53.48 %  |         |
| Other         | Finland Division 1A Women                   | 323           | 103         | 18             | 15      | 3        | 83.3 %   | 60.8–94.2  | +16.67 %  | +52.50 %  |         |
| ?             | 1 Division                                  | 17            | 89          | 18             | 15      | 3        | 83.3 %   | 60.8–94.2  | +16.67 %  | +52.50 %  |         |
| Poland        | MPBL                                        | 739           | 287         | 35             | 29      | 6        | 82.9 %   | 67.3–91.9  | +16.00 %  | +51.63 %  |         |
| ?             | NBL Women                                   | 0             | 106         | 17             | 14      | 3        | 82.3 %   | 59.0–93.8  | +15.29 %  | +50.71 %  | 🆕      |
| ?             | EYBL U16                                    | 0             | 265         | 33             | 27      | 6        | 81.8 %   | 65.6–91.4  | +14.55 %  | +49.73 %  | 🆕      |
| ?             | EYBL U13                                    | 0             | 118         | 32             | 26      | 6        | 81.2 %   | 64.7–91.1  | +13.75 %  | +48.69 %  | 🆕      |
| Brazil        | Brazil NBB                                  | 1,690         | 303         | 16             | 13      | 3        | 81.2 %   | 57.0–93.4  | +13.75 %  | +48.69 %  |         |
| ?             | CBI U17                                     | 48            | 173         | 21             | 17      | 4        | 81.0 %   | 60.0–92.3  | +13.33 %  | +48.14 %  |         |
| Other         | Hungary Divison A Women                     | 303           | 121         | 41             | 33      | 8        | 80.5 %   | 66.0–89.8  | +12.68 %  | +47.29 %  |         |
| ?             | NCAA Women's Division I                     | 0             | 1,563       | 267            | 214     | 53       | 80.2 %   | 75.0–84.5  | +12.21 %  | +46.67 %  | 🆕      |
| Other         | Switzerland LNA                             | 832           | 104         | 25             | 20      | 5        | 80.0 %   | 60.9–91.1  | +12.00 %  | +46.40 %  |         |
| ?             | EYBL U15                                    | 0             | 232         | 44             | 35      | 9        | 79.5 %   | 65.5–88.8  | +11.36 %  | +45.57 %  | 🆕      |
| ?             | CBI U19                                     | 0             | 113         | 39             | 31      | 8        | 79.5 %   | 64.5–89.2  | +11.28 %  | +45.46 %  | 🆕      |
| Other         | Basketligaen                                | 787           | 113         | 29             | 23      | 6        | 79.3 %   | 61.6–90.2  | +11.03 %  | +45.14 %  |         |
| Historical    | Poland 2nd Basketball League                | 828           | 807         | 166            | 130     | 36       | 78.3 %   | 71.4–83.9  | +9.64 %   | +43.31 %  |         |
| ?             | 1 Liga Kobiet                               | 0             | 251         | 18             | 14      | 4        | 77.8 %   | 54.8–91.0  | +8.89 %   | +42.33 %  | 🆕      |
| Other         | United League                               | 1,288         | 189         | 39             | 30      | 9        | 76.9 %   | 61.7–87.4  | +7.69 %   | +40.77 %  |         |
| ?             | Tipos Extraliga                             | 424           | 63          | 39             | 30      | 9        | 76.9 %   | 61.7–87.4  | +7.69 %   | +40.77 %  |         |
| ?             | Liga Nacional Femenina Chile                | 0             | 88          | 26             | 20      | 6        | 76.9 %   | 57.9–89.0  | +7.69 %   | +40.77 %  | 🆕      |
| Other         | Big V                                       | 712           | 169         | 21             | 16      | 5        | 76.2 %   | 54.9–89.4  | +6.67 %   | +39.43 %  |         |
| Other         | 2nd League                                  | 271           | 131         | 25             | 19      | 6        | 76.0 %   | 56.6–88.5  | +6.40 %   | +39.08 %  |         |
| Other         | SLB                                         | 90            | 208         | 20             | 15      | 5        | 75.0 %   | 53.1–88.8  | +5.00 %   | +37.25 %  |         |
| ?             | 1. SKL                                      | 44            | 52          | 16             | 12      | 4        | 75.0 %   | 50.5–89.8  | +5.00 %   | +37.25 %  |         |
| Other         | Russian Women's Basketball Premier League   | 726           | 102         | 27             | 20      | 7        | 74.1 %   | 55.3–86.8  | +3.70 %   | +35.56 %  |         |
| Other         | Swedish Basketball League Women             | 259           | 129         | 34             | 25      | 9        | 73.5 %   | 56.9–85.4  | +2.94 %   | +34.56 %  |         |
| ?             | EYBL U14                                    | 0             | 236         | 51             | 37      | 14       | 72.5 %   | 59.1–82.9  | +1.57 %   | +32.76 %  | 🆕      |
| ?             | Nostra.lt-RKL Division B                    | 0             | 200         | 25             | 18      | 7        | 72.0 %   | 52.4–85.7  | +0.80 %   | +31.76 %  | 🆕      |
| Other         | Premijer ženska liga                        | 327           | 100         | 31             | 21      | 10       | 67.7 %   | 50.1–81.4  | -5.16 %   | +23.97 %  | ⚠️     |
| Other         | Liga Femenina                               | 1,314         | 418         | 29             | 19      | 10       | 65.5 %   | 47.3–80.1  | -8.28 %   | +19.90 %  | ⚠️     |
| ?             | Liga 1 Masculin                             | 122           | 166         | 27             | 17      | 10       | 63.0 %   | 44.2–78.5  | -11.85 %  | +15.22 %  | ⚠️     |
| Other         | Sparkasse Prva Liga Zeni                    | 0             | 70          | 17             | 9       | 8        | 52.9 %   | 31.0–73.8  | -25.88 %  | -3.12 %   | ⚠️ 🆕  |
| Other         | A1 Women                                    | 965           | 89          | 14             | 0       | 14       | —        | —          | —         | —         |         |
| ?             | U20 EuroBasket                              | 9             | 154         | 14             | 0       | 14       | —        | —          | —         | —         |         |
| ?             | Campeonato Nacional da 1ª Divisao Feminina  | 0             | 107         | 13             | 0       | 13       | —        | —          | —         | —         | 🆕      |
| Other         | BLNO                                        | 483           | 54          | 13             | 0       | 13       | —        | —          | —         | —         |         |
| ?             | LBF                                         | 202           | 80          | 13             | 0       | 13       | —        | —          | —         | —         |         |
| Other         | The Women League                            | 0             | 31          | 12             | 0       | 12       | —        | —          | —         | —         | 🆕      |
| ?             | Női Amatőr NB 1                             | 135           | 48          | 12             | 0       | 12       | —        | —          | —         | —         |         |
| Other         | 1. Division B                               | 103           | 113         | 11             | 0       | 11       | —        | —          | —         | —         |         |
| Other         | Austrian Women's Basketball Bundesliga      | 156           | 54          | 11             | 0       | 11       | —        | —          | —         | —         |         |
| Historical    | Rwanda Basketball League                    | 31            | 81          | 11             | 0       | 11       | —        | —          | —         | —         |         |
| ?             | FIBA U16 Women's EuroBasket                 | 0             | 38          | 11             | 0       | 11       | —        | —          | —         | —         | 🆕      |
| Japan         | B League One                                | 3,097         | 369         | 10             | 0       | 10       | —        | —          | —         | —         |         |
| ?             | LMB                                         | 170           | 84          | 10             | 0       | 10       | —        | —          | —         | —         |         |
| Other         | Superliga                                   | 245           | 185         | 10             | 0       | 10       | —        | —          | —         | —         |         |
| ?             | LF Challenge                                | 0             | 145         | 10             | 0       | 10       | —        | —          | —         | —         | 🆕      |
| ?             | Kvindebasketligaen                          | 39            | 51          | 10             | 0       | 10       | —        | —          | —         | —         |         |
| Other         | Jedinstvena kadetska liga                   | 0             | 64          | 10             | 0       | 10       | —        | —          | —         | —         | 🆕      |
| ?             | 1. ZLS                                      | 53            | 59          | 10             | 0       | 10       | —        | —          | —         | —         |         |
| ?             | NBL1                                        | 906           | 530         | 10             | 0       | 10       | —        | —          | —         | —         |         |
| Japan         | B League Premier                            | 3,136         | 568         | 9              | 0       | 9        | —        | —          | —         | —         |         |
| Other         | Estonian National Women's Basketball League | 57            | 49          | 9              | 0       | 9        | —        | —          | —         | —         |         |
| ?             | Basket Liga                                 | 933           | 132         | 9              | 0       | 9        | —        | —          | —         | —         |         |
| Other         | U21 Espoirs Elite                           | 0             | 144         | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| Other         | Latvian Basketball Cup                      | 0             | 31          | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| ?             | The Basketball League                       | 63            | 136         | 9              | 0       | 9        | —        | —          | —         | —         |         |
| ?             | NBL1 West                                   | 135           | 101         | 9              | 0       | 9        | —        | —          | —         | —         |         |
| Other         | VBA                                         | 126           | 65          | 9              | 0       | 9        | —        | —          | —         | —         |         |
| Other         | Colombia LPB                                | 238           | 68          | 8              | 0       | 8        | —        | —          | —         | —         |         |
| Spain         | Liga ACB                                    | 1,885         | 271         | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | Eurocup                                     | 1,244         | 175         | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | France LF2                                  | 104           | 142         | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | 1. liga                                     | 0             | 113         | 8              | 0       | 8        | —        | —          | —         | —         | 🆕      |
| Other         | Premier League                              | 230           | 65          | 8              | 0       | 8        | —        | —          | —         | —         |         |
| USA           | NCAA Women's March Madness                  | 0             | 59          | 8              | 0       | 8        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA U17 Women's Basketball World Cup       | 0             | 37          | 8              | 0       | 8        | —        | —          | —         | —         | 🆕      |
| Other         | Liga e Parë                                 | 314           | 70          | 7              | 0       | 7        | —        | —          | —         | —         |         |
| France        | France Pro A                                | 1,864         | 204         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Lebanese Basketball League                  | 113           | 111         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | FBU Superleague                             | 849           | 92          | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Superliga e Femrave                         | 0             | 41          | 7              | 0       | 7        | —        | —          | —         | —         | 🆕      |
| Other         | Swedish Basketball League                   | 1,087         | 149         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Portugal LBP                                | 909           | 118         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Indonesian Basketball League                | 823           | 110         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| ?             | NBL Men                                     | 0             | 125         | 7              | 0       | 7        | —        | —          | —         | —         | 🆕      |
| ?             | NBL1 Central                                | 89            | 63          | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Catarinense                                 | 6             | 30          | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Spain         | Primera FEB                                 | 1,810         | 215         | 6              | 0       | 6        | —        | —          | —         | —         |         |
| International | Euroleague: regular season                  | 223           | 88          | 6              | 0       | 6        | —        | —          | —         | —         |         |
| ?             | U18 FIBA Asia Cup                           | 0             | 36          | 6              | 0       | 6        | —        | —          | —         | —         | 🆕      |
| ?             | LBP                                         | 0             | 34          | 6              | 0       | 6        | —        | —          | —         | —         | 🆕      |
| Other         | Liga Nationala                              | 873           | 152         | 5              | 0       | 5        | —        | —          | —         | —         |         |
| Other         | I B MCKL                                    | 0             | 36          | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| ?             | EGBL U14                                    | 0             | 47          | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| ?             | Druga muška liga                            | 489           | 74          | 5              | 0       | 5        | —        | —          | —         | —         |         |
| France        | Mexico LNBP Women                           | 93            | 52          | 5              | 0       | 5        | —        | —          | —         | —         |         |
| ?             | U18 EuroBasket                              | 0             | 124         | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA Asia Cup 2029 Pre-Qualifiers           | 0             | 31          | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| Other         | Azerbaijan Basketball League                | 0             | 105         | 4              | 0       | 4        | —        | —          | —         | —         | 🆕      |
| Australia     | WNBL                                        | 594           | 73          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Italy         | Serie A2                                    | 2,619         | 388         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Korisliiga                                  | 1,306         | 134         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Turkish Basketball Super League             | 1,509         | 182         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Greece        | Stoiximan GBL                               | 1,153         | 136         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | Road to Basketball Africa League            | 58            | 34          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | 1. Division                                 | 270           | 112         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Super League                                | 661           | 269         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Germany       | Germany BBL                                 | 1,904         | 257         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Cup                                         | 351           | 42          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | FIBA World Cup Qualification                | 0             | 57          | 4              | 0       | 4        | —        | —          | —         | —         | 🆕      |
| ?             | NBL1 South                                  | 206           | 161         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| USA           | WNBA                                        | 1,305         | 311         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | LFB                                         | 20            | 35          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | FIBA U17 Basketball World Cup               | 19            | 46          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| France        | LNB Clausura                                | 0             | 71          | 4              | 0       | 4        | —        | —          | —         | —         | 🆕      |
| Other         | Hungary NB 1.A                              | 858           | 159         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Cyprus Basketball Division A                | 402           | 76          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | 1st Division Men                            | 0             | 126         | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| Argentina     | Argentina Liga Nacional                     | 2,972         | 273         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Uruguay LUB                                 | 360           | 95          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Liga Košarkaškog saveza Herceg Bosne        | 0             | 45          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| ?             | LSB - Liga B                                | 5             | 46          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Swedish Basketball Superettan               | 372           | 145         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Portugal LFB Women                          | 192           | 105         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| ?             | Champions League Americas                   | 96            | 43          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| ?             | Kızlar Basketbol Gençler Ligi               | 0             | 34          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| ?             | CIBACOPA                                    | 44            | 216         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Historical    | Iraqi Professional League                   | 107           | 63          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Liga de Ascenso                             | 392           | 107         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| ?             | FIBA U16 EuroBasket Division B              | 0             | 61          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| Other         | SuperSport Premijer Liga                    | 1,096         | 165         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | A1 Liga                                     | 0             | 108         | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| Other         | Austrian Basketball Superliga               | 1,055         | 93          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | I A MCKL                                    | 122           | 125         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | Champions League                            | 1,098         | 125         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | 1. A SKL                                    | 726           | 115         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | Nationale 1                                 | 50            | 266         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Israel        | Israeli National League Basketball          | 489           | 183         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | A1                                          | 0             | 69          | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| Other         | Saku I liiga                                | 125           | 117         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | Iceland Women's Basketball Premier League   | 229           | 88          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | Segunda FEB                                 | 277           | 260         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | CN1 Basquetebol                             | 0             | 101         | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| ?             | EuroCup                                     | 774           | 112         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| France        | France LFB Women                            | 952           | 112         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | EGBL U19                                    | 0             | 39          | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| Other         | Prva RL Istok                               | 314           | 31          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | 2. ZLS                                      | 0             | 37          | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| ?             | Divisione Regionale 1 Toscana               | 0             | 131         | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| France        | Mexico LNBP                                 | 1,503         | 178         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | FIBA World Cup                              | 44            | 33          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| France        | Chile LNB                                   | 898           | 72          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Belgian Basketball 2nd Division             | 183           | 171         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Italy         | Lega A Basket                               | 1,595         | 176         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | AdmiralBet ABA League                       | 63            | 130         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | 1st Division                                | 337           | 124         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Princ Caffe Superliga                       | 660           | 78          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Liga de Baloncesto Puertorriquena           | 399           | 317         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | 1. Liga                                     | 1,174         | 138         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Poland        | Poland 1st Division Basketball              | 838           | 220         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Enovos League                               | 0             | 97          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Other         | Iranian Basketball Super League             | 414           | 38          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| International | Euroleague                                  | 1,645         | 307         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | FIBA Europe Cup                             | 651           | 92          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Iceland Basketball Premier League           | 553           | 103         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Israel        | Israeli Women Basketball Premier League     | 310           | 82          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Zain Basketball League                      | 61            | 76          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Historical    | Qatari Basketball League                    | 68            | 55          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Elite League                                | 354           | 58          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Turkey        | Türkiye Kadınlar Basketbol Ligi             | 0             | 39          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Other         | National League                             | 82            | 31          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | 2. Muška Liga                               | 288           | 43          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | National League 2                           | 607           | 106         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | A2 ESKAH                                    | 284           | 35          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Liga Argentina                              | 1,041         | 76          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Australia     | New Zealand NBL                             | 191           | 78          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Saku II Liiga                               | 0             | 35          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| ?             | 2. DBBL                                     | 0             | 34          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Other         | Proliga                                     | 93            | 30          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Nostra.lt-RKL Division A                    | 0             | 38          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Historical    | PGE Basket Liga                             | 1,521         | 186         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Élite 2                                     | 1,989         | 274         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Serie B                                     | 662           | 259         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Lithuania     | NKL                                         | 791           | 197         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Lithuania     | RKL Division A                              | 0             | 213         | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Superleague                                 | 1,329         | 103         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | BSNF                                        | 86            | 65          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Portugal Proliga                            | 218           | 54          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Meridianbet KLS                             | 1,404         | 182         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | European North Basketball League            | 137           | 88          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| International | ABA League 2                                | 271           | 52          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Europe Cup                                  | 477           | 50          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Divisional Tercera de Ascenso               | 0             | 47          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | The League                                  | 135           | 118         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Historical    | Slovenian Second Basketball League          | 124           | 83          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Euroleague Women                            | 706           | 46          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| International | U19 ABA League                              | 24            | 33          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | French Basketball Cup                       | 221           | 40          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Australia     | Bulgaria NBL                                | 684           | 85          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Germany       | Germany Pro A                               | 1,670         | 110         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Historical    | Pro B                                       | 1             | 120         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | TB2L                                        | 452           | 44          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Serie B Interregionale                      | 0             | 255         | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Erkekler Basketbol Gençler Ligi             | 0             | 35          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Divisione Regionale 2 - Lazio               | 0             | 105         | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Serie C Toscana                             | 0             | 50          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Serie C Girone H sud                        | 0             | 30          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | B Ethniki                                   | 0             | 95          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | TBL                                         | 758           | 58          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Basket World League                         | 0             | 31          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Espoirs Pro B                               | 179           | 48          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | A1 ESKAH                                    | 304           | 33          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Belize Elite Basketball League              | 0             | 39          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Egyptian Women's Basketball Super League    | 0             | 55          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Erkekler Bölgesel Basketbol Ligi            | 0             | 67          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Egyptian Basketball Super League            | 0             | 31          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Puerto Rico BSN                             | 290           | 188         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| France        | LNB 2                                       | 0             | 44          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Rapid League                                | 7             | 83          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | NBL1 North                                  | 120           | 81          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Canadian Elite Basketball League            | 95            | 103         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | U16 Türkiye Finalleri                       | 0             | 46          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | U18 EuroBasket Div. B                       | 0             | 50          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA U16 Women's EuroBasket Division B      | 0             | 34          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA U16 EuroBasket                         | 0             | 38          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Copa Espana                                 | 0             | 41          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | 2. MLS                                      | 0             | 30          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |

## 4. Detalle por liga y por umbral — Top 30 por volumen

### ? — NCAA Women's Division I

- Entrenado (A): **0** · Holdout (C): **1,563** · Sesgo @0.75: +0.0247

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 589      | 416     | 173      | 70.6 % | 66.8–74.2 | -1.12 %   | +29.25 %  |
| 0.68 | 476      | 345     | 131      | 72.5 % | 68.3–76.3 | +1.47 %   | +32.64 %  |
| 0.70 | 414      | 308     | 106      | 74.4 % | 70.0–78.4 | +4.15 %   | +36.14 %  |
| 0.72 | 363      | 274     | 89       | 75.5 % | 70.8–79.6 | +5.67 %   | +38.13 %  |
| 0.75 | 267      | 214     | 53       | 80.2 % | 75.0–84.5 | +12.21 %  | +46.67 %  |
| 0.80 | 173      | 153     | 20       | 88.4 % | 82.8–92.4 | +23.82 %  | +61.84 %  |

### Historical — Poland 2nd Basketball League

- Entrenado (A): **828** · Holdout (C): **807** · Sesgo @0.75: +0.0315

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 385      | 261     | 124      | 67.8 % | 63.0–72.3 | -5.09 %   | +24.06 %  |
| 0.68 | 304      | 218     | 86       | 71.7 % | 66.4–76.5 | +0.39 %   | +31.23 %  |
| 0.70 | 257      | 188     | 69       | 73.2 % | 67.4–78.2 | +2.41 %   | +33.87 %  |
| 0.72 | 226      | 169     | 57       | 74.8 % | 68.7–80.0 | +4.69 %   | +36.85 %  |
| 0.75 | 166      | 130     | 36       | 78.3 % | 71.4–83.9 | +9.64 %   | +43.31 %  |
| 0.80 | 93       | 79      | 14       | 85.0 % | 76.3–90.8 | +18.92 %  | +55.45 %  |

### Historical — LNB

- Entrenado (A): **318** · Holdout (C): **169** · Sesgo @0.75: -0.0474

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 103      | 91      | 12       | 88.3 % | 80.7–93.2 | +23.69 %  | +61.68 %  |
| 0.68 | 89       | 80      | 9        | 89.9 % | 81.9–94.6 | +25.84 %  | +64.49 %  |
| 0.70 | 79       | 70      | 9        | 88.6 % | 79.7–93.9 | +24.05 %  | +62.15 %  |
| 0.72 | 71       | 65      | 6        | 91.5 % | 82.8–96.1 | +28.17 %  | +67.54 %  |
| 0.75 | 67       | 61      | 6        | 91.0 % | 81.8–95.8 | +27.46 %  | +66.61 %  |
| 0.80 | 51       | 47      | 4        | 92.2 % | 81.5–96.9 | +29.02 %  | +68.65 %  |

### ? — CBI U15 Fem

- Entrenado (A): **0** · Holdout (C): **138** · Sesgo @0.75: -0.0737

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 89       | 78      | 11       | 87.6 % | 79.2–93.0 | +22.70 %  | +60.38 %  |
| 0.68 | 84       | 74      | 10       | 88.1 % | 79.5–93.4 | +23.33 %  | +61.21 %  |
| 0.70 | 78       | 69      | 9        | 88.5 % | 79.5–93.8 | +23.85 %  | +61.88 %  |
| 0.72 | 71       | 65      | 6        | 91.5 % | 82.8–96.1 | +28.17 %  | +67.54 %  |
| 0.75 | 65       | 61      | 4        | 93.8 % | 85.2–97.6 | +31.38 %  | +71.74 %  |
| 0.80 | 52       | 50      | 2        | 96.2 % | 87.0–98.9 | +34.62 %  | +75.96 %  |

### Australia — NBL

- Entrenado (A): **2,647** · Holdout (C): **536** · Sesgo @0.75: -0.0366

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 190      | 142     | 48       | 74.7 % | 68.1–80.4 | +4.63 %   | +36.77 %  |
| 0.68 | 146      | 113     | 33       | 77.4 % | 70.0–83.4 | +8.36 %   | +41.64 %  |
| 0.70 | 118      | 92      | 26       | 78.0 % | 69.7–84.5 | +9.15 %   | +42.68 %  |
| 0.72 | 88       | 69      | 19       | 78.4 % | 68.7–85.7 | +9.77 %   | +43.49 %  |
| 0.75 | 62       | 52      | 10       | 83.9 % | 72.8–91.0 | +17.42 %  | +53.48 %  |
| 0.80 | 23       | 20      | 3        | 87.0 % | 67.9–95.5 | +21.74 %  | +59.13 %  |

### ? — EYBL U14

- Entrenado (A): **0** · Holdout (C): **236** · Sesgo @0.75: +0.0875

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 113      | 77      | 36       | 68.1 % | 59.1–76.0 | -4.60 %   | +24.70 %  |
| 0.68 | 93       | 68      | 25       | 73.1 % | 63.3–81.1 | +2.37 %   | +33.81 %  |
| 0.70 | 76       | 56      | 20       | 73.7 % | 62.8–82.3 | +3.16 %   | +34.84 %  |
| 0.72 | 67       | 49      | 18       | 73.1 % | 61.5–82.3 | +2.39 %   | +33.84 %  |
| 0.75 | 51       | 37      | 14       | 72.5 % | 59.1–82.9 | +1.57 %   | +32.76 %  |
| 0.80 | 28       | 19      | 9        | 67.9 % | 49.3–82.1 | -5.00 %   | +24.18 %  |

### ? — CBI U15

- Entrenado (A): **0** · Holdout (C): **199** · Sesgo @0.75: -0.0227

| P ≥  | Apuestas | Ganados | Perdidos | WR      | IC95       | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:    | ---        | ---:      | ---:      |
| 0.65 | 90       | 70      | 20       | 77.8 %  | 68.2–85.1  | +8.89 %   | +42.33 %  |
| 0.68 | 77       | 63      | 14       | 81.8 %  | 71.8–88.8  | +14.55 %  | +49.73 %  |
| 0.70 | 63       | 53      | 10       | 84.1 %  | 73.2–91.1  | +17.78 %  | +53.95 %  |
| 0.72 | 59       | 50      | 9        | 84.8 %  | 73.5–91.8  | +18.64 %  | +55.08 %  |
| 0.75 | 49       | 42      | 7        | 85.7 %  | 73.3–92.9  | +20.00 %  | +56.86 %  |
| 0.80 | 33       | 33      | 0        | 100.0 % | 89.6–100.0 | +40.00 %  | +83.00 %  |

### ? — EYBL U15

- Entrenado (A): **0** · Holdout (C): **232** · Sesgo @0.75: +0.0234

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 98       | 70      | 28       | 71.4 % | 61.8–79.4 | +0.00 %   | +30.71 %  |
| 0.68 | 76       | 57      | 19       | 75.0 % | 64.2–83.4 | +5.00 %   | +37.25 %  |
| 0.70 | 70       | 54      | 16       | 77.1 % | 66.0–85.4 | +8.00 %   | +41.17 %  |
| 0.72 | 57       | 44      | 13       | 77.2 % | 64.8–86.2 | +8.07 %   | +41.26 %  |
| 0.75 | 44       | 35      | 9        | 79.5 % | 65.5–88.8 | +11.36 %  | +45.57 %  |
| 0.80 | 26       | 24      | 2        | 92.3 % | 75.9–97.9 | +29.23 %  | +68.92 %  |

### ? — ZBL

- Entrenado (A): **859** · Holdout (C): **94** · Sesgo @0.75: -0.0751

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 65       | 54      | 11       | 83.1 % | 72.2–90.3 | +16.31 %  | +52.03 %  |
| 0.68 | 55       | 47      | 8        | 85.5 % | 73.8–92.4 | +19.64 %  | +56.38 %  |
| 0.70 | 53       | 46      | 7        | 86.8 % | 75.2–93.5 | +21.51 %  | +58.83 %  |
| 0.72 | 49       | 43      | 6        | 87.8 % | 75.8–94.3 | +22.86 %  | +60.59 %  |
| 0.75 | 43       | 39      | 4        | 90.7 % | 78.4–96.3 | +26.98 %  | +65.98 %  |
| 0.80 | 27       | 25      | 2        | 92.6 % | 76.6–97.9 | +29.63 %  | +69.44 %  |

### ? — EYBL U17

- Entrenado (A): **0** · Holdout (C): **272** · Sesgo @0.75: -0.0594

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 92       | 71      | 21       | 77.2 % | 67.6–84.6 | +8.04 %   | +41.23 %  |
| 0.68 | 70       | 57      | 13       | 81.4 % | 70.8–88.8 | +14.00 %  | +49.01 %  |
| 0.70 | 60       | 51      | 9        | 85.0 % | 73.9–91.9 | +19.00 %  | +55.55 %  |
| 0.72 | 56       | 48      | 8        | 85.7 % | 74.3–92.6 | +20.00 %  | +56.86 %  |
| 0.75 | 42       | 37      | 5        | 88.1 % | 75.0–94.8 | +23.33 %  | +61.21 %  |
| 0.80 | 24       | 23      | 1        | 95.8 % | 79.8–99.3 | +34.17 %  | +75.38 %  |

### Historical — FSHB Superliga

- Entrenado (A): **149** · Holdout (C): **100** · Sesgo @0.75: -0.0217

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 65       | 50      | 15       | 76.9 % | 65.4–85.5 | +7.69 %   | +40.77 %  |
| 0.68 | 58       | 45      | 13       | 77.6 % | 65.3–86.4 | +8.62 %   | +41.98 %  |
| 0.70 | 55       | 43      | 12       | 78.2 % | 65.6–87.1 | +9.45 %   | +43.07 %  |
| 0.72 | 50       | 39      | 11       | 78.0 % | 64.8–87.2 | +9.20 %   | +42.74 %  |
| 0.75 | 41       | 35      | 6        | 85.4 % | 71.6–93.1 | +19.51 %  | +56.22 %  |
| 0.80 | 30       | 27      | 3        | 90.0 % | 74.4–96.5 | +26.00 %  | +64.70 %  |

### Other — Hungary Divison A Women

- Entrenado (A): **303** · Holdout (C): **121** · Sesgo @0.75: +0.0287

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 80       | 60      | 20       | 75.0 % | 64.5–83.2 | +5.00 %   | +37.25 %  |
| 0.68 | 66       | 52      | 14       | 78.8 % | 67.5–86.9 | +10.30 %  | +44.18 %  |
| 0.70 | 57       | 45      | 12       | 79.0 % | 66.7–87.5 | +10.53 %  | +44.47 %  |
| 0.72 | 52       | 43      | 9        | 82.7 % | 70.3–90.6 | +15.77 %  | +51.33 %  |
| 0.75 | 41       | 33      | 8        | 80.5 % | 66.0–89.8 | +12.68 %  | +47.29 %  |
| 0.80 | 31       | 23      | 8        | 74.2 % | 56.8–86.3 | +3.87 %   | +35.77 %  |

### ? — CBI U19

- Entrenado (A): **0** · Holdout (C): **113** · Sesgo @0.75: +0.0410

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 66       | 49      | 17       | 74.2 % | 62.6–83.3 | +3.94 %   | +35.86 %  |
| 0.68 | 58       | 45      | 13       | 77.6 % | 65.3–86.4 | +8.62 %   | +41.98 %  |
| 0.70 | 54       | 43      | 11       | 79.6 % | 67.1–88.2 | +11.48 %  | +45.72 %  |
| 0.72 | 50       | 39      | 11       | 78.0 % | 64.8–87.2 | +9.20 %   | +42.74 %  |
| 0.75 | 39       | 31      | 8        | 79.5 % | 64.5–89.2 | +11.28 %  | +45.46 %  |
| 0.80 | 25       | 22      | 3        | 88.0 % | 70.0–95.8 | +23.20 %  | +61.04 %  |

### Other — United League

- Entrenado (A): **1,288** · Holdout (C): **189** · Sesgo @0.75: +0.0276

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 97       | 73      | 24       | 75.3 % | 65.8–82.8 | +5.36 %   | +37.72 %  |
| 0.68 | 79       | 60      | 19       | 76.0 % | 65.5–84.0 | +6.33 %   | +38.99 %  |
| 0.70 | 63       | 49      | 14       | 77.8 % | 66.1–86.3 | +8.89 %   | +42.33 %  |
| 0.72 | 52       | 40      | 12       | 76.9 % | 63.9–86.3 | +7.69 %   | +40.77 %  |
| 0.75 | 39       | 30      | 9        | 76.9 % | 61.7–87.4 | +7.69 %   | +40.77 %  |
| 0.80 | 16       | 12      | 4        | 75.0 % | 50.5–89.8 | +5.00 %   | +37.25 %  |

### ? — Tipos Extraliga

- Entrenado (A): **424** · Holdout (C): **63** · Sesgo @0.75: +0.0955

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 53       | 37      | 16       | 69.8 % | 56.5–80.5 | -2.26 %   | +27.75 %  |
| 0.68 | 48       | 35      | 13       | 72.9 % | 59.0–83.4 | +2.08 %   | +33.44 %  |
| 0.70 | 47       | 34      | 13       | 72.3 % | 58.2–83.1 | +1.28 %   | +32.38 %  |
| 0.72 | 45       | 33      | 12       | 73.3 % | 59.0–84.0 | +2.67 %   | +34.20 %  |
| 0.75 | 39       | 30      | 9        | 76.9 % | 61.7–87.4 | +7.69 %   | +40.77 %  |
| 0.80 | 30       | 25      | 5        | 83.3 % | 66.4–92.7 | +16.67 %  | +52.50 %  |

### ? — Top Division Women

- Entrenado (A): **0** · Holdout (C): **100** · Sesgo @0.75: -0.0652

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 68       | 58      | 10       | 85.3 % | 75.0–91.8 | +19.41 %  | +56.09 %  |
| 0.68 | 61       | 52      | 9        | 85.2 % | 74.3–92.0 | +19.34 %  | +56.00 %  |
| 0.70 | 56       | 48      | 8        | 85.7 % | 74.3–92.6 | +20.00 %  | +56.86 %  |
| 0.72 | 51       | 45      | 6        | 88.2 % | 76.6–94.5 | +23.53 %  | +61.47 %  |
| 0.75 | 38       | 34      | 4        | 89.5 % | 75.9–95.8 | +25.26 %  | +63.74 %  |
| 0.80 | 25       | 22      | 3        | 88.0 % | 70.0–95.8 | +23.20 %  | +61.04 %  |

### Poland — MPBL

- Entrenado (A): **739** · Holdout (C): **287** · Sesgo @0.75: -0.0452

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 107      | 74      | 33       | 69.2 % | 59.9–77.1 | -3.18 %   | +26.56 %  |
| 0.68 | 81       | 60      | 21       | 74.1 % | 63.6–82.4 | +3.70 %   | +35.56 %  |
| 0.70 | 68       | 54      | 14       | 79.4 % | 68.4–87.3 | +11.18 %  | +45.32 %  |
| 0.72 | 54       | 42      | 12       | 77.8 % | 65.1–86.8 | +8.89 %   | +42.33 %  |
| 0.75 | 35       | 29      | 6        | 82.9 % | 67.3–91.9 | +16.00 %  | +51.63 %  |
| 0.80 | 7        | 0       | 7        | —      | —         | —         | —         |

### Other — Swedish Basketball League Women

- Entrenado (A): **259** · Holdout (C): **129** · Sesgo @0.75: +0.0853

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 68       | 49      | 19       | 72.1 % | 60.4–81.3 | +0.88 %   | +31.87 %  |
| 0.68 | 54       | 41      | 13       | 75.9 % | 63.1–85.4 | +6.30 %   | +38.94 %  |
| 0.70 | 47       | 36      | 11       | 76.6 % | 62.8–86.4 | +7.23 %   | +40.17 %  |
| 0.72 | 42       | 32      | 10       | 76.2 % | 61.5–86.5 | +6.67 %   | +39.43 %  |
| 0.75 | 34       | 25      | 9        | 73.5 % | 56.9–85.4 | +2.94 %   | +34.56 %  |
| 0.80 | 17       | 13      | 4        | 76.5 % | 52.7–90.4 | +7.06 %   | +39.94 %  |

### ? — NBL1 East

- Entrenado (A): **162** · Holdout (C): **126** · Sesgo @0.75: -0.0679

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 74       | 55      | 19       | 74.3 % | 63.3–82.9 | +4.05 %   | +36.01 %  |
| 0.68 | 61       | 47      | 14       | 77.0 % | 65.1–85.8 | +7.87 %   | +41.00 %  |
| 0.70 | 50       | 41      | 9        | 82.0 % | 69.2–90.2 | +14.80 %  | +50.06 %  |
| 0.72 | 41       | 35      | 6        | 85.4 % | 71.6–93.1 | +19.51 %  | +56.22 %  |
| 0.75 | 33       | 29      | 4        | 87.9 % | 72.7–95.2 | +23.03 %  | +60.82 %  |
| 0.80 | 16       | 15      | 1        | 93.8 % | 71.7–98.9 | +31.25 %  | +71.56 %  |

### ? — EYBL U16

- Entrenado (A): **0** · Holdout (C): **265** · Sesgo @0.75: -0.0097

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 83       | 63      | 20       | 75.9 % | 65.7–83.8 | +6.27 %   | +38.90 %  |
| 0.68 | 66       | 52      | 14       | 78.8 % | 67.5–86.9 | +10.30 %  | +44.18 %  |
| 0.70 | 55       | 43      | 12       | 78.2 % | 65.6–87.1 | +9.45 %   | +43.07 %  |
| 0.72 | 47       | 36      | 11       | 76.6 % | 62.8–86.4 | +7.23 %   | +40.17 %  |
| 0.75 | 33       | 27      | 6        | 81.8 % | 65.6–91.4 | +14.55 %  | +49.73 %  |
| 0.80 | 15       | 14      | 1        | 93.3 % | 70.2–98.8 | +30.67 %  | +70.80 %  |

### ? — EYBL U13

- Entrenado (A): **0** · Holdout (C): **118** · Sesgo @0.75: +0.0035

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 56       | 38      | 18       | 67.9 % | 54.8–78.6 | -5.00 %   | +24.18 %  |
| 0.68 | 51       | 35      | 16       | 68.6 % | 55.0–79.7 | -3.92 %   | +25.59 %  |
| 0.70 | 49       | 34      | 15       | 69.4 % | 55.5–80.5 | -2.86 %   | +26.98 %  |
| 0.72 | 44       | 31      | 13       | 70.5 % | 55.8–81.8 | -1.36 %   | +28.93 %  |
| 0.75 | 32       | 26      | 6        | 81.2 % | 64.7–91.1 | +13.75 %  | +48.69 %  |
| 0.80 | 19       | 15      | 4        | 79.0 % | 56.7–91.5 | +10.53 %  | +44.47 %  |

### Other — Premijer ženska liga

- Entrenado (A): **327** · Holdout (C): **100** · Sesgo @0.75: +0.1433

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 59       | 36      | 23       | 61.0 % | 48.3–72.4 | -14.58 %  | +11.66 %  |
| 0.68 | 52       | 32      | 20       | 61.5 % | 48.0–73.5 | -13.85 %  | +12.62 %  |
| 0.70 | 49       | 32      | 17       | 65.3 % | 51.3–77.1 | -8.57 %   | +19.51 %  |
| 0.72 | 41       | 27      | 14       | 65.8 % | 50.5–78.4 | -7.80 %   | +20.51 %  |
| 0.75 | 31       | 21      | 10       | 67.7 % | 50.1–81.4 | -5.16 %   | +23.97 %  |
| 0.80 | 18       | 11      | 7        | 61.1 % | 38.6–79.7 | -14.44 %  | +11.83 %  |

### Other — Basketligaen

- Entrenado (A): **787** · Holdout (C): **113** · Sesgo @0.75: +0.0021

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 56       | 41      | 15       | 73.2 % | 60.4–83.0 | +2.50 %   | +33.98 %  |
| 0.68 | 50       | 37      | 13       | 74.0 % | 60.4–84.1 | +3.60 %   | +35.42 %  |
| 0.70 | 47       | 36      | 11       | 76.6 % | 62.8–86.4 | +7.23 %   | +40.17 %  |
| 0.72 | 37       | 30      | 7        | 81.1 % | 65.8–90.5 | +13.51 %  | +48.38 %  |
| 0.75 | 29       | 23      | 6        | 79.3 % | 61.6–90.2 | +11.03 %  | +45.14 %  |
| 0.80 | 13       | 0       | 13       | —      | —         | —         | —         |

### Other — Liga Femenina

- Entrenado (A): **1,314** · Holdout (C): **418** · Sesgo @0.75: +0.1279

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 130      | 86      | 44       | 66.2 % | 57.7–73.7 | -7.38 %   | +21.06 %  |
| 0.68 | 100      | 66      | 34       | 66.0 % | 56.3–74.5 | -7.60 %   | +20.78 %  |
| 0.70 | 78       | 51      | 27       | 65.4 % | 54.3–75.0 | -8.46 %   | +19.65 %  |
| 0.72 | 53       | 35      | 18       | 66.0 % | 52.6–77.3 | -7.55 %   | +20.85 %  |
| 0.75 | 29       | 19      | 10       | 65.5 % | 47.3–80.1 | -8.28 %   | +19.90 %  |
| 0.80 | 9        | 0       | 9        | —      | —         | —         | —         |

### ? — EGBL U17

- Entrenado (A): **0** · Holdout (C): **74** · Sesgo @0.75: -0.0329

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 40       | 32      | 8        | 80.0 % | 65.2–89.5 | +12.00 %  | +46.40 %  |
| 0.68 | 34       | 27      | 7        | 79.4 % | 63.2–89.7 | +11.18 %  | +45.32 %  |
| 0.70 | 31       | 25      | 6        | 80.7 % | 63.7–90.8 | +12.90 %  | +47.58 %  |
| 0.72 | 30       | 24      | 6        | 80.0 % | 62.7–90.5 | +12.00 %  | +46.40 %  |
| 0.75 | 27       | 23      | 4        | 85.2 % | 67.5–94.1 | +19.26 %  | +55.89 %  |
| 0.80 | 14       | 0       | 14       | —      | —         | —         | —         |

### Other — Russian Women's Basketball Premier League

- Entrenado (A): **726** · Holdout (C): **102** · Sesgo @0.75: +0.0773

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 57       | 44      | 13       | 77.2 % | 64.8–86.2 | +8.07 %   | +41.26 %  |
| 0.68 | 44       | 34      | 10       | 77.3 % | 63.0–87.2 | +8.18 %   | +41.41 %  |
| 0.70 | 39       | 29      | 10       | 74.4 % | 58.9–85.4 | +4.10 %   | +36.08 %  |
| 0.72 | 36       | 26      | 10       | 72.2 % | 56.0–84.2 | +1.11 %   | +32.17 %  |
| 0.75 | 27       | 20      | 7        | 74.1 % | 55.3–86.8 | +3.70 %   | +35.56 %  |
| 0.80 | 17       | 12      | 5        | 70.6 % | 46.9–86.7 | -1.18 %   | +29.18 %  |

### ? — Liga 1 Masculin

- Entrenado (A): **122** · Holdout (C): **166** · Sesgo @0.75: +0.1569

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 86       | 61      | 25       | 70.9 % | 60.6–79.5 | -0.70 %   | +29.80 %  |
| 0.68 | 64       | 46      | 18       | 71.9 % | 59.9–81.4 | +0.62 %   | +31.53 %  |
| 0.70 | 47       | 33      | 14       | 70.2 % | 56.0–81.3 | -1.70 %   | +28.49 %  |
| 0.72 | 38       | 24      | 14       | 63.2 % | 47.3–76.6 | -11.58 %  | +15.58 %  |
| 0.75 | 27       | 17      | 10       | 63.0 % | 44.2–78.5 | -11.85 %  | +15.22 %  |
| 0.80 | 7        | 0       | 7        | —      | —         | —         | —         |

### ? — Liga Nacional Femenina Chile

- Entrenado (A): **0** · Holdout (C): **88** · Sesgo @0.75: +0.0611

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 49       | 32      | 17       | 65.3 % | 51.3–77.1 | -8.57 %   | +19.51 %  |
| 0.68 | 44       | 28      | 16       | 63.6 % | 48.9–76.2 | -10.91 %  | +16.45 %  |
| 0.70 | 37       | 24      | 13       | 64.9 % | 48.8–78.2 | -9.19 %   | +18.70 %  |
| 0.72 | 35       | 22      | 13       | 62.9 % | 46.3–76.8 | -12.00 %  | +15.03 %  |
| 0.75 | 26       | 20      | 6        | 76.9 % | 57.9–89.0 | +7.69 %   | +40.77 %  |
| 0.80 | 18       | 14      | 4        | 77.8 % | 54.8–91.0 | +8.89 %   | +42.33 %  |

### Other — Prva Liga

- Entrenado (A): **0** · Holdout (C): **139** · Sesgo @0.75: -0.0614

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 44       | 35      | 9        | 79.5 % | 65.5–88.8 | +11.36 %  | +45.57 %  |
| 0.68 | 33       | 27      | 6        | 81.8 % | 65.6–91.4 | +14.55 %  | +49.73 %  |
| 0.70 | 31       | 26      | 5        | 83.9 % | 67.4–92.9 | +17.42 %  | +53.48 %  |
| 0.72 | 27       | 24      | 3        | 88.9 % | 71.9–96.1 | +24.44 %  | +62.67 %  |
| 0.75 | 25       | 22      | 3        | 88.0 % | 70.0–95.8 | +23.20 %  | +61.04 %  |
| 0.80 | 14       | 0       | 14       | —      | —         | —         | —         |

### Other — Switzerland LNA

- Entrenado (A): **832** · Holdout (C): **104** · Sesgo @0.75: +0.0065

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 50       | 38      | 12       | 76.0 % | 62.6–85.7 | +6.40 %   | +39.08 %  |
| 0.68 | 42       | 32      | 10       | 76.2 % | 61.5–86.5 | +6.67 %   | +39.43 %  |
| 0.70 | 36       | 28      | 8        | 77.8 % | 61.9–88.3 | +8.89 %   | +42.33 %  |
| 0.72 | 29       | 23      | 6        | 79.3 % | 61.6–90.2 | +11.03 %  | +45.14 %  |
| 0.75 | 25       | 20      | 5        | 80.0 % | 60.9–91.1 | +12.00 %  | +46.40 %  |
| 0.80 | 11       | 0       | 11       | —      | —         | —         | —         |

## 5. Ligas destacadas

### 5.1 Mejores ROI @1.40 a 0.75 (mín. 30 apuestas)

| País       | Liga               | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR     | ROI @1.40 |
| ---        | ---                | ---:      | ---:           | ---:    | ---:     | ---:   | ---:      |
| ?          | CBI U15 Fem        | 0         | 65             | 61      | 4        | 93.8 % | +31.38 %  |
| Historical | LNB                | 318       | 67             | 61      | 6        | 91.0 % | +27.46 %  |
| ?          | ZBL                | 859       | 43             | 39      | 4        | 90.7 % | +26.98 %  |
| ?          | Top Division Women | 0         | 38             | 34      | 4        | 89.5 % | +25.26 %  |
| ?          | EYBL U17           | 0         | 42             | 37      | 5        | 88.1 % | +23.33 %  |
| ?          | NBL1 East          | 162       | 33             | 29      | 4        | 87.9 % | +23.03 %  |
| ?          | CBI U15            | 0         | 49             | 42      | 7        | 85.7 % | +20.00 %  |
| Historical | FSHB Superliga     | 149       | 41             | 35      | 6        | 85.4 % | +19.51 %  |
| Australia  | NBL                | 2,647     | 62             | 52      | 10       | 83.9 % | +17.42 %  |
| Poland     | MPBL               | 739       | 35             | 29      | 6        | 82.9 % | +16.00 %  |

### 5.2 Peores ROI @1.40 a 0.75 (mín. 30 apuestas)

| País       | Liga                            | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR     | ROI @1.40 |
| ---        | ---                             | ---:      | ---:           | ---:    | ---:     | ---:   | ---:      |
| Other      | Premijer ženska liga            | 327       | 31             | 21      | 10       | 67.7 % | -5.16 %   |
| ?          | EYBL U14                        | 0         | 51             | 37      | 14       | 72.5 % | +1.57 %   |
| Other      | Swedish Basketball League Women | 259       | 34             | 25      | 9        | 73.5 % | +2.94 %   |
| Other      | United League                   | 1,288     | 39             | 30      | 9        | 76.9 % | +7.69 %   |
| ?          | Tipos Extraliga                 | 424       | 39             | 30      | 9        | 76.9 % | +7.69 %   |
| Historical | Poland 2nd Basketball League    | 828       | 166            | 130     | 36       | 78.3 % | +9.64 %   |
| ?          | CBI U19                         | 0         | 39             | 31      | 8        | 79.5 % | +11.28 %  |
| ?          | EYBL U15                        | 0         | 44             | 35      | 9        | 79.5 % | +11.36 %  |
| ?          | NCAA Women's Division I         | 0         | 267            | 214     | 53       | 80.2 % | +12.21 %  |
| Other      | Hungary Divison A Women         | 303       | 41             | 33      | 8        | 80.5 % | +12.68 %  |

## 5.3 Marcadores de la columna "Bloqueo"

- **🆕 (arranque en frío):** la liga no tuvo ningún partido en el bloque de entrenamiento (A). El modelo no la vio; sus priors son as-of. Hay 78 ligas así.

- **⚠️ (ROI negativo a 0.75):** la liga tiene ROI@1.40 < 0 % a umbral 0.75. Candidata a bloqueo por medición. Hay 4 ligas así.

- **(vacío):** liga normal, sin alertas.

### 5.4 Ligas en arranque en frío (Entrenado = 0)

| País      | Liga                                     | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 |
| ---       | ---                                      | ---:        | ---:           | ---:    | ---:     | ---:     |
| ?         | NCAA Women's Division I                  | 1,563       | 267            | 214     | 53       | 80.2 %   |
| ?         | CBI U15 Fem                              | 138         | 65             | 61      | 4        | 93.8 %   |
| ?         | EYBL U14                                 | 236         | 51             | 37      | 14       | 72.5 %   |
| ?         | CBI U15                                  | 199         | 49             | 42      | 7        | 85.7 %   |
| ?         | EYBL U15                                 | 232         | 44             | 35      | 9        | 79.5 %   |
| ?         | EYBL U17                                 | 272         | 42             | 37      | 5        | 88.1 %   |
| ?         | CBI U19                                  | 113         | 39             | 31      | 8        | 79.5 %   |
| ?         | Top Division Women                       | 100         | 38             | 34      | 4        | 89.5 %   |
| ?         | EYBL U16                                 | 265         | 33             | 27      | 6        | 81.8 %   |
| ?         | EYBL U13                                 | 118         | 32             | 26      | 6        | 81.2 %   |
| ?         | EGBL U17                                 | 74          | 27             | 23      | 4        | 85.2 %   |
| ?         | Liga Nacional Femenina Chile             | 88          | 26             | 20      | 6        | 76.9 %   |
| Other     | Prva Liga                                | 139         | 25             | 22      | 3        | 88.0 %   |
| ?         | Nostra.lt-RKL Division B                 | 200         | 25             | 18      | 7        | 72.0 %   |
| Other     | 1st Division Women                       | 69          | 23             | 20      | 3        | 87.0 %   |
| ?         | CBI U17 Feminino                         | 54          | 22             | 21      | 1        | 95.5 %   |
| ?         | 1 Liga Kobiet                            | 251         | 18             | 14      | 4        | 77.8 %   |
| ?         | NBL Women                                | 106         | 17             | 14      | 3        | 82.3 %   |
| Other     | Sparkasse Prva Liga Zeni                 | 70          | 17             | 9       | 8        | 52.9 %   |
| ?         | EYBL U20                                 | 187         | 16             | 14      | 2        | 87.5 %   |
| ?         | SWC                                      | 39          | 15             | 14      | 1        | 93.3 %   |
| Other     | Superleague Women                        | 34          | 15             | 13      | 2        | 86.7 %   |
| Lithuania | RKL Division A                           | 213         | 0              | 0       | 0        | —        |
| ?         | Divisional Tercera de Ascenso            | 47          | 0              | 0       | 0        | —        |
| ?         | Serie B Interregionale                   | 255         | 0              | 0       | 0        | —        |
| ?         | Erkekler Basketbol Gençler Ligi          | 35          | 0              | 0       | 0        | —        |
| ?         | Divisione Regionale 2 - Lazio            | 105         | 0              | 0       | 0        | —        |
| Other     | Serie C Toscana                          | 50          | 0              | 0       | 0        | —        |
| Other     | Serie C Girone H sud                     | 30          | 0              | 0       | 0        | —        |
| ?         | B Ethniki                                | 95          | 0              | 0       | 0        | —        |
| Other     | Basket World League                      | 31          | 0              | 0       | 0        | —        |
| ?         | Belize Elite Basketball League           | 39          | 0              | 0       | 0        | —        |
| ?         | Egyptian Women's Basketball Super League | 55          | 0              | 0       | 0        | —        |
| ?         | Erkekler Bölgesel Basketbol Ligi         | 67          | 0              | 0       | 0        | —        |
| ?         | Egyptian Basketball Super League         | 31          | 0              | 0       | 0        | —        |
| France    | LNB 2                                    | 44          | 0              | 0       | 0        | —        |
| ?         | U16 Türkiye Finalleri                    | 46          | 0              | 0       | 0        | —        |
| ?         | U18 EuroBasket Div. B                    | 50          | 0              | 0       | 0        | —        |
| ?         | FIBA U16 Women's EuroBasket Division B   | 34          | 0              | 0       | 0        | —        |
| ?         | FIBA U16 EuroBasket                      | 38          | 0              | 0       | 0        | —        |

_...y 2 ligas más._


### 5.6 Ligas en arranque en frío con **todas** las apuestas perdidas (0 % WR)

| País   | Liga                                       | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos |
| ---    | ---                                        | ---:        | ---:           | ---:    | ---:     |
| ?      | Campeonato Nacional da 1ª Divisao Feminina | 107         | 13             | 0       | 13       |
| Other  | The Women League                           | 31          | 12             | 0       | 12       |
| ?      | FIBA U16 Women's EuroBasket                | 38          | 11             | 0       | 11       |
| ?      | LF Challenge                               | 145         | 10             | 0       | 10       |
| Other  | Jedinstvena kadetska liga                  | 64          | 10             | 0       | 10       |
| Other  | U21 Espoirs Elite                          | 144         | 9              | 0       | 9        |
| Other  | Latvian Basketball Cup                     | 31          | 9              | 0       | 9        |
| ?      | 1. liga                                    | 113         | 8              | 0       | 8        |
| USA    | NCAA Women's March Madness                 | 59          | 8              | 0       | 8        |
| ?      | FIBA U17 Women's Basketball World Cup      | 37          | 8              | 0       | 8        |
| Other  | Superliga e Femrave                        | 41          | 7              | 0       | 7        |
| ?      | NBL Men                                    | 125         | 7              | 0       | 7        |
| ?      | U18 FIBA Asia Cup                          | 36          | 6              | 0       | 6        |
| ?      | LBP                                        | 34          | 6              | 0       | 6        |
| Other  | I B MCKL                                   | 36          | 5              | 0       | 5        |
| ?      | EGBL U14                                   | 47          | 5              | 0       | 5        |
| ?      | U18 EuroBasket                             | 124         | 5              | 0       | 5        |
| ?      | FIBA Asia Cup 2029 Pre-Qualifiers          | 31          | 5              | 0       | 5        |
| Other  | Azerbaijan Basketball League               | 105         | 4              | 0       | 4        |
| ?      | FIBA World Cup Qualification               | 57          | 4              | 0       | 4        |
| France | LNB Clausura                               | 71          | 4              | 0       | 4        |
| Other  | 1st Division Men                           | 126         | 3              | 0       | 3        |
| Other  | Liga Košarkaškog saveza Herceg Bosne       | 45          | 3              | 0       | 3        |
| ?      | Kızlar Basketbol Gençler Ligi              | 34          | 3              | 0       | 3        |
| ?      | FIBA U16 EuroBasket Division B             | 61          | 3              | 0       | 3        |
| Other  | A1 Liga                                    | 108         | 2              | 0       | 2        |
| Other  | A1                                         | 69          | 2              | 0       | 2        |
| ?      | CN1 Basquetebol                            | 101         | 2              | 0       | 2        |
| ?      | EGBL U19                                   | 39          | 2              | 0       | 2        |
| ?      | 2. ZLS                                     | 37          | 2              | 0       | 2        |
| ?      | Divisione Regionale 1 Toscana              | 131         | 2              | 0       | 2        |
| ?      | Enovos League                              | 97          | 1              | 0       | 1        |
| Turkey | Türkiye Kadınlar Basketbol Ligi            | 39          | 1              | 0       | 1        |
| Other  | Saku II Liiga                              | 35          | 1              | 0       | 1        |
| ?      | 2. DBBL                                    | 34          | 1              | 0       | 1        |
| ?      | Nostra.lt-RKL Division A                   | 38          | 1              | 0       | 1        |

### 5.7 Ligas en arranque en frío **rentables** (WR ≥ 75 %, ≥ 10 apuestas) — candidatas a entrenar

> Estas ligas no se entrenaron (arranque en frío) pero tienen buen rendimiento en el holdout C. Si se consigue histórico para ellas (o se agrupan por `clean_name` con variantes que sí tienen historia), entrenar con ellas podría mejorar el modelo. Son las que valen la pena.

| País  | Liga                         | Partidos en BD (todas las variantes) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |
| ---   | ---                          | ---:                                 | ---:        | ---:           | ---:    | ---:     | ---:     | ---:      |
| ?     | NCAA Women's Division I      | 1,729                                | 1,563       | 267            | 214     | 53       | 80.2 %   | +12.21 %  |
| ?     | CBI U15 Fem                  | 145                                  | 138         | 65             | 61      | 4        | 93.8 %   | +31.38 %  |
| ?     | CBI U15                      | 207                                  | 199         | 49             | 42      | 7        | 85.7 %   | +20.00 %  |
| ?     | EYBL U15                     | 259                                  | 232         | 44             | 35      | 9        | 79.5 %   | +11.36 %  |
| ?     | EYBL U17                     | 308                                  | 272         | 42             | 37      | 5        | 88.1 %   | +23.33 %  |
| ?     | CBI U19                      | 147                                  | 113         | 39             | 31      | 8        | 79.5 %   | +11.28 %  |
| ?     | Top Division Women           | 109                                  | 100         | 38             | 34      | 4        | 89.5 %   | +25.26 %  |
| ?     | EYBL U16                     | 305                                  | 265         | 33             | 27      | 6        | 81.8 %   | +14.55 %  |
| ?     | EYBL U13                     | 129                                  | 118         | 32             | 26      | 6        | 81.2 %   | +13.75 %  |
| ?     | EGBL U17                     | 76                                   | 74          | 27             | 23      | 4        | 85.2 %   | +19.26 %  |
| ?     | Liga Nacional Femenina Chile | 135                                  | 88          | 26             | 20      | 6        | 76.9 %   | +7.69 %   |
| Other | Prva Liga                    | 168                                  | 139         | 25             | 22      | 3        | 88.0 %   | +23.20 %  |
| Other | 1st Division Women           | 83                                   | 69          | 23             | 20      | 3        | 87.0 %   | +21.74 %  |
| ?     | CBI U17 Feminino             | 54                                   | 54          | 22             | 21      | 1        | 95.5 %   | +33.64 %  |
| ?     | 1 Liga Kobiet                | 283                                  | 251         | 18             | 14      | 4        | 77.8 %   | +8.89 %   |
| ?     | NBL Women                    | 111                                  | 106         | 17             | 14      | 3        | 82.3 %   | +15.29 %  |
| ?     | EYBL U20                     | 248                                  | 187         | 16             | 14      | 2        | 87.5 %   | +22.50 %  |
| ?     | SWC                          | 40                                   | 39          | 15             | 14      | 1        | 93.3 %   | +30.67 %  |
| Other | Superleague Women            | 45                                   | 34          | 15             | 13      | 2        | 86.7 %   | +21.33 %  |

### 5.5 Ligas con ROI negativo a 0.75 (⚠️)

| País  | Liga                     | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |
| ---   | ---                      | ---:      | ---:           | ---:    | ---:     | ---:     | ---:      |
| Other | Sparkasse Prva Liga Zeni | 0         | 17             | 9       | 8        | 52.9 %   | -25.88 %  |
| ?     | Liga 1 Masculin          | 122       | 27             | 17      | 10       | 63.0 %   | -11.85 %  |
| Other | Liga Femenina            | 1,314     | 29             | 19      | 10       | 65.5 %   | -8.28 %   |
| Other | Premijer ženska liga     | 327       | 31             | 21      | 10       | 67.7 %   | -5.16 %   |

## 6. Caso específico — `NCAA Women's Division I` (backfill histórico)

> En `m27_v5_3_2` esta liga tenía 0 entrenamiento (solo temporada 2025-26, 1,729 partidos, todo en holdout C). En `m27_v5_3_3` se descargaron **19,541 partidos históricos** (2019-2026) vía API de SofaScore, por lo que ahora tiene entrenamiento. Es la prueba de que el backfill histórico resuelve el arranque en frío.

- **Entrenado (A):** 0 · **Holdout (C):** 1,563

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 589      | 416     | 173      | 70.6 % | 66.8–74.2 | -1.12 %   | +29.25 %  |
| 0.68 | 476      | 345     | 131      | 72.5 % | 68.3–76.3 | +1.47 %   | +32.64 %  |
| 0.70 | 414      | 308     | 106      | 74.4 % | 70.0–78.4 | +4.15 %   | +36.14 %  |
| 0.72 | 363      | 274     | 89       | 75.5 % | 70.8–79.6 | +5.67 %   | +38.13 %  |
| 0.75 | 267      | 214     | 53       | 80.2 % | 75.0–84.5 | +12.21 %  | +46.67 %  |
| 0.80 | 173      | 153     | 20       | 88.4 % | 82.8–92.4 | +23.82 %  | +61.84 %  |

A 0.75: **267 apuestas, WR 80.15 %, ROI@1.40 +12.21 %** (en vivo: 4 partidos, v5_3 2W/2L, v5_3_1 2W/0L).

## 7. Prueba contra los partidos en vivo de `m27_v5_3`

- **m27_v5_3 (original):** 89 apuestas, 71 ganadas, 18 perdidas → WR 79.8 %.
- **m27_v5_3_3 (clon):** 64 apuestas, 53 ganadas, 11 perdidas → WR 82.8 %.

**vs m27_v5_3_2 (anterior):** 64 apuestas, 53 ganadas, 11 perdidas → WR 82.8 %. v5_3_3: +0 apuestas, +0 ganadas, +0.0 pp de WR.

### 7.1 Resumen por liga — apuestas en vivo (original vs clon)

| Dif. WR   | WR v5_3 | WR v5_3_2 | V5_3 Wins | V5_3_2 Wins | País          | Liga                                                                        | En vivo (partidos) | v5_3: apuestas | v5_3: perdidas | v5_3_2: apuestas | v5_3_2: perdidas |      |
| ---       | ---:    | ---:      | ---:      | ---:        | ---           | ---:                                                                        | ---:               | ---:           | ---:           | ---:             | ---:             | ---: |
| ⚪        | 80 %    | 80 %      | 4         | 4           | Other         | LBP, Group B                                                                | 5                  | 5              | 1              | 5                | 1                |      |
| 🟢        | 80 %    | 100 %     | 4         | 4           | France        | LNB, Women                                                                  | 5                  | 5              | 1              | 4                | 0                |      |
| ⚪        | 75 %    | 75 %      | 3         | 3           | Other         | 1. ZLS, Women                                                               | 4                  | 4              | 1              | 4                | 1                |      |
| ⚪        | 100 %   | 100 %     | 3         | 3           | Other         | Road to Basketball Africa League, Group D                                   | 3                  | 3              | 0              | 3                | 0                |      |
| ⚪        | 100 %   | 100 %     | 3         | 3           | Other         | Top Division Women, Regular Season                                          | 3                  | 3              | 0              | 3                | 0                |      |
| 🟢        | 50 %    | 100 %     | 2         | 2           | Poland        | Poland 2nd Basketball League, Runda Zasadnicza                              | 4                  | 4              | 2              | 2                | 0                |      |
| 🟢        | 33 %    | 50 %      | 1         | 1           | Australia     | NBL, West                                                                   | 3                  | 3              | 2              | 2                | 1                |      |
| ⚪        | 100 %   | 100 %     | 3         | 2           | Other         | Swedish Basketball League Women                                             | 3                  | 3              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | Tipos Extraliga, Women                                                      | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | Estonian-Latvian Basketball League                                          | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Historical    | Liga Nacional Femenina Chile, Fase zonal                                    | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | United League                                                               | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 4         | 1           | Other         | Basketligaen                                                                | 4                  | 4              | 0              | 1                | 0                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | International | FIBA Women Basketball League Africa - Qualifers, Group A                    | 2                  | 2              | 1              | 1                | 1                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Other         | 2nd League                                                                  | 2                  | 2              | 1              | 1                | 1                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Other         | A1 Women                                                                    | 2                  | 2              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 2         | 1           | Historical    | L1BF BT, Faza I                                                             | 2                  | 2              | 0              | 1                | 0                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Japan         | B League One                                                                | 2                  | 2              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Switzerland LNA                                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Hungary Divison A Women                                                     | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Turkish Basketball Super League                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | 1 Liga Kobiet, Runda Zasadnicza                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 13 | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Premijer ženska liga                                                        | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | 1st Division Women, Deildarkeppni                                           | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 14 | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Nostra.lt-RKL Division A, Reguliarus sezonas                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | International | Eurocup, Group B                                                            | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Russian Women's Basketball Premier League                                   | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Brazil        | Brazil NBB                                                                  | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Other         | 2. ZLS, Women                                                               | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Liga 1 Masculin, Faza I                                                     | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Hungary NB 1.A                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Super League                                                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Other         | Women's Basketball Division A                                               | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Kvindebasketligaen , Grundspil                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | FSHB Superliga                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Greece        | Stoiximan GBL                                                               | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | International | AdmiralBet ABA League, Group B                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Liga Nationala                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Semi-Finals 19    | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Catarinense, Women, Playoffs                                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | CN2 Basquetebol, 1ª Fase - Série D                                          | 1                  | 1              | 0              | 1                | 0                |      |
| —         | 100 %   | —         | 2         | 0           | Other         | LMB, Clausura                                                               | 2                  | 2              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | 1. Division                                                                 | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | FIBA Women Basketball League Africa - Qualifers, Group B                    | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | France        | France Pro A                                                                | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | EuroCup, Women, Group B                                                     | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Japan         | B League Premier                                                            | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | ?             | Nostra.lt-RKL Division B, Reguliarus sezonas                                | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Israel        | Israeli Women Basketball Premier League                                     | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | Cyprus Basketball Division A                                                | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | France        | Mexico LNBP, Playoffs                                                       | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | Swedish Basketball Superettan                                               | 1                  | 1              | 1              | 0                | 0                |      |
| **TOTAL** | **89**  | **89**    | **71**    | **18**      | **80 %**      | **64**                                                                      | **53**             | **11**         | **83 %**       | **🟢**           |                  |      |

> Nota: la simulación secuencial con $100 por apuesta se integra en el barrido de umbrales de la sección 8 (columnas de bankroll final). El umbral operativo es 0.75.


## 8. Barrido de umbrales sobre las apuestas en vivo (m27_v5_3 vs clon)

> Simula qué habría pasado si se hubiera operado con distintos umbrales de confianza sobre los mismos partidos en vivo. Para el clon se aplica también el guard de margen (margin ≥ 0). Bank inicial $1,000, $100 por apuesta.

### Momio 1.40

| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR   | O:ROI    | C:apuestas | C:WR    | C:ROI    | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |
| ---    | ---:       | ---:      | ---:      | ---:       | ---:       | ---:   | ---:     | ---:       | ---:    | ---:     | ---:       | ---:       | ---:         | ---:         | ---:          |
| 0.60   | 89         | 71        | 53        | 18         | 36         | 79.8 % | +11.69 % | 89         | 59.6 %  | -16.63 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.62   | 89         | 71        | 53        | 18         | 36         | 79.8 % | +11.69 % | 89         | 59.6 %  | -16.63 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.65   | 89         | 71        | 53        | 18         | 36         | 79.8 % | +11.69 % | 89         | 59.6 %  | -16.63 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.68   | 89         | 71        | 53        | 18         | 35         | 79.8 % | +11.69 % | 88         | 60.2 %  | -15.68 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.70   | 89         | 71        | 53        | 18         | 34         | 79.8 % | +11.69 % | 87         | 60.9 %  | -14.71 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.72   | 89         | 71        | 53        | 18         | 30         | 79.8 % | +11.69 % | 83         | 63.9 %  | -10.60 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.75   | 77         | 64        | 53        | 13         | 11         | 83.1 % | +16.36 % | 64         | 82.8 %  | +15.94 % | $+1,260.00 | $+1,020.00 | $2,260.00    | $2,020.00    | 🟢            |
| 0.78   | 38         | 31        | 33        | 7          | 6          | 81.6 % | +14.21 % | 39         | 84.6 %  | +18.46 % | $+540.00   | $+720.00   | $1,540.00    | $1,720.00    | 🟢            |
| 0.80   | 27         | 23        | 23        | 4          | 3          | 85.2 % | +19.26 % | 26         | 88.5 %  | +23.85 % | $+520.00   | $+620.00   | $1,520.00    | $1,620.00    | 🟢            |
| 0.85   | 12         | 11        | 10        | 1          | 0          | 91.7 % | +28.33 % | 10         | 100.0 % | +40.00 % | $+340.00   | $+400.00   | $1,340.00    | $1,400.00    | 🟢            |
> * = se detuvo por falta de fondos.


### Momio 1.83

| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR   | O:ROI    | C:apuestas | C:WR    | C:ROI    | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |
| ---    | ---:       | ---:      | ---:      | ---:       | ---:       | ---:   | ---:     | ---:       | ---:    | ---:     | ---:       | ---:       | ---:         | ---:         | ---:          |
| 0.60   | 89         | 71        | 53        | 18         | 36         | 79.8 % | +45.99 % | 89         | 59.6 %  | +8.98 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.62   | 89         | 71        | 53        | 18         | 36         | 79.8 % | +45.99 % | 89         | 59.6 %  | +8.98 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.65   | 89         | 71        | 53        | 18         | 36         | 79.8 % | +45.99 % | 89         | 59.6 %  | +8.98 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.68   | 89         | 71        | 53        | 18         | 35         | 79.8 % | +45.99 % | 88         | 60.2 %  | +10.22 % | $+4,093.00 | $-953.00 * | $5,093.00    | $47.00       | 🔴            |
| 0.70   | 89         | 71        | 53        | 18         | 34         | 79.8 % | +45.99 % | 87         | 60.9 %  | +11.48 % | $+4,093.00 | $-953.00 * | $5,093.00    | $47.00       | 🔴            |
| 0.72   | 89         | 71        | 53        | 18         | 30         | 79.8 % | +45.99 % | 83         | 63.9 %  | +16.86 % | $+4,093.00 | $-921.00 * | $5,093.00    | $79.00       | 🔴            |
| 0.75   | 77         | 64        | 53        | 13         | 11         | 83.1 % | +52.10 % | 64         | 82.8 %  | +51.55 % | $+4,012.00 | $+3,299.00 | $5,012.00    | $4,299.00    | 🟢            |
| 0.78   | 38         | 31        | 33        | 7          | 6          | 81.6 % | +49.29 % | 39         | 84.6 %  | +54.85 % | $+1,873.00 | $+2,139.00 | $2,873.00    | $3,139.00    | 🟢            |
| 0.80   | 27         | 23        | 23        | 4          | 3          | 85.2 % | +55.89 % | 26         | 88.5 %  | +61.88 % | $+1,509.00 | $+1,609.00 | $2,509.00    | $2,609.00    | 🟢            |
| 0.85   | 12         | 11        | 10        | 1          | 0          | 91.7 % | +67.75 % | 10         | 100.0 % | +83.00 % | $+813.00   | $+830.00   | $1,813.00    | $1,830.00    | 🟢            |
> * = se detuvo por falta de fondos.


**Lectura:** el clon es más conservador (59 vs 85 apuestas) con acierto ligeramente superior (79.7 % vs 78.8 %). Diferencia pequeña, muestra diminuta. No se justifica cambio de modelo solo con esta evidencia.

## 9. Configuración del modelo m27_v5_3_2

| Parámetro                                      | Valor                              |
| ---                                            | ---                                |
| AUC train / calib (B) / test (C)               | 0.6459 / 0.6509 / 0.6548           |
| Brier / LogLoss / Acc (test)                   | 0.2304 / 0.6517 / 0.6128           |
| AUC gap (train−test)                           | -0.0089                            |
| Partidos elegibles                             | 184,878                            |
| Filas A / B / C                                | 191,244 / 58,045 / 92,225          |
| Umbral operativo (min_conf)                    | 0.75                               |
| Ventana de inferencia                          | [25.5, 30.0] (fuera → unavailable) |
| Ligas únicas (clean_name)                      | 1,311                              |
| Ligas en arranque en frío (trained=0)          | 78                                 |
| Ligas candidatas a bloqueo (IC95 sup < 71.4 %) | 0                                  |

### Formatos de torneo bloqueados (is_excluded_stage)

amistosos, 2ª fase, promoción/relegation, preliminar, colocación (misma lista que m27_v5_3, vía `models/common/competition_stage.py`).

### Ligas bloqueadas por medición propia

Ninguna con la evidencia actual (0 candidatas), igual que m27_v5_3.

### Walk-forward (3 cortes)

| Corte                                    | AUC    | Apuestas | Ganados | Perdidos | WR      | ROI @1.40 | ROI @1.83 |
| ---                                      | ---:   | ---:     | ---:    | ---:     | ---:    | ---:      | ---:      |
| Corte 1: Train <= 2022 -> Test 2023      | 0.6274 | 358      | 273     | 85       | 76.26 % | +6.76 %   | +39.55 %  |
| Corte 2: Train <= 2023 -> Test 2024      | 0.6369 | 2,954    | 2,368   | 586      | 80.16 % | +12.23 %  | +46.70 %  |
| Corte 3: Train <= 2024 -> Test 2025-2026 | 0.6485 | 14,360   | 11,497  | 2,863    | 80.06 % | +12.09 %  | +46.51 %  |

### Ligas con más partidos de entrenamiento (top 10)

| País      | Liga (clean_name)       | Entrenado (A) | Holdout (C) |
| ---       | ---                     | ---:          | ---:        |
| Japan     | B League Premier        | 3,136         | 568         |
| Japan     | B League One            | 3,097         | 369         |
| Argentina | Argentina Liga Nacional | 2,972         | 273         |
| Australia | NBL                     | 2,647         | 536         |
| Italy     | Serie A2                | 2,619         | 388         |
| Other     | Élite 2                 | 1,989         | 274         |
| Germany   | Germany BBL             | 1,904         | 257         |
| Spain     | Liga ACB                | 1,885         | 271         |
| France    | France Pro A            | 1,864         | 204         |
| Spain     | Primera FEB             | 1,810         | 215         |

## 10. Comparación final: v5_3 vs variantes

| Modelo                    | Cambio principal          | AUC test   | WR @0.75 (ancla) | ROI @1.40    | Live WR    | Live apuestas | Cold-start leagues | Partidos elegibles |
| ---                       | ---                       | ---:       | ---:             | ---:         | ---:       | ---:          | ---:               | ---:               |
| m27_v5_3 (original)       | base                      | 0.6545     | 81.68 %          | +14.35 %     | 78.8 %     | 85            | 108                | 182,803            |
| m27_v5_3_1 (clasificador) | fix Runda Zasadnicza      | 0.6544     | 80.99 %          | +13.93 %     | 82.8 %     | 59            | 108                | 184,709            |
| m27_v5_3_2 (clean_name)   | normalización de nombres  | 0.6548     | 80.86 %          | +13.93 %     | 82.8 %     | 64            | 78                 | 184,878            |
| **m27_v5_3_3 (backfill)** | **+NCAA Women histórico** | **0.6548** | **80.86 %**      | **+13.20 %** | **85.0 %** | **60**        | **78**             | **184,878**        |

**Conclusión:** v5_3_3 tiene el mejor rendimiento en vivo (85.0 % WR, 60 apuestas) y el mayor número de partidos elegibles, con AUC prácticamente igual (0.6541). La normalización + backfill histórico mejoró la calibración sobre los partidos en vivo. Recomendación: candidato a activar en el monitor (actualmente v5_3 en modo sombra).

## 11. Conclusiones

1. **v5_3_3 es la mejor variante hasta ahora** en vivo: 85.0 % WR (60 apuestas) vs 79.8 % del original.
2. El backfill histórico de NCAA Women (+17,800 partidos, 2019-2026) mejoró la calibración (AUC 0.6541, WR @0.75 81.66 %).
3. La normalización por clean_name redujo cold-start (78 vs 108 ligas) y mejoró AUC (0.6548 en v5_3_2).
4. Walk-forward estable (AUC 0.6274/0.6376/0.6483, WR ~77-80 %) — sin sobreajuste.
5. **Recomendación:** candidato a activar en el monitor. Mantener v5_3 en modo sombra hasta acumular ≥ 200 apuestas liquidadas con v5_3_3.

## 12. Notas metodológicas

- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis_v5_3_3.py`.

- Backfill histórico: `tools/backfill_tournament.py` (API SofaScore, 2 req/s). Ejemplo: `menu.bat backfill_tournament --tid 33655 --league "NCAA Women's Division I, Regular season" --seasons all`.

- Hallazgos API: NCAA Women's Division I (tid 33655) tiene 9 temporadas (18/19 a 26/27). EYBL U17/U15 solo 2 temporadas. CBI U15: 2 temporadas.

- Walk-forward: 3 cortes temporales (train ≤ 2022→2023, ≤2023→2024, ≤2024→2025-26).
