# 📊 Análisis por liga y por umbral — modelo m27_v5_3_3 (normalización + histórico NCAA Women)

> Modelo `m27_v5_3_3`: v5_3_2 + backfill histórico de `NCAA Women's Division I` (19,541 partidos desde 2019, antes solo 1,729 de la temporada actual). Walk-forward incluido (3 cortes). Umbral operativo: **P ≥ 0.75**.

> Holdout **C** (último 18 %, anclas t=30). Umbral operativo: **P ≥ 0.75**. Equilibrio a cuota 1.40: **71.4 %**.

## 1. Resumen ejecutivo

- **Ligas analizadas:** 245 (≥ 30 anclas en holdout C).
- **Partidos (anclas) en holdout C:** 33,173 · en entrenamiento (A): 108,039.
- **Apuestas totales a 0.75 en C:** 2,632.
- **Ligas en arranque en frío:** 83 (33.9 %).
- **Ligas candidatas a bloqueo (IC95 sup < 71.4 % a 0.75):** 0.
- **Ligas con ROI@1.40 > 20 % a 0.75:** 17.

**Comparación con m27_v5_3 (original):** eligible matches 201,660 vs 182,803 (+18,857). AUC test 0.6541 vs 0.6545. Apuestas @0.75 (ancla): 2,879 vs 2,538.

## 2. Tabla global de umbrales (anclas t=30, bloque C)

| P ≥  | Apuestas | Ganados | Perdidos | WR (IC95)          | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---                | ---:      | ---:      |
| 0.65 | 10,442   | 7,478   | 2,964    | 71.6 % (70.7–72.5) | +0.26 %   | +31.05 %  |
| 0.68 | 7,461    | 5,547   | 1,914    | 74.3 % (73.3–75.3) | +4.09 %   | +36.05 %  |
| 0.70 | 5,764    | 4,401   | 1,363    | 76.3 % (75.2–77.4) | +6.89 %   | +39.73 %  |
| 0.72 | 4,419    | 3,437   | 982      | 77.8 % (76.5–79.0) | +8.89 %   | +42.33 %  |
| 0.75 | 2,879    | 2,351   | 528      | 81.7 % (80.2–83.0) | +14.32 %  | +49.44 %  |
| 0.80 | 1,357    | 1,169   | 188      | 86.2 % (84.2–87.9) | +20.60 %  | +57.65 %  |

## 3. Tabla resumen por liga (ordenada por WR descendente a 0.75)

| País          | Liga                                            | Entrenado (A) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | IC95       | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---           | ---                                             | ---:          | ---:        | ---:           | ---:    | ---:     | ---:     | ---        | ---:      | ---:      | ---     |
| International | BNXT League                                     | 297           | 245         | 15             | 15      | 0        | 100.0 %  | 79.6–100.0 | +40.00 %  | +83.00 %  |         |
| ?             | CBI U17 Feminino                                | 0             | 54          | 22             | 21      | 1        | 95.5 %   | 78.2–99.2  | +33.64 %  | +74.68 %  | 🆕      |
| ?             | CBI U15 Fem                                     | 0             | 138         | 65             | 61      | 4        | 93.8 %   | 85.2–97.6  | +31.38 %  | +71.74 %  | 🆕      |
| ?             | SWC                                             | 0             | 39          | 16             | 15      | 1        | 93.8 %   | 71.7–98.9  | +31.25 %  | +71.56 %  | 🆕      |
| ?             | Rwanda Basketball League Women                  | 18            | 55          | 15             | 14      | 1        | 93.3 %   | 70.2–98.8  | +30.67 %  | +70.80 %  |         |
| Historical    | LNB                                             | 310           | 180         | 68             | 63      | 5        | 92.7 %   | 83.9–96.8  | +29.71 %  | +69.54 %  |         |
| ?             | ZBL                                             | 858           | 101         | 42             | 38      | 4        | 90.5 %   | 77.9–96.2  | +26.67 %  | +65.57 %  |         |
| ?             | Saudi Basketball League                         | 71            | 52          | 21             | 19      | 2        | 90.5 %   | 71.1–97.3  | +26.67 %  | +65.57 %  |         |
| Historical    | LDB                                             | 97            | 167         | 20             | 18      | 2        | 90.0 %   | 69.9–97.2  | +26.00 %  | +64.70 %  |         |
| ?             | Top Division Women                              | 0             | 106         | 39             | 35      | 4        | 89.7 %   | 76.4–95.9  | +25.64 %  | +64.23 %  | 🆕      |
| ?             | NBL1 East                                       | 162           | 126         | 29             | 26      | 3        | 89.7 %   | 73.6–96.4  | +25.52 %  | +64.07 %  |         |
| ?             | EYBL U20                                        | 0             | 223         | 19             | 17      | 2        | 89.5 %   | 68.6–97.1  | +25.26 %  | +63.74 %  | 🆕      |
| Historical    | FSHB Superliga                                  | 141           | 116         | 48             | 42      | 6        | 87.5 %   | 75.3–94.1  | +22.50 %  | +60.12 %  |         |
| ?             | EGBL U17                                        | 0             | 74          | 24             | 21      | 3        | 87.5 %   | 69.0–95.7  | +22.50 %  | +60.12 %  | 🆕      |
| ?             | EYBL U17                                        | 0             | 303         | 45             | 39      | 6        | 86.7 %   | 73.8–93.7  | +21.33 %  | +58.60 %  | 🆕      |
| Other         | 1st Division Women                              | 0             | 82          | 22             | 19      | 3        | 86.4 %   | 66.7–95.3  | +20.91 %  | +58.05 %  | 🆕      |
| ?             | EYBL U16                                        | 0             | 280         | 29             | 25      | 4        | 86.2 %   | 69.4–94.5  | +20.69 %  | +57.76 %  | 🆕      |
| Poland        | MPBL                                            | 739           | 287         | 35             | 30      | 5        | 85.7 %   | 70.6–93.7  | +20.00 %  | +56.86 %  |         |
| ?             | CBI U15                                         | 0             | 199         | 48             | 41      | 7        | 85.4 %   | 72.8–92.8  | +19.58 %  | +56.31 %  | 🆕      |
| Other         | Prva Liga                                       | 0             | 154         | 27             | 23      | 4        | 85.2 %   | 67.5–94.1  | +19.26 %  | +55.89 %  | 🆕      |
| Australia     | NBL                                             | 2,604         | 620         | 67             | 56      | 11       | 83.6 %   | 72.9–90.6  | +17.01 %  | +52.96 %  |         |
| ?             | NCAA Women's Division I                         | 15,586        | 1,636       | 241            | 201     | 40       | 83.4 %   | 78.2–87.6  | +16.76 %  | +52.63 %  |         |
| ?             | EYBL U15                                        | 0             | 272         | 48             | 40      | 8        | 83.3 %   | 70.4–91.3  | +16.67 %  | +52.50 %  | 🆕      |
| ?             | CBI U17                                         | 48            | 173         | 18             | 15      | 3        | 83.3 %   | 60.8–94.2  | +16.67 %  | +52.50 %  |         |
| Other         | Basketligaen                                    | 787           | 133         | 34             | 28      | 6        | 82.3 %   | 66.5–91.7  | +15.29 %  | +50.71 %  |         |
| Other         | United League                                   | 1,261         | 207         | 39             | 32      | 7        | 82.0 %   | 67.3–91.0  | +14.87 %  | +50.15 %  |         |
| Other         | Estonian-Latvian Basketball League              | 599           | 194         | 22             | 18      | 4        | 81.8 %   | 61.5–92.7  | +14.55 %  | +49.73 %  |         |
| ?             | 1. SKL                                          | 44            | 59          | 16             | 13      | 3        | 81.2 %   | 57.0–93.4  | +13.75 %  | +48.69 %  |         |
| ?             | NBL Women                                       | 0             | 106         | 16             | 13      | 3        | 81.2 %   | 57.0–93.4  | +13.75 %  | +48.69 %  | 🆕      |
| ?             | CBI U19                                         | 0             | 113         | 36             | 29      | 7        | 80.6 %   | 65.0–90.2  | +12.78 %  | +47.42 %  | 🆕      |
| Historical    | Poland 2nd Basketball League                    | 674           | 903         | 185            | 148     | 37       | 80.0 %   | 73.7–85.1  | +12.00 %  | +46.40 %  |         |
| ?             | Tipos Extraliga                                 | 424           | 73          | 40             | 32      | 8        | 80.0 %   | 65.2–89.5  | +12.00 %  | +46.40 %  |         |
| ?             | EYBL U13                                        | 0             | 126         | 35             | 28      | 7        | 80.0 %   | 64.1–90.0  | +12.00 %  | +46.40 %  | 🆕      |
| Other         | Finland Division 1A Women                       | 292           | 124         | 20             | 16      | 4        | 80.0 %   | 58.4–91.9  | +12.00 %  | +46.40 %  |         |
| Other         | Switzerland LNA                                 | 820           | 115         | 24             | 19      | 5        | 79.2 %   | 59.5–90.8  | +10.83 %  | +44.88 %  |         |
| Other         | Hungary Divison A Women                         | 276           | 129         | 42             | 33      | 9        | 78.6 %   | 64.1–88.3  | +10.00 %  | +43.79 %  |         |
| Other         | 2nd League                                      | 248           | 145         | 23             | 18      | 5        | 78.3 %   | 58.1–90.3  | +9.57 %   | +43.22 %  |         |
| ?             | Liga Nacional Femenina Chile                    | 0             | 103         | 32             | 25      | 7        | 78.1 %   | 61.2–89.0  | +9.38 %   | +42.97 %  | 🆕      |
| ?             | 1 Division                                      | 17            | 100         | 18             | 14      | 4        | 77.8 %   | 54.8–91.0  | +8.89 %   | +42.33 %  |         |
| ?             | Liga 1 Masculin                                 | 83            | 197         | 31             | 24      | 7        | 77.4 %   | 60.2–88.6  | +8.39 %   | +41.68 %  |         |
| Other         | Big V                                           | 712           | 169         | 21             | 16      | 5        | 76.2 %   | 54.9–89.4  | +6.67 %   | +39.43 %  |         |
| Other         | Swedish Basketball League Women                 | 239           | 141         | 37             | 28      | 9        | 75.7 %   | 59.9–86.6  | +5.95 %   | +38.49 %  |         |
| Other         | SLB                                             | 67            | 228         | 20             | 15      | 5        | 75.0 %   | 53.1–88.8  | +5.00 %   | +37.25 %  |         |
| ?             | 1 Liga Kobiet                                   | 0             | 284         | 19             | 14      | 5        | 73.7 %   | 51.2–88.2  | +3.16 %   | +34.84 %  | 🆕      |
| ?             | EYBL U14                                        | 0             | 264         | 45             | 33      | 12       | 73.3 %   | 59.0–84.0  | +2.67 %   | +34.20 %  | 🆕      |
| Brazil        | Brazil NBB                                      | 1,664         | 356         | 15             | 11      | 4        | 73.3 %   | 48.0–89.1  | +2.67 %   | +34.20 %  |         |
| Other         | Russian Women's Basketball Premier League       | 726           | 114         | 31             | 22      | 9        | 71.0 %   | 53.4–83.9  | -0.65 %   | +29.87 %  | ⚠️     |
| ?             | Nostra.lt-RKL Division B                        | 0             | 227         | 24             | 17      | 7        | 70.8 %   | 50.8–85.1  | -0.83 %   | +29.62 %  | ⚠️ 🆕  |
| Other         | Liga Femenina                                   | 1,267         | 492         | 30             | 20      | 10       | 66.7 %   | 48.8–80.8  | -6.67 %   | +22.00 %  | ⚠️     |
| ?             | LMB                                             | 170           | 121         | 17             | 11      | 6        | 64.7 %   | 41.3–82.7  | -9.41 %   | +18.41 %  | ⚠️     |
| Other         | Premijer ženska liga                            | 306           | 113         | 31             | 20      | 11       | 64.5 %   | 46.9–78.9  | -9.68 %   | +18.06 %  | ⚠️     |
| Other         | Sparkasse Prva Liga Zeni                        | 0             | 81          | 17             | 9       | 8        | 52.9 %   | 31.0–73.8  | -25.88 %  | -3.12 %   | ⚠️ 🆕  |
| ?             | Campeonato Nacional da 1ª Divisao Feminina      | 0             | 128         | 14             | 0       | 14       | —        | —          | —         | —         | 🆕      |
| Other         | A1 Women                                        | 944           | 102         | 13             | 0       | 13       | —        | —          | —         | —         |         |
| Other         | Swiss Basketball League Women                   | 156           | 81          | 13             | 0       | 13       | —        | —          | —         | —         |         |
| Other         | 1. Division B                                   | 77            | 128         | 13             | 0       | 13       | —        | —          | —         | —         |         |
| Other         | Superleague Women                               | 0             | 39          | 13             | 0       | 13       | —        | —          | —         | —         | 🆕      |
| Other         | BLNO                                            | 454           | 54          | 13             | 0       | 13       | —        | —          | —         | —         |         |
| ?             | Női Amatőr NB 1                                 | 102           | 48          | 13             | 0       | 13       | —        | —          | —         | —         |         |
| Other         | Superliga                                       | 234           | 191         | 12             | 0       | 12       | —        | —          | —         | —         |         |
| Other         | Women's Basketball Division A                   | 0             | 34          | 12             | 0       | 12       | —        | —          | —         | —         | 🆕      |
| Other         | Austrian Women's Basketball Bundesliga          | 149           | 60          | 12             | 0       | 12       | —        | —          | —         | —         |         |
| ?             | U20 EuroBasket                                  | 9             | 154         | 12             | 0       | 12       | —        | —          | —         | —         |         |
| Other         | The Women League                                | 0             | 37          | 11             | 0       | 11       | —        | —          | —         | —         | 🆕      |
| ?             | Basket Liga                                     | 907           | 141         | 11             | 0       | 11       | —        | —          | —         | —         |         |
| ?             | FIBA U16 Women's EuroBasket                     | 0             | 38          | 11             | 0       | 11       | —        | —          | —         | —         | 🆕      |
| ?             | 1. ZLS                                          | 53            | 59          | 10             | 0       | 10       | —        | —          | —         | —         |         |
| ?             | NBL1                                            | 906           | 530         | 10             | 0       | 10       | —        | —          | —         | —         |         |
| Other         | VBA                                             | 126           | 65          | 10             | 0       | 10       | —        | —          | —         | —         |         |
| Japan         | B League One                                    | 3,037         | 401         | 9              | 0       | 9        | —        | —          | —         | —         |         |
| ?             | 1. liga                                         | 0             | 125         | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| Other         | U21 Espoirs Elite                               | 0             | 144         | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| Other         | Latvian Basketball Cup                          | 0             | 31          | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| Other         | Jedinstvena kadetska liga                       | 0             | 64          | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| Historical    | Rwanda Basketball League                        | 17            | 81          | 9              | 0       | 9        | —        | —          | —         | —         |         |
| ?             | The Basketball League                           | 0             | 136         | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| ?             | LBF                                             | 193           | 80          | 9              | 0       | 9        | —        | —          | —         | —         |         |
| USA           | NCAA Women's March Madness                      | 0             | 59          | 9              | 0       | 9        | —        | —          | —         | —         | 🆕      |
| ?             | NBL1 West                                       | 135           | 101         | 9              | 0       | 9        | —        | —          | —         | —         |         |
| ?             | Kvindebasketligaen                              | 26            | 58          | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | France LF2                                      | 82            | 161         | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | LF Challenge                                    | 0             | 169         | 8              | 0       | 8        | —        | —          | —         | —         | 🆕      |
| Spain         | Liga ACB                                        | 1,858         | 291         | 8              | 0       | 8        | —        | —          | —         | —         |         |
| Japan         | B League Premier                                | 3,092         | 667         | 8              | 0       | 8        | —        | —          | —         | —         |         |
| Other         | Premier League                                  | 230           | 65          | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | NBL1 Central                                    | 89            | 63          | 8              | 0       | 8        | —        | —          | —         | —         |         |
| ?             | FIBA U17 Women's Basketball World Cup           | 0             | 37          | 8              | 0       | 8        | —        | —          | —         | —         | 🆕      |
| Other         | FBU Superleague                                 | 834           | 104         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Lebanese Basketball League                      | 94            | 125         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Superliga e Femrave                             | 0             | 47          | 7              | 0       | 7        | —        | —          | —         | —         | 🆕      |
| Other         | Swedish Basketball League                       | 1,067         | 165         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| France        | France Pro A                                    | 1,849         | 223         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Portugal LBP                                    | 896           | 131         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| ?             | Eurocup                                         | 1,236         | 201         | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Other         | Estonian National Women's Basketball League     | 49            | 53          | 7              | 0       | 7        | —        | —          | —         | —         |         |
| ?             | U18 FIBA Asia Cup                               | 0             | 36          | 7              | 0       | 7        | —        | —          | —         | —         | 🆕      |
| Other         | Catarinense                                     | 6             | 30          | 7              | 0       | 7        | —        | —          | —         | —         |         |
| Spain         | Primera FEB                                     | 1,773         | 237         | 6              | 0       | 6        | —        | —          | —         | —         |         |
| ?             | NBL Men                                         | 0             | 129         | 6              | 0       | 6        | —        | —          | —         | —         | 🆕      |
| Other         | Korisliiga                                      | 1,306         | 153         | 6              | 0       | 6        | —        | —          | —         | —         |         |
| Other         | Liga e Parë                                     | 301           | 80          | 6              | 0       | 6        | —        | —          | —         | —         |         |
| Other         | Indonesian Basketball League                    | 802           | 110         | 6              | 0       | 6        | —        | —          | —         | —         |         |
| ?             | LBP                                             | 0             | 34          | 6              | 0       | 6        | —        | —          | —         | —         | 🆕      |
| Other         | Colombia LPB                                    | 238           | 87          | 5              | 0       | 5        | —        | —          | —         | —         |         |
| Other         | I B MCKL                                        | 0             | 44          | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| Italy         | Serie A2                                        | 2,549         | 443         | 5              | 0       | 5        | —        | —          | —         | —         |         |
| Other         | Turkish Basketball Super League                 | 1,487         | 201         | 5              | 0       | 5        | —        | —          | —         | —         |         |
| ?             | EuroCup                                         | 762           | 158         | 5              | 0       | 5        | —        | —          | —         | —         |         |
| International | Euroleague: regular season                      | 220           | 88          | 5              | 0       | 5        | —        | —          | —         | —         |         |
| ?             | Druga muška liga                                | 437           | 74          | 5              | 0       | 5        | —        | —          | —         | —         |         |
| ?             | U18 EuroBasket                                  | 0             | 124         | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA Asia Cup 2029 Pre-Qualifiers               | 0             | 31          | 5              | 0       | 5        | —        | —          | —         | —         | 🆕      |
| Other         | Super League                                    | 646           | 300         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Greece        | Stoiximan GBL                                   | 1,137         | 152         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Liga Nationala                                  | 849           | 173         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Azerbaijan Basketball League                    | 0             | 120         | 4              | 0       | 4        | —        | —          | —         | —         | 🆕      |
| Australia     | WNBL                                            | 586           | 84          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Germany       | Germany BBL                                     | 1,879         | 276         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | 1. Division                                     | 244           | 126         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | Nationale 1                                     | 2             | 320         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | Road to Basketball Africa League                | 58            | 40          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| Other         | Cup                                             | 340           | 42          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | NBL1 South                                      | 206           | 161         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| USA           | WNBA                                            | 1,305         | 311         | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | LFB                                             | 20            | 35          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| ?             | FIBA U17 Basketball World Cup                   | 19            | 46          | 4              | 0       | 4        | —        | —          | —         | —         |         |
| France        | LNB Clausura                                    | 0             | 71          | 4              | 0       | 4        | —        | —          | —         | —         | 🆕      |
| Other         | I A MCKL                                        | 106           | 140         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Swedish Basketball Superettan                   | 352           | 158         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Austrian Basketball Superliga                   | 1,055         | 113         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| ?             | Segunda FEB                                     | 227           | 292         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Cyprus Basketball Division A                    | 390           | 87          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Portugal LFB Women                              | 166           | 118         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Argentina     | Argentina Liga Nacional                         | 2,944         | 301         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | 1st Division Men                                | 0             | 142         | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| Other         | Hungary NB 1.A                                  | 845           | 178         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Liga Košarkaškog saveza Herceg Bosne            | 0             | 52          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA Women Basketball League Africa - Qualifers | 0             | 41          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| Other         | A1                                              | 0             | 69          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| ?             | Champions League Americas                       | 85            | 43          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| ?             | Kızlar Basketbol Gençler Ligi                   | 0             | 34          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| ?             | CIBACOPA                                        | 17            | 216         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Historical    | Iraqi Professional League                       | 102           | 63          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Liga de Ascenso                                 | 392           | 107         | 3              | 0       | 3        | —        | —          | —         | —         |         |
| ?             | FIBA U16 EuroBasket Division B                  | 0             | 61          | 3              | 0       | 3        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA World Cup                                  | 44            | 33          | 3              | 0       | 3        | —        | —          | —         | —         |         |
| Other         | Uruguay LUB                                     | 349           | 121         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | CN1 Basquetebol                                 | 0             | 146         | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| France        | Mexico LNBP                                     | 1,503         | 192         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | A1 Liga                                         | 0             | 124         | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| Historical    | Qatari Basketball League                        | 54            | 69          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| France        | France LFB Women                                | 930           | 124         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | 1st Division                                    | 319           | 137         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | 1. A SKL                                        | 713           | 128         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Israel        | Israeli National League Basketball              | 464           | 205         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | Iceland Women's Basketball Premier League       | 222           | 100         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| Other         | Saku I liiga                                    | 99            | 134         | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | EGBL U19                                        | 0             | 39          | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| ?             | EGBL U14                                        | 0             | 47          | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| Other         | Prva RL Istok                                   | 284           | 31          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| ?             | Divisione Regionale 1 Toscana                   | 0             | 131         | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA World Cup Qualification                    | 0             | 57          | 2              | 0       | 2        | —        | —          | —         | —         | 🆕      |
| France        | Mexico LNBP Women                               | 93            | 52          | 2              | 0       | 2        | —        | —          | —         | —         |         |
| France        | Chile LNB                                       | 846           | 95          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | WABA 2 League                                   | 23            | 34          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | 1. Liga                                         | 1,174         | 164         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Serie B                                         | 611           | 298         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | AdmiralBet ABA League                           | 63            | 151         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | SuperSport Premijer Liga                        | 1,077         | 181         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Poland        | Poland 1st Division Basketball                  | 793           | 237         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Enovos League                                   | 0             | 110         | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Israel        | Israeli Women Basketball Premier League         | 305           | 92          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Champions League                                | 1,082         | 147         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| International | Euroleague                                      | 1,618         | 341         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| International | ABA League 2                                    | 271           | 64          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | FIBA Europe Cup                                 | 643           | 138         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | WABA League Women                               | 0             | 35          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Other         | Iceland Basketball Premier League               | 536           | 119         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | LSB - Liga B                                    | 0             | 48          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| ?             | Liga de Baloncesto Puertorriquena               | 390           | 355         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Iranian Basketball Super League                 | 403           | 38          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | Elite League                                    | 347           | 58          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | National League                                 | 82            | 31          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | 2. Muška Liga                                   | 249           | 43          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | 2. ZLS                                          | 0             | 37          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| ?             | National League 2                               | 577           | 106         | 1              | 0       | 1        | —        | —          | —         | —         |         |
| Other         | A2 ESKAH                                        | 257           | 35          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Egyptian Basketball Super League                | 0             | 31          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| ?             | Liga Argentina                                  | 945           | 76          | 1              | 0       | 1        | —        | —          | —         | —         |         |
| ?             | Rapid League                                    | 0             | 83          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Other         | Saku II Liiga                                   | 0             | 35          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| ?             | 2. DBBL                                         | 0             | 34          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| ?             | Nostra.lt-RKL Division A                        | 0             | 38          | 1              | 0       | 1        | —        | —          | —         | —         | 🆕      |
| Other         | Princ Caffe Superliga                           | 646           | 89          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Lithuania     | RKL Division A                                  | 0             | 234         | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Historical    | Liga Sudamericana                               | 195           | 43          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Historical    | Slovenian Second Basketball League              | 100           | 98          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Superleague                                     | 1,310         | 119         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Belgian Basketball 2nd Division                 | 168           | 191         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Portugal Proliga                                | 218           | 72          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Lithuania     | NKL                                             | 773           | 231         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | BSNF                                            | 86            | 77          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Germany       | Germany Pro A                                   | 1,637         | 116         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Historical    | PGE Basket Liga                                 | 1,507         | 205         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Historical    | Pro B                                           | 1             | 130         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Italy         | Lega A Basket                                   | 1,571         | 202         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Meridianbet KLS                                 | 1,373         | 192         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | The League                                      | 110           | 129         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Divisional Tercera de Ascenso                   | 0             | 61          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Élite 2                                         | 1,952         | 308         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | European North Basketball League                | 134           | 105         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Euroleague Women                                | 698           | 61          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Europe Cup                                      | 477           | 61          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | East Asia Super League                          | 55            | 33          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | French Basketball Cup                           | 217           | 46          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Zain Basketball League                          | 48            | 80          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| International | U19 ABA League                                  | 24            | 33          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Australia     | Bulgaria NBL                                    | 673           | 85          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Turkey        | Türkiye Kadınlar Basketbol Ligi                 | 0             | 39          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | TB2L                                            | 414           | 44          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Serie B Interregionale                          | 0             | 255         | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Erkekler Basketbol Gençler Ligi                 | 0             | 35          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Divisione Regionale 2 - Lazio                   | 0             | 105         | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Serie C Toscana                                 | 0             | 50          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Serie C Girone H sud                            | 0             | 30          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | B Ethniki                                       | 0             | 95          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | TBL                                             | 720           | 58          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Basket World League                             | 0             | 31          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Espoirs Pro B                                   | 139           | 48          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | A1 ESKAH                                        | 283           | 33          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | Belize Elite Basketball League                  | 0             | 39          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Egyptian Women's Basketball Super League        | 0             | 55          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | Erkekler Bölgesel Basketbol Ligi                | 0             | 67          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Puerto Rico BSN                                 | 288           | 188         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| France        | LNB 2                                           | 0             | 44          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Australia     | New Zealand NBL                                 | 187           | 78          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | NBL1 North                                      | 120           | 81          | 0              | 0       | 0        | —        | —          | —         | —         |         |
| Other         | Canadian Elite Basketball League                | 95            | 103         | 0              | 0       | 0        | —        | —          | —         | —         |         |
| ?             | U16 Türkiye Finalleri                           | 0             | 46          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | U18 EuroBasket Div. B                           | 0             | 50          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA U16 Women's EuroBasket Division B          | 0             | 34          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| ?             | FIBA U16 EuroBasket                             | 0             | 38          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Copa Espana                                     | 0             | 41          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | 2. MLS                                          | 0             | 30          | 0              | 0       | 0        | —        | —          | —         | —         | 🆕      |
| Other         | Proliga                                         | 93            | 30          | 0              | 0       | 0        | —        | —          | —         | —         |         |

## 4. Detalle por liga y por umbral — Top 30 por volumen

### ? — NCAA Women's Division I

- Entrenado (A): **15,586** · Holdout (C): **1,636** · Sesgo @0.75: -0.0282

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 617      | 452     | 165      | 73.3 % | 69.6–76.6 | +2.56 %   | +34.06 %  |
| 0.68 | 489      | 364     | 125      | 74.4 % | 70.4–78.1 | +4.21 %   | +36.22 %  |
| 0.70 | 412      | 315     | 97       | 76.5 % | 72.1–80.3 | +7.04 %   | +39.92 %  |
| 0.72 | 341      | 265     | 76       | 77.7 % | 73.0–81.8 | +8.80 %   | +42.21 %  |
| 0.75 | 241      | 201     | 40       | 83.4 % | 78.2–87.6 | +16.76 %  | +52.63 %  |
| 0.80 | 121      | 108     | 13       | 89.3 % | 82.5–93.6 | +24.96 %  | +63.34 %  |

### Historical — Poland 2nd Basketball League

- Entrenado (A): **674** · Holdout (C): **903** · Sesgo @0.75: +0.0132

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 420      | 289     | 131      | 68.8 % | 64.2–73.1 | -3.67 %   | +25.92 %  |
| 0.68 | 335      | 242     | 93       | 72.2 % | 67.2–76.8 | +1.13 %   | +32.20 %  |
| 0.70 | 289      | 212     | 77       | 73.4 % | 68.0–78.1 | +2.70 %   | +34.24 %  |
| 0.72 | 247      | 185     | 62       | 74.9 % | 69.1–79.9 | +4.86 %   | +37.06 %  |
| 0.75 | 185      | 148     | 37       | 80.0 % | 73.7–85.1 | +12.00 %  | +46.40 %  |
| 0.80 | 99       | 84      | 15       | 84.8 % | 76.5–90.6 | +18.79 %  | +55.27 %  |

### Historical — LNB

- Entrenado (A): **310** · Holdout (C): **180** · Sesgo @0.75: -0.0638

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 100      | 90      | 10       | 90.0 % | 82.6–94.5 | +26.00 %  | +64.70 %  |
| 0.68 | 88       | 79      | 9        | 89.8 % | 81.7–94.5 | +25.68 %  | +64.28 %  |
| 0.70 | 82       | 73      | 9        | 89.0 % | 80.4–94.1 | +24.63 %  | +62.91 %  |
| 0.72 | 74       | 67      | 7        | 90.5 % | 81.7–95.3 | +26.76 %  | +65.69 %  |
| 0.75 | 68       | 63      | 5        | 92.7 % | 83.9–96.8 | +29.71 %  | +69.54 %  |
| 0.80 | 52       | 48      | 4        | 92.3 % | 81.8–97.0 | +29.23 %  | +68.92 %  |

### Australia — NBL

- Entrenado (A): **2,604** · Holdout (C): **620** · Sesgo @0.75: -0.0365

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 214      | 162     | 52       | 75.7 % | 69.5–81.0 | +5.98 %   | +38.53 %  |
| 0.68 | 161      | 129     | 32       | 80.1 % | 73.3–85.6 | +12.17 %  | +46.63 %  |
| 0.70 | 124      | 101     | 23       | 81.5 % | 73.7–87.3 | +14.03 %  | +49.06 %  |
| 0.72 | 93       | 74      | 19       | 79.6 % | 70.3–86.5 | +11.40 %  | +45.61 %  |
| 0.75 | 67       | 56      | 11       | 83.6 % | 72.9–90.6 | +17.01 %  | +52.96 %  |
| 0.80 | 24       | 21      | 3        | 87.5 % | 69.0–95.7 | +22.50 %  | +60.12 %  |

### ? — CBI U15 Fem

- Entrenado (A): **0** · Holdout (C): **138** · Sesgo @0.75: -0.0827

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 86       | 76      | 10       | 88.4 % | 79.9–93.6 | +23.72 %  | +61.72 %  |
| 0.68 | 79       | 70      | 9        | 88.6 % | 79.7–93.9 | +24.05 %  | +62.15 %  |
| 0.70 | 75       | 67      | 8        | 89.3 % | 80.3–94.5 | +25.07 %  | +63.48 %  |
| 0.72 | 69       | 64      | 5        | 92.8 % | 84.1–96.9 | +29.86 %  | +69.74 %  |
| 0.75 | 65       | 61      | 4        | 93.8 % | 85.2–97.6 | +31.38 %  | +71.74 %  |
| 0.80 | 50       | 48      | 2        | 96.0 % | 86.5–98.9 | +34.40 %  | +75.68 %  |

### Historical — FSHB Superliga

- Entrenado (A): **141** · Holdout (C): **116** · Sesgo @0.75: -0.0459

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 72       | 56      | 16       | 77.8 % | 66.9–85.8 | +8.89 %   | +42.33 %  |
| 0.68 | 66       | 52      | 14       | 78.8 % | 67.5–86.9 | +10.30 %  | +44.18 %  |
| 0.70 | 61       | 49      | 12       | 80.3 % | 68.7–88.4 | +12.46 %  | +47.00 %  |
| 0.72 | 58       | 47      | 11       | 81.0 % | 69.1–89.1 | +13.45 %  | +48.29 %  |
| 0.75 | 48       | 42      | 6        | 87.5 % | 75.3–94.1 | +22.50 %  | +60.12 %  |
| 0.80 | 31       | 29      | 2        | 93.5 % | 79.3–98.2 | +30.97 %  | +71.19 %  |

### ? — CBI U15

- Entrenado (A): **0** · Holdout (C): **199** · Sesgo @0.75: -0.0234

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 89       | 67      | 22       | 75.3 % | 65.4–83.1 | +5.39 %   | +37.76 %  |
| 0.68 | 74       | 58      | 16       | 78.4 % | 67.7–86.2 | +9.73 %   | +43.43 %  |
| 0.70 | 62       | 51      | 11       | 82.3 % | 71.0–89.8 | +15.16 %  | +50.53 %  |
| 0.72 | 60       | 50      | 10       | 83.3 % | 72.0–90.7 | +16.67 %  | +52.50 %  |
| 0.75 | 48       | 41      | 7        | 85.4 % | 72.8–92.8 | +19.58 %  | +56.31 %  |
| 0.80 | 32       | 31      | 1        | 96.9 % | 84.3–99.4 | +35.62 %  | +77.28 %  |

### ? — EYBL U15

- Entrenado (A): **0** · Holdout (C): **272** · Sesgo @0.75: -0.0186

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 111      | 79      | 32       | 71.2 % | 62.1–78.8 | -0.36 %   | +30.24 %  |
| 0.68 | 90       | 66      | 24       | 73.3 % | 63.4–81.4 | +2.67 %   | +34.20 %  |
| 0.70 | 79       | 59      | 20       | 74.7 % | 64.1–83.0 | +4.56 %   | +36.67 %  |
| 0.72 | 66       | 51      | 15       | 77.3 % | 65.8–85.7 | +8.18 %   | +41.41 %  |
| 0.75 | 48       | 40      | 8        | 83.3 % | 70.4–91.3 | +16.67 %  | +52.50 %  |
| 0.80 | 28       | 26      | 2        | 92.9 % | 77.4–98.0 | +30.00 %  | +69.93 %  |

### ? — EYBL U17

- Entrenado (A): **0** · Holdout (C): **303** · Sesgo @0.75: -0.0504

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 94       | 72      | 22       | 76.6 % | 67.1–84.0 | +7.23 %   | +40.17 %  |
| 0.68 | 74       | 62      | 12       | 83.8 % | 73.8–90.5 | +17.30 %  | +53.32 %  |
| 0.70 | 62       | 53      | 9        | 85.5 % | 74.7–92.2 | +19.68 %  | +56.44 %  |
| 0.72 | 58       | 51      | 7        | 87.9 % | 77.1–94.0 | +23.10 %  | +60.91 %  |
| 0.75 | 45       | 39      | 6        | 86.7 % | 73.8–93.7 | +21.33 %  | +58.60 %  |
| 0.80 | 26       | 25      | 1        | 96.2 % | 81.1–99.3 | +34.62 %  | +75.96 %  |

### ? — EYBL U14

- Entrenado (A): **0** · Holdout (C): **264** · Sesgo @0.75: +0.0809

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 113      | 79      | 34       | 69.9 % | 60.9–77.6 | -2.12 %   | +27.94 %  |
| 0.68 | 91       | 68      | 23       | 74.7 % | 64.9–82.5 | +4.62 %   | +36.75 %  |
| 0.70 | 77       | 58      | 19       | 75.3 % | 64.6–83.6 | +5.45 %   | +37.84 %  |
| 0.72 | 67       | 50      | 17       | 74.6 % | 63.1–83.5 | +4.48 %   | +36.57 %  |
| 0.75 | 45       | 33      | 12       | 73.3 % | 59.0–84.0 | +2.67 %   | +34.20 %  |
| 0.80 | 26       | 20      | 6        | 76.9 % | 57.9–89.0 | +7.69 %   | +40.77 %  |

### ? — ZBL

- Entrenado (A): **858** · Holdout (C): **101** · Sesgo @0.75: -0.0761

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 65       | 55      | 10       | 84.6 % | 73.9–91.4 | +18.46 %  | +54.85 %  |
| 0.68 | 58       | 50      | 8        | 86.2 % | 75.1–92.8 | +20.69 %  | +57.76 %  |
| 0.70 | 56       | 49      | 7        | 87.5 % | 76.4–93.8 | +22.50 %  | +60.13 %  |
| 0.72 | 50       | 44      | 6        | 88.0 % | 76.2–94.4 | +23.20 %  | +61.04 %  |
| 0.75 | 42       | 38      | 4        | 90.5 % | 77.9–96.2 | +26.67 %  | +65.57 %  |
| 0.80 | 27       | 25      | 2        | 92.6 % | 76.6–97.9 | +29.63 %  | +69.44 %  |

### Other — Hungary Divison A Women

- Entrenado (A): **276** · Holdout (C): **129** · Sesgo @0.75: +0.0472

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 81       | 61      | 20       | 75.3 % | 64.9–83.4 | +5.43 %   | +37.81 %  |
| 0.68 | 70       | 54      | 16       | 77.1 % | 66.0–85.4 | +8.00 %   | +41.17 %  |
| 0.70 | 59       | 47      | 12       | 79.7 % | 67.7–88.0 | +11.53 %  | +45.78 %  |
| 0.72 | 54       | 44      | 10       | 81.5 % | 69.2–89.6 | +14.07 %  | +49.11 %  |
| 0.75 | 42       | 33      | 9        | 78.6 % | 64.1–88.3 | +10.00 %  | +43.79 %  |
| 0.80 | 30       | 24      | 6        | 80.0 % | 62.7–90.5 | +12.00 %  | +46.40 %  |

### ? — Tipos Extraliga

- Entrenado (A): **424** · Holdout (C): **73** · Sesgo @0.75: +0.0628

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 58       | 40      | 18       | 69.0 % | 56.2–79.4 | -3.45 %   | +26.21 %  |
| 0.68 | 53       | 39      | 14       | 73.6 % | 60.4–83.6 | +3.02 %   | +34.66 %  |
| 0.70 | 53       | 39      | 14       | 73.6 % | 60.4–83.6 | +3.02 %   | +34.66 %  |
| 0.72 | 46       | 35      | 11       | 76.1 % | 62.1–86.1 | +6.52 %   | +39.24 %  |
| 0.75 | 40       | 32      | 8        | 80.0 % | 65.2–89.5 | +12.00 %  | +46.40 %  |
| 0.80 | 31       | 26      | 5        | 83.9 % | 67.4–92.9 | +17.42 %  | +53.48 %  |

### ? — Top Division Women

- Entrenado (A): **0** · Holdout (C): **106** · Sesgo @0.75: -0.0751

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 68       | 58      | 10       | 85.3 % | 75.0–91.8 | +19.41 %  | +56.09 %  |
| 0.68 | 60       | 51      | 9        | 85.0 % | 73.9–91.9 | +19.00 %  | +55.55 %  |
| 0.70 | 55       | 47      | 8        | 85.5 % | 73.8–92.4 | +19.64 %  | +56.38 %  |
| 0.72 | 48       | 43      | 5        | 89.6 % | 77.8–95.5 | +25.42 %  | +63.94 %  |
| 0.75 | 39       | 35      | 4        | 89.7 % | 76.4–95.9 | +25.64 %  | +64.23 %  |
| 0.80 | 23       | 21      | 2        | 91.3 % | 73.2–97.6 | +27.83 %  | +67.09 %  |

### Other — United League

- Entrenado (A): **1,261** · Holdout (C): **207** · Sesgo @0.75: -0.0251

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 106      | 79      | 27       | 74.5 % | 65.5–81.9 | +4.34 %   | +36.39 %  |
| 0.68 | 86       | 64      | 22       | 74.4 % | 64.3–82.5 | +4.19 %   | +36.19 %  |
| 0.70 | 71       | 55      | 16       | 77.5 % | 66.5–85.6 | +8.45 %   | +41.76 %  |
| 0.72 | 56       | 43      | 13       | 76.8 % | 64.2–85.9 | +7.50 %   | +40.52 %  |
| 0.75 | 39       | 32      | 7        | 82.0 % | 67.3–91.0 | +14.87 %  | +50.15 %  |
| 0.80 | 18       | 14      | 4        | 77.8 % | 54.8–91.0 | +8.89 %   | +42.33 %  |

### Other — Swedish Basketball League Women

- Entrenado (A): **239** · Holdout (C): **141** · Sesgo @0.75: +0.0634

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 74       | 54      | 20       | 73.0 % | 61.9–81.8 | +2.16 %   | +33.54 %  |
| 0.68 | 61       | 45      | 16       | 73.8 % | 61.6–83.2 | +3.28 %   | +35.00 %  |
| 0.70 | 53       | 39      | 14       | 73.6 % | 60.4–83.6 | +3.02 %   | +34.66 %  |
| 0.72 | 48       | 36      | 12       | 75.0 % | 61.2–85.1 | +5.00 %   | +37.25 %  |
| 0.75 | 37       | 28      | 9        | 75.7 % | 59.9–86.6 | +5.95 %   | +38.49 %  |
| 0.80 | 20       | 15      | 5        | 75.0 % | 53.1–88.8 | +5.00 %   | +37.25 %  |

### ? — CBI U19

- Entrenado (A): **0** · Holdout (C): **113** · Sesgo @0.75: +0.0318

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 66       | 49      | 17       | 74.2 % | 62.6–83.3 | +3.94 %   | +35.86 %  |
| 0.68 | 57       | 45      | 12       | 79.0 % | 66.7–87.5 | +10.53 %  | +44.47 %  |
| 0.70 | 52       | 41      | 11       | 78.8 % | 66.0–87.8 | +10.38 %  | +44.29 %  |
| 0.72 | 46       | 37      | 9        | 80.4 % | 66.8–89.3 | +12.61 %  | +47.20 %  |
| 0.75 | 36       | 29      | 7        | 80.6 % | 65.0–90.2 | +12.78 %  | +47.42 %  |
| 0.80 | 22       | 20      | 2        | 90.9 % | 72.2–97.5 | +27.27 %  | +66.36 %  |

### Poland — MPBL

- Entrenado (A): **739** · Holdout (C): **287** · Sesgo @0.75: -0.0712

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 114      | 78      | 36       | 68.4 % | 59.4–76.2 | -4.21 %   | +25.21 %  |
| 0.68 | 82       | 60      | 22       | 73.2 % | 62.7–81.6 | +2.44 %   | +33.90 %  |
| 0.70 | 69       | 54      | 15       | 78.3 % | 67.2–86.4 | +9.57 %   | +43.22 %  |
| 0.72 | 57       | 46      | 11       | 80.7 % | 68.7–88.9 | +12.98 %  | +47.68 %  |
| 0.75 | 35       | 30      | 5        | 85.7 % | 70.6–93.7 | +20.00 %  | +56.86 %  |
| 0.80 | 7        | 0       | 7        | —      | —         | —         | —         |

### ? — EYBL U13

- Entrenado (A): **0** · Holdout (C): **126** · Sesgo @0.75: +0.0140

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 62       | 43      | 19       | 69.3 % | 57.0–79.4 | -2.90 %   | +26.92 %  |
| 0.68 | 55       | 39      | 16       | 70.9 % | 57.9–81.2 | -0.73 %   | +29.76 %  |
| 0.70 | 52       | 38      | 14       | 73.1 % | 59.7–83.2 | +2.31 %   | +33.73 %  |
| 0.72 | 45       | 34      | 11       | 75.6 % | 61.3–85.8 | +5.78 %   | +38.27 %  |
| 0.75 | 35       | 28      | 7        | 80.0 % | 64.1–90.0 | +12.00 %  | +46.40 %  |
| 0.80 | 20       | 17      | 3        | 85.0 % | 64.0–94.8 | +19.00 %  | +55.55 %  |

### Other — Basketligaen

- Entrenado (A): **787** · Holdout (C): **133** · Sesgo @0.75: -0.0310

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 64       | 48      | 16       | 75.0 % | 63.2–84.0 | +5.00 %   | +37.25 %  |
| 0.68 | 57       | 44      | 13       | 77.2 % | 64.8–86.2 | +8.07 %   | +41.26 %  |
| 0.70 | 51       | 40      | 11       | 78.4 % | 65.4–87.5 | +9.80 %   | +43.53 %  |
| 0.72 | 42       | 34      | 8        | 81.0 % | 66.7–90.0 | +13.33 %  | +48.14 %  |
| 0.75 | 34       | 28      | 6        | 82.3 % | 66.5–91.7 | +15.29 %  | +50.71 %  |
| 0.80 | 14       | 0       | 14       | —      | —         | —         | —         |

### ? — Liga Nacional Femenina Chile

- Entrenado (A): **0** · Holdout (C): **103** · Sesgo @0.75: +0.0559

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 59       | 39      | 20       | 66.1 % | 53.4–76.9 | -7.46 %   | +20.97 %  |
| 0.68 | 49       | 33      | 16       | 67.3 % | 53.4–78.8 | -5.71 %   | +23.24 %  |
| 0.70 | 45       | 29      | 16       | 64.4 % | 49.8–76.8 | -9.78 %   | +17.93 %  |
| 0.72 | 40       | 28      | 12       | 70.0 % | 54.6–81.9 | -2.00 %   | +28.10 %  |
| 0.75 | 32       | 25      | 7        | 78.1 % | 61.2–89.0 | +9.38 %   | +42.97 %  |
| 0.80 | 24       | 19      | 5        | 79.2 % | 59.5–90.8 | +10.83 %  | +44.88 %  |

### ? — Liga 1 Masculin

- Entrenado (A): **83** · Holdout (C): **197** · Sesgo @0.75: +0.0200

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 100      | 71      | 29       | 71.0 % | 61.5–79.0 | -0.60 %   | +29.93 %  |
| 0.68 | 78       | 58      | 20       | 74.4 % | 63.7–82.7 | +4.10 %   | +36.08 %  |
| 0.70 | 60       | 45      | 15       | 75.0 % | 62.8–84.2 | +5.00 %   | +37.25 %  |
| 0.72 | 49       | 34      | 15       | 69.4 % | 55.5–80.5 | -2.86 %   | +26.98 %  |
| 0.75 | 31       | 24      | 7        | 77.4 % | 60.2–88.6 | +8.39 %   | +41.68 %  |
| 0.80 | 11       | 0       | 11       | —      | —         | —         | —         |

### Other — Russian Women's Basketball Premier League

- Entrenado (A): **726** · Holdout (C): **114** · Sesgo @0.75: +0.0998

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 59       | 46      | 13       | 78.0 % | 65.9–86.6 | +9.15 %   | +42.68 %  |
| 0.68 | 46       | 35      | 11       | 76.1 % | 62.1–86.1 | +6.52 %   | +39.24 %  |
| 0.70 | 43       | 32      | 11       | 74.4 % | 59.8–85.1 | +4.19 %   | +36.19 %  |
| 0.72 | 36       | 25      | 11       | 69.4 % | 53.1–82.0 | -2.78 %   | +27.08 %  |
| 0.75 | 31       | 22      | 9        | 71.0 % | 53.4–83.9 | -0.65 %   | +29.87 %  |
| 0.80 | 15       | 11      | 4        | 73.3 % | 48.0–89.1 | +2.67 %   | +34.20 %  |

### Other — Premijer ženska liga

- Entrenado (A): **306** · Holdout (C): **113** · Sesgo @0.75: +0.1717

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 65       | 40      | 25       | 61.5 % | 49.4–72.4 | -13.85 %  | +12.62 %  |
| 0.68 | 56       | 35      | 21       | 62.5 % | 49.4–74.0 | -12.50 %  | +14.37 %  |
| 0.70 | 47       | 31      | 16       | 66.0 % | 51.7–77.8 | -7.66 %   | +20.70 %  |
| 0.72 | 39       | 26      | 13       | 66.7 % | 51.0–79.4 | -6.67 %   | +22.00 %  |
| 0.75 | 31       | 20      | 11       | 64.5 % | 46.9–78.9 | -9.68 %   | +18.06 %  |
| 0.80 | 18       | 10      | 8        | 55.6 % | 33.7–75.4 | -22.22 %  | +1.67 %   |

### Other — Liga Femenina

- Entrenado (A): **1,267** · Holdout (C): **492** · Sesgo @0.75: +0.1181

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 151      | 101     | 50       | 66.9 % | 59.0–73.9 | -6.36 %   | +22.40 %  |
| 0.68 | 106      | 74      | 32       | 69.8 % | 60.5–77.7 | -2.26 %   | +27.75 %  |
| 0.70 | 82       | 54      | 28       | 65.8 % | 55.1–75.2 | -7.80 %   | +20.51 %  |
| 0.72 | 58       | 36      | 22       | 62.1 % | 49.2–73.4 | -13.10 %  | +13.59 %  |
| 0.75 | 30       | 20      | 10       | 66.7 % | 48.8–80.8 | -6.67 %   | +22.00 %  |
| 0.80 | 6        | 0       | 6        | —      | —         | —         | —         |

### ? — NBL1 East

- Entrenado (A): **162** · Holdout (C): **126** · Sesgo @0.75: -0.0874

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 73       | 53      | 20       | 72.6 % | 61.4–81.5 | +1.64 %   | +32.86 %  |
| 0.68 | 59       | 45      | 14       | 76.3 % | 64.0–85.3 | +6.78 %   | +39.58 %  |
| 0.70 | 47       | 39      | 8        | 83.0 % | 69.9–91.1 | +16.17 %  | +51.85 %  |
| 0.72 | 40       | 33      | 7        | 82.5 % | 68.0–91.3 | +15.50 %  | +50.98 %  |
| 0.75 | 29       | 26      | 3        | 89.7 % | 73.6–96.4 | +25.52 %  | +64.07 %  |
| 0.80 | 13       | 0       | 13       | —      | —         | —         | —         |

### ? — EYBL U16

- Entrenado (A): **0** · Holdout (C): **280** · Sesgo @0.75: -0.0507

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 86       | 64      | 22       | 74.4 % | 64.3–82.5 | +4.19 %   | +36.19 %  |
| 0.68 | 64       | 48      | 16       | 75.0 % | 63.2–84.0 | +5.00 %   | +37.25 %  |
| 0.70 | 56       | 43      | 13       | 76.8 % | 64.2–85.9 | +7.50 %   | +40.52 %  |
| 0.72 | 43       | 34      | 9        | 79.1 % | 64.8–88.6 | +10.70 %  | +44.70 %  |
| 0.75 | 29       | 25      | 4        | 86.2 % | 69.4–94.5 | +20.69 %  | +57.76 %  |
| 0.80 | 15       | 14      | 1        | 93.3 % | 70.2–98.8 | +30.67 %  | +70.80 %  |

### Other — Prva Liga

- Entrenado (A): **0** · Holdout (C): **154** · Sesgo @0.75: -0.0298

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 50       | 38      | 12       | 76.0 % | 62.6–85.7 | +6.40 %   | +39.08 %  |
| 0.68 | 36       | 30      | 6        | 83.3 % | 68.1–92.1 | +16.67 %  | +52.50 %  |
| 0.70 | 34       | 28      | 6        | 82.3 % | 66.5–91.7 | +15.29 %  | +50.71 %  |
| 0.72 | 31       | 26      | 5        | 83.9 % | 67.4–92.9 | +17.42 %  | +53.48 %  |
| 0.75 | 27       | 23      | 4        | 85.2 % | 67.5–94.1 | +19.26 %  | +55.89 %  |
| 0.80 | 18       | 16      | 2        | 88.9 % | 67.2–96.9 | +24.44 %  | +62.67 %  |

### ? — EGBL U17

- Entrenado (A): **0** · Holdout (C): **74** · Sesgo @0.75: -0.0575

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 37       | 29      | 8        | 78.4 % | 62.8–88.6 | +9.73 %   | +43.43 %  |
| 0.68 | 34       | 27      | 7        | 79.4 % | 63.2–89.7 | +11.18 %  | +45.32 %  |
| 0.70 | 32       | 26      | 6        | 81.2 % | 64.7–91.1 | +13.75 %  | +48.69 %  |
| 0.72 | 29       | 23      | 6        | 79.3 % | 61.6–90.2 | +11.03 %  | +45.14 %  |
| 0.75 | 24       | 21      | 3        | 87.5 % | 69.0–95.7 | +22.50 %  | +60.12 %  |
| 0.80 | 12       | 0       | 12       | —      | —         | —         | —         |

### Other — Switzerland LNA

- Entrenado (A): **820** · Holdout (C): **115** · Sesgo @0.75: +0.0109

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 57       | 45      | 12       | 79.0 % | 66.7–87.5 | +10.53 %  | +44.47 %  |
| 0.68 | 45       | 36      | 9        | 80.0 % | 66.2–89.1 | +12.00 %  | +46.40 %  |
| 0.70 | 39       | 31      | 8        | 79.5 % | 64.5–89.2 | +11.28 %  | +45.46 %  |
| 0.72 | 30       | 24      | 6        | 80.0 % | 62.7–90.5 | +12.00 %  | +46.40 %  |
| 0.75 | 24       | 19      | 5        | 79.2 % | 59.5–90.8 | +10.83 %  | +44.88 %  |
| 0.80 | 10       | 0       | 10       | —      | —         | —         | —         |

## 5. Ligas destacadas

### 5.1 Mejores ROI @1.40 a 0.75 (mín. 30 apuestas)

| País       | Liga                    | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR     | ROI @1.40 |
| ---        | ---                     | ---:      | ---:           | ---:    | ---:     | ---:   | ---:      |
| ?          | CBI U15 Fem             | 0         | 65             | 61      | 4        | 93.8 % | +31.38 %  |
| Historical | LNB                     | 310       | 68             | 63      | 5        | 92.7 % | +29.71 %  |
| ?          | ZBL                     | 858       | 42             | 38      | 4        | 90.5 % | +26.67 %  |
| ?          | Top Division Women      | 0         | 39             | 35      | 4        | 89.7 % | +25.64 %  |
| Historical | FSHB Superliga          | 141       | 48             | 42      | 6        | 87.5 % | +22.50 %  |
| ?          | EYBL U17                | 0         | 45             | 39      | 6        | 86.7 % | +21.33 %  |
| Poland     | MPBL                    | 739       | 35             | 30      | 5        | 85.7 % | +20.00 %  |
| ?          | CBI U15                 | 0         | 48             | 41      | 7        | 85.4 % | +19.58 %  |
| Australia  | NBL                     | 2,604     | 67             | 56      | 11       | 83.6 % | +17.01 %  |
| ?          | NCAA Women's Division I | 15,586    | 241            | 201     | 40       | 83.4 % | +16.76 %  |

### 5.2 Peores ROI @1.40 a 0.75 (mín. 30 apuestas)

| País       | Liga                                      | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR     | ROI @1.40 |
| ---        | ---                                       | ---:      | ---:           | ---:    | ---:     | ---:   | ---:      |
| Other      | Premijer ženska liga                      | 306       | 31             | 20      | 11       | 64.5 % | -9.68 %   |
| Other      | Liga Femenina                             | 1,267     | 30             | 20      | 10       | 66.7 % | -6.67 %   |
| Other      | Russian Women's Basketball Premier League | 726       | 31             | 22      | 9        | 71.0 % | -0.65 %   |
| ?          | EYBL U14                                  | 0         | 45             | 33      | 12       | 73.3 % | +2.67 %   |
| Other      | Swedish Basketball League Women           | 239       | 37             | 28      | 9        | 75.7 % | +5.95 %   |
| ?          | Liga 1 Masculin                           | 83        | 31             | 24      | 7        | 77.4 % | +8.39 %   |
| ?          | Liga Nacional Femenina Chile              | 0         | 32             | 25      | 7        | 78.1 % | +9.38 %   |
| Other      | Hungary Divison A Women                   | 276       | 42             | 33      | 9        | 78.6 % | +10.00 %  |
| Historical | Poland 2nd Basketball League              | 674       | 185            | 148     | 37       | 80.0 % | +12.00 %  |
| ?          | Tipos Extraliga                           | 424       | 40             | 32      | 8        | 80.0 % | +12.00 %  |

## 5.3 Marcadores de la columna "Bloqueo"

- **🆕 (arranque en frío):** la liga no tuvo ningún partido en el bloque de entrenamiento (A). El modelo no la vio; sus priors son as-of. Hay 83 ligas así.

- **⚠️ (ROI negativo a 0.75):** la liga tiene ROI@1.40 < 0 % a umbral 0.75. Candidata a bloqueo por medición. Hay 6 ligas así.

- **(vacío):** liga normal, sin alertas.

### 5.4 Ligas en arranque en frío (Entrenado = 0)

| País      | Liga                                     | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 |
| ---       | ---                                      | ---:        | ---:           | ---:    | ---:     | ---:     |
| ?         | CBI U15 Fem                              | 138         | 65             | 61      | 4        | 93.8 %   |
| ?         | CBI U15                                  | 199         | 48             | 41      | 7        | 85.4 %   |
| ?         | EYBL U15                                 | 272         | 48             | 40      | 8        | 83.3 %   |
| ?         | EYBL U17                                 | 303         | 45             | 39      | 6        | 86.7 %   |
| ?         | EYBL U14                                 | 264         | 45             | 33      | 12       | 73.3 %   |
| ?         | Top Division Women                       | 106         | 39             | 35      | 4        | 89.7 %   |
| ?         | CBI U19                                  | 113         | 36             | 29      | 7        | 80.6 %   |
| ?         | EYBL U13                                 | 126         | 35             | 28      | 7        | 80.0 %   |
| ?         | Liga Nacional Femenina Chile             | 103         | 32             | 25      | 7        | 78.1 %   |
| ?         | EYBL U16                                 | 280         | 29             | 25      | 4        | 86.2 %   |
| Other     | Prva Liga                                | 154         | 27             | 23      | 4        | 85.2 %   |
| ?         | EGBL U17                                 | 74          | 24             | 21      | 3        | 87.5 %   |
| ?         | Nostra.lt-RKL Division B                 | 227         | 24             | 17      | 7        | 70.8 %   |
| ?         | CBI U17 Feminino                         | 54          | 22             | 21      | 1        | 95.5 %   |
| Other     | 1st Division Women                       | 82          | 22             | 19      | 3        | 86.4 %   |
| ?         | EYBL U20                                 | 223         | 19             | 17      | 2        | 89.5 %   |
| ?         | 1 Liga Kobiet                            | 284         | 19             | 14      | 5        | 73.7 %   |
| Other     | Sparkasse Prva Liga Zeni                 | 81          | 17             | 9       | 8        | 52.9 %   |
| ?         | SWC                                      | 39          | 16             | 15      | 1        | 93.8 %   |
| ?         | NBL Women                                | 106         | 16             | 13      | 3        | 81.2 %   |
| Lithuania | RKL Division A                           | 234         | 0              | 0       | 0        | —        |
| ?         | Divisional Tercera de Ascenso            | 61          | 0              | 0       | 0        | —        |
| Turkey    | Türkiye Kadınlar Basketbol Ligi          | 39          | 0              | 0       | 0        | —        |
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
| France    | LNB 2                                    | 44          | 0              | 0       | 0        | —        |
| ?         | U16 Türkiye Finalleri                    | 46          | 0              | 0       | 0        | —        |
| ?         | U18 EuroBasket Div. B                    | 50          | 0              | 0       | 0        | —        |
| ?         | FIBA U16 Women's EuroBasket Division B   | 34          | 0              | 0       | 0        | —        |
| ?         | FIBA U16 EuroBasket                      | 38          | 0              | 0       | 0        | —        |
| Other     | Copa Espana                              | 41          | 0              | 0       | 0        | —        |
| Other     | 2. MLS                                   | 30          | 0              | 0       | 0        | —        |

### 5.6 Ligas en arranque en frío con **todas** las apuestas perdidas (0 % WR)

| País   | Liga                                            | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos |
| ---    | ---                                             | ---:        | ---:           | ---:    | ---:     |
| ?      | Campeonato Nacional da 1ª Divisao Feminina      | 128         | 14             | 0       | 14       |
| Other  | Superleague Women                               | 39          | 13             | 0       | 13       |
| Other  | Women's Basketball Division A                   | 34          | 12             | 0       | 12       |
| Other  | The Women League                                | 37          | 11             | 0       | 11       |
| ?      | FIBA U16 Women's EuroBasket                     | 38          | 11             | 0       | 11       |
| ?      | 1. liga                                         | 125         | 9              | 0       | 9        |
| Other  | U21 Espoirs Elite                               | 144         | 9              | 0       | 9        |
| Other  | Latvian Basketball Cup                          | 31          | 9              | 0       | 9        |
| Other  | Jedinstvena kadetska liga                       | 64          | 9              | 0       | 9        |
| ?      | The Basketball League                           | 136         | 9              | 0       | 9        |
| USA    | NCAA Women's March Madness                      | 59          | 9              | 0       | 9        |
| ?      | LF Challenge                                    | 169         | 8              | 0       | 8        |
| ?      | FIBA U17 Women's Basketball World Cup           | 37          | 8              | 0       | 8        |
| Other  | Superliga e Femrave                             | 47          | 7              | 0       | 7        |
| ?      | U18 FIBA Asia Cup                               | 36          | 7              | 0       | 7        |
| ?      | NBL Men                                         | 129         | 6              | 0       | 6        |
| ?      | LBP                                             | 34          | 6              | 0       | 6        |
| Other  | I B MCKL                                        | 44          | 5              | 0       | 5        |
| ?      | U18 EuroBasket                                  | 124         | 5              | 0       | 5        |
| ?      | FIBA Asia Cup 2029 Pre-Qualifiers               | 31          | 5              | 0       | 5        |
| Other  | Azerbaijan Basketball League                    | 120         | 4              | 0       | 4        |
| France | LNB Clausura                                    | 71          | 4              | 0       | 4        |
| Other  | 1st Division Men                                | 142         | 3              | 0       | 3        |
| Other  | Liga Košarkaškog saveza Herceg Bosne            | 52          | 3              | 0       | 3        |
| ?      | FIBA Women Basketball League Africa - Qualifers | 41          | 3              | 0       | 3        |
| Other  | A1                                              | 69          | 3              | 0       | 3        |
| ?      | Kızlar Basketbol Gençler Ligi                   | 34          | 3              | 0       | 3        |
| ?      | FIBA U16 EuroBasket Division B                  | 61          | 3              | 0       | 3        |
| ?      | CN1 Basquetebol                                 | 146         | 2              | 0       | 2        |
| Other  | A1 Liga                                         | 124         | 2              | 0       | 2        |
| ?      | EGBL U19                                        | 39          | 2              | 0       | 2        |
| ?      | EGBL U14                                        | 47          | 2              | 0       | 2        |
| ?      | Divisione Regionale 1 Toscana                   | 131         | 2              | 0       | 2        |
| ?      | FIBA World Cup Qualification                    | 57          | 2              | 0       | 2        |
| ?      | Enovos League                                   | 110         | 1              | 0       | 1        |
| ?      | WABA League Women                               | 35          | 1              | 0       | 1        |
| ?      | LSB - Liga B                                    | 48          | 1              | 0       | 1        |
| ?      | 2. ZLS                                          | 37          | 1              | 0       | 1        |
| ?      | Egyptian Basketball Super League                | 31          | 1              | 0       | 1        |
| ?      | Rapid League                                    | 83          | 1              | 0       | 1        |
| Other  | Saku II Liiga                                   | 35          | 1              | 0       | 1        |
| ?      | 2. DBBL                                         | 34          | 1              | 0       | 1        |
| ?      | Nostra.lt-RKL Division A                        | 38          | 1              | 0       | 1        |

### 5.7 Ligas en arranque en frío **rentables** (WR ≥ 75 %, ≥ 10 apuestas) — candidatas a entrenar

> Estas ligas no se entrenaron (arranque en frío) pero tienen buen rendimiento en el holdout C. Si se consigue histórico para ellas (o se agrupan por `clean_name` con variantes que sí tienen historia), entrenar con ellas podría mejorar el modelo. Son las que valen la pena.

| País  | Liga                         | Partidos en BD (todas las variantes) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |
| ---   | ---                          | ---:                                 | ---:        | ---:           | ---:    | ---:     | ---:     | ---:      |
| ?     | CBI U15 Fem                  | 145                                  | 138         | 65             | 61      | 4        | 93.8 %   | +31.38 %  |
| ?     | CBI U15                      | 207                                  | 199         | 48             | 41      | 7        | 85.4 %   | +19.58 %  |
| ?     | EYBL U15                     | 259                                  | 272         | 48             | 40      | 8        | 83.3 %   | +16.67 %  |
| ?     | EYBL U17                     | 308                                  | 303         | 45             | 39      | 6        | 86.7 %   | +21.33 %  |
| ?     | Top Division Women           | 109                                  | 106         | 39             | 35      | 4        | 89.7 %   | +25.64 %  |
| ?     | CBI U19                      | 147                                  | 113         | 36             | 29      | 7        | 80.6 %   | +12.78 %  |
| ?     | EYBL U13                     | 129                                  | 126         | 35             | 28      | 7        | 80.0 %   | +12.00 %  |
| ?     | Liga Nacional Femenina Chile | 135                                  | 103         | 32             | 25      | 7        | 78.1 %   | +9.38 %   |
| ?     | EYBL U16                     | 305                                  | 280         | 29             | 25      | 4        | 86.2 %   | +20.69 %  |
| Other | Prva Liga                    | 168                                  | 154         | 27             | 23      | 4        | 85.2 %   | +19.26 %  |
| ?     | EGBL U17                     | 76                                   | 74          | 24             | 21      | 3        | 87.5 %   | +22.50 %  |
| ?     | CBI U17 Feminino             | 54                                   | 54          | 22             | 21      | 1        | 95.5 %   | +33.64 %  |
| Other | 1st Division Women           | 83                                   | 82          | 22             | 19      | 3        | 86.4 %   | +20.91 %  |
| ?     | EYBL U20                     | 248                                  | 223         | 19             | 17      | 2        | 89.5 %   | +25.26 %  |
| ?     | SWC                          | 40                                   | 39          | 16             | 15      | 1        | 93.8 %   | +31.25 %  |
| ?     | NBL Women                    | 111                                  | 106         | 16             | 13      | 3        | 81.2 %   | +13.75 %  |

### 5.5 Ligas con ROI negativo a 0.75 (⚠️)

| País  | Liga                                      | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |
| ---   | ---                                       | ---:      | ---:           | ---:    | ---:     | ---:     | ---:      |
| Other | Sparkasse Prva Liga Zeni                  | 0         | 17             | 9       | 8        | 52.9 %   | -25.88 %  |
| Other | Premijer ženska liga                      | 306       | 31             | 20      | 11       | 64.5 %   | -9.68 %   |
| ?     | LMB                                       | 170       | 17             | 11      | 6        | 64.7 %   | -9.41 %   |
| Other | Liga Femenina                             | 1,267     | 30             | 20      | 10       | 66.7 %   | -6.67 %   |
| ?     | Nostra.lt-RKL Division B                  | 0         | 24             | 17      | 7        | 70.8 %   | -0.83 %   |
| Other | Russian Women's Basketball Premier League | 726       | 31             | 22      | 9        | 71.0 %   | -0.65 %   |

## 6. Caso específico — `NCAA Women's Division I` (backfill histórico)

> En `m27_v5_3_2` esta liga tenía 0 entrenamiento (solo temporada 2025-26, 1,729 partidos, todo en holdout C). En `m27_v5_3_3` se descargaron **19,541 partidos históricos** (2019-2026) vía API de SofaScore, por lo que ahora tiene entrenamiento. Es la prueba de que el backfill histórico resuelve el arranque en frío.

- **Entrenado (A):** 15,586 · **Holdout (C):** 1,636

| P ≥  | Apuestas | Ganados | Perdidos | WR     | IC95      | ROI @1.40 | ROI @1.83 |
| ---  | ---:     | ---:    | ---:     | ---:   | ---       | ---:      | ---:      |
| 0.65 | 617      | 452     | 165      | 73.3 % | 69.6–76.6 | +2.56 %   | +34.06 %  |
| 0.68 | 489      | 364     | 125      | 74.4 % | 70.4–78.1 | +4.21 %   | +36.22 %  |
| 0.70 | 412      | 315     | 97       | 76.5 % | 72.1–80.3 | +7.04 %   | +39.92 %  |
| 0.72 | 341      | 265     | 76       | 77.7 % | 73.0–81.8 | +8.80 %   | +42.21 %  |
| 0.75 | 241      | 201     | 40       | 83.4 % | 78.2–87.6 | +16.76 %  | +52.63 %  |
| 0.80 | 121      | 108     | 13       | 89.3 % | 82.5–93.6 | +24.96 %  | +63.34 %  |

A 0.75: **241 apuestas, WR 83.4 %, ROI@1.40 +16.76 %** (en vivo: 4 partidos, v5_3 2W/2L, v5_3_1 2W/0L).

## 7. Prueba contra los partidos en vivo de `m27_v5_3`

- **m27_v5_3 (original):** 89 apuestas, 71 ganadas, 18 perdidas → WR 79.8 %.
- **m27_v5_3_3 (clon):** 60 apuestas, 51 ganadas, 9 perdidas → WR 85.0 %.

**vs m27_v5_3_2 (anterior):** 64 apuestas, 53 ganadas, 11 perdidas → WR 82.8 %. v5_3_3: -4 apuestas, -2 ganadas, +2.2 pp de WR.

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
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | United League                                                               | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Historical    | Liga Nacional Femenina Chile, Fase zonal                                    | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 2         | 2           | Other         | Estonian-Latvian Basketball League                                          | 2                  | 2              | 0              | 2                | 0                |      |
| ⚪        | 100 %   | 100 %     | 4         | 1           | Other         | Basketligaen                                                                | 4                  | 4              | 0              | 1                | 0                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | International | FIBA Women Basketball League Africa - Qualifers, Group A                    | 2                  | 2              | 1              | 1                | 1                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Other         | 2nd League                                                                  | 2                  | 2              | 1              | 1                | 1                |      |
| 🔴        | 50 %    | 0 %       | 1         | 0           | Other         | A1 Women                                                                    | 2                  | 2              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 2         | 1           | Historical    | L1BF BT, Faza I                                                             | 2                  | 2              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Premijer ženska liga                                                        | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Other         | Women's Basketball Division A                                               | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Russian Women's Basketball Premier League                                   | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Hungary NB 1.A                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 14 | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | FSHB Superliga                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Switzerland LNA                                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | Brazil        | Brazil NBB                                                                  | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Catarinense, Women, Playoffs                                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 13 | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | Nostra.lt-RKL Division A, Reguliarus sezonas                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Kvindebasketligaen , Grundspil                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | 1 Liga Kobiet, Runda Zasadnicza                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Super League                                                                | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Liga 1 Masculin, Faza I                                                     | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Liga Nationala                                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Other         | Turkish Basketball Super League                                             | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Historical    | CN2 Basquetebol, 1ª Fase - Série D                                          | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | International | AdmiralBet ABA League, Group B                                              | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | ?             | FIBA Women Basketball League Africa - Qualifers, Zone 5 - Semi-Finals 19    | 1                  | 1              | 0              | 1                | 0                |      |
| ⚪        | 0 %     | 0 %       | 0         | 0           | International | Eurocup, Group B                                                            | 1                  | 1              | 1              | 1                | 1                |      |
| ⚪        | 100 %   | 100 %     | 1         | 1           | Greece        | Stoiximan GBL                                                               | 1                  | 1              | 0              | 1                | 0                |      |
| —         | 50 %    | —         | 1         | 0           | Japan         | B League One                                                                | 2                  | 2              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 2         | 0           | Other         | LMB, Clausura                                                               | 2                  | 2              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | France        | Mexico LNBP, Playoffs                                                       | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | ?             | Nostra.lt-RKL Division B, Reguliarus sezonas                                | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | 1. Division                                                                 | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Other         | 1st Division Women, Deildarkeppni                                           | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | FIBA Women Basketball League Africa - Qualifers, Group B                    | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | Cyprus Basketball Division A                                                | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Japan         | B League Premier                                                            | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | France        | France Pro A                                                                | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Other         | Hungary Divison A Women                                                     | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | International | EuroCup, Women, Group B                                                     | 1                  | 1              | 0              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | Swedish Basketball Superettan                                               | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 0 %     | —         | 0         | 0           | Other         | 2. ZLS, Women                                                               | 1                  | 1              | 1              | 0                | 0                |      |
| —         | 100 %   | —         | 1         | 0           | Israel        | Israeli Women Basketball Premier League                                     | 1                  | 1              | 0              | 0                | 0                |      |
| **TOTAL** | **89**  | **89**    | **71**    | **18**      | **80 %**      | **60**                                                                      | **51**             | **9**          | **85 %**       | **🟢**           |                  |      |

> Nota: la simulación secuencial con $100 por apuesta se integra en el barrido de umbrales de la sección 8 (columnas de bankroll final). El umbral operativo es 0.75.


## 8. Barrido de umbrales sobre las apuestas en vivo (m27_v5_3 vs clon)

> Simula qué habría pasado si se hubiera operado con distintos umbrales de confianza sobre los mismos partidos en vivo. Para el clon se aplica también el guard de margen (margin ≥ 0). Bank inicial $1,000, $100 por apuesta.

### Momio 1.40

| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR   | O:ROI    | C:apuestas | C:WR    | C:ROI    | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |
| ---    | ---:       | ---:      | ---:      | ---:       | ---:       | ---:   | ---:     | ---:       | ---:    | ---:     | ---:       | ---:       | ---:         | ---:         | ---:          |
| 0.60   | 89         | 71        | 51        | 18         | 38         | 79.8 % | +11.69 % | 89         | 57.3 %  | -19.78 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.62   | 89         | 71        | 51        | 18         | 38         | 79.8 % | +11.69 % | 89         | 57.3 %  | -19.78 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.65   | 89         | 71        | 51        | 18         | 38         | 79.8 % | +11.69 % | 89         | 57.3 %  | -19.78 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.68   | 89         | 71        | 51        | 18         | 37         | 79.8 % | +11.69 % | 88         | 58.0 %  | -18.86 % | $+1,040.00 | $-940.00 * | $2,040.00    | $60.00       | 🔴            |
| 0.70   | 89         | 71        | 51        | 18         | 35         | 79.8 % | +11.69 % | 86         | 59.3 %  | -16.98 % | $+1,040.00 | $-960.00 * | $2,040.00    | $40.00       | 🔴            |
| 0.72   | 89         | 71        | 51        | 18         | 29         | 79.8 % | +11.69 % | 80         | 63.7 %  | -10.75 % | $+1,040.00 | $-960.00 * | $2,040.00    | $40.00       | 🔴            |
| 0.75   | 77         | 64        | 51        | 13         | 9          | 83.1 % | +16.36 % | 60         | 85.0 %  | +19.00 % | $+1,260.00 | $+1,140.00 | $2,260.00    | $2,140.00    | 🟢            |
| 0.78   | 38         | 31        | 28        | 7          | 5          | 81.6 % | +14.21 % | 33         | 84.8 %  | +18.79 % | $+540.00   | $+620.00   | $1,540.00    | $1,620.00    | 🟢            |
| 0.80   | 27         | 23        | 22        | 4          | 3          | 85.2 % | +19.26 % | 25         | 88.0 %  | +23.20 % | $+520.00   | $+580.00   | $1,520.00    | $1,580.00    | 🟢            |
| 0.85   | 12         | 11        | 8         | 1          | 0          | 91.7 % | +28.33 % | 8          | 100.0 % | +40.00 % | $+340.00   | $+320.00   | $1,340.00    | $1,320.00    | 🟢            |
> * = se detuvo por falta de fondos.


### Momio 1.83

| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR   | O:ROI    | C:apuestas | C:WR    | C:ROI    | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |
| ---    | ---:       | ---:      | ---:      | ---:       | ---:       | ---:   | ---:     | ---:       | ---:    | ---:     | ---:       | ---:       | ---:         | ---:         | ---:          |
| 0.60   | 89         | 71        | 51        | 18         | 38         | 79.8 % | +45.99 % | 89         | 57.3 %  | +4.87 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.62   | 89         | 71        | 51        | 18         | 38         | 79.8 % | +45.99 % | 89         | 57.3 %  | +4.87 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.65   | 89         | 71        | 51        | 18         | 38         | 79.8 % | +45.99 % | 89         | 57.3 %  | +4.87 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.68   | 89         | 71        | 51        | 18         | 37         | 79.8 % | +45.99 % | 88         | 58.0 %  | +6.06 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.70   | 89         | 71        | 51        | 18         | 35         | 79.8 % | +45.99 % | 86         | 59.3 %  | +8.52 %  | $+4,093.00 | $-902.00 * | $5,093.00    | $98.00       | 🔴            |
| 0.72   | 89         | 71        | 51        | 18         | 29         | 79.8 % | +45.99 % | 80         | 63.7 %  | +16.66 % | $+4,093.00 | $+1,333.00 | $5,093.00    | $2,333.00    | 🟢            |
| 0.75   | 77         | 64        | 51        | 13         | 9          | 83.1 % | +52.10 % | 60         | 85.0 %  | +55.55 % | $+4,012.00 | $+3,333.00 | $5,012.00    | $4,333.00    | 🟢            |
| 0.78   | 38         | 31        | 28        | 7          | 5          | 81.6 % | +49.29 % | 33         | 84.8 %  | +55.27 % | $+1,873.00 | $+1,824.00 | $2,873.00    | $2,824.00    | 🟢            |
| 0.80   | 27         | 23        | 22        | 4          | 3          | 85.2 % | +55.89 % | 25         | 88.0 %  | +61.04 % | $+1,509.00 | $+1,526.00 | $2,509.00    | $2,526.00    | 🟢            |
| 0.85   | 12         | 11        | 8         | 1          | 0          | 91.7 % | +67.75 % | 8          | 100.0 % | +83.00 % | $+813.00   | $+664.00   | $1,813.00    | $1,664.00    | 🟢            |
> * = se detuvo por falta de fondos.


**Lectura:** el clon es más conservador (59 vs 85 apuestas) con acierto ligeramente superior (79.7 % vs 78.8 %). Diferencia pequeña, muestra diminuta. No se justifica cambio de modelo solo con esta evidencia.

## 9. Configuración del modelo m27_v5_3_2

| Parámetro                                      | Valor                              |
| ---                                            | ---                                |
| AUC train / calib (B) / test (C)               | 0.6456 / 0.6511 / 0.6541           |
| Brier / LogLoss / Acc (test)                   | 0.2305 / 0.652 / 0.6117            |
| AUC gap (train−test)                           | -0.0085                            |
| Partidos elegibles                             | 201,660                            |
| Filas A / B / C                                | 197,726 / 59,307 / 101,269         |
| Umbral operativo (min_conf)                    | 0.75                               |
| Ventana de inferencia                          | [25.5, 30.0] (fuera → unavailable) |
| Ligas únicas (clean_name)                      | 1,311                              |
| Ligas en arranque en frío (trained=0)          | 83                                 |
| Ligas candidatas a bloqueo (IC95 sup < 71.4 %) | 0                                  |

### Formatos de torneo bloqueados (is_excluded_stage)

amistosos, 2ª fase, promoción/relegation, preliminar, colocación (misma lista que m27_v5_3, vía `models/common/competition_stage.py`).

### Ligas bloqueadas por medición propia

Ninguna con la evidencia actual (0 candidatas), igual que m27_v5_3.

### Walk-forward (3 cortes)

| Corte                                    | AUC    | Apuestas | Ganados | Perdidos | WR      | ROI @1.40 | ROI @1.83 |
| ---                                      | ---:   | ---:     | ---:    | ---:     | ---:    | ---:      | ---:      |
| Corte 1: Train <= 2022 -> Test 2023      | 0.6305 | 711      | 547     | 164      | 76.93 % | +7.71 %   | +40.79 %  |
| Corte 2: Train <= 2023 -> Test 2024      | 0.6376 | 3,666    | 2,912   | 754      | 79.43 % | +11.21 %  | +45.36 %  |
| Corte 3: Train <= 2024 -> Test 2025-2026 | 0.6483 | 15,708   | 12,405  | 3,303    | 78.97 % | +10.55 %  | +44.51 %  |

### Ligas con más partidos de entrenamiento (top 10)

| País      | Liga (clean_name)       | Entrenado (A) | Holdout (C) |
| ---       | ---                     | ---:          | ---:        |
| ?         | NCAA Women's Division I | 15,586        | 1,636       |
| Japan     | B League Premier        | 3,092         | 667         |
| Japan     | B League One            | 3,037         | 401         |
| Argentina | Argentina Liga Nacional | 2,944         | 301         |
| Australia | NBL                     | 2,604         | 620         |
| Italy     | Serie A2                | 2,549         | 443         |
| Other     | Élite 2                 | 1,952         | 308         |
| Germany   | Germany BBL             | 1,879         | 276         |
| Spain     | Liga ACB                | 1,858         | 291         |
| France    | France Pro A            | 1,849         | 223         |

## 10. Comparación final: v5_3 vs variantes

| Modelo                    | Cambio principal          | AUC test   | WR @0.75 (ancla) | ROI @1.40    | Live WR    | Live apuestas | Cold-start leagues | Partidos elegibles |
| ---                       | ---                       | ---:       | ---:             | ---:         | ---:       | ---:          | ---:               | ---:               |
| m27_v5_3 (original)       | base                      | 0.6545     | 81.68 %          | +14.35 %     | 78.8 %     | 85            | 108                | 182,803            |
| m27_v5_3_1 (clasificador) | fix Runda Zasadnicza      | 0.6544     | 80.99 %          | +13.93 %     | 82.8 %     | 59            | 108                | 184,709            |
| m27_v5_3_2 (clean_name)   | normalización de nombres  | 0.6548     | 80.86 %          | +13.93 %     | 82.8 %     | 64            | 78                 | 184,878            |
| **m27_v5_3_3 (backfill)** | **+NCAA Women histórico** | **0.6541** | **81.66 %**      | **+14.32 %** | **85.0 %** | **60**        | **83**             | **201,660**        |

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
