# 📊 Análisis por liga y por umbral — modelo m27_v5_3

> Generado a partir del holdout **C** (último 18 % de partidos, anclas t=30) del pipeline `tmp/m27_v5_3/train_m27_v5_3_pipeline.py`.
> Umbral operativo del modelo: **P ≥ 0.75**. Punto de equilibrio a cuota 1.40: **71.4 %** de acierto.
> `Entrenado` = anclas (t=30) del bloque A (70 %) por liga. `Holdout` = anclas del bloque C (18 %) por liga.

## 1. Resumen ejecutivo

- **Ligas analizadas:** 251 (con ≥ 30 anclas en el holdout C).
- **Partidos (anclas) en holdout C sumados:** 24,249.
- **Partidos (anclas) en entrenamiento (bloque A) sumados:** 73,151.
- **Apuestas totales a umbral 0.75 en C:** 1,855.
- **Ligas sin ningún partido en entrenamiento (arranque en frío):** 108 (43.0 %).
- **Ligas candidatas a bloqueo** (IC95 superior del WR < 71.4 % a 0.75): **0** (ninguna con la evidencia actual).
- **Ligas con ROI@1.40 > 20 % a 0.75:** 15.

## 2. Tabla global de umbrales (referencia, anclas t=30, bloque C)

| P ≥  | Apuestas | WR (IC95)          | ROI @1.40 | ROI @1.83 |
| ---  | ---      | ---                | ---       | ---       |
| 0.65 | 9,456    | 71.9 % (70.9–72.8) | +0.60 %   | +31.50 %  |
| 0.68 | 6,692    | 74.3 % (73.2–75.3) | +3.97 %   | +35.91 %  |
| 0.70 | 5,157    | 76.6 % (75.5–77.8) | +7.29 %   | +40.24 %  |
| 0.72 | 3,945    | 78.6 % (77.3–79.9) | +10.05 %  | +43.85 %  |
| 0.75 | 2,538    | 81.7 % (80.1–83.1) | +14.35 %  | +49.47 %  |
| 0.80 | 1,209    | 86.6 % (84.6–88.4) | +21.24 %  | +58.48 %  |

## 3. Tabla resumen por liga (ordenada por apuestas a 0.75)

> Nota: solo se incluyen ligas con ≥ 30 anclas en el holdout C (251 ligas). Las ligas con menos volumen quedan fuera; por eso la suma de apuestas aquí (1,855 a 0.75) es menor que la tabla global (2,538). El resto corresponde a ligas de cola.

| Liga                                                        | Entrenado (A) | Holdout (C) | Apuestas @0.75 | WR @0.75 | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---                                                         | ---:          | ---:        | ---:           | ---:     | ---       | ---:      | ---:      | ---     |
| NCAA Women's Division I, Regular season                     | 0             | 1,620       | 275            | 82.2 %   | 77.2–86.3 | +15.05 %  | +50.39 %  | 🆕       |
| FSHB Superliga, K.K.M 2025-26                               | 0             | 91          | 39             | 82.0 %   | 67.3–91.0 | +14.87 %  | +50.15 %  | 🆕       |
| Tipos Extraliga, Women                                      | 424           | 64          | 39             | 79.5 %   | 64.5–89.2 | +11.28 %  | +45.46 %  |         |
| NBL, East                                                   | 78            | 97          | 36             | 86.1 %   | 71.3–93.9 | +20.56 %  | +57.58 %  |         |
| LNB, Women                                                  | 184           | 55          | 36             | 91.7 %   | 78.2–97.1 | +28.33 %  | +67.75 %  |         |
| NBL1 East, Women                                            | 153           | 118         | 36             | 86.1 %   | 71.3–93.9 | +20.56 %  | +57.58 %  |         |
| ZBL, Women                                                  | 694           | 74          | 35             | 88.6 %   | 74.0–95.5 | +24.00 %  | +62.09 %  |         |
| Swedish Basketball League Women                             | 233           | 114         | 34             | 73.5 %   | 56.9–85.4 | +2.94 %   | +34.56 %  |         |
| United League                                               | 1,133         | 176         | 34             | 82.3 %   | 66.5–91.7 | +15.29 %  | +50.71 %  |         |
| Top Division Women, Regular Season                          | 0             | 89          | 34             | 88.2 %   | 73.4–95.3 | +23.53 %  | +61.47 %  | 🆕       |
| Hungary Divison A Women                                     | 276           | 104         | 34             | 76.5 %   | 60.0–87.6 | +7.06 %   | +39.94 %  |         |
| Basketligaen                                                | 651           | 111         | 32             | 78.1 %   | 61.2–89.0 | +9.38 %   | +42.97 %  |         |
| Premijer ženska liga                                        | 299           | 100         | 32             | 68.8 %   | 51.4–82.0 | -3.75 %   | +25.81 %  | ⚠️      |
| MPBL                                                        | 319           | 264         | 30             | 86.7 %   | 70.3–94.7 | +21.33 %  | +58.60 %  |         |
| Russian Women's Basketball Premier League                   | 726           | 103         | 27             | 74.1 %   | 55.3–86.8 | +3.70 %   | +35.56 %  |         |
| Prva Liga                                                   | 0             | 139         | 27             | 88.9 %   | 71.9–96.1 | +24.44 %  | +62.67 %  | 🆕       |
| CBI U15 , Classificatória A                                 | 0             | 76          | 27             | 92.6 %   | 76.6–97.9 | +29.63 %  | +69.44 %  | 🆕       |
| Big V, Women                                                | 108           | 82          | 26             | 80.8 %   | 62.1–91.5 | +13.08 %  | +47.81 %  |         |
| 1st Division Women                                          | 0             | 57          | 23             | 87.0 %   | 67.9–95.5 | +21.74 %  | +59.13 %  | 🆕       |
| Estonian-Latvian Basketball League                          | 594           | 163         | 22             | 81.8 %   | 61.5–92.7 | +14.55 %  | +49.73 %  |         |
| Nostra.lt-RKL Division B, Reguliarusis sezonas              | 0             | 182         | 22             | 72.7 %   | 51.8–86.8 | +1.82 %   | +33.09 %  | 🆕       |
| SLB, Women                                                  | 0             | 89          | 22             | 68.2 %   | 47.3–83.6 | -4.55 %   | +24.77 %  | ⚠️ 🆕    |
| 2nd League                                                  | 269           | 132         | 22             | 81.8 %   | 61.5–92.7 | +14.55 %  | +49.73 %  |         |
| CBI U19, Sede Rio                                           | 0             | 37          | 21             | 90.5 %   | 71.1–97.3 | +26.67 %  | +65.57 %  | 🆕       |
| NBL, West                                                   | 75            | 93          | 20             | 75.0 %   | 53.1–88.8 | +5.00 %   | +37.25 %  |         |
| CBI U15 , Classificatória B                                 | 0             | 78          | 20             | 75.0 %   | 53.1–88.8 | +5.00 %   | +37.25 %  | 🆕       |
| BNXT League                                                 | 205           | 231         | 19             | 94.7 %   | 75.4–99.1 | +32.63 %  | +73.37 %  |         |
| Finland Division 1A Women                                   | 294           | 95          | 19             | 84.2 %   | 62.4–94.5 | +17.89 %  | +54.11 %  |         |
| Switzerland LNA                                             | 689           | 91          | 19             | 73.7 %   | 51.2–88.2 | +3.16 %   | +34.84 %  |         |
| Saudi Basketball League, Premier League                     | 0             | 45          | 19             | 94.7 %   | 75.4–99.1 | +32.63 %  | +73.37 %  | 🆕       |
| 1 Division, Women                                           | 17            | 84          | 18             | 83.3 %   | 60.8–94.2 | +16.67 %  | +52.50 %  |         |
| Sparkasse Prva Liga Zeni                                    | 0             | 66          | 17             | 52.9 %   | 31.0–73.8 | -25.88 %  | -3.12 %   | ⚠️ 🆕    |
| NBL Women , Regular Season                                  | 0             | 106         | 17             | 82.3 %   | 59.0–93.8 | +15.29 %  | +50.71 %  | 🆕       |
| Brazil NBB                                                  | 1,436         | 294         | 16             | 81.2 %   | 57.0–93.4 | +13.75 %  | +48.69 %  |         |
| Swiss Basketball League Women                               | 147           | 69          | 16             | 87.5 %   | 64.0–96.5 | +22.50 %  | +60.13 %  |         |
| Liga 1 Masculin, Faza II                                    | 0             | 119         | 16             | 68.8 %   | 44.4–85.8 | -3.75 %   | +25.81 %  | ⚠️ 🆕    |
| LDB, Primeira fase                                          | 0             | 128         | 16             | 87.5 %   | 64.0–96.5 | +22.50 %  | +60.13 %  | 🆕       |
| 1. SKL, Women                                               | 44            | 52          | 15             | 86.7 %   | 62.1–96.3 | +21.33 %  | +58.60 %  |         |
| Rwanda Basketball League, Regular Season                    | 0             | 65          | 15             | 80.0 %   | 54.8–93.0 | +12.00 %  | +46.40 %  | 🆕       |
| CBI U17, Sede Curitiba                                      | 0             | 67          | 14             | —        | —         | —         | —         | 🆕       |
| LNB, Clausura                                               | 0             | 49          | 14             | —        | —         | —         | —         | 🆕       |
| Liga Nacional Femenina Chile, Fase regular                  | 0             | 34          | 13             | —        | —         | —         | —         | 🆕       |
| 1. Division B                                               | 93            | 105         | 12             | —        | —         | —         | —         |         |
| A1 Women                                                    | 813           | 82          | 12             | —        | —         | —         | —         |         |
| EYBL U14, Group E                                           | 0             | 52          | 12             | —        | —         | —         | —         | 🆕       |
| LMB, Clausura                                               | 120           | 73          | 11             | —        | —         | —         | —         |         |
| Superleague Women                                           | 0             | 32          | 11             | —        | —         | —         | —         | 🆕       |
| B League One                                                | 3,075         | 372         | 11             | —        | —         | —         | —         |         |
| Liga Femenina, Women, Clausura                              | 0             | 115         | 11             | —        | —         | —         | —         | 🆕       |
| Austrian Women's Basketball Bundesliga                      | 144           | 50          | 10             | —        | —         | —         | —         |         |
| EYBL U13, Group D                                           | 0             | 35          | 10             | —        | —         | —         | —         | 🆕       |
| Basket Liga, Women                                          | 780           | 108         | 10             | —        | —         | —         | —         |         |
| Jedinstvena kadetska liga                                   | 0             | 64          | 10             | —        | —         | —         | —         | 🆕       |
| Rwanda Basketball League Women, Regular Season              | 0             | 42          | 10             | —        | —         | —         | —         | 🆕       |
| NBL1 West, Women                                            | 127           | 95          | 10             | —        | —         | —         | —         |         |
| B League Premier                                            | 3,125         | 605         | 9              | —        | —         | —         | —         |         |
| LF Challenge, Regular Season                                | 0             | 154         | 9              | —        | —         | —         | —         | 🆕       |
| Swedish Basketball League                                   | 912           | 127         | 9              | —        | —         | —         | —         |         |
| Latvian Basketball Cup                                      | 0             | 31          | 9              | —        | —         | —         | —         | 🆕       |
| EYBL U15, Group D                                           | 0             | 41          | 9              | —        | —         | —         | —         | 🆕       |
| 1. ZLS, Women                                               | 53            | 54          | 9              | —        | —         | —         | —         |         |
| VBA                                                         | 119           | 57          | 9              | —        | —         | —         | —         |         |
| NBL Men , Regular Season                                    | 0             | 126         | 8              | —        | —         | —         | —         | 🆕       |
| Estonian National Women's Basketball League                 | 55            | 49          | 8              | —        | —         | —         | —         |         |
| EYBL U17, Group E                                           | 0             | 49          | 8              | —        | —         | —         | —         | 🆕       |
| The Basketball League, Regular Season                       | 30            | 126         | 8              | —        | —         | —         | —         |         |
| NBL, Playoffs                                               | 58            | 41          | 8              | —        | —         | —         | —         |         |
| NBL1 Central, Women                                         | 84            | 58          | 8              | —        | —         | —         | —         |         |
| Liga 1 Masculin, Faza I                                     | 101           | 57          | 7              | —        | —         | —         | —         |         |
| 1. liga, Women                                              | 0             | 87          | 7              | —        | —         | —         | —         | 🆕       |
| Liga e Parë                                                 | 277           | 72          | 7              | —        | —         | —         | —         |         |
| France Pro A                                                | 1,737         | 189         | 7              | —        | —         | —         | —         |         |
| Superliga e Femrave                                         | 0             | 44          | 7              | —        | —         | —         | —         | 🆕       |
| Portugal LBP                                                | 766           | 104         | 7              | —        | —         | —         | —         |         |
| Superliga, Women                                            | 0             | 38          | 7              | —        | —         | —         | —         | 🆕       |
| Indonesian Basketball League                                | 758           | 91          | 7              | —        | —         | —         | —         |         |
| LBF, Women                                                  | 177           | 61          | 7              | —        | —         | —         | —         |         |
| NCAA Women's March Madness                                  | 0             | 59          | 7              | —        | —         | —         | —         | 🆕       |
| CBI U19, Sede Curitiba                                      | 0             | 44          | 7              | —        | —         | —         | —         | 🆕       |
| NBL                                                         | 1,965         | 317         | 6              | —        | —         | —         | —         |         |
| Liga Femenina, Women                                        | 1,169         | 193         | 6              | —        | —         | —         | —         |         |
| Liga ACB                                                    | 1,756         | 258         | 6              | —        | —         | —         | —         |         |
| Euroleague: regular season                                  | 220           | 88          | 6              | —        | —         | —         | —         |         |
| Kvindebasketligaen , Grundspil                              | 0             | 42          | 5              | —        | —         | —         | —         | 🆕       |
| Primera FEB                                                 | 1,654         | 213         | 5              | —        | —         | —         | —         |         |
| Turkish Basketball Super League                             | 1,376         | 169         | 5              | —        | —         | —         | —         |         |
| Campeonato Nacional da 1ª Divisao Feminina, 1ª Fase - Sul   | 0             | 52          | 5              | —        | —         | —         | —         | 🆕       |
| Campeonato Nacional da 1ª Divisao Feminina, 1ª Fase - Norte | 0             | 38          | 5              | —        | —         | —         | —         | 🆕       |
| I B MCKL                                                    | 0             | 38          | 5              | —        | —         | —         | —         | 🆕       |
| FBU Superleague                                             | 732           | 75          | 5              | —        | —         | —         | —         |         |
| U21 Espoirs Elite                                           | 0             | 136         | 5              | —        | —         | —         | —         | 🆕       |
| EYBL U20, Group E                                           | 0             | 37          | 5              | —        | —         | —         | —         | 🆕       |
| EYBL U15, Group F                                           | 0             | 31          | 5              | —        | —         | —         | —         | 🆕       |
| CBI U15 Fem, CBI U15 Fem                                    | 0             | 34          | 5              | —        | —         | —         | —         | 🆕       |
| EYBL U16, CHALLENGE CUP                                     | 0             | 37          | 5              | —        | —         | —         | —         | 🆕       |
| NBL1 South, Women                                           | 198           | 153         | 5              | —        | —         | —         | —         |         |
| NBL1, East                                                  | 148           | 118         | 5              | —        | —         | —         | —         |         |
| Uruguay LUB                                                 | 250           | 93          | 4              | —        | —         | —         | —         |         |
| WNBL                                                        | 553           | 71          | 4              | —        | —         | —         | —         |         |
| 1. Division                                                 | 255           | 101         | 4              | —        | —         | —         | —         |         |
| Cyprus Basketball Division A                                | 348           | 69          | 4              | —        | —         | —         | —         |         |
| Germany BBL                                                 | 1,737         | 246         | 4              | —        | —         | —         | —         |         |
| Serie A2, Women, Group A                                    | 139           | 119         | 4              | —        | —         | —         | —         |         |
| Liga Nationala                                              | 742           | 157         | 4              | —        | —         | —         | —         |         |
| Eurocup, Group B                                            | 279           | 69          | 4              | —        | —         | —         | —         |         |
| Iraqi Professional League, Iraqi Professional league        | 0             | 53          | 4              | —        | —         | —         | —         | 🆕       |
| LNB Clausura                                                | 0             | 67          | 4              | —        | —         | —         | —         | 🆕       |
| Argentina Liga Nacional                                     | 2,645         | 264         | 3              | —        | —         | —         | —         |         |
| 1st Division Men                                            | 0             | 98          | 3              | —        | —         | —         | —         | 🆕       |
| Lebanese Basketball League                                  | 86            | 89          | 3              | —        | —         | —         | —         |         |
| Saku I liiga                                                | 118           | 103         | 3              | —        | —         | —         | —         |         |
| Korisliiga                                                  | 1,116         | 106         | 3              | —        | —         | —         | —         |         |
| France LF2, Women, Regular Season                           | 103           | 110         | 3              | —        | —         | —         | —         |         |
| EYBL U14, Group A                                           | 0             | 31          | 3              | —        | —         | —         | —         | 🆕       |
| Swedish Basketball Superettan                               | 350           | 138         | 3              | —        | —         | —         | —         |         |
| Segunda FEB, Group West                                     | 127           | 119         | 3              | —        | —         | —         | —         |         |
| 1. A SKL                                                    | 640           | 112         | 3              | —        | —         | —         | —         |         |
| Stoiximan GBL                                               | 1,017         | 129         | 3              | —        | —         | —         | —         |         |
| A1                                                          | 0             | 68          | 3              | —        | —         | —         | —         | 🆕       |
| Kızlar Basketbol Gençler Ligi - A Grubu                     | 0             | 34          | 3              | —        | —         | —         | —         | 🆕       |
| EYBL U13, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 0             | 30          | 3              | —        | —         | —         | —         | 🆕       |
| NBL1, Central                                               | 88            | 51          | 3              | —        | —         | —         | —         |         |
| CBI U17, Fase Final                                         | 0             | 49          | 3              | —        | —         | —         | —         | 🆕       |
| Super League                                                | 644           | 248         | 2              | —        | —         | —         | —         |         |
| Israeli National League Basketball                          | 441           | 179         | 2              | —        | —         | —         | —         |         |
| Qatari Basketball League, Regular Season                    | 43            | 52          | 2              | —        | —         | —         | —         |         |
| Colombia LPB                                                | 206           | 53          | 2              | —        | —         | —         | —         |         |
| France LFB Women                                            | 864           | 111         | 2              | —        | —         | —         | —         |         |
| Nationale 1, Group A                                        | 0             | 132         | 2              | —        | —         | —         | —         | 🆕       |
| I A MCKL                                                    | 116           | 121         | 2              | —        | —         | —         | —         |         |
| Serie A2                                                    | 639           | 265         | 2              | —        | —         | —         | —         |         |
| Hungary NB 1.A                                              | 773           | 143         | 2              | —        | —         | —         | —         |         |
| Portugal LFB Women                                          | 175           | 95          | 2              | —        | —         | —         | —         |         |
| Azerbaijan Basketball League, A Group                       | 0             | 40          | 2              | —        | —         | —         | —         | 🆕       |
| A1 Liga                                                     | 0             | 106         | 2              | —        | —         | —         | —         | 🆕       |
| Liga Košarkaškog saveza Herceg Bosne                        | 0             | 43          | 2              | —        | —         | —         | —         | 🆕       |
| Eurocup, Group A                                            | 270           | 74          | 2              | —        | —         | —         | —         |         |
| EYBL U15, Group C                                           | 0             | 32          | 2              | —        | —         | —         | —         | 🆕       |
| Iceland Women's Basketball Premier League                   | 203           | 69          | 2              | —        | —         | —         | —         |         |
| Israeli Women Basketball Premier League                     | 300           | 70          | 2              | —        | —         | —         | —         |         |
| EYBL U14, Group G                                           | 0             | 30          | 2              | —        | —         | —         | —         | 🆕       |
| 2. Muška Liga                                               | 280           | 43          | 2              | —        | —         | —         | —         |         |
| Prva RL Istok                                               | 306           | 31          | 2              | —        | —         | —         | —         |         |
| Mexico LNBP Women                                           | 93            | 52          | 2              | —        | —         | —         | —         |         |
| CBI U19, Fase Final                                         | 0             | 32          | 2              | —        | —         | —         | —         | 🆕       |
| Liga de Ascenso                                             | 360           | 88          | 2              | —        | —         | —         | —         |         |
| Liga de Baloncesto Puertorriquena , Seccion Este            | 90            | 73          | 1              | —        | —         | —         | —         |         |
| Iceland Basketball Premier League                           | 461           | 92          | 1              | —        | —         | —         | —         |         |
| Euroleague                                                  | 1,203         | 305         | 1              | —        | —         | —         | —         |         |
| Azerbaijan Basketball League, B Group                       | 0             | 42          | 1              | —        | —         | —         | —         | 🆕       |
| 1. Liga, Group East                                         | 338           | 69          | 1              | —        | —         | —         | —         |         |
| Poland 1st Division Basketball                              | 763           | 206         | 1              | —        | —         | —         | —         |         |
| 1st Division                                                | 330           | 126         | 1              | —        | —         | —         | —         |         |
| Austrian Basketball Superliga                               | 919           | 79          | 1              | —        | —         | —         | —         |         |
| SuperSport Premijer Liga                                    | 891           | 155         | 1              | —        | —         | —         | —         |         |
| Princ Caffe Superliga                                       | 588           | 80          | 1              | —        | —         | —         | —         |         |
| Zain Basketball League, Premier League                      | 0             | 43          | 1              | —        | —         | —         | —         | 🆕       |
| EYBL U16, Group B                                           | 0             | 31          | 1              | —        | —         | —         | —         | 🆕       |
| EYBL U17, Group A                                           | 0             | 30          | 1              | —        | —         | —         | —         | 🆕       |
| EYBL U20, Group B                                           | 0             | 30          | 1              | —        | —         | —         | —         | 🆕       |
| EYBL U14, Group D                                           | 0             | 34          | 1              | —        | —         | —         | —         | 🆕       |
| Elite League                                                | 337           | 56          | 1              | —        | —         | —         | —         |         |
| Türkiye Kadınlar Basketbol Ligi                             | 0             | 36          | 1              | —        | —         | —         | —         | 🆕       |
| 2. ZLS, Women                                               | 0             | 34          | 1              | —        | —         | —         | —         | 🆕       |
| Divisione Regionale 1 Toscana, Girone A                     | 0             | 66          | 1              | —        | —         | —         | —         | 🆕       |
| CIBACOPA , Primera Vuelta                                   | 0             | 96          | 1              | —        | —         | —         | —         | 🆕       |
| A2 ESKAH                                                    | 276           | 35          | 1              | —        | —         | —         | —         |         |
| Superliga                                                   | 199           | 117         | 1              | —        | —         | —         | —         |         |
| EYBL U15, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 0             | 32          | 1              | —        | —         | —         | —         | 🆕       |
| Chile LNB                                                   | 678           | 47          | 1              | —        | —         | —         | —         |         |
| NBL1, West                                                  | 141           | 108         | 1              | —        | —         | —         | —         |         |
| Big V                                                       | 594           | 82          | 1              | —        | —         | —         | —         |         |
| Brazil NBB, Playoffs                                        | 246           | 30          | 1              | —        | —         | —         | —         |         |
| WNBA                                                        | 1,206         | 292         | 1              | —        | —         | —         | —         |         |
| Mexico LNBP                                                 | 1,265         | 162         | 1              | —        | —         | —         | —         |         |
| Saku II Liiga                                               | 0             | 31          | 1              | —        | —         | —         | —         | 🆕       |
| 1. liga, Women, základní část                               | 0             | 31          | 1              | —        | —         | —         | —         | 🆕       |
| ABA League 2                                                | 251           | 44          | 0              | —        | —         | —         | —         |         |
| European North Basketball League, Group B                   | 62            | 31          | 0              | —        | —         | —         | —         |         |
| Liga Femenina                                               | 64            | 59          | 0              | —        | —         | —         | —         |         |
| Liga de Baloncesto Puertorriquena , Seccion Sur             | 86            | 75          | 0              | —        | —         | —         | —         |         |
| Lega A Basket                                               | 1,403         | 164         | 0              | —        | —         | —         | —         |         |
| NKL                                                         | 700           | 172         | 0              | —        | —         | —         | —         |         |
| Liga de Baloncesto Puertorriquena , Seccion Norte           | 85            | 73          | 0              | —        | —         | —         | —         |         |
| Liga de Baloncesto Puertorriquena , Seccion Oeste           | 85            | 72          | 0              | —        | —         | —         | —         |         |
| 1. Liga, Group West                                         | 309           | 63          | 0              | —        | —         | —         | —         |         |
| PGE Basket Liga                                             | 1,346         | 169         | 0              | —        | —         | —         | —         |         |
| Élite 2                                                     | 1,869         | 270         | 0              | —        | —         | —         | —         |         |
| AdmiralBet ABA League, Group B                              | 0             | 59          | 0              | —        | —         | —         | —         | 🆕       |
| SLB                                                         | 80            | 115         | 0              | —        | —         | —         | —         |         |
| Nationale 1, Group B                                        | 0             | 139         | 0              | —        | —         | —         | —         | 🆕       |
| Meridianbet KLS                                             | 1,257         | 183         | 0              | —        | —         | —         | —         |         |
| RKL Division A                                              | 0             | 198         | 0              | —        | —         | —         | —         | 🆕       |
| Superleague                                                 | 396           | 86          | 0              | —        | —         | —         | —         |         |
| Chile LNB, Playoffs                                         | 197           | 34          | 0              | —        | —         | —         | —         |         |
| Belgian Basketball 2nd Division                             | 156           | 169         | 0              | —        | —         | —         | —         |         |
| Portugal Proliga, South                                     | 105           | 31          | 0              | —        | —         | —         | —         |         |
| Serie B, Group A                                            | 576           | 220         | 0              | —        | —         | —         | —         |         |
| Segunda FEB, Group East                                     | 131           | 117         | 0              | —        | —         | —         | —         |         |
| AdmiralBet ABA League, Group A                              | 0             | 58          | 0              | —        | —         | —         | —         | 🆕       |
| Enovos League, Regular Season                               | 0             | 90          | 0              | —        | —         | —         | —         | 🆕       |
| Slovenian Second Basketball League, Regular Season          | 0             | 76          | 0              | —        | —         | —         | —         | 🆕       |
| The League                                                  | 122           | 90          | 0              | —        | —         | —         | —         |         |
| French Basketball Cup                                       | 219           | 46          | 0              | —        | —         | —         | —         |         |
| Iranian Basketball Super League                             | 366           | 38          | 0              | —        | —         | —         | —         |         |
| U19 ABA League                                              | 0             | 33          | 0              | —        | —         | —         | —         | 🆕       |
| EuroCup, Women, Playoffs                                    | 285           | 55          | 0              | —        | —         | —         | —         |         |
| EYBL U20, Group D                                           | 0             | 30          | 0              | —        | —         | —         | —         | 🆕       |
| Bulgaria NBL                                                | 683           | 80          | 0              | —        | —         | —         | —         |         |
| Germany Pro A                                               | 1,519         | 99          | 0              | —        | —         | —         | —         |         |
| Pro B, South                                                | 0             | 57          | 0              | —        | —         | —         | —         | 🆕       |
| Druga muška liga - Centar                                   | 105           | 34          | 0              | —        | —         | —         | —         |         |
| Serie B Interregionale, Girone C                            | 0             | 40          | 0              | —        | —         | —         | —         | 🆕       |
| Serie B Interregionale, Girone E                            | 0             | 33          | 0              | —        | —         | —         | —         | 🆕       |
| Erkekler Basketbol Gençler Ligi - A Grubu                   | 0             | 35          | 0              | —        | —         | —         | —         | 🆕       |
| Serie B Interregionale, Girone D                            | 0             | 46          | 0              | —        | —         | —         | —         | 🆕       |
| Divisione Regionale 2 - Lazio, Girone C                     | 0             | 30          | 0              | —        | —         | —         | —         | 🆕       |
| Serie C Toscana                                             | 0             | 49          | 0              | —        | —         | —         | —         | 🆕       |
| Serie B Interregionale, Girone A                            | 0             | 45          | 0              | —        | —         | —         | —         | 🆕       |
| Serie B Interregionale, Girone B                            | 0             | 39          | 0              | —        | —         | —         | —         | 🆕       |
| Serie C Girone H sud                                        | 0             | 30          | 0              | —        | —         | —         | —         | 🆕       |
| B Ethniki - Group 1                                         | 0             | 32          | 0              | —        | —         | —         | —         | 🆕       |
| B Ethniki - Group 2                                         | 0             | 30          | 0              | —        | —         | —         | —         | 🆕       |
| B Ethniki - Group 3                                         | 0             | 33          | 0              | —        | —         | —         | —         | 🆕       |
| TBL                                                         | 712           | 56          | 0              | —        | —         | —         | —         |         |
| Divisione Regionale 1 Toscana, Girone B                     | 0             | 60          | 0              | —        | —         | —         | —         | 🆕       |
| Basket World League                                         | 0             | 31          | 0              | —        | —         | —         | —         | 🆕       |
| Pro B, North                                                | 0             | 55          | 0              | —        | —         | —         | —         | 🆕       |
| Serie B Interregionale, Girone F                            | 0             | 42          | 0              | —        | —         | —         | —         | 🆕       |
| A1 ESKAH                                                    | 284           | 32          | 0              | —        | —         | —         | —         |         |
| Belize Elite Basketball League, Regular Season              | 0             | 31          | 0              | —        | —         | —         | —         | 🆕       |
| Egyptian Women's Basketball Super League, Regular Season    | 0             | 55          | 0              | —        | —         | —         | —         | 🆕       |
| EYBL U16, SUPERFINAL - 12./15.03.2026 - Riga, Latvia        | 0             | 32          | 0              | —        | —         | —         | —         | 🆕       |
| NKL, Playoffs                                               | 79            | 32          | 0              | —        | —         | —         | —         |         |
| Puerto Rico BSN                                             | 245           | 156         | 0              | —        | —         | —         | —         |         |
| LNB 2                                                       | 0             | 44          | 0              | —        | —         | —         | —         | 🆕       |
| NBL1, South                                                 | 194           | 135         | 0              | —        | —         | —         | —         |         |
| Liga Argentina, Playoffs                                    | 74            | 56          | 0              | —        | —         | —         | —         |         |
| CIBACOPA , Segunda Vuelta                                   | 0             | 88          | 0              | —        | —         | —         | —         | 🆕       |
| New Zealand NBL                                             | 180           | 73          | 0              | —        | —         | —         | —         |         |
| Rapid League, Regular Season                                | 0             | 83          | 0              | —        | —         | —         | —         | 🆕       |
| NBL1, North                                                 | 113           | 71          | 0              | —        | —         | —         | —         |         |
| NBL1 North, Women                                           | 113           | 74          | 0              | —        | —         | —         | —         |         |
| Canadian Elite Basketball League                            | 95            | 103         | 0              | —        | —         | —         | —         |         |
| Serie B, Playoffs                                           | 61            | 43          | 0              | —        | —         | —         | —         |         |
| CBI U17, Sede Vila Velha                                    | 0             | 57          | 0              | —        | —         | —         | —         | 🆕       |
| Puerto Rico BSN, Playoffs                                   | 43            | 32          | 0              | —        | —         | —         | —         |         |
| Copa Espana                                                 | 0             | 41          | 0              | —        | —         | —         | —         | 🆕       |
| LDB, Fase Final                                             | 0             | 39          | 0              | —        | —         | —         | —         | 🆕       |
| BSNF, Women                                                 | 0             | 60          | 0              | —        | —         | —         | —         | 🆕       |
| Divisional Tercera de Ascenso, Fase Regular DTA 2026        | 0             | 41          | 0              | —        | —         | —         | —         | 🆕       |

## 4. Detalle por liga y por umbral — Top 30 por volumen de apuestas

### NCAA Women's Division I, Regular season

- Partidos entrenados (A): **0** · Holdout (C): **1,620** · Sesgo @0.75: +0.0026

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 611      | 72.0 % | 68.3–75.4 | +0.82 %   | +31.78 %  |         |
| 0.68 | 484      | 73.8 % | 69.7–77.5 | +3.26 %   | +34.98 %  |         |
| 0.70 | 432      | 75.7 % | 71.4–79.5 | +5.97 %   | +38.52 %  |         |
| 0.72 | 368      | 78.3 % | 73.8–82.2 | +9.57 %   | +43.22 %  |         |
| 0.75 | 275      | 82.2 % | 77.2–86.3 | +15.05 %  | +50.39 %  |         |
| 0.80 | 178      | 88.8 % | 83.3–92.6 | +24.27 %  | +62.44 %  |         |

### FSHB Superliga, K.K.M 2025-26

- Partidos entrenados (A): **0** · Holdout (C): **91** · Sesgo @0.75: +0.0002

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 60       | 75.0 % | 62.8–84.2 | +5.00 %   | +37.25 %  |         |
| 0.68 | 51       | 80.4 % | 67.5–89.0 | +12.55 %  | +47.12 %  |         |
| 0.70 | 51       | 80.4 % | 67.5–89.0 | +12.55 %  | +47.12 %  |         |
| 0.72 | 49       | 79.6 % | 66.4–88.5 | +11.43 %  | +45.65 %  |         |
| 0.75 | 39       | 82.0 % | 67.3–91.0 | +14.87 %  | +50.15 %  |         |
| 0.80 | 24       | 91.7 % | 74.2–97.7 | +28.33 %  | +67.75 %  |         |

### Tipos Extraliga, Women

- Partidos entrenados (A): **424** · Holdout (C): **64** · Sesgo @0.75: +0.0677

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 52       | 71.2 % | 57.7–81.7 | -0.38 %   | +30.21 %  |         |
| 0.68 | 47       | 74.5 % | 60.5–84.7 | +4.26 %   | +36.28 %  |         |
| 0.70 | 46       | 73.9 % | 59.7–84.4 | +3.48 %   | +35.26 %  |         |
| 0.72 | 44       | 75.0 % | 60.6–85.4 | +5.00 %   | +37.25 %  |         |
| 0.75 | 39       | 79.5 % | 64.5–89.2 | +11.28 %  | +45.46 %  |         |
| 0.80 | 30       | 83.3 % | 66.4–92.7 | +16.67 %  | +52.50 %  |         |

### NBL, East

- Partidos entrenados (A): **78** · Holdout (C): **97** · Sesgo @0.75: -0.0481

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 61       | 80.3 % | 68.7–88.4 | +12.46 %  | +47.00 %  |         |
| 0.68 | 53       | 83.0 % | 70.8–90.8 | +16.23 %  | +51.92 %  |         |
| 0.70 | 45       | 84.4 % | 71.2–92.3 | +18.22 %  | +54.53 %  |         |
| 0.72 | 40       | 87.5 % | 73.9–94.5 | +22.50 %  | +60.12 %  |         |
| 0.75 | 36       | 86.1 % | 71.3–93.9 | +20.56 %  | +57.58 %  |         |
| 0.80 | 17       | 82.3 % | 59.0–93.8 | +15.29 %  | +50.71 %  |         |

### LNB, Women

- Partidos entrenados (A): **184** · Holdout (C): **55** · Sesgo @0.75: -0.0201

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 44       | 90.9 % | 78.8–96.4 | +27.27 %  | +66.36 %  |         |
| 0.68 | 44       | 90.9 % | 78.8–96.4 | +27.27 %  | +66.36 %  |         |
| 0.70 | 39       | 89.7 % | 76.4–95.9 | +25.64 %  | +64.23 %  |         |
| 0.72 | 38       | 92.1 % | 79.2–97.3 | +28.95 %  | +68.55 %  |         |
| 0.75 | 36       | 91.7 % | 78.2–97.1 | +28.33 %  | +67.75 %  |         |
| 0.80 | 32       | 90.6 % | 75.8–96.8 | +26.87 %  | +65.84 %  |         |

### NBL1 East, Women

- Partidos entrenados (A): **153** · Holdout (C): **118** · Sesgo @0.75: -0.0477

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 78       | 75.6 % | 65.1–83.8 | +5.90 %   | +38.42 %  |         |
| 0.68 | 62       | 79.0 % | 67.4–87.3 | +10.65 %  | +44.63 %  |         |
| 0.70 | 59       | 78.0 % | 65.9–86.6 | +9.15 %   | +42.68 %  |         |
| 0.72 | 46       | 82.6 % | 69.3–90.9 | +15.65 %  | +51.17 %  |         |
| 0.75 | 36       | 86.1 % | 71.3–93.9 | +20.56 %  | +57.58 %  |         |
| 0.80 | 19       | 89.5 % | 68.6–97.1 | +25.26 %  | +63.74 %  |         |

### ZBL, Women

- Partidos entrenados (A): **694** · Holdout (C): **74** · Sesgo @0.75: -0.0450

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 52       | 84.6 % | 72.5–92.0 | +18.46 %  | +54.85 %  |         |
| 0.68 | 45       | 84.4 % | 71.2–92.3 | +18.22 %  | +54.53 %  |         |
| 0.70 | 44       | 86.4 % | 73.3–93.6 | +20.91 %  | +58.05 %  |         |
| 0.72 | 41       | 85.4 % | 71.6–93.1 | +19.51 %  | +56.22 %  |         |
| 0.75 | 35       | 88.6 % | 74.0–95.5 | +24.00 %  | +62.09 %  |         |
| 0.80 | 25       | 88.0 % | 70.0–95.8 | +23.20 %  | +61.04 %  |         |

### Swedish Basketball League Women

- Partidos entrenados (A): **233** · Holdout (C): **114** · Sesgo @0.75: +0.0858

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 64       | 70.3 % | 58.2–80.1 | -1.56 %   | +28.67 %  |         |
| 0.68 | 53       | 71.7 % | 58.4–82.0 | +0.38 %   | +31.21 %  |         |
| 0.70 | 49       | 69.4 % | 55.5–80.5 | -2.86 %   | +26.98 %  |         |
| 0.72 | 41       | 73.2 % | 58.1–84.3 | +2.44 %   | +33.90 %  |         |
| 0.75 | 34       | 73.5 % | 56.9–85.4 | +2.94 %   | +34.56 %  |         |
| 0.80 | 19       | 68.4 % | 46.0–84.6 | -4.21 %   | +25.21 %  |         |

### United League

- Partidos entrenados (A): **1,133** · Holdout (C): **176** · Sesgo @0.75: -0.0248

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 94       | 74.5 % | 64.8–82.2 | +4.26 %   | +36.28 %  |         |
| 0.68 | 75       | 76.0 % | 65.2–84.2 | +6.40 %   | +39.08 %  |         |
| 0.70 | 62       | 77.4 % | 65.6–86.0 | +8.39 %   | +41.68 %  |         |
| 0.72 | 49       | 79.6 % | 66.4–88.5 | +11.43 %  | +45.65 %  |         |
| 0.75 | 34       | 82.3 % | 66.5–91.7 | +15.29 %  | +50.71 %  |         |
| 0.80 | 15       | 73.3 % | 48.0–89.1 | +2.67 %   | +34.20 %  |         |

### Top Division Women, Regular Season

- Partidos entrenados (A): **0** · Holdout (C): **89** · Sesgo @0.75: -0.0480

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 62       | 82.3 % | 71.0–89.8 | +15.16 %  | +50.53 %  |         |
| 0.68 | 56       | 82.1 % | 70.2–90.0 | +15.00 %  | +50.32 %  |         |
| 0.70 | 49       | 83.7 % | 71.0–91.5 | +17.14 %  | +53.12 %  |         |
| 0.72 | 44       | 86.4 % | 73.3–93.6 | +20.91 %  | +58.05 %  |         |
| 0.75 | 34       | 88.2 % | 73.4–95.3 | +23.53 %  | +61.47 %  |         |
| 0.80 | 24       | 87.5 % | 69.0–95.7 | +22.50 %  | +60.12 %  |         |

### Hungary Divison A Women

- Partidos entrenados (A): **276** · Holdout (C): **104** · Sesgo @0.75: +0.0710

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 69       | 72.5 % | 61.0–81.6 | +1.45 %   | +32.61 %  |         |
| 0.68 | 57       | 73.7 % | 61.0–83.4 | +3.16 %   | +34.84 %  |         |
| 0.70 | 50       | 74.0 % | 60.4–84.1 | +3.60 %   | +35.42 %  |         |
| 0.72 | 44       | 77.3 % | 63.0–87.2 | +8.18 %   | +41.41 %  |         |
| 0.75 | 34       | 76.5 % | 60.0–87.6 | +7.06 %   | +39.94 %  |         |
| 0.80 | 26       | 76.9 % | 57.9–89.0 | +7.69 %   | +40.77 %  |         |

### Basketligaen

- Partidos entrenados (A): **651** · Holdout (C): **111** · Sesgo @0.75: +0.0201

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 57       | 75.4 % | 62.9–84.8 | +5.61 %   | +38.05 %  |         |
| 0.68 | 50       | 76.0 % | 62.6–85.7 | +6.40 %   | +39.08 %  |         |
| 0.70 | 46       | 76.1 % | 62.1–86.1 | +6.52 %   | +39.24 %  |         |
| 0.72 | 42       | 76.2 % | 61.5–86.5 | +6.67 %   | +39.43 %  |         |
| 0.75 | 32       | 78.1 % | 61.2–89.0 | +9.38 %   | +42.97 %  |         |
| 0.80 | 18       | 83.3 % | 60.8–94.2 | +16.67 %  | +52.50 %  |         |

### Premijer ženska liga

- Partidos entrenados (A): **299** · Holdout (C): **100** · Sesgo @0.75: +0.1344

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 61       | 59.0 % | 46.5–70.5 | -17.38 %  | +8.00 %   | ⚠️      |
| 0.68 | 55       | 60.0 % | 46.8–71.9 | -16.00 %  | +9.80 %   |         |
| 0.70 | 52       | 61.5 % | 48.0–73.5 | -13.85 %  | +12.62 %  |         |
| 0.72 | 41       | 65.8 % | 50.5–78.4 | -7.80 %   | +20.51 %  |         |
| 0.75 | 32       | 68.8 % | 51.4–82.0 | -3.75 %   | +25.81 %  |         |
| 0.80 | 20       | 60.0 % | 38.7–78.1 | -16.00 %  | +9.80 %   |         |

### MPBL

- Partidos entrenados (A): **319** · Holdout (C): **264** · Sesgo @0.75: -0.0813

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 104      | 68.3 % | 58.8–76.4 | -4.42 %   | +24.93 %  |         |
| 0.68 | 81       | 72.8 % | 62.3–81.3 | +1.98 %   | +33.30 %  |         |
| 0.70 | 63       | 81.0 % | 69.6–88.8 | +13.33 %  | +48.14 %  |         |
| 0.72 | 50       | 80.0 % | 67.0–88.8 | +12.00 %  | +46.40 %  |         |
| 0.75 | 30       | 86.7 % | 70.3–94.7 | +21.33 %  | +58.60 %  |         |
| 0.80 | 5        | —      | —         | —         | —         |         |

### Russian Women's Basketball Premier League

- Partidos entrenados (A): **726** · Holdout (C): **103** · Sesgo @0.75: +0.0757

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 56       | 78.6 % | 66.2–87.3 | +10.00 %  | +43.79 %  |         |
| 0.68 | 43       | 76.7 % | 62.3–86.8 | +7.44 %   | +40.44 %  |         |
| 0.70 | 38       | 73.7 % | 58.0–85.0 | +3.16 %   | +34.84 %  |         |
| 0.72 | 36       | 72.2 % | 56.0–84.2 | +1.11 %   | +32.17 %  |         |
| 0.75 | 27       | 74.1 % | 55.3–86.8 | +3.70 %   | +35.56 %  |         |
| 0.80 | 17       | 70.6 % | 46.9–86.7 | -1.18 %   | +29.18 %  |         |

### Prva Liga

- Partidos entrenados (A): **0** · Holdout (C): **139** · Sesgo @0.75: -0.0733

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 44       | 77.3 % | 63.0–87.2 | +8.18 %   | +41.41 %  |         |
| 0.68 | 37       | 83.8 % | 68.9–92.3 | +17.30 %  | +53.32 %  |         |
| 0.70 | 32       | 84.4 % | 68.2–93.1 | +18.12 %  | +54.41 %  |         |
| 0.72 | 27       | 88.9 % | 71.9–96.1 | +24.44 %  | +62.67 %  |         |
| 0.75 | 27       | 88.9 % | 71.9–96.1 | +24.44 %  | +62.67 %  |         |
| 0.80 | 16       | 93.8 % | 71.7–98.9 | +31.25 %  | +71.56 %  |         |

### CBI U15 , Classificatória A

- Partidos entrenados (A): **0** · Holdout (C): **76** · Sesgo @0.75: -0.0917

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 39       | 84.6 % | 70.3–92.8 | +18.46 %  | +54.85 %  |         |
| 0.68 | 32       | 93.8 % | 79.9–98.3 | +31.25 %  | +71.56 %  |         |
| 0.70 | 29       | 93.1 % | 78.0–98.1 | +30.34 %  | +70.38 %  |         |
| 0.72 | 28       | 92.9 % | 77.4–98.0 | +30.00 %  | +69.93 %  |         |
| 0.75 | 27       | 92.6 % | 76.6–97.9 | +29.63 %  | +69.44 %  |         |
| 0.80 | 21       | 95.2 % | 77.3–99.2 | +33.33 %  | +74.29 %  |         |

### Big V, Women

- Partidos entrenados (A): **108** · Holdout (C): **82** · Sesgo @0.75: +0.0164

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 51       | 70.6 % | 57.0–81.3 | -1.18 %   | +29.18 %  |         |
| 0.68 | 44       | 77.3 % | 63.0–87.2 | +8.18 %   | +41.41 %  |         |
| 0.70 | 37       | 78.4 % | 62.8–88.6 | +9.73 %   | +43.43 %  |         |
| 0.72 | 34       | 76.5 % | 60.0–87.6 | +7.06 %   | +39.94 %  |         |
| 0.75 | 26       | 80.8 % | 62.1–91.5 | +13.08 %  | +47.81 %  |         |
| 0.80 | 13       | —      | —         | —         | —         |         |

### 1st Division Women

- Partidos entrenados (A): **0** · Holdout (C): **57** · Sesgo @0.75: -0.0394

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 36       | 75.0 % | 58.9–86.2 | +5.00 %   | +37.25 %  |         |
| 0.68 | 29       | 82.8 % | 65.5–92.4 | +15.86 %  | +51.45 %  |         |
| 0.70 | 28       | 82.1 % | 64.4–92.1 | +15.00 %  | +50.32 %  |         |
| 0.72 | 25       | 84.0 % | 65.3–93.6 | +17.60 %  | +53.72 %  |         |
| 0.75 | 23       | 87.0 % | 67.9–95.5 | +21.74 %  | +59.13 %  |         |
| 0.80 | 15       | 93.3 % | 70.2–98.8 | +30.67 %  | +70.80 %  |         |

### Estonian-Latvian Basketball League

- Partidos entrenados (A): **594** · Holdout (C): **163** · Sesgo @0.75: -0.0454

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 86       | 79.1 % | 69.3–86.3 | +10.70 %  | +44.70 %  |         |
| 0.68 | 67       | 82.1 % | 71.3–89.4 | +14.93 %  | +50.22 %  |         |
| 0.70 | 51       | 86.3 % | 74.3–93.2 | +20.78 %  | +57.88 %  |         |
| 0.72 | 40       | 87.5 % | 73.9–94.5 | +22.50 %  | +60.12 %  |         |
| 0.75 | 22       | 81.8 % | 61.5–92.7 | +14.55 %  | +49.73 %  |         |
| 0.80 | 4        | —      | —         | —         | —         |         |

### Nostra.lt-RKL Division B, Reguliarusis sezonas

- Partidos entrenados (A): **0** · Holdout (C): **182** · Sesgo @0.75: +0.0640

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 63       | 74.6 % | 62.7–83.7 | +4.44 %   | +36.52 %  |         |
| 0.68 | 48       | 72.9 % | 59.0–83.4 | +2.08 %   | +33.44 %  |         |
| 0.70 | 40       | 80.0 % | 65.2–89.5 | +12.00 %  | +46.40 %  |         |
| 0.72 | 33       | 75.8 % | 59.0–87.2 | +6.06 %   | +38.64 %  |         |
| 0.75 | 22       | 72.7 % | 51.8–86.8 | +1.82 %   | +33.09 %  |         |
| 0.80 | 8        | —      | —         | —         | —         |         |

### SLB, Women

- Partidos entrenados (A): **0** · Holdout (C): **89** · Sesgo @0.75: +0.1444

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 56       | 66.1 % | 53.0–77.1 | -7.50 %   | +20.91 %  |         |
| 0.68 | 47       | 68.1 % | 53.8–79.6 | -4.68 %   | +24.60 %  |         |
| 0.70 | 41       | 70.7 % | 55.5–82.4 | -0.98 %   | +29.44 %  |         |
| 0.72 | 29       | 65.5 % | 47.3–80.1 | -8.28 %   | +19.90 %  |         |
| 0.75 | 22       | 68.2 % | 47.3–83.6 | -4.55 %   | +24.77 %  |         |
| 0.80 | 15       | 73.3 % | 48.0–89.1 | +2.67 %   | +34.20 %  |         |

### 2nd League

- Partidos entrenados (A): **269** · Holdout (C): **132** · Sesgo @0.75: -0.0254

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 67       | 73.1 % | 61.5–82.3 | +2.39 %   | +33.84 %  |         |
| 0.68 | 50       | 74.0 % | 60.4–84.1 | +3.60 %   | +35.42 %  |         |
| 0.70 | 39       | 74.4 % | 58.9–85.4 | +4.10 %   | +36.08 %  |         |
| 0.72 | 31       | 74.2 % | 56.8–86.3 | +3.87 %   | +35.77 %  |         |
| 0.75 | 22       | 81.8 % | 61.5–92.7 | +14.55 %  | +49.73 %  |         |
| 0.80 | 7        | —      | —         | —         | —         |         |

### CBI U19, Sede Rio

- Partidos entrenados (A): **0** · Holdout (C): **37** · Sesgo @0.75: -0.0617

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 32       | 78.1 % | 61.2–89.0 | +9.38 %   | +42.97 %  |         |
| 0.68 | 29       | 82.8 % | 65.5–92.4 | +15.86 %  | +51.45 %  |         |
| 0.70 | 29       | 82.8 % | 65.5–92.4 | +15.86 %  | +51.45 %  |         |
| 0.72 | 28       | 82.1 % | 64.4–92.1 | +15.00 %  | +50.32 %  |         |
| 0.75 | 21       | 90.5 % | 71.1–97.3 | +26.67 %  | +65.57 %  |         |
| 0.80 | 13       | —      | —         | —         | —         |         |

### NBL, West

- Partidos entrenados (A): **75** · Holdout (C): **93** · Sesgo @0.75: +0.0605

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 52       | 80.8 % | 68.1–89.2 | +13.08 %  | +47.81 %  |         |
| 0.68 | 39       | 76.9 % | 61.7–87.4 | +7.69 %   | +40.77 %  |         |
| 0.70 | 27       | 77.8 % | 59.2–89.4 | +8.89 %   | +42.33 %  |         |
| 0.72 | 25       | 76.0 % | 56.6–88.5 | +6.40 %   | +39.08 %  |         |
| 0.75 | 20       | 75.0 % | 53.1–88.8 | +5.00 %   | +37.25 %  |         |
| 0.80 | 11       | —      | —         | —         | —         |         |

### CBI U15 , Classificatória B

- Partidos entrenados (A): **0** · Holdout (C): **78** · Sesgo @0.75: +0.0743

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 41       | 75.6 % | 60.7–86.2 | +5.85 %   | +38.37 %  |         |
| 0.68 | 34       | 76.5 % | 60.0–87.6 | +7.06 %   | +39.94 %  |         |
| 0.70 | 32       | 75.0 % | 57.9–86.7 | +5.00 %   | +37.25 %  |         |
| 0.72 | 29       | 75.9 % | 57.9–87.8 | +6.21 %   | +38.83 %  |         |
| 0.75 | 20       | 75.0 % | 53.1–88.8 | +5.00 %   | +37.25 %  |         |
| 0.80 | 12       | —      | —         | —         | —         |         |

### BNXT League

- Partidos entrenados (A): **205** · Holdout (C): **231** · Sesgo @0.75: -0.1718

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 79       | 74.7 % | 64.1–83.0 | +4.56 %   | +36.67 %  |         |
| 0.68 | 54       | 72.2 % | 59.1–82.4 | +1.11 %   | +32.17 %  |         |
| 0.70 | 35       | 71.4 % | 54.9–83.7 | +0.00 %   | +30.71 %  |         |
| 0.72 | 28       | 78.6 % | 60.5–89.8 | +10.00 %  | +43.79 %  |         |
| 0.75 | 19       | 94.7 % | 75.4–99.1 | +32.63 %  | +73.37 %  |         |
| 0.80 | 2        | —      | —         | —         | —         |         |

### Finland Division 1A Women

- Partidos entrenados (A): **294** · Holdout (C): **95** · Sesgo @0.75: -0.0337

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 43       | 72.1 % | 57.3–83.3 | +0.93 %   | +31.93 %  |         |
| 0.68 | 35       | 74.3 % | 57.9–85.8 | +4.00 %   | +35.94 %  |         |
| 0.70 | 29       | 86.2 % | 69.4–94.5 | +20.69 %  | +57.76 %  |         |
| 0.72 | 25       | 88.0 % | 70.0–95.8 | +23.20 %  | +61.04 %  |         |
| 0.75 | 19       | 84.2 % | 62.4–94.5 | +17.89 %  | +54.11 %  |         |
| 0.80 | 9        | —      | —         | —         | —         |         |

### Switzerland LNA

- Partidos entrenados (A): **689** · Holdout (C): **91** · Sesgo @0.75: +0.0610

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 45       | 75.6 % | 61.3–85.8 | +5.78 %   | +38.27 %  |         |
| 0.68 | 36       | 77.8 % | 61.9–88.3 | +8.89 %   | +42.33 %  |         |
| 0.70 | 29       | 72.4 % | 54.3–85.3 | +1.38 %   | +32.52 %  |         |
| 0.72 | 22       | 77.3 % | 56.6–89.9 | +8.18 %   | +41.41 %  |         |
| 0.75 | 19       | 73.7 % | 51.2–88.2 | +3.16 %   | +34.84 %  |         |
| 0.80 | 6        | —      | —         | —         | —         |         |

### Saudi Basketball League, Premier League

- Partidos entrenados (A): **0** · Holdout (C): **45** · Sesgo @0.75: -0.1213

| P ≥  | Apuestas | WR     | IC95      | ROI @1.40 | ROI @1.83 | Bloqueo |
| ---  | ---:     | ---:   | ---       | ---:      | ---:      | ---     |
| 0.65 | 32       | 81.2 % | 64.7–91.1 | +13.75 %  | +48.69 %  |         |
| 0.68 | 27       | 85.2 % | 67.5–94.1 | +19.26 %  | +55.89 %  |         |
| 0.70 | 26       | 88.5 % | 71.0–96.0 | +23.85 %  | +61.88 %  |         |
| 0.72 | 24       | 91.7 % | 74.2–97.7 | +28.33 %  | +67.75 %  |         |
| 0.75 | 19       | 94.7 % | 75.4–99.1 | +32.63 %  | +73.37 %  |         |
| 0.80 | 13       | —      | —         | —         | —         |         |

## 5. Ligas problemáticas y destacadas

### 5.1 Candidatas a bloqueo (IC95 superior del WR < 71.4 % a 0.75)

_Ninguna con la evidencia actual._


### 5.2 Arranque en frío (0 partidos en entrenamiento)

108 ligas aparecen solo en el holdout C, es decir, el modelo **no las vio en entrenamiento**. Para ellas los priors (ELO, H2H, win-rate) provienen del estado as-of, pero la liga no aportó filas al ajuste. Conviene vigilar su calibración antes de confiar en ellas.

| Liga                                               | Holdout (C) | Apuestas @0.75 | WR @0.75 |
| ---                                                | ---:        | ---:           | ---:     |
| NCAA Women's Division I, Regular season            | 1,620       | 275            | 82.2 %   |
| RKL Division A                                     | 198         | 0              | —        |
| Nostra.lt-RKL Division B, Reguliarusis sezonas     | 182         | 22             | 72.7 %   |
| LF Challenge, Regular Season                       | 154         | 9              | —        |
| Prva Liga                                          | 139         | 27             | 88.9 %   |
| Nationale 1, Group B                               | 139         | 0              | —        |
| U21 Espoirs Elite                                  | 136         | 5              | —        |
| Nationale 1, Group A                               | 132         | 2              | —        |
| LDB, Primeira fase                                 | 128         | 16             | 87.5 %   |
| NBL Men , Regular Season                           | 126         | 8              | —        |
| Liga 1 Masculin, Faza II                           | 119         | 16             | 68.8 %   |
| Liga Femenina, Women, Clausura                     | 115         | 11             | —        |
| NBL Women , Regular Season                         | 106         | 17             | 82.3 %   |
| A1 Liga                                            | 106         | 2              | —        |
| 1st Division Men                                   | 98          | 3              | —        |
| CIBACOPA , Primera Vuelta                          | 96          | 1              | —        |
| FSHB Superliga, K.K.M 2025-26                      | 91          | 39             | 82.0 %   |
| Enovos League, Regular Season                      | 90          | 0              | —        |
| Top Division Women, Regular Season                 | 89          | 34             | 88.2 %   |
| SLB, Women                                         | 89          | 22             | 68.2 %   |
| CIBACOPA , Segunda Vuelta                          | 88          | 0              | —        |
| 1. liga, Women                                     | 87          | 7              | —        |
| Rapid League, Regular Season                       | 83          | 0              | —        |
| CBI U15 , Classificatória B                        | 78          | 20             | 75.0 %   |
| CBI U15 , Classificatória A                        | 76          | 27             | 92.6 %   |
| Slovenian Second Basketball League, Regular Season | 76          | 0              | —        |
| A1                                                 | 68          | 3              | —        |
| CBI U17, Sede Curitiba                             | 67          | 14             | —        |
| LNB Clausura                                       | 67          | 4              | —        |
| Sparkasse Prva Liga Zeni                           | 66          | 17             | 52.9 %   |
| Divisione Regionale 1 Toscana, Girone A            | 66          | 1              | —        |
| Rwanda Basketball League, Regular Season           | 65          | 15             | 80.0 %   |
| Jedinstvena kadetska liga                          | 64          | 10             | —        |
| Divisione Regionale 1 Toscana, Girone B            | 60          | 0              | —        |
| BSNF, Women                                        | 60          | 0              | —        |
| NCAA Women's March Madness                         | 59          | 7              | —        |
| AdmiralBet ABA League, Group B                     | 59          | 0              | —        |
| AdmiralBet ABA League, Group A                     | 58          | 0              | —        |
| 1st Division Women                                 | 57          | 23             | 87.0 %   |
| Pro B, South                                       | 57          | 0              | —        |

_...y 68 ligas más con menos volumen._


### 5.3 Mejores ligas por ROI @1.40 a 0.75 (mín. 30 apuestas)

| Liga                                    | Entrenado | Apuestas @0.75 | WR     | ROI @1.40 | ROI @1.83 |
| ---                                     | ---:      | ---:           | ---:   | ---:      | ---:      |
| LNB, Women                              | 184       | 36             | 91.7 % | +28.33 %  | +67.75 %  |
| ZBL, Women                              | 694       | 35             | 88.6 % | +24.00 %  | +62.09 %  |
| Top Division Women, Regular Season      | 0         | 34             | 88.2 % | +23.53 %  | +61.47 %  |
| MPBL                                    | 319       | 30             | 86.7 % | +21.33 %  | +58.60 %  |
| NBL, East                               | 78        | 36             | 86.1 % | +20.56 %  | +57.58 %  |
| NBL1 East, Women                        | 153       | 36             | 86.1 % | +20.56 %  | +57.58 %  |
| United League                           | 1,133     | 34             | 82.3 % | +15.29 %  | +50.71 %  |
| NCAA Women's Division I, Regular season | 0         | 275            | 82.2 % | +15.05 %  | +50.39 %  |
| FSHB Superliga, K.K.M 2025-26           | 0         | 39             | 82.0 % | +14.87 %  | +50.15 %  |
| Tipos Extraliga, Women                  | 424       | 39             | 79.5 % | +11.28 %  | +45.46 %  |
| Basketligaen                            | 651       | 32             | 78.1 % | +9.38 %   | +42.97 %  |
| Hungary Divison A Women                 | 276       | 34             | 76.5 % | +7.06 %   | +39.94 %  |
| Swedish Basketball League Women         | 233       | 34             | 73.5 % | +2.94 %   | +34.56 %  |
| Premijer ženska liga                    | 299       | 32             | 68.8 % | -3.75 %   | +25.81 %  |

### 5.4 Ligas con peor ROI @1.40 a 0.75 (mín. 30 apuestas)

| Liga                                    | Entrenado | Apuestas @0.75 | WR     | ROI @1.40 | ROI @1.83 |
| ---                                     | ---:      | ---:           | ---:   | ---:      | ---:      |
| Premijer ženska liga                    | 299       | 32             | 68.8 % | -3.75 %   | +25.81 %  |
| Swedish Basketball League Women         | 233       | 34             | 73.5 % | +2.94 %   | +34.56 %  |
| Hungary Divison A Women                 | 276       | 34             | 76.5 % | +7.06 %   | +39.94 %  |
| Basketligaen                            | 651       | 32             | 78.1 % | +9.38 %   | +42.97 %  |
| Tipos Extraliga, Women                  | 424       | 39             | 79.5 % | +11.28 %  | +45.46 %  |
| FSHB Superliga, K.K.M 2025-26           | 0         | 39             | 82.0 % | +14.87 %  | +50.15 %  |
| NCAA Women's Division I, Regular season | 0         | 275            | 82.2 % | +15.05 %  | +50.39 %  |
| United League                           | 1,133     | 34             | 82.3 % | +15.29 %  | +50.71 %  |
| NBL, East                               | 78        | 36             | 86.1 % | +20.56 %  | +57.58 %  |
| NBL1 East, Women                        | 153       | 36             | 86.1 % | +20.56 %  | +57.58 %  |
| MPBL                                    | 319       | 30             | 86.7 % | +21.33 %  | +58.60 %  |
| Top Division Women, Regular Season      | 0         | 34             | 88.2 % | +23.53 %  | +61.47 %  |
| ZBL, Women                              | 694       | 35             | 88.6 % | +24.00 %  | +62.09 %  |
| LNB, Women                              | 184       | 36             | 91.7 % | +28.33 %  | +67.75 %  |

## 6. Resultados en vivo del monitoreo (realidad, no backtest)

> Estos son los **partidos reales** que el modelo vio en producción a través de `monitor_v3`. A diferencia del holdout C (backtest sobre datos históricos), aquí el modelo opera sobre partidos nuevos, con datos capturados en vivo. **Muestra muy pequeña**: solo 93 apuestas liquidadas, por lo que el IC95 es ancho y no permite conclusiones firmes. Pero sirve para contrastar la euforia del backtest con la realidad.

- **Apuestas totales registradas:** 94 (72 ganadas, 21 perdidas, 1 pendientes).
- **Acierto en vivo (sobre liquidadas):** 77.4 % (backtest @0.75: 81.7 %).
- **Confianza promedio en vivo:** 79.2 %.

### 6.1 Por liga — backtest vs vivo (con recomendación)

| Liga                                                                        | Entrenado (A) | Backtest @0.75 (WR) | Vivo: apuestas | Vivo: ganadas | Vivo: perdidas | Vivo: WR | Recomendación               |
| ---                                                                         | ---:          | ---:                | ---:           | ---:          | ---:           | ---:     | ---                         |
| B League Premier                                                            | 3,125         | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| B League One                                                                | 3,075         | —                   | 2              | 1             | 1              | 50.0 %   | 👀 vigilar (muestra pequeña) |
| France Pro A                                                                | 1,737         | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Brazil NBB                                                                  | 1,436         | 81.2 %              | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Turkish Basketball Super League                                             | 1,376         | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| United League                                                               | 1,133         | 82.3 %              | 2              | 2             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Stoiximan GBL                                                               | 1,017         | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| A1 Women                                                                    | 813           | —                   | 2              | 1             | 1              | 50.0 %   | 👀 vigilar (muestra pequeña) |
| Hungary NB 1.A                                                              | 773           | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Liga Nationala                                                              | 742           | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Russian Women's Basketball Premier League                                   | 726           | 74.1 %              | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Switzerland LNA                                                             | 689           | 73.7 %              | 2              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Basketligaen                                                                | 651           | 78.1 %              | 4              | 4             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Super League                                                                | 644           | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Estonian-Latvian Basketball League                                          | 594           | 81.8 %              | 2              | 2             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Tipos Extraliga, Women                                                      | 424           | 79.5 %              | 2              | 2             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Swedish Basketball Superettan                                               | 350           | —                   | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Cyprus Basketball Division A                                                | 348           | —                   | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Israeli Women Basketball Premier League                                     | 300           | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Premijer ženska liga                                                        | 299           | 68.8 %              | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Eurocup, Group B                                                            | 279           | —                   | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Hungary Divison A Women                                                     | 276           | 76.5 %              | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| 2nd League                                                                  | 269           | 81.8 %              | 2              | 1             | 1              | 50.0 %   | 👀 vigilar (muestra pequeña) |
| 1. Division                                                                 | 255           | —                   | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Swedish Basketball League Women                                             | 233           | 73.5 %              | 3              | 3             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| LNB, Women                                                                  | 184           | 91.7 %              | 5              | 4             | 1              | 80.0 %   | 👀 vigilar (muestra pequeña) |
| Swiss Basketball League Women                                               | 147           | 87.5 %              | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| LMB, Clausura                                                               | 120           | —                   | 2              | 2             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Liga 1 Masculin, Faza I                                                     | 101           | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| NBL, West                                                                   | 75            | 75.0 %              | 3              | 1             | 2              | 33.3 %   | 👀 vigilar (muestra pequeña) |
| 1. ZLS, Women                                                               | 53            | —                   | 4              | 3             | 1              | 75.0 %   | 👀 vigilar (muestra pequeña) |
| Poland 2nd Basketball League, Runda Zasadnicza                              | 0             | —                   | 5              | 2             | 3              | 40.0 %   | ⚠️ bloquear                 |
| LBP, Group B                                                                | 0             | —                   | 5              | 4             | 1              | 80.0 %   | 👀 vigilar (muestra pequeña) |
| FSHB Superliga                                                              | 0             | —                   | 3              | 2             | 1              | 66.7 %   | 👀 vigilar (muestra pequeña) |
| Road to Basketball Africa League, Group D                                   | 0             | —                   | 3              | 3             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| L1BF BT, Faza I                                                             | 0             | —                   | 2              | 2             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| FIBA Women Basketball League Africa - Qualifers, Group A                    | 0             | —                   | 2              | 1             | 1              | 50.0 %   | 👀 vigilar (muestra pequeña) |
| Catarinense, Women, Playoffs                                                | 0             | —                   | 2              | 2             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Women's Basketball Division A                                               | 0             | —                   | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |
| Superleague Women                                                           | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Kvindebasketligaen , Grundspil                                              | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Top Division Women, Regular Season                                          | 0             | 88.2 %              | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| 1st Division Women, Deildarkeppni                                           | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| AdmiralBet ABA League, Group B                                              | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| FIBA Women Basketball League Africa - Qualifers, Group B                    | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Liga Femenina, Women, Apertura, Group B                                     | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Mexico LNBP, Playoffs                                                       | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Women's Basketball League Asia, Group B                                     | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| 1 Liga Kobiet, Runda Zasadnicza                                             | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| EuroCup, Women, Group B                                                     | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Liga Nacional Femenina Chile, Fase zonal                                    | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 13 | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 14 | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| FIBA Women Basketball League Africa - Qualifers, Zone 5 - Semi-Finals 19    | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Nostra.lt-RKL Division B, Reguliarus sezonas                                | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| Nostra.lt-RKL Division A, Reguliarus sezonas                                | 0             | —                   | 1              | 1             | 0              | 100.0 %  | 👀 vigilar (muestra pequeña) |
| 2. ZLS, Women                                                               | 0             | —                   | 1              | 0             | 1              | 0.0 %    | 👀 vigilar (muestra pequeña) |

**Lectura honesta:** con 93 apuestas liquidadas el IC95 del acierto es de ±~10 pp, así que el 76.6 % vivo y el 81.7 % de backtest son **indistinguibles estadísticamente**. Lo que sí se ve es que varias ligas con WR de backtest > 85 % (NCAA Women, BNXT, Saudi, etc.) no han aparecido aún en vivo o han rendido por debajo, lo que confirma que **no hay que confiar en muestras pequeñas**. El modelo sigue en modo sombra y no alerta hasta tener ≥ 200 apuestas liquidadas.

> Nota: cuando la columna `Backtest @0.75` muestra `—` y `Entrenado = 0`, la liga no tuvo ningún partido con ese **nombre exacto** en el holdout/entrenamiento. Causas posibles: (a) arranque en frío (liga nueva), o (b) la liga se filtró en el entrenamiento porque `classify_stage` la catalogó como fase descartada (p. ej. `Runda Zasadnicza` = temporada regular en polaco, pero el clasificador la marca como `SEGUNDA_FASE`). En esos casos la DB sí tiene histórico (ej. `Poland 2nd Basketball League, Runda Zasadnicza` = 1072 partidos), pero el modelo nunca la vio; solo hay datos en vivo, que es la única evidencia disponible.

## 7. Bug conocido del clasificador de fases (`classify_stage`)

> Varios nombres de liga se catalogan como fase descartada (`SEGUNDA_FASE`, `AMISTOSOS`, `PRELIMINAR`, `PROMOCION`) aunque su `stage_detail` sea `Regular Season`. Consecuencia: el modelo **no las entrena ni las opera** (Entrenado = 0, Backtest = `—`), aunque la DB sí tenga histórico. Ejemplo claro: `Poland 2nd Basketball League, Runda Zasadnicza` (1072 partidos, cobertura 100 %) se marca como `SEGUNDA_FASE` porque el clasificador asocia "Runda" con segunda fase.

| Liga                                           | Partidos (DB) | stage            | stage_detail   | Clasificado como |
| ---                                            | ---:          | ---              | ---            | ---              |
| Poland 2nd Basketball League, Runda Zasadnicza | 852           | Runda Zasadnicza | Regular Season | SEGUNDA_FASE     |
| Liga Argentina, Group Stage                    | 518           | Group Stage      | Group Stage    | SEGUNDA_FASE     |
| NBA G League Tip-Off Tournament                | 377           | Regular season   | Regular Season | AMISTOSOS        |
| 1 Liga Kobiet, Runda Zasadnicza                | 270           | Runda Zasadnicza | Regular Season | SEGUNDA_FASE     |
| Club Friendlies                                | 198           | Regular season   | Regular Season | AMISTOSOS        |
| National Basketball Championship, Women        | 126           | Regular season   | Regular Season | SEGUNDA_FASE     |
| The Basketball Tournament                      | 123           | Regular season   | Regular Season | AMISTOSOS        |
| Women's Super League Basketball , Championship | 108           | Regular season   | Regular Season | SEGUNDA_FASE     |

**Ligas claramente mal clasificadas (stage_detail = Regular Season pero bloqueadas):** `Poland 2nd Basketball League, Runda Zasadnicza`, `1 Liga Kobiet, Runda Zasadnicza`, `National Basketball Championship, Women`, `Women's Super League Basketball, Championship`, `NBA G League Tip-Off Tournament` (stage_detail Regular Season). `Liga Argentina, Group Stage` y `Women's Super League Basketball, Championship` tienen stage_detail distinto y requieren revisión manual.

**Corrección aplicada:** se eliminó `'runda zasadnicza'` de `SEGUNDA_FASE_MARKERS` en `models/common/competition_stage.py` (era un error: "Runda Zasadnicza" = temporada regular, no segunda fase). Ligas desbloqueadas: `Poland 2nd Basketball League, Runda Zasadnicza` (852) y `1 Liga Kobiet, Runda Zasadnicza` (270). Los casos `championship` y `group stage` (National Basketball Championship, Women; Women's Super League Basketball, Championship; Liga Argentina, Group Stage; etc.) quedan bloqueados a la espera de revisión manual, porque el marcador es ambiguo (puede ser torneo real o temporada regular).

## 8. Clon `m27_v5_3_1` — prueba contra los partidos en vivo de `m27_v5_3`

> Se clonó el modelo `m27_v5_3` a `m27_v5_3_1` aplicando la corrección del clasificador (Runda Zasadnicza como temporada regular). **Ligas añadidas al entrenamiento:** `Poland 2nd Basketball League, Runda Zasadnicza` (852 partidos) y `1 Liga Kobiet, Runda Zasadnicza` (270 partidos). Los 94 partidos en vivo de `m27_v5_3` **no se incluyeron en el entrenamiento** (caen en el bloque C por ser los más recientes) y se usaron como holdout de prueba.

- **Entrenamiento:** 85 de los 94 partidos en vivo tuvieron ancla (t=30) evaluable.
- **m27_v5_3 (original):** 85 apuestas, 67 ganadas, 18 perdidas → WR 78.8 %.
- **m27_v5_3_1 (clon):** 59 apuestas, 47 ganadas, 12 perdidas → WR 79.7 %.

> **Caso específico — `Poland 2nd Basketball League, Runda Zasadnicza`:** 4 partidos en vivo. `m27_v5_3` apostó los 4 (2 ganadas, 2 perdidas → 50 %). `m27_v5_3_1` apostó solo 2 (los 2 que ganó) y pasó los 2 que perdió → 2/2 = **100 %**. Muestra diminuta (4 partidos), pero la señal es positiva: el modelo entrenado con la liga evitó las 2 apuestas perdedoras.

| Liga                                                                        | m27_v5_3 (orig) | m27_v5_3_1 (clon) | Vivo: apuestas | Vivo: ganadas | Vivo: perdidas |
| ---                                                                         | ---             | ---               | ---:           | ---:          | ---:           |
| Liga 1 Masculin, Faza I                                                     | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| 1 Liga Kobiet, Runda Zasadnicza                                             | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| 1. Division                                                                 | 0 % (0/1)       | — (0/0)           | 1              | 0             | 0              |
| 1. ZLS, Women                                                               | 75 % (3/4)      | 75 % (3/4)        | 4              | 3             | 1              |
| 1st Division Women, Deildarkeppni                                           | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| 2. ZLS, Women                                                               | 0 % (0/1)       | 0 % (0/1)         | 1              | 0             | 1              |
| 2nd League                                                                  | 50 % (1/2)      | 0 % (0/1)         | 2              | 0             | 1              |
| A1 Women                                                                    | 50 % (1/2)      | 0 % (0/1)         | 2              | 0             | 1              |
| AdmiralBet ABA League, Group B                                              | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| B League One                                                                | 50 % (1/2)      | 0 % (0/1)         | 2              | 0             | 1              |
| B League Premier                                                            | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Basketligaen                                                                | 100 % (4/4)     | 100 % (1/1)       | 4              | 1             | 0              |
| Brazil NBB                                                                  | 0 % (0/1)       | 0 % (0/1)         | 1              | 0             | 1              |
| Catarinense, Women, Playoffs                                                | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Cyprus Basketball Division A                                                | 0 % (0/1)       | — (0/0)           | 1              | 0             | 0              |
| Estonian-Latvian Basketball League                                          | 100 % (2/2)     | 100 % (2/2)       | 2              | 2             | 0              |
| EuroCup, Women, Group B                                                     | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Eurocup, Group B                                                            | 0 % (0/1)       | 0 % (0/1)         | 1              | 0             | 1              |
| FIBA Women Basketball League Africa - Qualifers, Group A                    | 50 % (1/2)      | 0 % (0/1)         | 2              | 0             | 1              |
| FIBA Women Basketball League Africa - Qualifers, Group B                    | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 13 | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| FIBA Women Basketball League Africa - Qualifers, Zone 5 - Quarter-Finals 14 | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| FIBA Women Basketball League Africa - Qualifers, Zone 5 - Semi-Finals 19    | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| FSHB Superliga                                                              | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| France Pro A                                                                | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Hungary Divison A Women                                                     | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Hungary NB 1.A                                                              | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Israeli Women Basketball Premier League                                     | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Kvindebasketligaen , Grundspil                                              | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| L1BF BT, Faza I                                                             | 100 % (2/2)     | 100 % (2/2)       | 2              | 2             | 0              |
| LBP, Group B                                                                | 80 % (4/5)      | 80 % (4/5)        | 5              | 4             | 1              |
| LMB, Clausura                                                               | 100 % (2/2)     | 100 % (1/1)       | 2              | 1             | 0              |
| LNB, Women                                                                  | 80 % (4/5)      | 100 % (4/4)       | 5              | 4             | 0              |
| Liga Nacional Femenina Chile, Fase zonal                                    | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Liga Nationala                                                              | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Mexico LNBP, Playoffs                                                       | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| NBL, West                                                                   | 33 % (1/3)      | 33 % (1/3)        | 3              | 1             | 2              |
| Nostra.lt-RKL Division A, Reguliarus sezonas                                | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Nostra.lt-RKL Division B, Reguliarus sezonas                                | 100 % (1/1)     | — (0/0)           | 1              | 0             | 0              |
| Poland 2nd Basketball League, Runda Zasadnicza                              | 50 % (2/4)      | 100 % (2/2)       | 4              | 2             | 0              |
| Premijer ženska liga                                                        | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Road to Basketball Africa League, Group D                                   | 100 % (3/3)     | 100 % (3/3)       | 3              | 3             | 0              |
| Russian Women's Basketball Premier League                                   | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Stoiximan GBL                                                               | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Super League                                                                | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Swedish Basketball League Women                                             | 100 % (3/3)     | 100 % (2/2)       | 3              | 2             | 0              |
| Swedish Basketball Superettan                                               | 0 % (0/1)       | — (0/0)           | 1              | 0             | 0              |
| Switzerland LNA                                                             | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Tipos Extraliga, Women                                                      | 100 % (2/2)     | 100 % (2/2)       | 2              | 2             | 0              |
| Top Division Women, Regular Season                                          | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| Turkish Basketball Super League                                             | 100 % (1/1)     | 100 % (1/1)       | 1              | 1             | 0              |
| United League                                                               | 100 % (2/2)     | 100 % (2/2)       | 2              | 2             | 0              |
| Women's Basketball Division A                                               | 0 % (0/1)       | 0 % (0/1)         | 1              | 0             | 1              |

**Lectura:** el clon es **más conservador** (59 vs 85 apuestas) con un acierto ligeramente superior (79.7 % vs 78.8 %), pero la diferencia es pequeña y la muestra es diminuta (94 partidos). La inclusión de las 2 ligas polacas no cambia materialmente el resultado en vivo; el modelo sigue en modo sombra. **No se justifica un cambio de modelo con esta evidencia.**

## 9. Conclusiones

1. **El modelo mantiene el acierto por encima del equilibrio en la gran mayoría de ligas** a umbral 0.75: 251 de 251 ligas no son candidatas a bloqueo (IC95 superior ≥ 71.4 %).
2. **0 ligas** tienen el límite superior de su IC95 por debajo del equilibrio a 0.75 → candidatas a bloqueo por medición. Conviene revisarlas antes de operar, pero con muestras pequeñas el IC es ancho; no bloquear por un único corte.
3. **108 ligas (43.0 %)** son arranque en frío: aparecen solo en el holdout y no aportaron filas al entrenamiento. Sus priors son as-of, pero la calibración por liga es incierta; vigilar.
4. **El volumen se concentra**: las ligas con más apuestas a 0.75 son las que dominan el resultado global; las de cola aportan poco volumen pero permiten diversificación.
5. **Subir el umbral a 0.80** mejora WR y ROI en todas las ligas, pero reduce el volumen a menos de la mitad; la decisión de umbral debe basarse en EV real (cuota Q4), no solo en P.
6. **Bajar el umbral a 0.65–0.68** aumenta volumen pero acerca el WR al equilibrio (71.4 %); solo justificable si la cuota real del mercado Q4 es superior a 1.40.
7. La lista de bloqueo por medición queda **vacía en producción** hoy (0 ligas) porque el filtro de formatos de torneo ya descartó los formatos tóxicos (amistosos, 2ª fase, promoción, preliminar, colocación). La lista se repoblará sola cuando acumule evidencia.

## 10. Notas metodológicas

- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis.py` (re-entrena el ensamble, calibra con Platt simétrico, evalúa en C).
- `matches_holdout` = anclas (t=30) en el bloque C por liga; `matches_train` = anclas en el bloque A.
- `green()` aplica además el guard de margen (margin ≥ 0 para apuestas al local), igual que el pipeline.
- IC95 = Wilson. Bloqueo candidato si el límite superior del IC95 del WR < 71.4 % (equilibrio a cuota 1.40).
- Ligas con < 30 anclas en C se omiten por falta de muestra.
- Resultados en vivo: `tmp/m27_v5_3_audit/live_league_stats.py` sobre `bet_monitor_log_v3` (apuestas liquidadas del monitoreo).
