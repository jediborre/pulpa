# Diagnóstico del "día malo" de m27_v5_3 (2026-10-20)

> Qué se revisó, qué seMidió y qué se descartó como causa. Todo reproducible con
> los scripts de `tmp/m27_v5_3_audit/`.

## 1. Los números de hoy

| modelo | apuestas hoy | ganadas | perdidas | WR | IC95 |
|---|---|---|---|---|---|
| `m27_v5_3` | **5** | 2 | 3 | 40.0 % | [11.8, 76.9] |
| `m27_v5_2` | 3 | 2 | 1 | 66.7 % | [20.8, 93.9] |
| `m27_v3` | 24 | 10 | 14 | 41.7 % | [24.5, 61.2] |
| `v6_2` | 36 | 22 | 14 | 61.1 % | [44.9, 75.2] |

Con **n = 5** el intervalo de confianza va del 12 % al 77 %: hoy **no dice nada**
sobre la calidad del modelo. Si el modelo acierta el 80 % de las veces, sacar 2/5
tiene una probabilidad del **6.5 %** (P(X ≤ 2 | n=5, p=0.8)).

## 2. La liquidación es correcta

Las 5 apuestas se verificaron una a una contra `quarter_scores_v2`:

| partido | Q3 | Q4 | resultado |
|---|---|---|---|
| REG BBC vs Dynamo | 34-8 | 16-20 | loss (margen −4) |
| Denso Iris vs Indian Railways | 29-4 | 29-12 | win (+17) |
| UGMK Ekaterinburg vs BC Samara | 19-12 | 22-6 | win (+16) |
| Türk Telekom vs Šiauliai | 22-24 | 17-27 | loss (−10) |
| Flamengo vs Botafogo | 20-21 | 12-14 | loss (−2) |

Las 5 liquidaciones cuadran. **No hay bug de liquidación.**

## 3. El camino vivo es fiel al offline (test decisivo)

Para las 37 apuestas con equivalente en el dataset, se recalcularon las
predicciones offline **sobre esos mismos partidos al mismo minuto**:

- WR vivo: **70.3 %**
- WR offline: **70.3 %** (idéntico)
- Diferencia media de confianza: **+0.019**

El monitor no está "calculando otra cosa" que el backtest.

## 4. Hipótesis descartadas una a una

| hipótesis | resultado |
|---|---|
| **Efecto basura** (el Q4 de un partido decidido es basura) | ❌ el modelo es **mejor** ahí: WR **83.4 %** a 0.75 en partidos con 25+ de ventaja, bien calibrado (+0.6 pp) |
| **Apuestas de remontada** (elegir al que va perdiendo) | ❌ ese bucket da 76.8 % a 0.75, bien calibrado (−0.3 pp). Hoy salieron 2 de 5, mala suerte |
| **Población de ligas** | ❌ el backtest en las 33 ligas de las apuestas vivas da **78.2 %** (no 69 %); y el tamaño de la liga no cambia nada (80.8 % vs 80.6 %) |
| **Calibración rota** | ❌ en el backtest el bucket "el elegido va ganando" da 81.2 % con confianza media 0.812 → **+0.0 pp** de desviación |
| **Liquidación** | ❌ las 5 cuadran con el Q4 real |

## 5. Lo que sí es cierto: el hueco vivo–backtest es general

| modelo | apuestas vivas | WR vivo |
|---|---|---|
| `m27_v5_3` | 45 | 68.9 % |
| `m27_v5_2` | 14 | 78.6 % |
| `m27_v3` | 312 | 57.4 % |
| `v6_2` | 448 | 54.0 % |

`m27_v5_3` es **el mejor de la familia m27**, pero su 68.9 % con 45 apuestas está
9 pp por debajo de lo que dice el backtest (78 % de confianza media). Con n=45 eso
son ~1.3 desviaciones estándar: **no es significativo todavía**, y `v6_2`/`m27_v3`
(ya retirados del monitoreo) están mucho peor con muestras mucho mayores.

**El único tramo con desviación clara** es la banda de confianza alta en vivo:
0.75-0.80 → 76.5 % (+0.4 pp, bien calibrada); 0.80-0.85 → 63.6 % (−18.5 pp, n=11);
0.85+ → 75.0 % (−14.3 pp, n=4). Con n=11 y n=4 es ruido, pero es donde mirar
cuando la muestra crezca.

## 6. Un hallazgo puntual: las copas por subtorneo

La apuesta de hoy en **Eurocup, Group B** cayó en una liga cuyo WR histórico a 0.75
en el backtest es del **50 %** (n=12). Se mantienen eliminatorias y copas porque
globalmente miden 82.8 %, pero **por subtorneo** hay que mirarlas: es exactamente
el caso que P1 (política por población) debe capturar.

## 7. Corrección de percepción (lo que hace ver "pesimo" al modelo)

En la tarjeta de RESULTADO, las filas `NO BET` se dibujaban con 🔴 en el texto y
con la palabra "NO BET" en **rojo** en la imagen. Un NO BET es una **no apuesta**,
no una derrota: la tarjeta se leía como una racha de derrotas.

- texto: 🔴 → ⚪️
- imagen: etiqueta "NO BET", porcentaje y "HUBIERA GANADO/PERDIDO" en gris neutro
  (las apuestas reales conservan el semáforo y el verde/rojo de resultado)

## 8. Qué hacer

1. **No cambiar nada por hoy.** n=5 no es evidencia.
2. Dejar correr hasta el punto de decisión pre-registrado (**~200 apuestas
   resueltas**); hoy hay 45.
3. Vigilar específicamente la banda **0.80-0.85**: si a las ~100 apuestas de esa
   banda el WR sigue por debajo del 75 %, el calibrador Platt está sobreconfiado
   en la cola y habría que capar la confianza (o subir el umbral solo ahí).
4. Medir el WR por **subtorneo** (Eurocup/Group B y parecidos) y aplicar el filtro
   de P1.
