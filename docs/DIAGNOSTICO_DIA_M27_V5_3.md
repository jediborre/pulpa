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

---

# Segunda vuelta: el mapa de fugas del WR (2026-10-20, 21:00)

El usuario tenía razón en el fondo: **el WR es la métrica**, y la calibración es
otra cosa. Aquí está la cuenta completa.

## 9. El número honesto

| | n | WR | esperado (backtest) | gap | z |
|---|---|---|---|---|---|
| bets de `m27_v5_3` a 0.75+ (vivos, 3 días) | 46 | **68.9 %** | 80.7 % | **−12 pp** | ≈ **2.0** (p≈0.03) |

Con n=46 son ~2 desviaciones: **no es concluyente, pero tampoco es "está bien"**.
Es una bandera amarilla real.

## 10. Mapa de fugas: dónde se pierde el acierto

| banda de confianza | n | WR esperado | WR real | gap |
|---|---|---|---|---|
| 0.75-0.80 | 30 | 77.3 % | 70.0 % | −7.3 pp |
| 0.80-0.85 | 12 | 83.3 % | **58.3 %** | **−25.0 pp** |
| 0.85+ | 4 | 90.8 % | 75.0 % | −15.8 pp |

El hueco se concentra en la banda alta. Con n=12, z≈−2.0 → suggestive.

## 11. Se descartaron todas las causas "debug"

| causa | veredicto |
|---|---|
| liquidación | ❌ las 5 de hoy cuadran con el Q4 real de `quarter_scores_v2` |
| features (marcador) | ❌ vivo vs BD: diferencia media **−0.01**, solo el 8 % difiere ≥3 puntos |
| predicción | ❌ el backtest sobre los **mismos 37 partidos** da **70.3 %** y el vivo **70.3 %** |
| features de mercado | ❌ la ruta viva usa las cuotas **pre-partido**, igual que el entrenamiento (`_parse_odds` con `isLive=False`) |
| minuto | ❌ acierto y calibración planos de 27 a 30 |
| efecto basura | ❌ en partidos decididos el modelo es **mejor** (83.4 %) |
| mezcla de ligas | ❌ el backtest en las 33 ligas vivas da 78.1 % (no 69 %) |
| cobertura de liga/equipo | ❌ en el backtest las ligas desconocidas son las **mejores** (82.3 %); exigir historial **baja** el WR (80.7 % → 78.4 %) |

Y la comprobación global: el acierto vivo de **todas** las predicciones es
**59.5 %** con una confianza media de **60.6 %** → el modelo está calibrado
globalmente. El mal día está en la submuestra 0.75+ (n=6 hoy).

## 12. El problema de fondo: 6 apuestas/día

La superficie de beneficio **viva** (se calcula desde la serie por minuto, sin
esperar a que se acumulen apuestas):

| umbral | bets/día | WR backtest | WR vivo (hoy) | u@1.40 backtest | u@1.83 backtest |
|---|---|---|---|---|---|
| 0.65 | 20 | 71.6 % | 65.0 % | +20 u | +967 u |
| 0.70 | 8 | 76.0 % | 50.0 % | +349 u | +2.158 u |
| 0.72 | 6 | 78.0 % | 33.3 % | +386 u | +1.795 u |
| 0.75 | **6** | 80.8 % | 33.3 % | +356 u | +1.327 u |
| 0.80 | 3 | 86.2 % | 33.3 % | +268 u | +760 u |

A 0.75 el modelo dispara **6 veces al día**: llegar a 200 apuestas resueltas (el
punto de decisión pre-registrado) lleva **un mes**. A 0.65 serían 20/día → diez
días. El muestreo vivo actual no puede decidir nada.

## 13. La regla que decide el umbral: el precio

Punto de equilibrio por unidad: `WR_equilibrio = 1 − 1/cuota`

| cuota | WR de equilibrio | WR a 0.75 en vivo (46 bets) |
|---|---|---|
| 1.40 | **71.4 %** | 68.9 % → por debajo del equilibrio |
| 1.83 | **54.6 %** | 68.9 % → 14 pp de margen |

Con la cuota que de verdad se consiga, el umbral cambia:
- **a 1.40**: 0.75 es el único umbral con WR esperado suficiente; bajar el umbral
  destruye el beneficio. Y el WR vivo actual (68.9 %) está **por debajo** del
  equilibrio → no se debe bajar sin más.
- **a 1.83**: 0.70-0.72 es óptimo (1.795-2.158 u en el backtest) y da el doble de
  muestra viva. Con un WR vivo de 68.9 % el margen sigue siendo enorme.

**Y aquí está el punto ciego**: la cuota del mercado de Q4 **no existe en
SofaScore** (ver `docs/ANALISIS_POR_MINUTO.md`), así que no se puede medir el EV
ni calibrar el umbral por precio. La "cuota" de la tarjeta es el ML in-play del
partido completo: sirve de ilustración, no como precio.

## 14. Recomendación

1. **No tocar el modelo.** El camino vivo es fiel; lo que falla es la muestra
   (6 bets/día) y el precio desconocido.
2. **Bajar el umbral de ANÁLISIS a 0.65-0.70** (no el de apuesta): la serie por
   minuto ya permite calcular la curva de WR viva en cualquier umbral. Así el
   veredicto llega en 10 días en vez de un mes.
3. **Dejar la apuesta en 0.75** hasta que la curva viva lo contradiga.
4. **Pre-registrar el test de la banda alta**: si a las ~100 bets de 0.80-0.85 el
   WR sigue < 75 %, capar la confianza (o exigir margen mayor en esa banda).
5. **Pedir la cuota real** que se consigue en la apuesta de Q4: con eso se
   cierra el EV y el umbral se decide con números, no con supuestos.

