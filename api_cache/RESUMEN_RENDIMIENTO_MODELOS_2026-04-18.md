# Resumen de rendimiento real de modelos en `api_cache`

Fecha de corte: **2026-04-18**  
Ventana analizada: archivos `api_cache/*.json` entre **2026-04-12** y **2026-04-18**.  
Metodologia: conteo de picks resueltos en `outcome in {hit, miss}`; `push` se separa.  
Importante: esto mide **rendimiento real observado en cache**, no benchmark offline ni metricas de training.

## 1. Lectura rapida

Si la pregunta es **"cual ha sido el modelo mas efectivo en vivo, con evidencia real en estos JSON?"**, la respuesta hoy es:

1. **v12**: mejor resultado real de la muestra y ya corre con engine live propio.
2. **v13**: sigue siendo de los mejores modelos live "puros", con mejor logica contextual que los legacy, pero por debajo de v12 en esta ventana.
3. **v2 / v4**: no son los mas sofisticados, pero traen el mejor balance entre volumen y estabilidad entre los modelos legacy.
4. **v15**: prometedor en idea, pero la muestra real aun es chica y el confidence no esta bien alineado con el resultado observado.
5. **v16**: hoy no hay evidencia real suficiente para defenderlo; en estos JSON se ve flojo y muy inestable.
6. **v9**: el peor de los modelos con volumen grande en esta muestra.

## 2. Ranking por evidencia real

| Modelo | Tipo | Bets totales | Resueltas | Hit rate real | Push | IC 95% hit rate | Lectura operativa |
|---|---|---:|---:|---:|---:|---:|---|
| v12 | live engine propio | 145 | 141 | **85.11%** | 4 | 78.30% - 90.05% | Mejor resultado real de la muestra |
| v2 | legacy live via `infer_match` | 955 | 904 | **76.66%** | 51 | 73.79% - 79.30% | Muy buen piso, mucho volumen |
| v4 | legacy live via `infer_match` | 913 | 866 | **76.10%** | 47 | 73.15% - 78.82% | Muy parecido a v2, algo mas contextual |
| v13 | live engine propio | 168 | 161 | **74.53%** | 7 | 67.29% - 80.64% | Gana al break-even, pero debajo de v12 |
| v6 | legacy live via `infer_match` | 907 | 860 | 73.37% | 47 | 70.32% - 76.22% | Rentable por poco margen, menos solido |
| v15 | live engine propio | 44 | 44 | 70.45% | 0 | 55.78% - 81.84% | Muestra demasiado chica; aun no concluyente |
| v9 | legacy live via `infer_match` | 1076 | 1017 | 56.83% | 59 | 53.77% - 59.85% | Mal resultado real, no operable |
| v16 | live engine propio | 22 | 21 | 47.62% | 1 | 28.34% - 67.63% | Muy flojo y sin base estadistica |

## 3. Que hace cada modelo y por que podria funcionar

### v2

- Base clasica con `league/team buckets`.
- Usa `LogReg + RF + GB`.
- Teoricamente funciona cuando el problema esta dominado por contexto estructural de liga/equipo y no tanto por dinamica fina en vivo.
- En la practica: sorprende por estabilidad. No es el mas "inteligente", pero no se rompe facil.

### v4

- Toma v2 y agrega **pressure/comeback features**.
- Mete contexto de marcador, urgencia y ritmo via play-by-play.
- Teoricamente mejora cuando la dinamica del partido importa mas que el promedio historico.
- En la practica: casi empata a v2, lo que sugiere que el extra de contexto ayuda sin meter demasiado ruido.

### v6

- Agrega **Monte Carlo** y features de momentum.
- Modela la continuacion del juego "posesion por posesion" con varianza.
- Teoricamente deberia capturar mejor escenarios de remontada o inercia.
- En la practica: mejora teorica interesante, pero el resultado real queda debajo de v2/v4. Probable causa: mas complejidad y mas sensibilidad al ruido del feed.

### v9

- Version optimizada para velocidad con ensemble simple (`LogReg + GB`) y features de **racha/momentum**.
- Teoricamente debia ser mas rapido y mas afinado.
- En la practica: en estos JSON sale claramente mal. Posible razon: sobrepondera señales de momentum que en live se revierten mucho, o quedo peor calibrado que las versiones previas.

### v12

- Es el modelo mas conservador de los live engines fuertes.
- Mezcla **clasificacion + regresion + risk management + filtro de ligas**.
- Reusa lo mejor de v4, v6 y v9, y encima mete gates de volatilidad, calidad de datos y ligas "apostables".
- Teoricamente es fuerte porque no solo predice, tambien **decide cuando no apostar**.
- En la practica: eso se ve en los datos. Tiene el mejor hit rate de toda la muestra con volumen razonable.

### v13

- Engine live dedicado.
- Usa **dynamic cutoffs**, **seleccion por pace bucket**, separacion por genero y regresion auxiliar para validar la lectura del partido.
- Esta mas orientado a contexto live real que los legacy.
- Teoricamente puede ser mas efectivo porque elige modelo segun el ritmo del partido y no aplica una sola receta a todo.
- En la practica: funciona bien y supera break-even, pero hoy queda por debajo de v12. Aun asi, entre los modelos claramente "live-aware", sigue siendo de los mas defendibles.

### v15

- Modelo por liga orientado a **ROI con odds 1.40**.
- Usa ensemble por liga, calibracion explicita, threshold aprendido por liga y varios gates secuenciales.
- Teoricamente es mas serio a nivel producto: no busca solo accuracy, busca rentabilidad y control de cobertura.
- En la practica real de estos JSON todavia no alcanza. Queda en 70.45%, con muy pocas apuestas, y ademas su confianza media en misses (`0.914`) sale incluso mayor que en hits (`0.898`), mala senal de calibracion en esta muestra.

### v16

- Es la evolucion mas ambiciosa: selector inteligente de ligas, multi-snapshot, regresion como feature, forecasting de serie con **TimesFM**, quantiles, drift detection y versionado semanal.
- Teoricamente deberia ser el mas avanzado.
- En la practica de `api_cache`: no. Tiene poca cobertura, dias vacios y un 47.62% real. Hoy esta mucho mas cerca de experimento prometedor que de modelo ganador.

## 4. Observaciones reales importantes

### 4.1 Lo "en vivo" no es igual en todos

- **v12, v13, v15 y v16** usan engine propio live, el mismo flujo que replica el bot.
- **v2, v4, v6 y v9** corren via `training.infer_match` con `force_version`; siguen operando sobre snapshot real, pero su diseno es mas legacy.
- O sea: si tu criterio es "quiero algo realmente live", los candidatos serios son **v12 y v13** hoy; **v15/v16** aun no lo respaldan los resultados reales.

### 4.2 Cobertura vs precision

- **v12**: alto acierto con cobertura media.
- **v13**: cobertura algo mayor que v12, pero menor precision.
- **v2/v4**: muchisimo volumen y buen piso.
- **v15/v16**: muy poca cobertura; hoy no alcanzan muestra para coronarlos.

### 4.3 Estabilidad diaria

- **v12** cierra fuerte al final de la ventana: 94.59% el 2026-04-17 y 95.45% el 2026-04-18.
- **v13** es mas volatil: 87% y 88% en dias buenos, pero tambien 58%-64% en dias flojos.
- **v15** y **v16** tienen dias vacios o con 1-2 apuestas; no sirve sacar conclusiones fuertes de eso.
- **v9** es consistentemente debil: casi todos sus dias quedan en la zona 50%-60%.

### 4.4 Por target

| Modelo | Q3 hit rate | Q4 hit rate | Nota |
|---|---:|---:|---|
| v12 | 83.33% | 86.67% | Fuerte en ambos, ligeramente mejor en Q4 |
| v13 | 77.97% | 72.55% | Bueno en Q3, mas fragil en Q4 |
| v15 | 64.29% | 81.25% | Q4 se ve mejor, pero con muy poca muestra |
| v16 | 50.00% | 42.86% | Flojo en ambos |
| v2 | 74.61% | 78.65% | Muy parejo y sano |
| v4 | 74.17% | 77.93% | Muy parecido a v2 |
| v6 | 71.67% | 75.00% | Aceptable pero inferior |
| v9 | 57.93% | 55.73% | Malo en ambos |

## 5. Conclusiones practicas

### Si quieres el modelo mas efectivo hoy, basado en cache real

- **Ganador actual: `v12`**.
- No solo gana en hit rate; tambien tiene un volumen ya util y una logica live real, no solo teorica.

### Si quieres el mejor candidato "live-aware" despues de v12

- **`v13`**.
- Sigue teniendo sentido operativo porque su arquitectura esta claramente pensada para live real: pace buckets, cutoffs dinamicos y lectura contextual.
- Pero con estos datos no puedo decir honestamente que sea mejor que v12. Hoy no lo es.

### Si quieres algo robusto por volumen mientras sigues evaluando los live engines

- **`v2` o `v4`**.
- No son los mas modernos, pero dan un baseline real duro de tumbar.

### Modelos que hoy no compraria como principales

- **`v15`**: todavia necesita mas muestra real y revisar calibracion.
- **`v16`**: demasiado experimental para elegirlo hoy.
- **`v9`**: resultado real claramente insuficiente.

## 6. Recomendacion operativa al 2026-04-18

Si tu objetivo es elegir **un modelo principal en vivo**, mi recomendacion seria:

1. **Usar `v12` como principal**.
2. **Seguir monitoreando `v13` como segundo candidato live**.
3. Tomar **`v2` o `v4` como benchmark estable** para no enganarte con versiones mas nuevas pero menos rentables.
4. No promover **`v15` ni `v16`** a produccion principal hasta juntar una muestra real bastante mayor.

## 7. Lo que falta para estar mas seguro

Esta conclusion sale de una ventana corta, del **12 al 18 de abril de 2026**. Para una decision mas dura de producto yo haria despues:

- consolidado rolling de 30 dias;
- ROI real por modelo y por target, no solo hit rate;
- comparacion por liga;
- calibration curve real (`confidence` vs acierto) para v12, v13, v15 y v16;
- portfolio mix: por ejemplo `v12` para Q4 y `v13` solo en ciertos contextos si demuestra edge.

## Veredicto final

**A la fecha 2026-04-18, el modelo mas efectivo en vivo dentro de `api_cache` es `v12`.**  
**`v13` sigue siendo un modelo live serio y contextual, pero no lidera el rendimiento real de esta muestra.**  
**`v16` es el mas avanzado en teoria, pero no el mejor en produccion real hoy.**
