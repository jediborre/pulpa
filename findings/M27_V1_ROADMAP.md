# M27_V1 Roadmap

Roadmap de investigacion y desarrollo para `m27_v1`, su extension a `m30`, y mejoras futuras del pipeline Q4.

## Objetivo general

- Mejorar la calidad de senal de `m27_v1` sin perder interpretabilidad.
- Extender aprendizajes de `m27_v1` hacia `m30` y posibles modelos expertos.
- Entender mejor volatilidad, cobertura insuficiente y comportamiento en live.
- Identificar features nuevas con valor real y evitar ruido o leakage.

## Prioridad alta

### 1. Features de marcador por cuarto

- Revisar otros marcadores individuales de `Q1` y `Q2` para evaluar si agregan senal.
- Revisar combinaciones de marcadores parciales y cerrados:
  - `Q1 winner + Q2 winner`
  - `halftime leader + q3 partial leader`
  - bins de diferencia en `Q1`, `Q2`, halftime y minuto 27
  - remontada parcial por cuarto
- Evaluar si conviene modelar patrones especificos de combinacion y no solo difs agregados.

Preguntas:
- Que combinaciones son robustas y no solo ruido de pocas muestras.
- Que informacion adicional aporta cada cuarto sobre el ganador de `Q4`.

Criterio de exito:
- Subida consistente en `roc_auc`, `log_loss` o ROI sin inflar mucho el feature set.

### 2. Llevar mejoras de m27_v1 a Q4 m30

- Revisar si el set de features de `m27_v1` se puede adaptar a `m30`.
- Probar una variante `m30_v1` con:
  - bloques de presion/remontada
  - ventanas recientes
  - graph slopes cortos
  - menos ruido categorico
- Comparar `m30_v1` contra `m30` actual y contra `m27_v1`.

Preguntas:
- Si `m30` conserva ventaja offline al usar features mas limpias.
- Si `m30` deja de sobreapostar o mejora precision real de edge.

### 3. Investigar ligas volatiles y ligas sin apuesta

- Revisar ligas volatiles puntualmente.
- Investigar por que algunas ligas no apuestan:
  - falta edge
  - falta confianza
  - falta graph points
  - falta play-by-play
  - gate demasiado duro
- Revisar si se puede ser mas especifico con la definicion de volatilidad por liga o por tipo de partido.

Preguntas:
- Que le falta al modelo para tomar decision en ligas donde casi no entra.
- Si el problema es del modelo, del gate o de la data live.

Criterio de exito:
- Clasificar ligas en `bloquear`, `vigilar`, `preferir`, `sin muestra` y reducir falsos rechazos.

### 4. Modelo adaptado a baja cobertura live

- Revisar si se puede hacer un modelo adaptado para juegos con menos puntos en la grafica o en el play-by-play.
- Separar al menos dos rutas:
  - `coverage_ok`
  - `coverage_low`
- Ver si una version de respaldo puede reconstruir mejor minuto 27 aunque el timing de live venga adelantado o incompleto.

Preguntas:
- Si un modelo especializado para coverage baja puede rescatar partidos hoy excluidos.
- Si el snapshot sintetico reproduce suficientemente bien el `min 27` real.

## Prioridad media

### 5. Fouls y senal de contexto fisico

- Revisar si se pueden sacar los fouls desde la data disponible.
- Evaluar si fouls sirven como feature para `Q4`:
  - fouls acumulados
  - fouls recientes
  - fouls por equipo
  - diferencial de fouls

Preguntas:
- Si los fouls anticipan ritmo, bonus, remontadas o cierre apretado.
- Si la calidad del scrape alcanza para usarlos sin meter ruido.

### 6. Ensemble y promedio de algoritmos

- Jugar con los pesos del promedio entre modelos en `m27_v1`.
- Probar alternativas a `avg_prob_xgb_hist_gb`:
  - pesos fijos distintos
  - pesos por split temporal
  - pesos por liga o coverage
- Revisar si un promedio simple sigue siendo la mejor opcion.

Preguntas:
- Si uno de los modelos domina en ciertos segmentos y conviene ponderarlo distinto.

Criterio de exito:
- Mejorar `log_loss`, `brier` y ROI sin inestabilidad entre cortes.

### 7. Ensemble y promedio de algoritmos por segmento o liga

- Jugar con los pesos del promedio entre modelos en `m27_v1`.
- Probar alternativas a `avg_prob_xgb_hist_gb`:
  - pesos fijos distintos
  - pesos por split temporal
  - pesos por liga o coverage
  - pesos por volatilidad
- Revisar si un promedio simple sigue siendo la mejor opcion.
- Revisar si conviene dar mas peso a `xgb` en ciertas ligas y a `hist_gb` en otras.

Preguntas:
- Si uno de los modelos domina en ciertos segmentos y conviene ponderarlo distinto.
- Si existen ligas donde `xgb` o `hist_gb` son consistentemente superiores en cortes temporales.

Criterio de exito:
- Mejorar `log_loss`, `brier` y ROI sin inestabilidad entre cortes.

### 8. Team of experts m27_v1 + m30_v1

- Revisar si se puede hacer un `team of experts` entre `m27_v1` y un futuro `m30_v1`.
- Posibles esquemas:
  - selector por confianza
  - selector por cobertura
  - selector por volatilidad
  - mezcla ponderada
  - regla de desempate por edge

Preguntas:
- Cuando debe decidir `m27` y cuando conviene esperar a `m30`.
- Si el experto final debe elegir modelo o solo ajustar stake.

### 9. Estrategias o variantes por liga

- Revisar si algunas ligas justifican una estrategia puntual en lugar de usar la misma logica global.
- Identificar ligas donde convenga:
  - una regla operativa distinta
  - un gate distinto
  - un stake distinto
  - una variante propia de `m27`
  - una variante propia de `m30`
- Evaluar si ciertas ligas necesitan features unicas por contexto de ritmo, volatilidad, cobertura o estructura de parciales.
- Separar claramente entre:
  - ajuste ligero de politica
  - variante de modelo por liga
  - estrategia completa por liga

Preguntas:
- Que ligas tienen patron suficientemente estable como para justificar trato especial.
- Si el beneficio viene de features unicas, del gate o de una logica de apuesta distinta.
- Si conviene hacer variantes por grupos de ligas en vez de una por liga individual.

Criterio de exito:
- Mejorar ROI y estabilidad en ligas objetivo sin fragmentar demasiado la muestra ni volver inmantenible el pipeline.

### 10. Momentum y aceleracion independientes

- Generar graficas independientes por cuarto.
- Revisar si hay aceleracion del ritmo, momentum independiente y cambios bruscos por ventana.
- Separar:
  - ritmo base
  - ritmo reciente
  - aceleracion
  - swings del margen

Preguntas:
- Si el ritmo de anotacion reciente aporta mas que el margen bruto.
- Si la aceleracion anticipa remontadas o colapsos en Q4.

## Prioridad exploratoria

### 11. Historial de equipos remontadores

- Considerar si scrapeando historico de partidos anteriores se puede estimar si ciertos equipos son mas remontadores.
- Posibles features:
  - frecuencia historica de comeback
  - rendimiento historico cerrando `Q4`
  - perfil de equipo cuando va abajo al halftime o al minuto 27

Riesgos:
- mucho costo de scrape
- leakage indirecto si no se controla el corte temporal
- features inestables en ligas con poca data

### 12. Juegos con mucha desventaja del pick

- Revisar que pasa con juegos donde el pick entra con desventaja amplia.
- Estudiar si esos juegos se emparejan o si el modelo esta comprando remontadas poco realistas.
- Cruzar:
  - deficit actual
  - deficit de halftime
  - ritmo reciente
  - volatilidad

Preguntas:
- Si conviene limitar apuestas en deficits altos aunque el modelo vea edge.

### 13. Ver si es posible usarlo en live

- Revisar si el flujo actual se puede observar y explotar en live real.
- Validar:
  - timing de llegada de datos
  - cobertura de graph y play-by-play
  - reconstruccion consistente del snapshot
  - latencia de inferencia

Preguntas:
- Si la misma logica de backtest es suficientemente fiel en live.
- Que condiciones minimas de cobertura deben cumplirse para habilitar apuesta.

## Orden sugerido de ejecucion

1. Consolidar politica de ligas y causas de `no bet`.
2. Analizar features nuevas de `Q1` / `Q2` y combinaciones de marcador.
3. Prototipar `m30_v1` con la filosofia de `m27_v1`.
4. Medir pesos alternativos del ensemble `m27_v1`, incluyendo pesos por liga o segmento.
5. Probar ruta `coverage_low`.
6. Revisar si algunas ligas justifican estrategia propia, gate propio o variante dedicada de `m27` / `m30`.
7. Diseñar `team of experts` entre `m27_v1` y `m30_v1` solo despues de cerrar mejor el ensemble interno.
8. Explorar fouls y momentum/aceleracion.
9. Evaluar viabilidad real de live.

## Notas

- Cualquier mejora de ROI usando ligas filtradas debe considerarse provisional hasta validarla fuera de la misma muestra usada para definir la politica.
- Features nuevas deben revisarse contra leakage antes de pasar a entrenamiento formal.

## Opinion tecnica

### Cambios con mejor pinta

- Lo mas prometedor me parece seguir profundizando en features de contexto de `Q1`, `Q2`, halftime y `Q3` parcial, porque va en la misma direccion que ya hizo fuerte a `m27_v1`: menos ruido y mas estado real del partido.
- Tambien me parece muy buena apuesta intentar llevar la filosofia de `m27_v1` a `m30`, porque `m30` ya tenia fuerza offline y probablemente estaba cargando demasiado ruido estructural o una logica menos afinada.
- El trabajo de ligas volatiles, ligas sin apuesta y causas de `no bet` tiene mucho valor practico. No necesariamente mejora AUC, pero si puede mejorar ROI real y operatividad.
- Tambien veo valor en revisar ligas muy particulares para detectar si ameritan una estrategia puntual o incluso una variante propia de `m27` o `m30` con features unicas, pero solo donde la muestra realmente aguante.

### Cambios utiles pero con mas riesgo

- La blacklist por ligas puede subir ROI en backtest, pero tiene riesgo claro de sobreajuste si se define con la misma muestra donde luego se evalua.
- El `team of experts` entre `m27_v1` y `m30_v1` suena fuerte, pero puede meter complejidad demasiado pronto si antes no esta claro que `m30_v1` realmente aporta algo distinto.
- Ajustar pesos del promedio entre algoritmos puede ayudar, pero esperaria ganancias pequenas salvo que se detecte un patron muy claro por segmento, coverage o liga. Por ejemplo, podria pasar que `xgb` tenga mas peso en ciertas ligas y `hist_gb` en otras si los cortes temporales muestran una ventaja consistente.
- Las estrategias o variantes por liga pueden valer mucho en casos puntuales, pero solo si se evita fragmentar demasiado la muestra. Preferiria empezar por grupos de ligas o por 2 o 3 ligas muy claras antes de abrir una rama distinta para cada una.

### Cambios que primero requieren validar data

- Fouls me interesa, pero solo si la calidad del scrape realmente aguanta. Si la captura es inconsistente, es mas facil que meta ruido a que ayude.
- Un modelo especial para baja cobertura live me parece valioso, pero solo si antes se mide bien cuantos partidos se pierden por coverage baja y si ese subconjunto tiene tamano suficiente para entrenar algo estable.
- Ver si se puede usar bien en live es importante, pero lo trataria como validacion de infraestructura y timing, no solo como tema de modelado.

### Cambios mas exploratorios

- Scraping historico para identificar equipos remontadores puede ser potente, pero es el tipo de mejora que mas facil se vuelve costosa y fragil. No la pondria adelante de mejoras que ya estan al alcance con la data actual.
- Analizar aceleracion, momentum independiente y graficas por cuarto si me parece buena linea, pero conviene hacerlo de forma incremental para no inflar el feature space con derivadas que luego no sostengan senal.

### Mi orden de preferencia real

1. Profundizar marcador por cuartos, bins y combinaciones no leaky.
2. Revisar ligas volatiles, causas de `no bet` y coverage insuficiente.
3. Construir `m30_v1` con la misma filosofia de `m27_v1`.
4. Probar un filtro o ruta especial para coverage baja.
5. Ajustar ensemble, incluyendo pesos por liga o segmento.
6. Revisar si hay 2 o 3 ligas que justifiquen estrategia propia o variantes dedicadas de `m27` / `m30`.
7. Despues de eso pensar en `team of experts`.
8. Solo luego entrar fuerte a fouls, historico de remontadas y live real.