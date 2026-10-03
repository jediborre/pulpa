# V13 — Análisis de timing del monitor live

## El problema central de Q4

### Q3 vs Q4: estructura de tiempo completamente diferente

```
PARTIDO DE BALONCESTO (4 cuartos × 12 min)
─────────────────────────────────────────────────────────────────────

Q1 ──── Q2 ──── [DESCANSO ~15 min] ──── Q3 ──── Q4
0       12      24                       36       48

              ↑                                ↑
         Bot despierta Q3              Bot despierta Q4
         (min 20, hay pausa)           (min 32, juego en curso)
```

### Timeline real en Q3 (caso favorable)

```
Tiempo real (ejemplo: partido 20:00 hora local)
──────────────────────────────────────────────────────────────────

20:00  Partido inicia
20:24  Q2 termina → DESCANSO empieza
20:24  Bot detecta min ≥ 20 (WAKE_BEFORE=4), despierta
20:25  Scrape ejecutado: Q1+Q2 completos, gp hasta min 24
20:25  Inferencia ejecutada
20:25  Señal enviada por Telegram ✅ AÚN DURANTE EL DESCANSO
20:38  Q3 inicia (si descanso dura ~14 min)

→ VENTANA PARA APOSTAR: ~13 min reales
```

### Timeline real en Q4 (caso problemático)

```
Tiempo real (ejemplo: partido 20:00 hora local)
──────────────────────────────────────────────────────────────────

21:15  Q3 termina → Q4 inicia INMEDIATAMENTE (sin pausa)
21:15  Bot aún dormido (esperaba min 32 que equivale a ~21:17-21:19)
21:17  Bot detecta min ≥ 32 (WAKE_BEFORE_Q4=4, target=36-4=32)
21:18  Scrape ejecutado: gp llegan hasta min ~32-33, NO hasta 36
21:18  Inferencia: modelo ve snapshot "incompleto" vs lo que aprendió (min 36)
21:18  Señal: probablemente NO_BET (confianza baja por datos parciales)
21:20  Segundo tick (CONFIRM_TICKS=2, +100s): más gp disponibles
21:20  Inferencia 2: ahora sí confia → BET
21:21  Señal BET enviada ⚠️ Q4 lleva ya 6 min jugados de 12

→ VENTANA PARA APOSTAR REAL: ~6 min (50% del cuarto ya jugado)
→ PROBLEMAS: (1) señal tardía (2) flip de opinión (3) ventaja ya establecida
```

---

## Por qué el modelo da puntuación baja con snapshot parcial Q4

El modelo V12 fue entrenado con:
- graph_points acumulados desde min 0 hasta min **36** (fin de Q3)
- Estas features incluyen: `gp_count`, `gp_slope_3m`, `gp_slope_5m`, `gp_acceleration`

Cuando llega un snapshot live con gp hasta min 32:
- `gp_count` = ~18 puntos (menos que los ~24 de entrenamiento)
- `gp_slope_5m` = pendiente de min27→32 (diferente a min31→36 que aprendió)
- `gp_acceleration` = calculado sobre base más corta → valor distorsionado

El modelo encuentra un vector de features que **no se parece a ninguna muestra de entrenamiento** en los valores de graph_points, por lo que su probabilidad converge hacia 0.5 → confianza 0% → NO_BET.

---

## Solución propuesta para V13

### Opción A: Entrenar con cutoff adelantado (recomendada, más simple)

Entrenar Q4 con graph_points hasta min **29-31** en vez de 36.  
En producción, el monitor pide datos a min ~32 → el modelo ve un snapshot  
comparable al de entrenamiento → confianza correcta.

```python
# Antes (V12)     →  Después (V13)
Q4_GRAPH_CUTOFF = 36   →  Q4_GRAPH_CUTOFF = 31

# El monitor también baja su target:
MONITOR_Q4_MINUTE = 36 →  MONITOR_Q4_MINUTE = 31
```

**Ventaja**: Modelo calibrado para el momento real de inferencia.  
**Desventaja**: Se pierden features del tramo final Q3 (min 31-36). Sin embargo,  
esos datos raramente están disponibles en el momento live, así que la pérdida  
es ficticia — el modelo V12 también los pierde pero no fue entrenado para eso.

### Opción B: Entrenar con snapshot aleatorizado (más robusto, más complejo)

Para cada partido de entrenamiento, elegir aleatoriamente `snapshot_minute ∈ [28, 36]`  
y construir features con esos datos. Añadir `snapshot_minute` como feature.

**Ventaja**: El modelo es explícitamente robusto a cualquier momento de captura.  
**Desventaja**: Requiere más muestras, el dataset se multiplica por ~4.

### Opción C: Post-procesado de confianza por timing

Mantener V12 y ajustar la confianza en `infer_match_v13.py` según cuántos  
graph_points llegaron vs los esperados:

```python
# Si gp_count < gp_full_expected * 0.8:
#   confidence *= (gp_count / gp_full_expected)  ← recalibrar confianza
```

**Ventaja**: No requiere reentrenar.  
**Desventaja**: Hack sobre un modelo mal alineado; no resuelve el problema raíz.

**Recomendación**: Opción A para V13.0, Opción B para V13.1 si los datos son suficientes.

---

## Impacto en la ventana de apuesta

Con Opción A (cutoff Q4 en min 31):

```
MONITOR_Q4_MINUTE = 31
WAKE_BEFORE = 2

→ Bot despierta a min 29 (juego en curso, Q3 cerca del fin)
→ Verifica: _has_scores("Q3") → espera si Q3 no terminó
→ Q3 termina ~min 36 real
→ Scrape + inferencia al instante → señal en ~min 36
→ Q4 inicia → ventana completa de ~12 min para apostar ✅
```

El truco es verificar `_has_scores(Q3)` para asegurar que Q3 terminó  
aunque el minuto de clock sea 29-31. El bot espera ese trigger.

---

## Configuración de bet_monitor para V13

```python
# bet_monitor.py (V13)
Q3_MINUTE = 22               # antes 24; obtener datos más pronto en Q2
Q4_MINUTE = 31               # antes 36; crucial: alinear con cutoff de entrenamiento

WAKE_BEFORE_MINUTES = 2      # reducido de 4; menos margen muerto

# Para Q4: verificar que Q3 esté completo antes de inferir
# El loop ya tiene esta lógica en _has_scores(data, "Q1","Q2","Q3")
# Solo hay que asegurarse que la condición se evalúe ANTES de correr inferencia.

NO_BET_CONFIRM_TICKS      = 2   # Q3: mantener comportamiento
# Para Q4: usar 0 (sin confirmación; implementar por cuarto)
```

---

## Nota sobre partidos de bajo anotaje

El mensaje "puntos insuficientes de gráfica" que aparece en Q4 con marcador  
avanzado tiene DOS causas distintas:

1. **Umbral desalineado** (ya corregido en V12): el gate de V12 exigía más gp  
   que el gate del monitor → el monitor pasaba partidos que V12 rechazaba.

2. **Partidos de bajo ritmo** (aún existe en V13): partidos con pocos puntos  
   tienden a tener **también pocos eventos en el play-by-play**. El gate de  
   `MIN_PBP_Q4` puede ser demasiado estricto para partidos defensivos.  
   V13 debe analizar si el gate de PBP es necesario o si con graph_points es suficiente.

---

## Checklist antes de entrenar V13

- [ ] Verificar que el DB tiene suficientes partidos con Q3+Q4 completos
- [ ] Confirmar que `config.py` será fuente única (eliminar duplicados)
- [ ] Implementar muestras con `snapshot_minute` como feature opcional
- [ ] Añadir calibración Platt/Isotonic al ensemble
- [ ] Evaluar con simulación de timing live (no solo accuracy estática)
- [ ] Medir: ¿cuántos partidos Q4 habrían recibido señal correcta en < 2 min?
