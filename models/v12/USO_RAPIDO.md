# V12 - Guía Rápida de Uso

## 🚀 Comandos Principales

### 1. Análisis en Vivo (Virtual Bookmaker)
```bash
# VER LÍNEAS JUSTAS de un partido en vivo
python training/v12/live_engine/virtual_bookmaker.py --match-id <ID> --quarter Q4

# Demo con partidos recientes
python training/v12/live_engine/virtual_bookmaker.py --demo

# Output JSON (para integrar con otras herramientas)
python training/v12/live_engine/virtual_bookmaker.py --match-id <ID> --json
```

**Cuándo usar**: Durante un quarter (min 4-9), cuando quieres saber si hay value en tu casa de apuestas.

### 2. Predicción Pre-Bet
```bash
# Predecir ganador del quarter
python training/v12/v12_cli.py predict <ID> --target q4

# Con línea de sportsbook para over/under
python training/v12/v12_cli.py predict <ID> --line 185.5
```

**Cuándo usar**: Antes de que empiece Q3 o Q4.

### 3. Validación
```bash
# Verificar data leakage y simular bankroll
python training/v12/v12_cli.py validate

# Solo simulación con odds pesimistas
python training/v12/v12_cli.py validate --simulation --odds 1.41
```

**Cuándo usar**: Para entender las limitaciones reales del modelo.

### 4. Live Betting (Escenarios)
```bash
# Ver ejemplos de comeback betting
python training/v12/v12_cli.py live --scenarios

# Simular comebacks históricos
python training/v12/v12_cli.py live --simulate --limit 500
```

**Cuándo usar**: Para entender cómo funcionan las apuestas en vivo.

---

## 📋 Flujo de Trabajo Recomendado

### Antes del Partido:
1. `predict <ID>` → Ver predicción pre-bet
2. Si confidence > 65% → WATCHLIST

### Durante el Quarter (min 4-9):
1. Abre SofaScore → Ve score y graph
2. `virtual_bookmaker.py --match-id <ID>` → Obtén líneas justas
3. Compara con tu casa de apuestas
4. Si odds de tu casa > threshold → APUESTA
5. Si odds de tu casa < threshold → SKIP

### Después del Quarter:
1. Anota resultado en spreadsheet
2. Trackea: qué apostaste, odds, resultado
3. Revisa hit rate acumulado

---

## 🎯 Ejemplo Completo

```bash
# Partido: Lakers vs Celtics, Q4 acaba de empezar

# Paso 1: Predicción pre-bet
python training/v12/v12_cli.py predict 15879747 --target q4
# Output: V12 predice Lakers 55% confidence

# Paso 2: Min 6 del Q4, score 8-10
python training/v12/live_engine/virtual_bookmaker.py --match-id 15879747 --quarter Q4

# Output:
#   Proyección: 20-22 (diff -2)
#   Handicap justo: Lakers +2.5
#   Threshold: 2.30
#   O/U justo: 42 pts
#   Threshold Over: 2.10

# Paso 3: Comparas con tu casa
#   Tu casa: Lakers +3.5 @ 2.40 → 2.40 > 2.30 → VALUE ✅
#   Apuestas: Lakers +3.5 @ 2.40

# Paso 4: Resultado final Q4: 22-21
#   Lakers +3.5 GANA (perdió por 1, cubre +3.5)
#   Profit: +$28 en apuesta de $20
```

---

## ⚠️ Reglas de Oro

1. **NUNCA apuestes sin comparar líneas** - Usa virtual_bookmaker.py SIEMPRE
2. **Solo apuesta si odds > threshold** - Si es menor, SKIP
3. **Bankroll management** - Máx 5% por apuesta
4. **Trackea TODO** - Sin datos no sabes si funcionó
5. **Sé paciente** - Solo 2-3 oportunidades buenas por día

---

## 📚 Documentación Completa

- `V12_LIVE_BOOKMAKER_GUIA.md` - Guía completa del sistema
- `HONEST_ASSESSMENT.md` - Evaluación realista (LEER PRIMERO)
- `LIVE_BETTING_STRATEGY.md` - Estrategia de comebacks
- `REAL_LIVE_BETTING_STRATEGY.md` - Estrategia multi-mercado
- `DATA_LEAKAGE_ANALYSIS.md` - Análisis de fugas de datos

---

*Última actualización: 2026-04-13*
