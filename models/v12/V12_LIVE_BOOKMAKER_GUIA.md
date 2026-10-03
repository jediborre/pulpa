# V12 LIVE BOOKMAKER - Sistema Completo

## 🎯 Concepto: Tu Casa Virtual vs Casa Real

### Problema:
- Casas de apuestas son gandallas - NUNCA regalan valor
- Odds de 1.41 requieren 70.9% hit rate (imposible con modelo de 52%)
- Cuando hay ventaja clara, quitan el mercado 1X2
- Se adaptan RÁPIDO al marcador

### Solución:
**Tu propia casa virtual** que calcula líneas justas y te dice:
- "Si tu casa ofrece MEJOR que esto → VALUE"
- "Si tu casa ofrece PEOR que esto → NO VALUE"

---

## 📊 Ejemplo de Output del Sistema

```
======================================================================
V12 LIVE BOOKMAKER - Análisis en Vivo
======================================================================
Match: Lakers vs Celtics, Q4
Marcador: 8-3 (6 min jugados, 6 restantes)

──────────────────────────────────────────────────────────────────────
MERCADOS CON LÍNEAS JUSTAS:
──────────────────────────────────────────────────────────────────────

  Over 30 pts en quarter (actual: 11)
    Nuestra prob:  55.3%
    Odds justas:   1.81
    Confianza:     MEDIUM
    → Si tu casa ofrece Over 30 a 2.08 o mayor → VALUE

  Under 30 pts en quarter (actual: 11)
    Nuestra prob:  44.7%
    Odds justas:   2.24
    → Si tu casa ofrece Under 30 a 2.57 o mayor → VALUE
```

**Tú comparas manualmente:**
- Tu casa ofrece Over 30 @ 1.90 → 1.90 < 2.08 → ❌ NO VALUE
- Tu casa ofrece Over 30 @ 2.20 → 2.20 > 2.08 → ✅ VALUE

---

## 🔧 Cómo Funciona

### 1. Calcula Líneas Justas (Sin Margen de Casa):
```python
# 1X2
Monte Carlo pesimista → prob_home = 45%, prob_away = 55%
Fair odds: Home 2.22, Away 1.82

# Handicap
Proyección diff: 7 pts
Fair handicap: Away -7.5
Prob Away cubre: 58%
Fair odds: 1.72

# Over/Under
Proyección total: 45 pts
Fair O/U: 45
Prob Over: 52%
Fair odds Over: 1.92
```

### 2. Agrega Margen del 15% (Para Comparación):
```
Fair odds × 1.15 = Threshold de VALUE

Si fair odds = 2.00
Threshold = 2.30

→ Solo apuesta si tu casa da 2.30 o mayor
```

### 3. Tú Comparas con Tu Casa Real:
```
Nuestro threshold: 2.30
Casa real ofrece:  2.50
→ VALUE! Edge: +8.7%

Nuestro threshold: 2.30
Casa real ofrece:  2.10
→ NO VALUE (margen muy ajustado)
```

---

## 📋 Uso Práctico

### Comando:
```bash
# Demo con datos históricos
python training/v12/live_engine/virtual_bookmaker.py --demo

# Match específico
python training/v12/live_engine/virtual_bookmaker.py --match-id 15879747

# Output JSON (para integración)
python training/v12/live_engine/virtual_bookmaker.py --match-id 15879747 --json
```

### Durante Partido Real (Manual):
1. Abre SofaScore → Ve el quarter en vivo
2. Anota: score actual, minuto, momentum del graph
3. Corre el script con esos datos
4. Compara output con tu casa de apuestas
5. Si hay value → apuesta

### Flujo Ideal:
```
Min 6 del Q4:
  Score: Home 8 - Away 10
  
  Corre: virtual_bookmaker.py
  
  Output:
    Fair Over/Under: 42 pts
    Threshold Over: 2.15
    Threshold Under: 2.30
  
  Tu casa:
    Over 41.5 @ 2.20 → 2.20 > 2.15 → VALUE ✅
    Under 41.5 @ 1.75 → 1.75 < 2.30 → NO VALUE ❌
  
  Decisión: BET Over 41.5 @ 2.20
```

---

## 📁 Archivos Creados

```
training/v12/
├── live_engine/
│   ├── live_betting.py          # Motor anterior (con simulación)
│   ├── virtual_bookmaker.py     # ✅ NUEVO: Casa virtual completa
│   └── comeback_simulation.json # Datos de simulación
│
├── V12_LIVE_BOOKMAKER.md        # ✅ Documentación completa
├── LIVE_BETTING_STRATEGY.md     # Estrategia de comebacks
├── REAL_LIVE_BETTING_STRATEGY.md # Estrategia realista
├── HONEST_ASSESSMENT.md         # Evaluación honesta
├── DATA_LEAKAGE_ANALYSIS.md     # Análisis de fugas
└── DOCUMENTO_FINAL.md           # Resumen ejecutivo
```

---

## 🎯 Comparación: V12 Original vs V12 Live Bookmaker

| Aspecto | V12 Original (Pre-Bet) | V12 Live Bookmaker |
|---------|----------------------|-------------------|
| Cuándo | Antes del quarter | Durante el quarter (min 3-9) |
| Data | Solo histórico | Score en vivo + graph + PBP |
| Mercados | Solo 1X2 | 1X2, Handicap, O/U |
| Necesita odds? | Sí (para comparación) | ❌ NO (genera líneas justas) |
| Método | Predice ganador | Compara líneas justas vs reales |
| Hit rate necesario | 70.9% (@ 1.41) | 45-55% (@ 2.0+) |
| Viabilidad | ❌ NO (imposible) | ✅ POSIBLE |
| Trabajo | Bajo (automático) | Medio (manual comparación) |

---

## ⚠️ Limitaciones

### Lo Que el Sistema NO Hace:
1. ❌ NO se conecta automáticamente a casas de apuestas
2. ❌ NO scrapea odds en tiempo real
3. ❌ NO apuesta por ti
4. ❌ NO garantiza ganancias

### Lo Que el Sistema SÍ Hace:
1. ✅ Calcula líneas justas basadas en datos
2. ✅ Usa Monte Carlo pesimista (como la casa)
3. ✅ Incluye MAE real (5.33 pts) como margen
4. ✅ Genera thresholds de value claros
5. ✅ Te dice exactamente qué buscar

---

## 💡 Estrategia de Uso

### Cuándo Usar:
- Q3/Q4, min 4-9 (suficiente data, tiempo para reaccionar)
- Diferencia 3-10 pts (no muy obvio, no muy volátil)
- Momentum claro en graph (≥3 en valor absoluto)

### Cuándo NO Usar:
- < 3 min restantes (muy volátil)
- > 9 min transcurridos (poco tiempo para value)
- Diferencia > 15 pts (casa quitó mercados)
- Momentum neutral (sin señal clara)

---

## 📊 Ejemplo Real Paso a Paso

### Situación:
```
Q4, min 7 (5 min jugados, 7 restantes)
Score: River 12 - Boca 8
Graph: Momentum +5 (River pressure)
Total acumulado: 70 - 65
```

### Paso 1: Correr Script
```bash
python training/v12/live_engine/virtual_bookmaker.py --match-id <id> --quarter Q4
```

### Paso 2: Output del Sistema
```
PROYECCIONES:
  River: 12 → 21 pts
  Boca:  8 → 19 pts
  Diff:  +4 → +2 pts
  Total: 20 → 40 pts

MERCADOS:
  Handicap justo: Boca +2.5
  Fair odds Boca +2.5: 1.95
  Threshold: 2.24
  
  O/U justo: 40 pts
  Fair odds Over 40: 1.88
  Threshold: 2.16
  Fair odds Under 40: 2.12
  Threshold: 2.44
```

### Paso 3: Comparar con Casa Real
```
Tu casa ofrece:
  River -2.5 @ 1.90
  Boca +2.5 @ 1.90  → 1.90 < 2.24 → ❌ NO VALUE
  
  Over 39.5 @ 1.95  → 1.95 < 2.16 → ❌ NO VALUE
  Under 39.5 @ 1.85 → 1.85 < 2.44 → ❌ NO VALUE
```

### Resultado: NO BET
```
"La casa tiene las líneas bien ajustadas.
No hay value en ningún mercado.
Esperar mejor oportunidad."
```

### Otra Situación con VALUE:
```
Q4, min 5 (7 min restantes)
Score: Lakers 10 - Celtics 16
Momentum: +8 (Lakers pressure fuerte)

Virtual Bookmaker dice:
  Proyección: Lakers cierran brecha
  Handicap justo: Lakers +5.5
  Threshold: 2.30

Casa real ofrece:
  Lakers +7.5 @ 2.40 → 2.40 > 2.30 → ✅ VALUE!

APUESTA: Lakers +7.5 @ 2.40
```

---

## 🚀 Próximos Pasos

### Automatización Futura:
1. Scraping de SofaScore en vivo
2. Input manual de odds desde tu casa
3. Cálculo automático de value
4. Alerts por Telegram

### Por Ahora:
1. Usa `--demo` para entender el sistema
2. Durante partidos, corre manualmente
3. Compara con tu casa
4. Trackea resultados en spreadsheet

---

## ✅ Conclusión

**V12 Live Bookmaker = Tu asesor de apuestas personal.**

No te dice qué apostar. Te dice:
- "Las líneas justas son X"
- "Si tu casa ofrece mejor que Y, hay value"
- "Si ofrece peor, NO apuestes"

**Es como tener un quant en tu bolsillo evaluando si la casa se equivocó.**

---

*Generado: 2026-04-13*
*V12 Live Bookmaker - Tu Casa Virtual*
