# V12 LIVE - Estrategia REALISTA con Todos los Mercados

## 🎰 La Realidad de las Casas de Apuestas

### Cómo Funciona la Casa:

**1. Se adapta RÁPIDO al marcador:**
```
Q4, min 39 (3 min restantes)
Score: Home 4 - Away 10

Casa ofrece:
  1 (Home): 5.26    (implies 19%)
  2 (Away): 1.20    (implies 83%)
  1X: 3.69          (Home no pierde)
  2X: 1.10          (Away no pierde)
```

**2. Si la ventaja es GRANDE, QUitan el 1X2:**
```
Q4, min 42, Home 2 - Away 18 (diff 16)
❌ 1X2 REMOVIDO
✅ Solo handicap y over/under disponibles
```

**3. Nunca regalan odds altas:**
- Max odds realistas: 5.0-8.0 (no 21.0 como simulé)
- Casa SIEMPRE agrega 10-20% de margen
- Si probabilidad real es 20%, te da odds de 4.0 (25% implied)

---

## 📊 Todos los Mercados Disponibles

### 1. **1X2 (Moneyline)**
```
Cuándo disponible: Diferencia < 10-12 pts
Odds típicas: 1.05 - 8.00
Ventaja: Simple
Desventaja: Casa lo quita si es muy obvio
```

### 2. **Handicap (Spread)**
```
Cuándo disponible: SIEMPRE (casa lo mantiene)
Ejemplo: Home +6.5 @ 1.90 / Away -6.5 @ 1.90
Ventaja: Siempre disponible, odds razonables
Desventaja: Necesitas cubrir el spread, no solo ganar
```

### 3. **Over/Under (Total de puntos del quarter)**
```
Cuándo disponible: SIEMPRE
Ejemplo: O/U 45.5 pts @ 1.90
Ventaja: No importa quién gana, solo el total
Desventaja: Necesitas predecir ritmo del juego
```

### 4. **Doble Oportunidad (1X, 2X, 12)**
```
Cuándo disponible: A veces
Ejemplo: 2X (Away no pierde) @ 1.10
Ventaja: Mayor probabilidad de ganar
Desventaja: Odds MUY bajas (1.05-1.20)
```

---

## 🎯 Ejemplo Real: Q4, min 39, Home 4 - Away 10

### Análisis de la Situación:
```
Quedan: 3 minutos (180 segundos)
Diferencia: 6 puntos a favor de Away
Pace actual:
  Home: 4 pts / 9 min = 0.44 pts/min ← MUY bajo
  Away: 10 pts / 9 min = 1.11 pts/min

Proyección simple:
  Home final: 4 + (0.44 * 3) = 5.3 pts
  Away final: 10 + (1.11 * 3) = 13.3 pts
  Diferencia final estimada: ~8 pts
```

### Mercado 1: 1X2
```
Home: 5.26 (implies 19%)
Away: 1.20 (implies 83%)

V12 Monte Carlo (pesimista):
  Home comeback prob: ~15% (pace muy bajo)
  Fair odds: 6.67
  Casa ofrece: 5.26
  
  ❌ NO VALUE (casa te da MENOS de lo justo)
```

### Mercado 2: Handicap Home +6.5
```
Home +6.5 @ 1.90
Significa: Home necesita NO perder por más de 6 pts

Escenarios:
  Si Q4 termina 10-16 (diff 6): Home +6.5 GANA ✅
  Si Q4 termina 10-17 (diff 7): Home +6.5 PIERDE ❌
  
Proyección:
  Diff final estimada: 8 pts
  Home +6.5 necesita diff <= 6
  
  Probabilidad: ~35% (necesitan frenar el ritmo)
  
  Fair odds: 2.86
  Casa ofrece: 1.90
  
  ⚠️ MARGINAL (1.90 vs 2.86 fair)
```

### Mercado 3: Handicap Away -6.5
```
Away -6.5 @ 1.90
Significa: Away necesita ganar por 7+ pts

Proyección:
  Diff final estimada: 8 pts
  Away -6.5 necesita diff >= 7
  
  Probabilidad: ~65%
  
  Fair odds: 1.54
  Casa ofrece: 1.90
  
  ✅ VALUE! (1.90 > 1.54 fair)
  Expected ROI: 0.65 * (1.90 - 1) - (1 - 0.65) = +0.235 = +23.5%
```

### Mercado 4: Over/Under Q4 Total
```
Si línea O/U es 45 pts:
  Current total: 4 + 10 = 14 pts (en 9 min)
  Pace: 14/9 = 1.56 pts/min
  Proyección: 14 + (1.56 * 3) = 18.7 pts en Q4
  
  ❌ 18.7 << 45 → UNDER claro
  PERO: Probablemente la casa ya ajustó la línea a ~20
```

---

## 💡 La Estrategia REALISTA

### Cuándo el 1X2 está disponible:

**Home trailing (diferencia 3-8 pts, 3-6 min left):**
```
1. Evaluar comeback prob con Monte Carlo pesimista
2. Si casa da odds 4.0-6.0 y tu prob es 20-30%:
   → VALUE BET en trailing team
3. Si odds < 3.0: SKIP (no vale la pena)
```

**Away dominating (diferencia 8+ pts):**
```
1. 1X2 probablemente NO disponible o odds muy bajas
2. Mejor opción: Handicap
```

### Cuándo el 1X2 NO está disponible (casa lo quitó):

**Opción A: Handicap**
```
Buscar handicap donde:
- Tu proyección diff final > handicap away
- O tu proyección diff final < handicap home

Ejemplo:
  Casa ofrece: Home +8.5 @ 1.90 / Away -8.5 @ 1.90
  Tu proyección: diff final = 10 pts
  
  Away -8.5: prob ~60%
  Fair odds: 1.67
  Casa ofrece: 1.90 → VALUE!
```

**Opción B: Over/Under**
```
Si puedes predecir ritmo:
- Pace alto → OVER
- Pace bajo → UNDER

Tu modelo de regresión (MAE 5.33 pts) da edge aquí
```

---

## 🔧 Modelo Monte Carlo PESIMISTA

### Ajuste Realista (la casa es gandalla):

```python
def monte_carlo_pesimista(score_home, score_away, mins_left):
    """
    Monte Carlo con varianza ALTA (más realista).
    La casa SIEMPRE te da peores odds.
    """
    # Pace actual
    elapsed = 12.0 - mins_left
    home_ppm = score_home / elapsed if elapsed > 0 else 1.5
    away_ppm = score_away / elapsed if elapsed > 0 else 1.5
    
    # Varianza MUY ALTA (basketball es volátil)
    var_home = max(1.0, home_ppm * 2.0)  # 2x en vez de 1.5x
    var_away = max(1.0, away_ppm * 2.0)
    
    sims = 10000
    sim_home = normal(score_home + home_ppm * mins_left, sqrt(var_home * mins_left), sims)
    sim_away = normal(score_away + away_ppm * mins_left, sqrt(var_away * mins_left), sims)
    
    home_wins = sum(sim_home > sim_away)
    ties = sum(abs(sim_home - sim_away) < 0.5)
    
    home_prob = (home_wins + 0.5 * ties) / sims
    
    # AJUSTE PESIMISTA: reducir prob del trailing team
    # La casa sabe algo que tú no
    if home_prob < 0.5:
        home_prob *= 0.85  # 15% penalty
    else:
        home_prob = 1.0 - (1.0 - home_prob) * 0.85
    
    return home_prob
```

### Ejemplo con modelo pesimista:

```
Q4, min 39: Home 4 - Away 10, 3 min left

Monte Carlo optimista: Home prob = 18%
Monte Carlo pesimista: Home prob = 15% (después del 15% penalty)

Fair odds pesimista: 6.67
Casa ofrece: 5.26

❌ NO VALUE (incluso más claro que antes)
```

---

## 📋 Flujo de Decisión REALISTA

```
1. ¿1X2 disponible?
   ├── SÍ → Ir a paso 2
   └── NO → Ir a paso 3 (Handicap)

2. 1X2 disponible:
   ├── Trailing team odds > 4.0?
   │   ├── SÍ → Calcular comeback prob
   │   │   ├── Prob * odds > 1.20? → BET
   │   │   └── NO → SKIP
   │   └── NO → SKIP (odds muy bajas)
   └── Leading team odds > 1.50?
       ├── SÍ → Lock bet (pero ROI bajo)
       └── NO → SKIP

3. Handicap disponible:
   ├── Calcular diff final proyectada
   ├── ¿Tu proyección > handicap away?
   │   ├── SÍ → BET Away -handicap
   │   └── NO → Ir a paso 4
   └── ¿Tu proyección < handicap home?
       ├── SÍ → BET Home +handicap
       └── NO → SKIP

4. Over/Under disponible:
   ├── Calcular pace total
   ├── ¿Pace > línea O/U?
   │   ├── SÍ → BET OVER
   │   └── NO → BET UNDER
   └── Verificar edge > 10%
```

---

## 💰 Ejemplo Completo con Handicap

```
PARTIDO: Lakers vs Celtics, Q4, min 39
SCORE: Lakers 4 - Celtics 10

V12 LIVE ANÁLISIS:
  Pace Lakers: 0.44 pts/min (MUY bajo)
  Pace Celtics: 1.11 pts/min
  Proyección diff final: ~8 pts

MERCADOS DISPONIBLES:
  ❌ 1X2: Disponible pero sin value (5.26 vs 6.67 fair)
  
  ✅ Handicap:
    Lakers +8.5 @ 1.90
    Celtics -8.5 @ 1.90
  
  Tu proyección: diff = 8 pts
  Celtics -8.5 necesita diff >= 9
  
  Prob Celtics cubre: ~55%
  Fair odds: 1.82
  Casa ofrece: 1.90
  
  ✅ VALUE! Edge: 1.90 - 1.82 = +0.08
  Expected ROI: 0.55 * 0.90 - 0.45 = +0.045 = +4.5%

DECISIÓN:
  BET $30 en Celtics -8.5 @ 1.90
  - Si cubre: +$27
  - Si no: -$30
  - Expected value: +$1.35 (marginal pero positivo)
```

---

## ⚡ Cuándo SÍ Hay Value Claro

### Escenario IDEAL (handicap):

```
Q4, min 36 (6 min left)
Score: Home 15 - Away 22 (diff 7)

Casa ofrece:
  Home +7.5 @ 1.95
  Away -7.5 @ 1.85

Tu análisis:
  Home pace: 2.5 pts/min (subiendo)
  Away pace: 3.7 pts/min (estable)
  Graph momentum: +8 (home pressure)
  
  Proyección: Home cierra brecha
  Diff final est: 5-6 pts
  
  Home +7.5 necesita diff <= 7
  Prob: ~70%
  Fair odds: 1.43
  Casa ofrece: 1.95
  
  ✅✅ VALUE CLARO!
  Expected ROI: 0.70 * 0.95 - 0.30 = +0.365 = +36.5%
```

**Este es el tipo de apuesta que buscas:**
- Momentum a favor del trailing team
- Handicap generoso (casa sobre-ajustó)
- Suficiente tiempo para que se refleje el momentum

---

## 🎯 Recomendación Final

### NO apostar:
- ❌ 1X2 cuando odds < 3.0 (no hay value)
- ❌ Comebacks con <3 min left (muy volátil)
- ❌ Cualquier cosa con ROI esperado < 10%

### SÍ apostar:
- ✅ Handicap cuando momentum ≠ marcador actual
- ✅ O/U cuando tu MAE de 5.33 pts da edge
- ✅ 1X2 trailing team SOLO si odds > 4.0 y prob > 25%

### La Única Vía Real de Ganarle a la Casa:

**Handicap betting con momentum detection:**
1. Identificar quarter donde trailing team tiene momentum
2. Esperar a que casa ofrezca handicap basado en score actual
3. Apostar a favor del trailing team +handicap
4. Necesitas 55-60% hit rate para ser rentable (factible con V12 LIVE)

---

*Generado: 2026-04-13*
*V12 LIVE - Estrategia Realista Multi-Mercado*
