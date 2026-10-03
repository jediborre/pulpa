# V12 LIVE BOOKMAKER - Tu Propia Casa Virtual

## 🎯 Filosofía: Pensar Como la Casa, No Como el Apostador

### La Mentalidad de la Casa:
1. **NUNCA regala valor** - Siempre agrega margen del 10-20%
2. **Se adapta RÁPIDO** al marcador y momentum
3. **Protege sus mercados** - Quita 1X2 si es obvio, ajusta handicaps agresivamente
4. **Solo ofrece apuestas donde TIENE edge** - Si no, no ofrece nada
5. **Es pesimista** - Asume el peor escenario para sus líneas

### Nuestra Ventaja vs Casas Reales:
- **TENEMOS el modelo** (MAE, Monte Carlo, momentum)
- **Podemos ser MÁS estrictos** que la casa rival
- **Solo apostamos cuando NUESTRA casa dice que hay value en SU casa**
- **Es como tener una casa interna evaluando a otra casa**

---

## 📐 Arquitectura del Sistema

### Flujo de Decisión:

```
┌─────────────────────────────────────────────────────┐
│  V12 LIVE BOOKMAKER (Tu Casa Virtual)               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Recibe datos en vivo:                            │
│     - Score actual (quarter, home, away)             │
│     - Graph points (momentum)                        │
│     - Play-by-play events                            │
│     - Tiempo restante                                │
│                                                      │
│  2. Calcula probabilidades INTERNAS (pesimistas):   │
│     - Monte Carlo con varianza ALTA                  │
│     - Ajuste por momentum (si graph lo justifica)    │
│     - Penalización del 15% al trailing team          │
│     - Usa MAE del modelo (±5.33 pts) como margen    │
│                                                      │
│  3. Genera SUS propias líneas "justas":             │
│     - 1X2 fair odds                                 │
│     - Handicap justo (diff esperada ± MAE)           │
│     - Over/Under justo (total esperado ± MAE)        │
│                                                      │
│  4. Compara con casa REAL:                           │
│     - ¿Nuestras odds < Sus odds? → VALUE             │
│     - ¿Nuestras odds > Sus odds? → NO VALUE          │
│     - Margen mínimo: 15% de edge                     │
│                                                      │
│  5. Recomienda apuesta SOLO si:                      │
│     - Edge > 15%                                     │
│     - Confianza > 60%                                │
│     - Mercado disponible                             │
│     - ROI esperado > 10%                             │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Cálculo de Líneas "Justas" (Internas)

### A. 1X2 Fair Odds:

```python
def calcular_1x2_fair(home_score, away_score, mins_left, graph_momentum):
    """
    Calcula 1X2 odds JUSTOS (sin margen de casa).
    Luego aplica 15% penalty al trailing team (pesimismo).
    """
    # Monte Carlo pesimista
    prob_home = monte_carlo_pesimista(
        home_score, away_score, mins_left, graph_momentum
    )
    prob_away = 1.0 - prob_home
    
    # Penalización al trailing team (la casa sabe que comebacks fallan)
    if prob_home < 0.5:
        prob_home *= 0.85  # 15% penalty
    else:
        prob_away *= 0.85
    
    # Normalizar
    total = prob_home + prob_away
    prob_home /= total
    prob_away /= total
    
    # Fair odds (sin margen)
    fair_home = 1.0 / prob_home if prob_home > 0.01 else 999
    fair_away = 1.0 / prob_away if prob_away > 0.01 else 999
    
    return fair_home, fair_away, prob_home, prob_away
```

### B. Handicap Justo:

```python
def calcular_handicap_fair(home_score, away_score, mins_left, mae=5.33):
    """
    Handicap justo = diff proyectada ± MAE del modelo.
    La casa TÍPICAMENTE pone handicap = diff proyectada.
    Nosotros somos MÁS conservadores: usamos diff + MAE.
    """
    # Proyección de puntos restantes
    elapsed = 12.0 - mins_left
    home_ppm = max(1.5, home_score / elapsed) if elapsed > 0 else 1.5
    away_ppm = max(1.5, away_score / elapsed) if elapsed > 0 else 1.5
    
    # Proyección final
    home_final = home_score + home_ppm * mins_left
    away_final = away_score + away_ppm * mins_left
    diff_proyectada = away_final - home_final  # Positivo = away ahead
    
    # Handicap justo (con margen MAE)
    # Usamos el MAE como "buffer" conservador
    handicap_justo = round(diff_proyectada + mae * 0.5)  # Half MAE como conservador
    
    # Probabilidad de cubrir handicap
    prob_away_cubre = calcular_prob_cubre_handicap(
        diff_proyectada, handicap_justo, mae
    )
    
    return handicap_justo, prob_away_cubre
```

### C. Over/Under Justo:

```python
def calcular_ou_fair(home_score, away_score, mins_left, mae=5.33):
    """
    O/U justo = total proyectado ± MAE.
    """
    elapsed = 12.0 - mins_left
    home_ppm = max(1.5, home_score / elapsed) if elapsed > 0 else 1.5
    away_ppm = max(1.5, away_score / elapsed) if elapsed > 0 else 1.5
    
    total_actual = home_score + away_score
    total_proyectado = total_actual + (home_ppm + away_ppm) * mins_left
    
    # Línea O/U justa (con MAE como margen)
    ou_line = round(total_proyectado)
    
    # Probabilidad de OVER
    prob_over = calcular_prob_over(total_proyectado, ou_line, mae)
    
    return ou_line, prob_over
```

---

## ⚖️ Comparación con Casa Real

### Detección de Value:

```python
def detectar_value_vs_casa_real(
    nuestras_probs,        # Nuestras probabilidades internas
    sus_odds,              # Odds de la casa real
    min_edge=0.15,         # 15% edge mínimo
    min_confidence=0.60,   # 60% confianza mínima
):
    """
    Compara nuestras probabilidades con odds de la casa real.
    Solo recomienda si hay edge CLARO.
    """
    recomendaciones = []
    
    for mercado, nuestra_prob in nuestras_probs.items():
        su_odd = sus_odds.get(mercado)
        if su_odd is None:
            continue  # Mercado no disponible
        
        # Su probabilidad implícita
        su_prob_implícita = 1.0 / su_odd
        
        # Nuestro edge
        edge = nuestra_prob - su_prob_implícita
        
        # ROI esperado
        roi = nuestra_prob * (su_odd - 1) - (1 - nuestra_prob)
        
        # Filtros ESTRICTOS (como la casa)
        if edge < min_edge:
            continue  # No hay suficiente edge
        
        if nuestra_prob < min_confidence:
            continue  # No es suficientemente confiable
        
        if roi < 0.10:
            continue  # ROI muy bajo
        
        recomendaciones.append({
            "mercado": mercado,
            "nuestra_prob": round(nuestra_prob, 4),
            "su_odd": su_odd,
            "su_prob_implícita": round(su_prob_implícita, 4),
            "edge": round(edge, 4),
            "roi_esperado": round(roi, 4),
            "confianza": "ALTA" if nuestra_prob > 0.70 else "MEDIA",
            "stake_recomendado": calcular_kelly(nuestra_prob, su_odd),
        })
    
    # Ordenar por ROI
    recomendaciones.sort(key=lambda x: x["roi_esperado"], reverse=True)
    
    return recomendaciones
```

---

## 🎯 Ejemplo Completo: Q4, min 39, 4-10

### Paso 1: Nuestras Líneas Internas

```python
# Datos en vivo
home_score = 4
away_score = 10
mins_left = 3.0
graph_momentum = +5  # Home pressure subiendo
mae = 5.33

# 1X2 Fair
fair_home, fair_away, prob_home, prob_away = calcular_1x2_fair(
    4, 10, 3.0, +5
)
→ fair_home = 7.50 (prob 13.3%)
→ fair_away = 1.18 (prob 86.7%)

# Handicap Justo
handicap, prob_cubre = calcular_handicap_fair(4, 10, 3.0, 5.33)
→ diff proyectada: ~8 pts
→ handicap justo: Away -7.5
→ prob_away_cubre: 58%

# O/U Justo
ou_line, prob_over = calcular_ou_fair(4, 10, 3.0, 5.33)
→ total proyectado: ~18.7 pts
→ O/U justo: 19 pts
→ prob_over: 45%
```

### Paso 2: Casa Real Ofrece

```
1X2:
  Home: 5.26
  Away: 1.20

Handicap:
  Home +7.5 @ 1.90
  Away -7.5 @ 1.90

O/U:
  Over 18.5 @ 1.90
  Under 18.5 @ 1.90
```

### Paso 3: Comparación Estricta

```
1X2 Home:
  Nuestra prob: 13.3%
  Su odd: 5.26 (implies 19.0%)
  Edge: 13.3% - 19.0% = -5.7% ❌ NO VALUE

1X2 Away:
  Nuestra prob: 86.7%
  Su odd: 1.20 (implies 83.3%)
  Edge: 86.7% - 83.3% = +3.4% ⚠️ MUY BAJO

Handicap Away -7.5:
  Nuestra prob: 58%
  Su odd: 1.90 (implies 52.6%)
  Edge: 58% - 52.6% = +5.4% ⚠️ BAJO (< 15%)

O/U Under 18.5:
  Nuestra prob: 55%
  Su odd: 1.90 (implies 52.6%)
  Edge: 55% - 52.6% = +2.4% ❌ MUY BAJO
```

### Resultado:

```
❌ NO RECOMENDACIÓN

Razón: Ningún mercado tiene edge > 15%.
La casa ya priceó correctamente este escenario.

Mensaje al usuario:
"NO hay value en ningún mercado ahora.
La casa tiene las líneas bien ajustadas.
Esperar mejor oportunidad."
```

---

## ✅ Cuándo SÍ Recomienda

### Escenario con Value Real:

```
Q4, min 36 (6 min left)
Score: Home 15 - Away 22
Graph momentum: +12 (home pressure FUERTE)

Nuestras líneas:
  1X2: Home 3.50 (28.6%) / Away 1.40 (71.4%)
  Handicap: Away -6.5 (prob 52%)
  O/U: 42 pts

Casa real ofrece:
  1X2: Home 4.50 / Away 1.25
  Handicap: Home +7.5 @ 2.10 / Away -7.5 @ 1.75
  O/U: Over 41.5 @ 1.90

Comparación:
  Home +7.5:
    Nuestra prob: 62% (momentum justifica)
    Su odd: 2.10 (implies 47.6%)
    Edge: 62% - 47.6% = +14.4% ✅
    ROI: 0.62 * 1.10 - 0.38 = +0.302 = +30.2%
    
    ✅ RECOMENDACIÓN:
    Mercado: Home +7.5
    Stake: $25 (Kelly fraccional 25%)
    Confianza: ALTA
    ROI esperado: +30.2%
```

---

## 📊 Matriz de Decisión

| Nuestras Líneas vs Casa Real | Edge > 15% | Edge 5-15% | Edge < 5% |
|------------------------------|-----------|-----------|----------|
| Confianza > 70% | ✅✅ BET FUERTE | ⚠️ BET MARGINAL | ❌ SKIP |
| Confianza 55-70% | ✅ BET | ❌ SKIP | ❌ SKIP |
| Confianza < 55% | ❌ SKIP | ❌ SKIP | ❌ SKIP |

**Solo BET FUERTE o BET es recomendado.** Marginal solo para usuarios avanzados.

---

## 🔧 Implementación Técnica

### Estructura de Datos:

```python
@dataclass
class LiveInput:
    """Datos en vivo del partido."""
    match_id: str
    quarter: str
    minute: float  # Minuto actual del quarter (0-12)
    home_score: int
    away_score: int
    graph_points: list[dict]  # SofaScore graph data
    pbp_events: list[dict]    # Play-by-play events
    sportsbook_odds: dict     # Odds actuales de la casa real


@dataclass
class BookmakerLines:
    """Líneas justas de NUESTRA casa virtual."""
    fair_home_odds: float
    fair_away_odds: float
    fair_handicap: float  # Positivo = away favored
    fair_handicap_prob: float
    fair_ou_line: float
    fair_over_prob: float
    confidence: str  # "high", "medium", "low"
    reasoning: str


@dataclass
class ValueOpportunity:
    """Oportunidad de value detectada."""
    market: str  # "1x2_home", "handicap_away", "over", etc.
    our_probability: float
    their_odds: float
    their_implied_prob: float
    edge: float  # our_prob - their_implied
    expected_roi: float
    recommended_stake: float
    confidence: str
    action: str  # "BET_STRONG", "BET", "AVOID"
    reasoning: str
```

### Flujo Principal:

```python
def v12_live_bookmaker(input: LiveInput) -> list[ValueOpportunity]:
    """
    Sistema completo de casa virtual.
    Genera líneas justas y compara con casa real.
    """
    mins_left = 12.0 - input.minute
    graph_momentum = calcular_momentum(input.graph_points)
    
    # 1. Calcular nuestras líneas justas
    nuestras_lineas = BookmakerLines(
        fair_home_odds=calcular_1x2_fair(
            input.home_score, input.away_score, 
            mins_left, graph_momentum
        ),
        fair_handicap=calcular_handicap_fair(
            input.home_score, input.away_score,
            mins_left, mae=5.33
        ),
        fair_ou_line=calcular_ou_fair(
            input.home_score, input.away_score,
            mins_left, mae=5.33
        ),
        confidence=evaluar_confianza(input),
    )
    
    # 2. Comparar con casa real
    oportunidades = detectar_value_vs_casa_real(
        nuestras_probs=extraer_probs(nuestras_lineas),
        sus_odds=input.sportsbook_odds,
        min_edge=0.15,
        min_confidence=0.60,
    )
    
    # 3. Filtrar solo recomendaciones fuertes
    recomendaciones_finales = [
        opp for opp in oportunidades
        if opp["action"] in ["BET_STRONG", "BET"]
    ]
    
    return recomendaciones_finales
```

---

## 📱 Interfaz de Usuario (Concepto)

### Output para el Usuario:

```
═══════════════════════════════════════════
V12 LIVE BOOKMAKER - Análisis en Vivo
═══════════════════════════════════════════

Partido: Lakers vs Celtics
Quarter: Q4 | Minuto: 36:00 | Restante: 6:00
Marcador: Lakers 15 - Celtics 22

───────────────────────────────────────────
NUESTRAS LÍNEAS JUSTAS:
───────────────────────────────────────────
1X2:        Home 3.50 | Away 1.40
Handicap:   Home +6.5 | Away -6.5
Over/Under: 42 pts

───────────────────────────────────────────
CASA REAL OFRECE:
───────────────────────────────────────────
1X2:        Home 4.50 | Away 1.25
Handicap:   Home +7.5 @ 2.10 | Away -7.5 @ 1.75
Over/Under: Over 41.5 @ 1.90

───────────────────────────────────────────
ANÁLISIS DE VALUE:
───────────────────────────────────────────
✅ Home +7.5 @ 2.10
   Nuestra prob: 62%
   Su prob implícita: 47.6%
   Edge: +14.4%
   ROI esperado: +30.2%
   Stake recomendado: $25
   Confianza: ALTA
   
❌ 1X2 Home @ 4.50
   Nuestra prob: 28.6%
   Su prob implícita: 22.2%
   Edge: +6.4% (BAJO)
   → NO RECOMENDADO

❌ Over 41.5 @ 1.90
   Nuestra prob: 52%
   Su prob implícita: 52.6%
   Edge: -0.6% (NEGATIVO)
   → NO RECOMENDADO

───────────────────────────────────────────
RECOMENDACIÓN FINAL:
───────────────────────────────────────────
APUESTA: Home +7.5 @ 2.10
STAKE: $25 (2.5% del bankroll)
ROI ESPERADO: +30.2%
RAZONAMIENTO: 
  Lakers tienen momentum fuerte (+12 graph).
  House overreacted al score actual.
  Proyección: diff final ~5-6 pts.
  Home +7.5 cubre cómodamente.
═══════════════════════════════════════════
```

---

## ⚠️ Reglas Estrictas (Como la Casa)

### NUNCA Recomendar Si:

1. **Edge < 15%** - No hay valor real
2. **Confianza < 60%** - Demasiado incierto
3. **ROI < 10%** - No vale el riesgo
4. **MAE cruza la línea** - Si total_proyectado ± MAE cruza la línea O/U, SKIP
5. **Momentum < umbral** - Si graph_momentum < 5 en valor absoluto, SKIP
6. **< 3 min restantes** - Demasiado volátil
7. **Casa quitó el mercado** - Si 1X2 no disponible, no forzar

### SIEMPRE Recomendar Si:

1. **Edge > 25%** - Value claro
2. **Confianza > 70%** - Alta certeza
3. **ROI > 20%** - Excelente expectativa
4. **Momentum confirma** - Graph apoya la proyección
5. **MAE no cruza** - Línea está fuera del rango de error

---

## 🎯 Próximos Pasos para Implementar

### Fase 1: Manual (HOY)
1. Usa `v12_cli.py live --scenarios` para ver ejemplos
2. Durante partidos en vivo, calcula líneas manualmente
3. Compara con tu casa de apuestas
4. Anota resultados en spreadsheet

### Fase 2: Semi-Automático (1-2 semanas)
1. Script que scrapea SofaScore en vivo
2. Input manual de odds de tu casa
3. Output en terminal con recomendaciones
4. Tracking automático de resultados

### Fase 3: Totalmente Automático (1 mes)
1. Scraping automático de SofaScore + casas de apuestas
2. Alerts por Telegram cuando hay value
3. Dashboard web en tiempo real
4. Base de datos de todas las recomendaciones

---

## 💡 Conclusión

**V12 LIVE Bookmaker = Tu casa virtual evaluando casas reales.**

- Calcula líneas JUSTAS (sin margen)
- Compara con líneas REALES (con margen)
- Solo apuesta cuando TU línea es MEJOR que la de ellos
- Sé MÁS estricto que la casa (edge > 15%, confianza > 60%)

**Esto es lo más cercano a "ganarle a la casa" que puedes llegar:**
No adivinando quién gana, sino encontrando CUÁNDO la casa se equivocó en sus líneas.

---

*Generado: 2026-04-13*
*V12 LIVE Bookmaker - Pensando Como la Casa*
