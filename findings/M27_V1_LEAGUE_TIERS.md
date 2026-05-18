# M27_V1 League Tiers

Snapshot de politica inicial por liga para `m27_v1`, basado en el reporte ROI mas reciente.

## Objetivo

- `blacklist`: ligas donde ya hay dano suficiente como para evitar seguir arriesgando.
- `watchlist`: ligas con volumen medio y PnL negativo; conviene vigilarlas o endurecer umbral.
- `whitelist_soft`: ligas con mejor senal relativa; no implica confianza ciega.

## Blacklist

- `Liga 1 Masculin, Faza II`
- `Germany BBL`
- `EYBL U17, CHALLENGE CUP`

## Watchlist

- `CIBACOPA , Segunda Vuelta`
- `LMB, Apertura`
- `Colombia LPB`
- `France Pro A`
- `Elite 2`
- `Meridianbet KLS`
- `EYBL U20, CHALLENGE CUP`
- `Liga Nacional Femenina Chile, Fase regular`

## Whitelist Soft

- `B1 League`
- `BNXT League`
- `Puerto Rico BSN`
- `Israeli National League Basketball`
- `China CBA`
- `Serie A2, Women, Playoffs`
- `LF Challenge, Regular Season`

## Uso recomendado

- Correr `m27_v1` raw para no perder visibilidad del modelo base.
- Correr `m27_v1_blacklist` para medir si el filtro mejora ROI con menos volumen.
- Revisar estas listas cada vez que cambie materialmente el sample o las reglas de stake.