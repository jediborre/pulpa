# SofaScore Basketball Extractor

Extractor de partidos de basketball de SofaScore usando Playwright.

La estrategia evita requests directos desde fuera del navegador. En su lugar:

1. Abre una pagina real de SofaScore con Playwright.
2. Deja que SofaScore plante cookies y sesion.
3. Ejecuta las llamadas JSON desde el contexto del navegador.

Con eso el flujo se comporta como un navegador legitimo y evita el 403 que aparece con requests directos.

## Que extrae

- Metadata del partido: equipos, fecha, hora UTC, venue, liga.
- Marcador final.
- Parciales por cuarto: Q1, Q2, Q3, Q4 y OT si existe.
- Play-by-play de jugadas de anotacion.
- Graph points de momentum/presion.
- Features tabulares para ML.

## Base de datos

El archivo SQLite por defecto es:

```text
matches.db
```

Tablas principales:

- `matches`: partido consolidado.
- `quarter_scores`: parciales por cuarto.
- `play_by_play`: jugadas de anotacion.
- `graph_points`: puntos de la grafica de momentum.
- `discovered_ft_matches`: IDs FT descubiertos por fecha.
- `backfill_state`: cursor persistente para reanudar extraccion.

## Requisitos

Instalar dependencias Python:

```bash
pip install -r requirements.txt
```

Instalar Chromium para Playwright:

```bash
playwright install chromium
```

Si usas el entorno virtual del proyecto en Windows:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\playwright.exe install chromium
```

## Ejecucion basica

Entrar a la carpeta del proyecto:

```bash
cd match
```

Ver ayuda general:

```bash
python cli.py --help
```

## 1. Extraer un partido individual por URL

```bash
python cli.py scrape "https://www.sofascore.com/basketball/match/brooklyn-nets-new-york-knicks/wtbsLtb#id:14442355"
```

Esto:

1. Obtiene el `match_id` desde la URL.
2. Carga la pagina del partido con Playwright.
3. Consulta desde el contexto del navegador:
   - `/api/v1/event/{id}`
   - `/api/v1/event/{id}/incidents`
   - `/api/v1/event/{id}/graph`
4. Guarda el resultado en SQLite.

## 2. Ver un partido guardado en JSON

```bash
python cli.py show 14442355
```

## 3. Listar partidos guardados

```bash
python cli.py list
```

## 4. Backfill FT por fecha, resumible

Este es el flujo para ML a escala.

### Que hace

1. Abre `https://www.sofascore.com/basketball` con Playwright.
2. Llama desde el navegador al endpoint diario:

```text
https://api.sofascore.com/api/v1/sport/basketball/scheduled-events/YYYY-MM-DD
```

3. Filtra solo eventos con `status.type = finished`.
4. Guarda los IDs en `discovered_ft_matches`.
5. Luego ejecuta la extraccion individual por `match_id`.

### Primera corrida

```bash
python cli.py backfill-ft --days 7 --start-date 2026-03-20 --reset-cursor --ingest-limit 200
```

Significado:

- `--days 7`: camina 7 dias hacia atras en esta corrida.
- `--start-date 2026-03-20`: fecha inicial del cursor.
- `--reset-cursor`: reinicia el cursor persistido.
- `--ingest-limit 200`: procesa hasta 200 partidos pendientes en esta corrida.

### Continuar desde donde quedo

```bash
python cli.py backfill-ft --days 7 --ingest-limit 200
```

No hace falta volver a indicar `start-date` si ya existe cursor guardado.

### Ver el estado del backfill

```bash
python cli.py backfill-status
```

Muestra:

- `resume_key`
- `resume_cursor`
- cantidad total descubierta
- procesados
- pendientes
- pendientes con error
- rango de fechas descubierto

## 5. Procesar solo pendientes

Si ya descubriste muchos IDs y quieres vaciar la cola sin seguir descubriendo fechas nuevas:

```bash
python cli.py process-pending --limit 500
```

Esto toma solamente filas pendientes de `discovered_ft_matches` y ejecuta la extraccion individual.

## 6. Descubrir IDs sin ingerir detalle

Si quieres primero poblar IDs y despues procesarlos por separado:

```bash
python cli.py backfill-ft --days 30 --no-ingest
```

Luego:

```bash
python cli.py process-pending --limit 500
```

## 7. Exportar features para ML

### Un registro por partido

```bash
python cli.py export-features --out features.csv
```

O en JSONL:

```bash
python cli.py export-features --format jsonl --out features.jsonl
```

### Un registro por cuarto

```bash
python cli.py export-features-quarters --out features_quarters.csv
```

Esto genera una fila por `Q1`, `Q2`, `Q3`, `Q4` por partido, util para modelos in-game.

## 8. Reconstruir la grafica con seaborn

```bash
python cli.py plot-graph 14442355 --out graph_14442355.png
```

La grafica:

- usa los `graph_points` guardados en DB.
- marca los cuatro cuartos.
- usa `FT` en el titulo.
- incluye los parciales Q1-Q4 en el texto del titulo.

## Flujo recomendado completo

### Opcion A: pipeline continuo resumible

1. Descubrir e ingerir en una sola corrida:

```bash
python cli.py backfill-ft --days 7 --ingest-limit 200
```

2. Revisar estado:

```bash
python cli.py backfill-status
```

3. Continuar despues:

```bash
python cli.py backfill-ft --days 7 --ingest-limit 200
```

### Opcion B: primero IDs, luego detalle

1. Descubrir muchos IDs FT:

```bash
python cli.py backfill-ft --days 90 --no-ingest --reset-cursor --start-date 2026-03-20
```

2. Procesar pendientes en lotes:

```bash
python cli.py process-pending --limit 500
```

3. Repetir hasta vaciar cola:

```bash
python cli.py process-pending --limit 500
```

## Uso con otra base de datos

Todos los comandos aceptan `--db`:

```bash
python cli.py backfill-status --db historical_matches.db
python cli.py backfill-ft --db historical_matches.db --days 30 --ingest-limit 300
python cli.py process-pending --db historical_matches.db --limit 300
```

## Notas operativas

- No todos los partidos FT traen todos los endpoints completos.
- Si un partido falla, el error queda en `discovered_ft_matches.last_error`.
- Un partido con error queda pendiente para reintento futuro.
- El cursor persistido permite seguir hacia atras en otra corrida.
- `process-pending` sirve para continuar exactamente donde se quedo el detalle.

## Comandos disponibles

```bash
python cli.py scrape <match_url>
python cli.py show <match_id>
python cli.py list
python cli.py backfill-ft [--days N] [--start-date YYYY-MM-DD] [--stop-date YYYY-MM-DD] [--ingest-limit N] [--no-ingest] [--reset-cursor]
python cli.py backfill-status
python cli.py process-pending [--limit N]
python cli.py export-features [--format csv|jsonl] [--out PATH]
python cli.py export-features-quarters [--format csv|jsonl] [--out PATH]
python cli.py plot-graph <match_id> [--out PATH]
```