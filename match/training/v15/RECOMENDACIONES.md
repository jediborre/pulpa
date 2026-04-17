# V15 - Recomendaciones operativas

Gu�a pr�ctica para operar el modelo semana a semana: entrenar, monitorear, detectar drift y reaccionar r�pido. Documento vivo: actualizarlo cuando cambien umbrales o se agreguen ligas.

> Contexto: [README.md](README.md) describe la arquitectura. Este archivo se enfoca **s�lo** en operaci�n.

---

## TL;DR


| Qu�           | Cu�ndo                                              | Comando                      |
| ------------- | --------------------------------------------------- | ---------------------------- |
| Re-entrenar   | 1� semana (lunes) o tras 3 d�as sin reentrenamiento | `cli_menu -> 1 -> 1`         |
| Test ROI      | despu�s de cada entrenamiento                       | `cli_menu -> 2 -> 1`         |
| Revisar plots | despu�s de cada entrenamiento                       | `cli_menu -> 2 -> 5`         |
| Resumen PROD  | ante cualquier duda                                 | `cli_menu -> 2 -> 4`         |
| Kill-switch   | si una liga empieza a perder                        | editar `league_overrides.py` |


Bandera roja: **hit rate global < 72% durante 2 semanas seguidas** �! revisar calibraci�n o subir `MIN_CONFIDENCE_BASE`.

---

## Workflow semanal

### Lunes (despu�s de que llegan los resultados del domingo)

```bash
# 1. Scan r�pido para ver que hay data nueva
python -m training.v15.cli_menu  # [4] Config -> [3] Escanear DB

# 2. Entrenar PROD
python -m training.v15.cli train \
    --train-days 72 --val-days 15 --cal-days 3 --holdout-days 0 \
    --min-samples-train 200 --active-days 14

# 3. Revisar resultados
python training/v15/reports/_summarize_prod.py

# 4. Revisar las 9 gr�ficas de diagn�stico
python -m training.v15.cli plots
# �! abrir training/v15/model_outputs/plots/*.png
```

### Durante la semana

Antes de cada partido:

```python
from training.v15.inference import V15Engine
engine = V15Engine.load()
pred = engine.predict(...)
if pred.signal == "BET":
    # apostar stake fijo a odds >= 1.40
    ...
# guardar pred.to_json() en tu DB para auditor�a posterior
```

**Regla de oro**: nunca apostar si `pred.signal == "NO_BET"`, aunque el modelo parezca "casi" confiado. Los gates est�n ah� por algo.

### Post-jornada (lunes siguiente o antes)

1. Actualizar la DB con resultados reales.
2. Comparar predicciones guardadas vs resultados �! calcular hit rate real de la semana.
3. Si hit rate semanal < 70% en ligas del portfolio �! investigar (ver [Checklist de drift](#checklist-de-drift)).

---

## Configuraci�n de producci�n (ganadora)

```
--train-days 72  --val-days 15  --cal-days 3  --holdout-days 0
--min-samples-train 200  --active-days 14
```

**Por qu� estos valores**:

- `train=72d` es un balance entre volumen y actualidad. M�s corto (50d) deja 100k muestras afuera. M�s largo (>90d) incluye temporadas que ya terminaron (NCAA).
- `val=15d` suficiente para aprender threshold por liga sin fragmentar demasiado.
- `cal=3d` para calibraci�n isot�nica. M�s no mejora.
- `holdout=0d` producci�n: no queremos "gastar" datos en un holdout que en breve deja de importar; los datos nuevos son el test real.
- `min_samples_train=200` deja entrar Brazil NBB (136 � 2 cuartos H" 270), Euroleague (136), Argentina LN (130). Con 300 se perd�an.
- `active_days=14` filtra NCAA terminada y otras ligas inactivas que s�lo a�aden ruido.

Estos valores salieron de un barrido contra holdout de 7 d�as comparando 3 configs (ver README, secci�n Evaluaci�n). Vuelta a re-barrer si la temporada cambia dr�sticamente (p.ej. nueva liga de invierno).

---

## Overrides activos

Derivados del `val_roi` del �ltimo entrenamiento. Se aplican en `training/v15/league_overrides.py`.


| Liga                    | Acci�n               | Raz�n                                             |
| ----------------------- | -------------------- | ------------------------------------------------- |
| Brazil NBB              | `force_nobet` (full) | val_roi q3=-100%, q4=-49%. Modelo no generaliza.  |
| Israeli National League | `force_nobet` (full) | val_roi q3=-20%, q4=-5.5%. Data ruidosa.          |
| NBA                     | `force_nobet_q4`     | q4 val_roi -27.6%, hit 51.7%. Variance late-game. |
| China CBA               | `force_nobet_q3`     | q3 val_roi -36%. Q4 es el �nico rentable.         |
| Argentina Liga Nacional | `force_nobet_q4`     | q4 val_roi -20.8%.                                |
| B2 League               | `force_nobet_q4`     | q4 val_roi -11.3%. Q3 rentable (+15.3%).          |
| Superliga               | `force_nobet_q3`     | q3 val_roi -12.8%.                                |
| �lite 2                 | `force_nobet_q3`     | q3 val_roi -36%.                                  |


**Regla para actualizarlos** (revisar cada re-entrenamiento):

1. Correr `python training/v15/reports/_summarize_prod.py`.
2. Cualquier (liga, target) con `val_roi < -5%` �! agregar `force_nobet_q3` o `force_nobet_q4`.
3. Cualquier liga con AMBOS targets `val_roi < 0%` �! `force_nobet` completo.
4. Volver a correr `test-roi` para verificar mejora global.

El CLI `cli_menu -> 4 -> 1` muestra todos los overrides activos.

---

## Checklist de drift

Cuando el hit rate **cae sostenidamente**, revisar en este orden:

### 1. Calibraci�n

Abrir `plots/05_calibration_curves.png`. Si las curvas se alejan de la diagonal, el modelo sobre/sub-predice.

**Fix**: re-entrenar con `--cal-days 5` y verificar. Si no mejora, revisar si la liga cambi� reglas (p.ej. NBA "Elam ending" en playoffs) o si hay un nuevo equipo dominante.

### 2. Train-val gap

Abrir `plots/01_train_val_gap.png`. Barras por encima de 0.15 = overfitting.

**Fix**:

- Agregar esa liga a `league_overrides.py` con `min_confidence_q3/q4` m�s alto (0.80+).
- Si sigue, `force_nobet_`*.
- Si varias ligas muestran gap �! reducir m�s la capacidad del modelo en `models.py` (m�s `min_samples_leaf`, menor `max_depth`).

### 3. Threshold aprendido

Abrir `plots/04_threshold_curves.png`. Si una liga que antes operaba bien a 0.75 ahora tiene su pico a 0.85, el modelo ya no captura bien esa liga.

**Fix**: subir `min_confidence_qN` en overrides de esa liga.

### 4. Volatilidad creciente

Abrir `plots/06_probability_distribution.png`. Si la distribuci�n se aplasta cerca de 0.5 (menos predicciones confiadas), algo cambi� en los inputs (feed de datos, disponibilidad de PBP, etc.).

**Fix**: revisar `plots/08_samples_per_league.png` para ver si alguna liga perdi� volumen. Verificar que el feed de PBP siga completo.

### 5. Una liga espec�fica se rompi�

Correr:

```bash
python -m training.v15.cli test-roi --odds 1.40 --min-bets 3 --top 50
```

Detectar ligas con `hit_rate < 0.65` y bets >= 10. Esas son candidatas para `force_nobet_*`.

---

## Kill-switches de emergencia

### Emergencia nivel 1: parar una liga

```python
# en league_overrides.py
LEAGUE_OVERRIDES["liga-problem�tica"] = {
    "force_nobet": True,
    "notes": "pauseada por drift detectado YYYY-MM-DD",
}
```

No requiere re-entrenamiento, se aplica al cargar `V15Engine`.

### Emergencia nivel 2: parar un target de una liga

```python
LEAGUE_OVERRIDES["NBA"] = {
    "force_nobet_q4": True,
    "notes": "Q4 NBA inestable en playoffs",
}
```

### Emergencia nivel 3: parar todo

En `config.py`:

```python
MIN_CONFIDENCE_BASE = 0.99
```

Efectivamente ninguna predicci�n pasa el gate de confianza. Revertir cuando est� solucionado.

### Emergencia nivel 4: rollback a PROD anterior

Los archivos `*_PROD.json` se conservan. Para volver al �ltimo conocido-bueno:

```bash
cd training/v15/model_outputs
cp training_summary_v15_PROD.json training_summary_v15.json
# los .joblib tambi�n se pueden restaurar si se hizo backup
```

Tip: antes de cada re-entrenamiento, hacer backup de los `.joblib` que importan:

```bash
cd training/v15/model_outputs
mkdir -p backup_$(date +%Y%m%d)
cp *.joblib backup_$(date +%Y%m%d)/
cp training_summary_v15.json backup_$(date +%Y%m%d)/
```

---

## Gesti�n de bankroll (sugerencia, no implementado en c�digo)

El modelo solo produce se�ales BET/NO_BET. El staking es responsabilidad externa. Recomendaciones:

- **Stake fijo** (flat betting) a 1u por apuesta. Simple y defensivo.
- **Kelly fraccionado (1/4 Kelly)**: stake = 0.25 � (p � (odds - 1) - (1 - p)) / (odds - 1), con p = `pred.probability`. Hace sentido cuando el modelo est� bien calibrado (curvas de `05_calibration_curves.png` cerca de la diagonal).
- **Nunca flat betting si el modelo no est� calibrado**   subestimas ventaja en bets altos y sobre-apost�s en bets bajos.

Con odds 1.40 y hit rate 75%, edge H" 5%. Bankroll management conservador: **1-2% del bankroll por apuesta**.

---

## Preguntas frecuentes operativas

### �Cu�ntas apuestas esperar por d�a?

Con config PROD actual y 11 ligas activas:

- D�a alto-volumen (jueves/s�bado): 6-15 bets (Q3 + Q4 combinados).
- D�a bajo (martes/mi�rcoles): 1-5 bets.
- Promedio semanal: 30-50 bets.

### �Con qu� frecuencia re-entrenar?

- �ptimo: **diario**. El dataset crece ~300 partidos/d�a.
- Aceptable: **semanal**. Dejar de apostar si no se re-entren� en 10 d�as.
- Riesgoso: **mensual**. Thresholds aprendidos envejecen.

### �Puedo bajar el threshold para tener m�s cobertura?

S�, editando `MIN_CONFIDENCE_BASE` en `config.py`. Pero **cuidado**:

- Cada 0.01 de bajada t�picamente agrega ~20% de cobertura pero -2% de hit rate.
- Por debajo de 0.72 el margen sobre break-even desaparece.
- Mejor alternativa: bajar por liga en `league_overrides.py` con `min_confidence_q3/q4` solo para ligas donde el modelo es confiable.

### �C�mo agregar una nueva liga?

No hace falta c�digo. Simplemente:

1. Cargar partidos de esa liga en `matches.db`.
2. Esperar a tener e"300 partidos (o e"200 si la liga est� activa y quer�s pronto).
3. Re-entrenar. La liga entrar� autom�ticamente al cat�logo.
4. Al siguiente re-entrenamiento, revisar `_summarize_prod.py` y decidir si vale la pena activarla (val_roi > 0%).

### �C�mo activar ligas femeninas?

En `config.py` cambiar:

```python
ALLOWED_GENDERS = ("men", "women")
```

Pero primero verificar volumen: si las ligas femeninas no tienen e"500 muestras cada una, los modelos van a ser fr�giles. Recomendado esperar a tener 2 temporadas completas de data.

### �Por qu� NO_BET si el modelo dice 0.78?

Porque pas� el gate de confianza pero fall� otro:

- Volatilidad muy alta (e"8 cambios de liderato).
- Racha muy grande (`current_run > 14`).
- Regresor no confirma al clasificador.
- Datos insuficientes (`graph_points < 14` o `pbp_events < 12`).

Revisar `pred.debug["gates"]` para ver cu�l fall�.

---

## Checklist pre-producci�n (primera vez)

- `matches.db` con al menos 90 d�as de datos.
- `python -m training.v15.cli_menu -> 4 -> 3` muestra 2k+ partidos/semana.
- Entrenar con config PROD.
- `test-roi` global muestra hit rate e" 72% y ROI e" 0%.
- Revisar `plots/05_calibration_curves.png`: l�nea cerca de diagonal.
- Revisar `plots/01_train_val_gap.png`: barras < 0.20.
- Al menos 8 (liga, target) con `val_roi > 0%`.
- Hacer backup de `model_outputs/*.joblib` antes de ir a vivo.
- Preparar logging de cada `pred.to_json()` a DB externa para auditor�a.
- Definir stop-loss diario (p.ej. -5u �! pausar hasta ma�ana).

