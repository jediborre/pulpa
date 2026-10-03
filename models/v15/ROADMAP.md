# V15 - Roadmap, deuda t�cnica y opini�n honesta

Este documento dice lo que el README no dice porque no es "marketing". Cosas que faltan, cosas que podr�an romperse, y mi lectura honesta del nivel de madurez del modelo.

> Si le�ste [README.md](README.md), ya sab�s lo que V15 hace bien. Ac� vamos al otro lado.

---

## 1. Deuda heredada de versiones anteriores

### 1.1 Lo que v14 empez� y v15 no termin�

La versi�n 14 hab�a dejado planeada una "Fase 3" que nunca se ejecut� completamente. Haciendo inventario del directorio `training/v14/`:


| Feature de v14                                                    | Estado en v14                                     | Estado en v15                                                     | Prioridad                                |
| ----------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------- |
| Trajectory features (lead_changes, current_run, last5_diff, etc.) | implementado                                      | **portado e integrado**                                           | ya hecho                                 |
| `LiveSample` protocol entre dataset e inferencia                  | implementado                                      | reemplazado por `Sample` equivalente                              | equivalente                              |
| `TimesFM` (Google Research time-series)                           | **esqueleto �nicamente**, `TIMESFM_ENABLED=False` | no implementado                                                   | **media-alta**                           |
| Segmentaci�n por pace                                             | segmentaba modelos por bucket                     | v15 usa pace como feature (una sola instancia de modelo por liga) | decisi�n consciente, no deuda            |
| Monitor/telegram integration (v12/v14)                            | implementado en v12, compatible en v14            | **no portado** a v15                                              | **alta si el bot es parte del producto** |


### 1.2 TimesFM: qu� era y por qu� importa

v14 dej� el esqueleto para integrar **TimesFM** de Google Research: un modelo pre-entrenado de forecasting de time series que, aplicado a la serie de `graph_points` (diferencia de puntos a lo largo del partido), puede dar un "point_forecast" y "quantile_forecast" de c�mo va a evolucionar el marcador en los pr�ximos N minutos.

Ver `training/v14/features.py:_timesfm_features`   la funci�n existe, est� documentada, pero termina con `raise NotImplementedError`.

**Por qu� vale la pena implementarlo**:

- Es un modelo que ya vio millones de series temporales y generaliza mejor que cualquier feature manual.
- Las quantile forecasts dan naturalmente una medida de incertidumbre (p90-p10) que puede usarse como gate adicional.
- Ser�a una feature a�adida al ensemble, no una sustituci�n   riesgo bajo.

**Por qu� no se implement�**:

- Requiere cargar un modelo pre-entrenado (~500MB), cuesta RAM en inferencia.
- Necesita fine-tuning con datos propios para ser �til de verdad (el dataset actual es chico para eso).
- Latencia de forecast: ~50-100ms por predicci�n. Aceptable pero no gratis.
- Tooling: `timesfm` paquete no est� s�per estable todav�a.

**Cuando tenga sentido volver a esto**: cuando tengas 6+ meses de data hist�rica y quieras squeeze del 3-5% extra de hit rate en ligas de alta volatilidad (NBA Q4, por ejemplo).

### 1.3 Telegram bot / monitor

v12 ten�a `telegram_v12_live_handler.py` y `telegram_integration.py`. v14 los manten�a compatibles. En v15 no existen.

Si el flujo de producci�n real es "el bot monitorea partidos en vivo, dispara predicci�n al llegar al minuto 22/31, env�a al usuario por Telegram"   **es deuda cr�tica que falta portar**.

No lo hicimos porque no estaba claro si el proyecto va a seguir usando Telegram o mover a una UI web. Migraci�n sugerida si se mantiene Telegram:

1. Copiar `training/v12/telegram_v12_live_handler.py` �! `training/v15/telegram_handler.py`.
2. Reemplazar las llamadas a `infer_match_v12` por `V15Engine.predict()`.
3. Adaptar el formato de mensaje al payload JSON de v15 (el `debug` ahora es mucho m�s rico).

Estimaci�n: 4-6 horas de trabajo.

---

## 2. Limitaciones t�cnicas conocidas

### 2.1 Dataset chico

**90 d�as de historia es poco**. Riesgos:

- Algunas ligas operan con `n_train H" 200-400` muestras. Es el piso aceptable; no es �ptimo.
- No cubrimos un ciclo completo de playoffs en ligas con temporadas largas.
- Eventos raros (back-to-back, lesiones de stars, reglas temporales) est�n subrepresentados.
- El split temporal asume "la pr�xima semana se parece a las �ltimas 3"   en transiciones regular-season �! playoffs esto puede fallar.

**Mitigaci�n actual**: `active_days=14`, bloquear ligas con malos resultados de validaci�n.

**Mitigaci�n futura**: scraper o feed que acumule data de al menos 2 temporadas completas.

### 2.2 No hay detector autom�tico de drift

El pipeline emite advertencias de `train_val_gap` pero **no hay un sistema automatizado que detecte si un modelo entrenado hace 5 d�as empez� a perder**. Hoy depende de que el operador corra `test-roi` manualmente.

**Siguiente paso natural**: un script cron que:

1. Corra test-roi sobre las �ltimas 48hs cada 6 horas.
2. Env�e alerta si el hit rate rolling cae bajo 72%.
3. (Opcional) Dispare re-entrenamiento autom�tico.

### 2.3 Calibraci�n conf�a mucho en `cal_days=3`

La calibraci�n isot�nica con solo 3 d�as de datos puede ser ruidosa. En el barrido esto funcion� porque las curvas se ve�an razonables, pero si una liga tiene pocas muestras en esos 3 d�as, `CalibratedClassifierCV` puede producir una curva err�tica.

**Siguiente paso**: usar `temperature scaling` (un �nico escalar) como alternativa cuando `len(cal) < 100`. M�s estable con poca data.

### 2.4 El portfolio no tiene stops

El modelo emite se�ales pero no sabe cu�ndo parar. Si ten�s 10 losses seguidas, el modelo seguir� sugiriendo bets. El stop-loss es responsabilidad del c�digo que consume el engine.

**Recomendaci�n**: implementar en el caller:

```python
if session_pnl < -5:  # 5 unidades perdidas
    # pausar hasta ma�ana
```

### 2.5 Las ligas con `force_nobet` no se re-eval�an solas

Si bloqueaste Brazil NBB por malos resultados pero dos semanas despu�s la liga cambi� (nuevo coach, equipo dominante desapareci�), seguir� bloqueada hasta que edites `league_overrides.py` manualmente.

**Siguiente paso**: en cada entrenamiento, si una liga bloqueada muestra `val_roi > +5%` durante 2 semanas, emitir un recomendado de re-activaci�n en los logs.

### 2.6 No hay versionado de modelos

Se guarda `training_summary_v15_PROD.json` como backup pero los `.joblib` se sobreescriben. No hay "modelo v15-2026W16" vs "modelo v15-2026W17".

**Siguiente paso**: en `train.py`, guardar modelos en `model_outputs/weekly/{YYYY-WW}/` y mantener symlink `latest`.

### 2.7 Feature importance es superficial

`plots/07_feature_importance.png` muestra importance promedio por ensemble. No desagrega por liga ni analiza interacciones. Si un modelo usa mal una feature (p.ej. `graph_slope` con peso bajo porque est� correlacionado con `current_run`), no lo detectamos.

**Siguiente paso**: SHAP values por liga. Costo de implementaci�n bajo (una librer�a m�s), alto valor para debug.

---

## 3. Oportunidades no exploradas

### 3.1 Features que faltan

Candidatas que no prob�:

- **Score differential variance en los �ltimos 6 minutos antes del cutoff** (se�al de tensi�n del partido).
- **Fouls cumulados por equipo** (si est� en el PBP): equipo en bonus early predice m�s FT �! m�s scoring.
- **D�as de descanso desde �ltimo partido** (fatiga). Requiere join con schedule.
- **Tipo de competici�n** (regular vs playoffs). Hoy se mezcla en el feature `league`.
- **Encoding expl�cito de home/away advantage por liga**   hoy est� impl�cito en las walk-forward stats.

### 3.2 Modelos alternativos

No hemos probado:

- **LightGBM**   suele matchar a XGBoost con menor tiempo de training. Podr�a reemplazar uno de los 4 ensembles.
- **Quantile regression** en el regressor (en vez de punto). Dar�a intervalos de confianza usables como gate.
- **Stacking con meta-learner** en vez de promedio ponderado simple.
- **Redes neuronales chicas** (3-4 layers) sobre las features num�ricas. A este volumen de data probablemente no mejora, pero vale el experimento.

### 3.3 Multi-target joint modeling

Hoy Q3 y Q4 se entrenan como modelos independientes. **Saben que est�n relacionados** (el resultado de Q3 es un input directo para Q4) pero no se comunican durante el training.

Un modelo joint (multi-task learning) podr�a capturar dependencias, pero duplica la complejidad. **Veredicto**: bajo retorno para el esfuerzo actual.

### 3.4 Aprendizaje online

Cada re-entrenamiento tira el modelo anterior y entrena de cero con 72 d�as. **Es ineficiente**. Frameworks como `river` o fine-tuning incremental reducir�an tiempo de entrenamiento a segundos.

**Veredicto**: no urgente (90s es aceptable), pero �til si en el futuro se entrena diariamente.

---

## 4. Opini�n honesta sobre el modelo

### 4.1 Lo que funciona bien

- **La arquitectura por liga es acertada.** Los resultados del barrido muestran que ligas como Euroleague, B1 League y LF Challenge son consistentemente rentables mientras NBA Q4 y Brazil NBB no. Un modelo global hubiera promediado estas realidades y perdido plata en todas.
- **Los gates hacen su trabajo.** Pasar de 350 matches en holdout a 85 bets via filtering es exactamente lo que quer�s: el modelo dice "no s�" la mayor�a del tiempo, solo apuesta cuando est� confiado.
- **Calibraci�n + threshold �ptimo por liga.** Combinado, este es el secret sauce. No hacerlo deja un 3-5% de ROI en la mesa.
- **El CLI y los debug outputs** hacen que auditar una predicci�n sea trivial. Esto importa cuando perd�s una apuesta y quer�s saber por qu� el modelo estaba confiado.

### 4.2 Lo que me preocupa

- **Tama�o de muestra de validaci�n chico.** Varias ligas tienen `n_val H" 60-100`. Con esa n, el intervalo de confianza del `val_roi` es �15-20 puntos porcentuales. Estamos tomando decisiones (force_nobet) con data que es apenas estad�sticamente significativa. Una liga con `val_roi=-5%` y n=60 podr�a perfectamente ser rentable en la realidad y la estamos bloqueando.
**Mitigaci�n**: usar ventanas de validaci�n rolling (no un �nico val fijo) y exigir se�ales robustas en m�ltiples ventanas antes de bloquear.
- **Odds 1.40 es muy poco margen.** El break-even es 71.43%; operamos target 75%. Un drop de 3 puntos porcentuales en hit rate pasa de "rentable" a "a p�rdida". Modelos de ML en deportes rara vez mantienen �3% estable en distintos reg�menes. Si aparece un "outlier month" (jueces distintos, lesiones raras), puede perder plata varias semanas seguidas.
**Mitigaci�n**: considerar buscar casas con odds 1.50+ para los mismos mercados. Aumenta el colch�n matem�tico de 71.4% a 66.7%.
- **El filtro de ligas activas puede estar sobreajustado al presente.** En 14 d�as, "NCAA termin�" fue obvio, pero a veces ligas que juegan cada 2 semanas (European cups) pueden caer fuera del filtro. Revisar el corte manualmente.
- **No hay se�al temprana de "temporada terminando".** NBA Q4 de regular season se comporta muy diferente a Q4 de playoffs. El modelo no lo sabe; si pasamos de regular a playoffs y dejamos NBA activa, puede empezar a fallar silenciosamente.
**Mitigaci�n**: agregar feature booleana `is_playoffs` derivada de alguna metadata en DB.

### 4.3 Nivel de madurez honesto

- **Prototipo funcional**: ' s�.
- **MVP operable por un operador t�cnico**: ' s�   con supervisi�n diaria.
- **Sistema aut�nomo de producci�n**: L' no todav�a. Necesita al menos:
  - Detector autom�tico de drift.
  - Stop-loss configurable.
  - Versionado de modelos.
  - Logging estructurado de cada predicci�n vs resultado.
  - Alertas cuando hit rate rolling cae bajo threshold.

### 4.4 Expectativas realistas

**Con el estado actual**, esperar:

- **Hit rate sostenido 73-78%** en el portfolio filtrado, con algunas semanas de 65% y algunas de 82%.
- **ROI mensual 2-8%** si se respetan las ligas del portfolio y se hace re-entrenamiento semanal.
- **Drawdowns de -3 a -5 unidades** cada 2-3 semanas son normales, no raz�n para p�nico.

**Banderas para tomar en serio**:

- 3 semanas consecutivas con ROI < 0.
- Una liga del portfolio con 20+ bets al 60% hit rate   no es variance, es cambio estructural.
- `train_val_gap` creciendo semana a semana en la misma liga.

### 4.5 Consejo final

El modelo es **razonablemente robusto pero no infalible**. Tratalo como un **asistente de decisi�n** que te ahorra tiempo y corrige tu bias humano, no como una "impresora de dinero". La diferencia entre +5% ROI y perder plata es disciplina operativa: re-entrenar a tiempo, respetar los gates, no bajar thresholds "porque hoy estoy seguro".

El camino a un modelo verdaderamente s�lido pasa por **acumular m�s data** (6-12 meses m�nimo) y **automatizar el monitoreo**. El modelo en s� ya tiene la arquitectura correcta; ahora necesita tiempo y tooling.

---

## 5. Prioridades sugeridas (pr�ximas iteraciones)

En orden de retorno / esfuerzo:

1. **Alta prioridad**
  - Detector autom�tico de drift con alertas (cron + `test-roi`).
  - Versionado de modelos por semana (`model_outputs/weekly/{YYYY-WW}/`).
  - Logging estructurado de cada predicci�n a una DB separada.
  - Portear telegram bot de v12 a v15 si se sigue usando.
2. **Media prioridad**
  - SHAP values por liga en `plots.py`.
  - Re-activaci�n autom�tica de ligas bloqueadas cuando mejoran.
  - Temperature scaling como fallback cuando `len(cal) < 100`.
  - Feature `is_playoffs` cuando el dataset lo permita.
3. **Baja prioridad (nice-to-have)**
  - LightGBM como cuarto ensemble member.
  - Stacking con meta-learner.
  - SHAP dashboard interactivo.
  - Integraci�n TimesFM (Fase 3 de v14).
4. **Estrat�gico (requiere tiempo calendario)**
  - Acumular 6 meses de data.
  - Incorporar ligas femeninas cuando haya e"500 muestras/liga.
  - Medir ROI real de operaci�n vs ROI simulado para validar calibraci�n.
  - Explorar odds 1.50+ para aumentar margen matem�tico.

---

## 6. Si vuelvo a arrancar de cero (lecciones)

Pensando en una v16 te�rica, lo que har�a distinto:

- **Empezar con holdout rolling desde d�a 1**: no diferir a un "pseudo-holdout" ad-hoc.
- **Modelado joint Q3 + Q4** desde el inicio, con un meta-feature que sea el output de Q3 durante el Q4 training.
- **Stop expl�cito en el engine**: si se detecta drift estructural, el `V15Engine.predict` deber�a poder devolver `NO_BET` por s� mismo con reason `"drift_alert"`, no esperar a que el operador bloquee manualmente.
- **Separar datos hist�ricos (DB fuente) de la DB de predicciones/auditor�a**. Hoy dependemos de un �nico SQLite.
- **Simular bankroll evolution** en el propio test-roi, no solo hit rate y ROI. Un ROI +5% con drawdowns de -15u es peor operativamente que +3% con drawdowns de -5u.

Ninguna de estas es un problema bloqueante hoy; son optimizaciones para cuando el modelo ya est� operando con confianza.