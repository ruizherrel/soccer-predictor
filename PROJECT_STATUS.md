# Estado del proyecto — soccer-predictor

Última actualización: 2026-09-10 (+MLB — primer deporte además de fútbol, con selector de deporte en la app; 13 competencias configuradas en total, todas funcionando)

## Qué es

App de Streamlit que predice resultados de fútbol (1X2: local/empate/visitante) y, desde ahora, de béisbol MLB (local/visitante) usando Elo, Pi ratings, un modelo Poisson/Dixon-Coles (solo fútbol) y un clasificador XGBoost entrenado por liga/competencia. Desplegada en Streamlit Community Cloud, código en GitHub.

- **Repo**: https://github.com/ruizherrel/soccer-predictor (público)
- **App en vivo**: `https://soccer-predictor-c4fjnwvjqjuz2rci5kcdxc.streamlit.app/`
- **Redeploy**: automático en cada push a `master`. A veces solo hace un "hot reload" y no un reinicio completo del proceso — si tras un push a `config.py` o `ingest.py` sale `AttributeError: module '...' has no attribute '...'`, usa **Manage app → Reboot app** en Streamlit Cloud.

## Ligas soportadas

| Liga | Código | Estado | Fuente de datos |
|---|---|---|---|
| Premier League (Inglaterra) | E0 | ✅ Funcionando | football-data.co.uk (o espejo GitHub `datasets/football-datasets` si el sitio está caído) |
| La Liga (España) | SP1 | ✅ Funcionando | igual que arriba |
| Bundesliga (Alemania) | D1 | ✅ Funcionando | igual que arriba |
| Serie A (Italia) | I1 | ✅ Funcionando | igual que arriba |
| Ligue 1 (Francia) | F1 | ✅ Funcionando | igual que arriba |
| Liga MX (México) | MEX | ✅ Funcionando, datos vigentes | TheSportsDB (API gratuita, key `123`), no football-data.co.uk |
| MLS (Estados Unidos) | MLS | ✅ Funcionando, datos vigentes | TheSportsDB |
| Eredivisie (Holanda) | N1 | ✅ Funcionando, datos vigentes | TheSportsDB (football-data.co.uk tiene el certificado SSL roto, ver abajo) |
| Primeira Liga (Portugal) | P1 | ✅ Funcionando, datos vigentes | TheSportsDB (igual que Holanda) |
| UEFA Champions League | UCL | ✅ Funcionando, con limitaciones (ver abajo) | TheSportsDB (football-data.co.uk no cubre competencias continentales) |
| UEFA Conference League | UECL | ✅ Funcionando, con limitaciones (ver abajo) | TheSportsDB |
| Liga de Expansión MX (México, 2ª división) | MX2 | ✅ Funcionando, datos vigentes | TheSportsDB |
| Copa Libertadores | LIB | ✅ Funcionando, con limitaciones (ver abajo) | TheSportsDB |
| MLB (Béisbol, Estados Unidos) | MLB | ✅ Funcionando, con datos parciales (ver abajo) | TheSportsDB |

**football-data.co.uk sigue con el certificado SSL roto** (verificado independientemente con curl/WebFetch/requests, sin cambios desde que se armó el proyecto) — por eso México, MLS, Holanda y Portugal usan TheSportsDB en su lugar. E0/SP1/D1/I1/F1 siguen funcionando porque tienen espejo de respaldo en GitHub (`datasets/football-datasets`); si ese espejo alguna vez falla también, esas 5 quedarían igual de bloqueadas hasta que football-data.co.uk se arregle.

**Limitaciones de las competencias continentales (UCL, UECL, LIB)**: solo cubren la fase de grupos/liga (jornadas 1-8 en Champions, 1-6 en Conference y Libertadores), no la eliminatoria posterior — el aviso aparece en la app. En Libertadores específicamente, las rondas de eliminación directa usan una numeración no secuencial en la fuente de datos (ej. octavos = jornada 16, cuartos = jornada 125, sin patrón que las conecte) que no encaja con el mecanismo de descarga que usa este proyecto en todas las demás ligas, así que ni siquiera se intentó. **UEFA Europa League no se pudo agregar**: no tiene datos utilizables en TheSportsDB bajo ningún ID encontrado (se probó por nombre exacto, por ID, por 3 formatos de temporada, y por eventos próximos — todo vacío), a diferencia de Champions, Conference y Libertadores que sí funcionan. Es un hueco real de su cobertura gratuita, no arreglable desde este código.

**Ajustes numéricos para competencias con pocos datos por equipo**: con pocos partidos por equipo y un grupo de clubes que cambia cada año por clasificación (Champions/Conference: ~100-180 clubes en 3 temporadas; Libertadores: 82 clubes en 510 partidos), el ajuste Poisson normal en `poisson_model.py` puede fallar de dos formas distintas, ambas ya vistas en producción: (1) diverge directamente (`ValueError`/parámetros no finitos) — resuelto con un fallback a regresión regularizada (ridge, alpha=0.001), activo solo cuando el ajuste normal falla; (2) converge pero produce un coeficiente tan extremo que `exp()` desborda a infinito, lo cual XGBoost rechaza de plano al entrenar — resuelto limitando (`clip`) el predictor lineal a ±5 antes de exponenciar (goles esperados entre ~0.007 y ~148, un rango ya absurdamente amplio para cualquier partido real, así que nunca afecta a un valor genuino). Ninguno de los dos cambios afecta a las ligas domésticas (confirmado: reentrenarlas todas después de estos cambios produjo modelos idénticos byte a byte).

**Alias de equipo encontrados y corregidos** (patrón repetido en cada liga vía TheSportsDB: nombra al mismo club distinto entre temporadas — ej. con/sin prefijo "FC"/"SC", con/sin acento, o abreviatura vs nombre completo): ver `MEXICO_TEAM_NAME_MAP`, `UCL_TEAM_NAME_MAP`, `MLS_TEAM_NAME_MAP`, `N1_TEAM_NAME_MAP`, `P1_TEAM_NAME_MAP`, `UECL_TEAM_NAME_MAP`, `MX2_TEAM_NAME_MAP`, `LIB_TEAM_NAME_MAP` en `ingest.py`. Si un futuro refresh muestra más equipos de los reales para alguna liga, revisar si hay un alias nuevo sin mapear antes de confiar en los ratings — la búsqueda automática (`_names_roughly_match`) ayuda pero no es infalible: ya produjo falsos positivos peligrosos (ej. "Inter Milan" contra el "Milan" de Serie A, que es el AC Milan, un club distinto; o "Universidad Católica" de Chile contra la "Universidad Católica del Ecuador", dos clubes reales distintos que coinciden en la misma Libertadores 2026) y falsos negativos (alias sin ninguna subcadena en común, como "TNS"/"The New Saints" o nombres con guión vs espacio).

**Rating cruzado liga doméstica → competencias europeas**: un equipo debutante o que regresa a Champions/Conference League (ej. Roma tras 7 años fuera de Champions) ya no arranca con el rating genérico de "equipo nuevo" — se siembra con su Elo/Pi doméstico más reciente, vía `dataset.CROSS_LEAGUE_TEAM_MAPS` (`UCL_DOMESTIC_TEAM_MAP` y `UECL_DOMESTIC_TEAM_MAP`), cruzado contra las 7 ligas domésticas que cubrimos. Equipos de ligas que no cubrimos (Noruega, Turquía, Kazajistán, etc.) siguen sin este respaldo — limitación conocida, no un bug. `EloRatingSystem`/`PiRatingSystem` aceptan un `seed_ratings` opcional por equipo (vacío por defecto, sin efecto en ninguna otra liga).

**MLB (béisbol) — primer deporte no-fútbol**: la app ahora tiene un selector de deporte (⚽ Fútbol / ⚾ Béisbol) arriba del selector de liga, que filtra las opciones según el nuevo campo `"sport"` en `config.LEAGUES`. MLB reusa toda la arquitectura existente (Elo, Pi, clasificador XGBoost de 3 clases) en vez de un pipeline binario aparte, con estas diferencias reales:
- **Hiperparámetros de Elo propios por liga** (antes eran constantes globales usadas por todas): `elo_k=4` y `elo_home_adv=24` para MLB en vez de los `20`/`80` de fútbol (metodología pública de Elo de MLB de FiveThirtyEight — un partido de MLB es mucho menos informativo dado que hay ~162 partidos/temporada, y la ventaja de local es menor). `use_dixon_coles=False` porque esa corrección es específica de la tendencia del fútbol a subestimar empates bajos. `build_ratings_features`/`build_poisson_features`/`build_live_features` en `dataset.py` resuelven estos overrides desde `config.LEAGUES[league]`, con los globales como default — confirmado que no afecta a ninguna otra liga (reentrenar las 12 restantes produjo modelos idénticos byte a byte, **excepto Copa Libertadores** — ver nota abajo, es un problema preexistente no relacionado).
- **El béisbol nunca empata**, y esto rompía el entrenamiento de raíz: el `XGBClassifier` de sklearn (xgboost 3.4.1) exige que las 3 clases (visitante/empate/local) aparezcan al menos una vez en los datos de entrenamiento, y MLB nunca tiene ejemplos de la clase "empate". Se resolvió en `xgb_model._ensure_all_classes_present` agregando una fila sintética con esa clase y `sample_weight=0` (no influye en el ajuste en absoluto, solo satisface la validación de sklearn) — es un no-op para cualquier liga que ya tenga las 3 clases, que hoy es cualquier liga que no sea MLB. Verificado en vivo: la probabilidad de empate que predice el modelo para partidos reales de MLB sale ~0.00001%, no exactamente 0 pero indistinguible en la práctica.
- **Datos incompletos, limitación real de la fuente gratuita** (aviso visible en la app): TheSportsDB solo tiene historial de MLB desde 2025 (2022-2024 no tienen partidos), y su endpoint por jornada recorta cada semana a 50 partidos cuando la liga completa juega ~90/semana — se probó el endpoint por fecha como alternativa y resultó todavía más limitado (tope de exactamente 3 eventos/día). No hay forma gratuita de traer el calendario completo. 2371 partidos descargados (2025+2026), 30 equipos reales, sin alias de nombre encontrados (a diferencia de cada otra liga de TheSportsDB agregada hasta ahora).
- **Hallazgo separado, no relacionado con MLB**: reentrenar Copa Libertadores dos veces seguidas con el código sin cambios produce modelos distintos (no bytes idénticos) — su ajuste Poisson ya emite `SingularMatrixWarning` ("los parámetros del modelo no están determinados de forma única"), así que el resultado exacto depende de detalles no deterministas del solver (probablemente el mismo tipo de problema documentado arriba en "Ajustes numéricos para competencias con pocos datos por equipo", pero esta vez afectando reproducibilidad en vez de crashear). No es un bug introducido esta sesión — ya existía, solo que nadie había reentrenado esa liga dos veces seguidas para notarlo. No se investigó más a fondo por estar fuera del alcance de la tarea de MLB.

## Funcionalidades agregadas

- **Selector de deporte y liga** en la app: primero se elige el deporte (⚽ Fútbol / ⚾ Béisbol), luego la liga dentro de ese deporte — cada una con modelo y datos independientes (no se mezclan escalas de fuerza entre ligas ni entre deportes). Para béisbol la UI se ajusta automáticamente: solo 2 columnas de probabilidad (sin "Empate"), sin el aviso de empate probable, y una cuadrícula de marcador más ancha etiquetada en "carreras" en vez de "goles".
- **Altitud y distancia de viaje** (solo México, `src/soccer_predictor/geo.py`): captura el efecto real de jugar en Ciudad de México (~2240 msnm) contra equipos de costa. No se hizo para las ligas europeas por baja variación de altitud ahí.
- **Detección de valor (+EV) con Criterio de Kelly** (`src/soccer_predictor/value_betting.py`), corrido dentro de `scripts/run_backtest.py`: compara las probabilidades del modelo contra las cuotas de cierre históricas de Bet365 (cuando están disponibles). Ahora mismo no muestra nada útil porque ninguna liga tiene columnas de cuotas (todas se obtuvieron del espejo de GitHub sin cuotas, mientras football-data.co.uk sigue caído) — funcionará solo en cuanto se refresque una liga desde la fuente primaria.
- **Detección de próximo partido real** (`ingest.fetch_upcoming_fixtures`, vía TheSportsDB): la app avisa si el partido elegido es uno programado de verdad — con fecha **y hora en horario del centro de México** (America/Mexico_City) — o si es una predicción hipotética. Confiable al 100% en las ligas que ya vienen de TheSportsDB (México, MLS, Holanda, Portugal, Champions/Conference League); para las que vienen de football-data.co.uk usa coincidencia de nombres conservadora (solo si es igual o substring después de quitar acentos) — puede no encontrar coincidencia por diferencias de nombre entre fuentes, pero nunca muestra un partido equivocado. Nunca puede tumbar la predicción principal aunque TheSportsDB falle (ya pasó en producción con un 503).
- **Aviso de empate probable**: si la probabilidad de empate supera en 5+ puntos porcentuales la tasa histórica de empates de esa liga, se muestra una nota (probado empíricamente: exigir que el empate sea la opción más probable de las tres casi nunca se cumple, por eso el umbral es relativo a la tasa histórica).

## Problemas de entorno resueltos

- **McAfee bloquea el desarrollo local**: su protección web/firewall corta el WebSocket de Streamlit en el navegador (Firefox, Chrome, VS Code) tras la carga inicial. El usuario no tiene permisos de administrador para agregar la excepción. Mientras tanto, verificar cambios en la app desplegada, no en localhost.
- **MKL roto en el entorno conda de esta máquina**: cualquier operación de álgebra lineal (hasta un SVD trivial de numpy) tronaba el proceso sin traceback en esta CPU (Intel i7-9750H). Se resolvió cambiando el backend a OpenBLAS: `conda install -n soccer-predictor -c conda-forge -y "libblas=*=*openblas" "liblapack=*=*openblas" "libcblas=*=*openblas"`.
- **Bug de espejo de GitHub**: los nombres de carpeta usados como respaldo (`spanish-la-liga`, etc.) no coincidían con los reales (`la-liga`, etc.) — nunca se había notado porque football-data.co.uk siempre había funcionado. Corregido.

## Presupuesto investigado para lo que falta (no implementado, requiere pago)

| Necesitas | Costo mínimo real |
|---|---|
| xG básico + lesiones (Sportmonks, 5-8 ligas) | ~€30-100/mes |
| Datos de evento completos (xT, VAEP, redes de pases — StatsBomb/Opta) | €50,000+/año, contrato enterprise |
| Carga física/GPS de equipos rivales | No es un producto comprable a terceros (dato propietario de cada club) |

## Cómo correr el pipeline completo

```cmd
cd C:\Users\manuelruiz\soccer-predictor
"C:\Users\manuelruiz\AppData\Local\anaconda3\envs\soccer-predictor\python.exe" scripts\refresh_data.py
"C:\Users\manuelruiz\AppData\Local\anaconda3\envs\soccer-predictor\python.exe" scripts\train.py
"C:\Users\manuelruiz\AppData\Local\anaconda3\envs\soccer-predictor\python.exe" scripts\run_backtest.py
```
Cada script acepta `--league <CODIGO>` para correr una sola liga.
