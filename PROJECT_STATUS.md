# Estado del proyecto — soccer-predictor

Última actualización: 2026-09-10 (+Holanda, +Portugal, +Conference League — las 10 ligas configuradas ya funcionan)

## Qué es

App de Streamlit que predice resultados 1X2 (local/empate/visitante) de fútbol usando Elo, Pi ratings, un modelo Poisson/Dixon-Coles y un clasificador XGBoost entrenado por liga. Desplegada en Streamlit Community Cloud, código en GitHub.

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

**football-data.co.uk sigue con el certificado SSL roto** (verificado independientemente con curl/WebFetch/requests, sin cambios desde que se armó el proyecto) — por eso México, MLS, Holanda y Portugal usan TheSportsDB en su lugar. E0/SP1/D1/I1/F1 siguen funcionando porque tienen espejo de respaldo en GitHub (`datasets/football-datasets`); si ese espejo alguna vez falla también, esas 5 quedarían igual de bloqueadas hasta que football-data.co.uk se arregle.

**Limitaciones de Champions League y Conference League**: ambas solo cubren la fase de liga del formato suizo (jornadas 1-8 en Champions, 1-6 en Conference), no la eliminatoria posterior — el aviso aparece en la app. **UEFA Europa League no se pudo agregar**: no tiene datos utilizables en TheSportsDB bajo ningún ID encontrado (se probó por nombre exacto, por ID, por 3 formatos de temporada, y por eventos próximos — todo vacío), a diferencia de Champions y Conference que sí funcionan. Es un hueco real de su cobertura gratuita, no arreglable desde este código. Con pocos partidos por equipo y un grupo de clubes que cambia cada año por clasificación, el ajuste Poisson normal diverge numéricamente en ambas — resuelto con un fallback a regresión regularizada (ridge, alpha=0.001) en `poisson_model.py`, activo solo cuando el ajuste normal falla.

**Alias de equipo encontrados y corregidos** (patrón repetido en cada liga vía TheSportsDB: nombra al mismo club distinto entre temporadas — ej. con/sin prefijo "FC"/"SC", con/sin acento, o abreviatura vs nombre completo): ver `MEXICO_TEAM_NAME_MAP`, `UCL_TEAM_NAME_MAP`, `MLS_TEAM_NAME_MAP`, `N1_TEAM_NAME_MAP`, `P1_TEAM_NAME_MAP`, `UECL_TEAM_NAME_MAP` en `ingest.py`. Si un futuro refresh muestra más equipos de los reales para alguna liga, revisar si hay un alias nuevo sin mapear antes de confiar en los ratings — la búsqueda automática (`_names_roughly_match`) ayuda pero no es infalible: ya produjo falsos positivos peligrosos (ej. "Inter Milan" contra el "Milan" de Serie A, que es el AC Milan, un club distinto) y falsos negativos (alias sin ninguna subcadena en común, como "TNS"/"The New Saints" o nombres con guión vs espacio).

**Rating cruzado liga doméstica → competencias europeas**: un equipo debutante o que regresa a Champions/Conference League (ej. Roma tras 7 años fuera de Champions) ya no arranca con el rating genérico de "equipo nuevo" — se siembra con su Elo/Pi doméstico más reciente, vía `dataset.CROSS_LEAGUE_TEAM_MAPS` (`UCL_DOMESTIC_TEAM_MAP` y `UECL_DOMESTIC_TEAM_MAP`), cruzado contra las 7 ligas domésticas que cubrimos. Equipos de ligas que no cubrimos (Noruega, Turquía, Kazajistán, etc.) siguen sin este respaldo — limitación conocida, no un bug. `EloRatingSystem`/`PiRatingSystem` aceptan un `seed_ratings` opcional por equipo (vacío por defecto, sin efecto en ninguna otra liga).

## Funcionalidades agregadas

- **Selector de liga** en la app, con modelo y datos independientes por liga (no se mezclan escalas de fuerza entre ligas).
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
