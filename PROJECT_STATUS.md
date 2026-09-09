# Estado del proyecto — soccer-predictor

Última actualización: 2026-09-09 (sesión ampliada: +Champions League, +MLS)

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
| MLS (Estados Unidos) | MLS | ✅ Funcionando, datos vigentes | TheSportsDB (football-data.co.uk tiene el mismo certificado SSL roto para su endpoint de MLS) |
| UEFA Champions League | UCL | ✅ Funcionando, con limitaciones (ver abajo) | TheSportsDB (football-data.co.uk no cubre competencias continentales) |
| Eredivisie (Holanda) | N1 | ⚠️ Pendiente | football-data.co.uk (sin espejo de respaldo configurado) |
| Primeira Liga (Portugal) | P1 | ⚠️ Pendiente | football-data.co.uk (sin espejo de respaldo configurado) |

**Limitaciones de Champions League**: solo cubre la fase de liga (jornadas 1-8 del formato suizo desde 2024-25), no la eliminatoria (octavos en adelante) — el aviso aparece en la app. Con solo 8 partidos/equipo por temporada y un grupo de clubes que cambia cada año por clasificación (~100 equipos distintos en 3 temporadas, muchos de una sola aparición), el ajuste Poisson normal diverge numéricamente para esta liga — se resolvió con un fallback a regresión regularizada (ridge, alpha=0.001) en `poisson_model.py`, activo solo cuando el ajuste normal falla, sin afectar a las demás ligas.

**Alias de equipo encontrados y corregidos** (mismo patrón que México: TheSportsDB nombra al mismo club distinto entre temporadas): Champions League ("Atletico Madrid"/"Atlético Madrid", "Paris SG"/"Paris Saint-Germain") y MLS ("New York City"/"New York City FC", "Seattle Sounders"/"Seattle Sounders FC") — ver `UCL_TEAM_NAME_MAP` y `MLS_TEAM_NAME_MAP` en `ingest.py`. Si un futuro refresh muestra más equipos de los reales, revisar si hay un alias nuevo sin mapear antes de confiar en los ratings.

**Pendiente de Holanda/Portugal**: football-data.co.uk tiene el certificado SSL roto desde que se armó este proyecto (verificado independientemente con curl/WebFetch/requests). En cuanto se recupere:
```
python scripts/refresh_data.py --league N1
python scripts/refresh_data.py --league P1
python scripts/train.py --league N1
python scripts/train.py --league P1
```
y luego commit + push.

## Funcionalidades agregadas

- **Selector de liga** en la app, con modelo y datos independientes por liga (no se mezclan escalas de fuerza entre ligas).
- **Altitud y distancia de viaje** (solo México, `src/soccer_predictor/geo.py`): captura el efecto real de jugar en Ciudad de México (~2240 msnm) contra equipos de costa. No se hizo para las ligas europeas por baja variación de altitud ahí.
- **Detección de valor (+EV) con Criterio de Kelly** (`src/soccer_predictor/value_betting.py`), corrido dentro de `scripts/run_backtest.py`: compara las probabilidades del modelo contra las cuotas de cierre históricas de Bet365 (cuando están disponibles). Ahora mismo no muestra nada útil porque ninguna liga tiene columnas de cuotas (todas se obtuvieron del espejo de GitHub sin cuotas, mientras football-data.co.uk sigue caído) — funcionará solo en cuanto se refresque una liga desde la fuente primaria.
- **Detección de próximo partido real** (`ingest.fetch_upcoming_fixtures`, vía TheSportsDB): la app avisa si el partido elegido es uno programado de verdad, o si es una predicción hipotética. Confiable al 100% en México (misma fuente que ya usamos); para las ligas europeas usa coincidencia de nombres conservadora (solo si es igual o substring después de quitar acentos) — puede no encontrar coincidencia por diferencias de nombre entre fuentes, pero nunca muestra un partido equivocado.
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
