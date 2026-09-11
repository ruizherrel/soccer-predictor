"""Streamlit app: pick two Premier League teams, see 1X2 probabilities and a
predicted scoreline heatmap.

Run with: streamlit run src/soccer_predictor/app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

import config
from soccer_predictor import dataset, ingest, odds, value_betting, xgb_model

# Spanish labels for the SHAP explanation panel, with {home}/{away}
# placeholders filled in with the actual team names at prediction time.
# Kept in sync by hand with xgb_model.FEATURE_COLUMNS and
# xgb_model.PYTHAGOREAN_FEATURE_COLUMNS; any feature column missing here
# just falls back to its raw name (see _feature_label) instead of crashing.
_FEATURE_LABEL_TEMPLATES = {
    "elo_home": "Rating Elo de {home}",
    "elo_away": "Rating Elo de {away}",
    "elo_diff": "Diferencia de Elo (local − visitante)",
    "pi_home": "Pi-rating de {home}",
    "pi_away": "Pi-rating de {away}",
    "pi_diff": "Diferencia de Pi-rating (local − visitante)",
    "home_ppg_last5": "Puntos por partido de {home} (últimos 5)",
    "home_gf_last5": "{unit} a favor de {home} (últimos 5)",
    "home_ga_last5": "{unit} en contra de {home} (últimos 5)",
    "home_ppg_last10": "Puntos por partido de {home} (últimos 10)",
    "home_gf_last10": "{unit} a favor de {home} (últimos 10)",
    "home_ga_last10": "{unit} en contra de {home} (últimos 10)",
    "home_rest_days": "Días de descanso de {home}",
    "away_ppg_last5": "Puntos por partido de {away} (últimos 5)",
    "away_gf_last5": "{unit} a favor de {away} (últimos 5)",
    "away_ga_last5": "{unit} en contra de {away} (últimos 5)",
    "away_ppg_last10": "Puntos por partido de {away} (últimos 10)",
    "away_gf_last10": "{unit} a favor de {away} (últimos 10)",
    "away_ga_last10": "{unit} en contra de {away} (últimos 10)",
    "away_rest_days": "Días de descanso de {away}",
    "poisson_lambda_home": "{unit} esperados (Poisson) de {home}",
    "poisson_lambda_away": "{unit} esperados (Poisson) de {away}",
    "home_altitude_m": "Altitud del estadio de {home}",
    "altitude_delta_m": "Diferencia de altitud (local − visitante)",
    "away_travel_km": "Distancia de viaje de {away}",
    "pyth_home_pct": "Expectativa Pitagórica de {home}",
    "pyth_away_pct": "Expectativa Pitagórica de {away}",
    "pyth_diff": "Diferencia Pitagórica (local − visitante)",
}


def _feature_label(column: str, home_team: str, away_team: str, unit: str) -> str:
    template = _FEATURE_LABEL_TEMPLATES.get(column, column)
    return template.format(home=home_team, away=away_team, unit=unit)


# Popular nicknames fans actually search for, which don't appear anywhere
# in the underlying data (confirmed live: TheSportsDB has only ever called
# this club "CD Guadalajara" across every Liga MX season on record, never
# "Chivas" -- this isn't a data alias bug, the nickname just doesn't exist
# in any source this project pulls from). Shown in the team dropdowns via
# format_func so typing the nickname finds the team -- Streamlit's
# selectbox search filters the format_func-rendered label sent to the
# frontend, not the underlying option value (confirmed in
# streamlit/elements/widgets/selectbox.py: `selectbox_proto.options[:] =
# formatted_options`), so this isn't just cosmetic. Deliberately limited to
# nicknames unambiguous enough to be confident about; extend as needed.
TEAM_NICKNAMES: dict[str, str] = {
    "CD Guadalajara": "Chivas",
    "América": "Águilas",
    "Cruz Azul": "La Máquina",
    "Monterrey": "Rayados",
    "Tijuana": "Xolos",
    "Pachuca": "Tuzos",
    "Toluca": "Diablos Rojos",
}


def _team_display_name(name: str) -> str:
    nickname = TEAM_NICKNAMES.get(name)
    return f"{name} ({nickname})" if nickname else name


# Below this market-implied probability, this model has been observed
# (live testing, 2026-09-11: Real Madrid vs. Vallecano's away win at 4.3%
# market-implied showed a model probability of 11%; Levante vs. Barcelona's
# home win at 6.3% showed 23.6%; Chelsea vs. Hull's away win at 8.5% showed
# 22.3%) to substantially overestimate a big underdog's chances relative to
# the market -- a known weakness of tree ensembles on rare/extreme outcomes
# (few blowout examples in training data). A large "edge" entirely driven
# by this effect is much more likely to be model miscalibration than real
# value, so it's excluded from the value-bet highlight below (but still
# shown in the table for transparency).
MIN_MARKET_PROB_FOR_VALUE_ALERT = 0.15

st.set_page_config(page_title="Predictor de fútbol", page_icon="⚽")
st.title("⚽ Predictor de partidos")

# Bigger text for the team/league dropdowns (both the closed selectbox and
# its open option list) — data-baseweb is a stable attribute of the
# underlying widget library, unlikely to shift across Streamlit versions
# the way generated class names do.
st.html(
    """
    <style>
    div[data-baseweb="select"] * { font-size: 1.15rem !important; }
    ul[role="listbox"] li { font-size: 1.15rem !important; }
    </style>
    """
)

# Auto-select the current value's text when a dropdown search box gets
# focus, so typing immediately replaces it instead of requiring the user
# to clear it by hand first — most noticeable on mobile, where retyping
# over an existing team name is fiddly. st.html (unlike the deprecated
# st.components.v1.html this replaced) isn't iframed, so this runs
# directly against the app's own `document` — no reaching across a frame
# boundary needed. A document-level "focusin" listener survives
# Streamlit's re-renders where a per-widget listener wouldn't.
st.html(
    """
    <script>
    (function () {
        if (document.__teamSelectAllBound) return;
        document.__teamSelectAllBound = true;
        document.addEventListener("focusin", function (e) {
            var el = e.target;
            if (el && el.tagName === "INPUT" && el.closest('div[data-baseweb="select"]')) {
                setTimeout(function () { el.select(); }, 0);
            }
        });
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)


@st.cache_data(ttl=3600)
def _load_matches(league: str):
    return ingest.load_matches(league)


@st.cache_resource
def _load_model(league: str):
    return xgb_model.load_model(league)


@st.cache_resource
def _load_explainer(league: str):
    model, _feature_columns = _load_model(league)
    return shap.TreeExplainer(model)


@st.cache_data(ttl=6 * 3600)
def _load_upcoming_fixtures(league: str):
    # This is a nice-to-have (tells the user whether the matchup is a real
    # scheduled fixture) that depends on a third-party API TheSportsDB) with
    # no uptime guarantee — seen live returning 503s. It must never take
    # down the actual prediction below it, so any failure here just means
    # "couldn't check" rather than a crashed page.
    try:
        return ingest.fetch_upcoming_fixtures(league)
    except Exception:
        logging.getLogger(__name__).exception("fetch_upcoming_fixtures failed for %s", league)
        return pd.DataFrame(columns=["date", "home_team", "away_team"])


@st.cache_data(ttl=6 * 3600)
def _load_odds(league: str):
    # odds.fetch_odds already never raises internally (see its docstring),
    # but the same "third-party API must never take down the prediction"
    # principle applies here as for _load_upcoming_fixtures above.
    return odds.fetch_odds(league)


SPORT_OPTIONS = {"⚽ Fútbol": "soccer", "⚾ Béisbol": "baseball"}
sport_label = st.selectbox("Deporte", options=list(SPORT_OPTIONS))
sport = SPORT_OPTIONS[sport_label]

league_options = [code for code in config.LEAGUES if config.LEAGUES[code]["sport"] == sport]
league = st.selectbox(
    "Liga",
    options=league_options,
    format_func=lambda code: config.LEAGUES[code]["name"],
)

notice = config.LEAGUES[league].get("notice")
if notice:
    # A collapsed expander instead of a permanently-open warning box: on
    # mobile, the full-height box was pushing the team dropdowns down far
    # enough that the on-screen keyboard (opened by the dropdown's
    # search-to-filter field) covered most of the option list.
    with st.expander("⚠️ Limitaciones de esta competencia"):
        st.write(notice)

matches = _load_matches(league)
teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Equipo local", teams, index=0, format_func=_team_display_name)
with col2:
    away_team = st.selectbox(
        "Equipo visitante", teams, index=min(1, len(teams) - 1), format_func=_team_display_name
    )

if home_team == away_team:
    st.warning("Elige dos equipos distintos.")
    st.stop()

if not config.model_path(league).exists():
    st.error(
        "No se encontró un modelo entrenado para esta liga. Corre primero "
        "`python scripts/train.py` desde la carpeta del proyecto."
    )
    st.stop()

with st.spinner("Buscando si hay un partido real programado..."):
    upcoming_fixtures = _load_upcoming_fixtures(league)
    fixture_date = (
        ingest.find_upcoming_fixture(upcoming_fixtures, home_team, away_team)
        if not upcoming_fixtures.empty
        else None
    )

if fixture_date is not None:
    mexico_time = fixture_date.tz_convert(ZoneInfo("America/Mexico_City"))
    st.success(
        f"📅 Partido real programado para el {mexico_time.strftime('%d/%m/%Y')} a las "
        f"{mexico_time.strftime('%H:%M')} (hora del centro de México)."
    )
else:
    st.caption(
        "No encontramos un partido programado próximamente entre estos dos equipos según nuestros "
        "datos — esta es una predicción hipotética con la forma y el rating más recientes disponibles, "
        "no necesariamente su próximo partido real."
    )

if st.button("Predecir", type="primary"):
    is_baseball = sport == "baseball"
    unit = "Carreras" if is_baseball else "Goles"

    with st.spinner("Calculando ratings y probabilidades..."):
        live_row, poisson_model = dataset.build_live_features(matches, home_team, away_team, league)
        model, feature_columns = _load_model(league)
        probs = xgb_model.predict_proba(model, live_row, feature_columns)[0]  # (away, draw, home)
        p_away, p_draw, p_home = probs

    if is_baseball:
        # MLB never draws, so the 3-class classifier's draw output should
        # come out ~0 on real data (verified empirically before shipping
        # this) — no separate binary model needed, just no "Empate" column
        # shown since it would always read ~0%.
        st.subheader("Probabilidad de victoria")
        m1, m2 = st.columns(2)
        m1.metric(f"Gana {home_team}", f"{p_home:.1%}")
        m2.metric(f"Gana {away_team}", f"{p_away:.1%}")
    else:
        st.subheader("Probabilidades 1X2")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Gana {home_team}", f"{p_home:.1%}")
        m2.metric("Empate", f"{p_draw:.1%}")
        m3.metric(f"Gana {away_team}", f"{p_away:.1%}")

        # Draw being the single most-likely outcome (beating both home and
        # away) is genuinely rare in this model — checked empirically across
        # ~300 matchups for E0, it essentially never happens even in close
        # games, since home/away probabilities are rarely both below it at
        # once. A league-relative threshold (comfortably above that league's
        # own historical draw rate) actually fires for real "unusually
        # draw-prone" matchups instead. Doesn't apply to baseball, which has
        # no draws at all.
        league_draw_rate = matches["result"].eq("D").mean()
        if p_draw > league_draw_rate + 0.05:
            st.info(
                f"⚖️ Este partido tiene una probabilidad de empate notablemente alta ({p_draw:.1%} vs. "
                f"{league_draw_rate:.1%} de tasa histórica en esta liga) — el empate es el resultado más "
                "difícil de acertar, pero también el que más suele subestimar el mercado de apuestas."
            )

    if is_baseball:
        bar_x = [f"Gana {home_team}", f"Gana {away_team}"]
        bar_y = [p_home, p_away]
        bar_colors = ["#2ca02c", "#d62728"]
    else:
        bar_x = [f"Gana {home_team}", "Empate", f"Gana {away_team}"]
        bar_y = [p_home, p_draw, p_away]
        bar_colors = ["#2ca02c", "#7f7f7f", "#d62728"]

    fig_bar = go.Figure(
        go.Bar(
            x=bar_x,
            y=bar_y,
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in bar_y],
            textposition="auto",
        )
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
    st.plotly_chart(fig_bar, width="stretch")

    with st.expander("🔍 ¿Por qué esta predicción?"):
        outcome_names = [away_team, "Empate", home_team]  # CLASS_ORDER = (away, draw, home)
        predicted_idx = int(np.argmax(probs))
        predicted_label = outcome_names[predicted_idx]

        explainer = _load_explainer(league)
        shap_exp = explainer(live_row[feature_columns])
        shap_row = shap_exp.values[0, :, predicted_idx]

        imp_df = pd.DataFrame(
            {
                "label": [_feature_label(c, home_team, away_team, unit) for c in feature_columns],
                "shap": shap_row,
            }
        ).dropna(subset=["shap"])
        imp_df["abs_shap"] = imp_df["shap"].abs()
        top_imp = imp_df.sort_values("abs_shap", ascending=False).head(8).sort_values("shap")

        if top_imp.empty:
            st.caption("No hay suficiente información disponible para explicar esta predicción.")
        else:
            st.caption(
                f"Variables que más influyeron en que el modelo prediga **{predicted_label}** "
                "(valores SHAP). Verde = empuja hacia esa predicción, rojo = empuja en contra."
            )
            fig_shap = go.Figure(
                go.Bar(
                    x=top_imp["shap"],
                    y=top_imp["label"],
                    orientation="h",
                    marker_color=["#2ca02c" if v > 0 else "#d62728" for v in top_imp["shap"]],
                )
            )
            fig_shap.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title=f"Impacto en la predicción de '{predicted_label}'",
            )
            st.plotly_chart(fig_shap, width="stretch")

    # Market odds only exist for real scheduled matches -- a hypothetical
    # matchup (fixture_date is None) has nothing to compare against, so
    # this section only ever renders alongside a confirmed real fixture.
    if fixture_date is not None:
        if not odds.is_configured():
            st.caption(
                "💰 Para comparar contra cuotas de mercado y ver apuestas de valor (+EV), configura "
                "`ODDS_API_KEY` (gratis hasta 500 consultas/mes en the-odds-api.com)."
            )
        elif league not in odds.SPORT_KEYS:
            pass  # Genuinely not covered by this provider (e.g. MX2) -- nothing useful to say every time.
        else:
            odds_df = _load_odds(league)
            match_odds = odds.find_match_odds(odds_df, home_team, away_team) if not odds_df.empty else None

            if match_odds is None:
                st.caption("💰 No se encontraron cuotas de mercado para este partido todavía.")
            else:
                model_prob_by_outcome = {"home": p_home, "draw": p_draw, "away": p_away}
                outcomes = [("home", home_team, match_odds["odds_home"]), ("away", away_team, match_odds["odds_away"])]
                if not is_baseball:
                    outcomes.insert(1, ("draw", "Empate", match_odds["odds_draw"]))

                raw_implied = {
                    key: 1.0 / dec_odds
                    for key, _label, dec_odds in outcomes
                    if pd.notna(dec_odds) and dec_odds > 0
                }
                norm_factor = sum(raw_implied.values())

                computed = []
                for key, label, dec_odds in outcomes:
                    if key not in raw_implied:
                        continue
                    model_prob = model_prob_by_outcome[key]
                    market_prob = raw_implied[key] / norm_factor
                    edge = model_prob * dec_odds - 1.0
                    low_confidence = market_prob < MIN_MARKET_PROB_FOR_VALUE_ALERT
                    computed.append(
                        {
                            "label": label,
                            "dec_odds": dec_odds,
                            "market_prob": market_prob,
                            "model_prob": model_prob,
                            "edge": edge,
                            "low_confidence": low_confidence,
                            # Zeroed rather than just unhighlighted -- suggesting a
                            # Kelly stake here would contradict not trusting this edge.
                            "kelly_pct": 0.0 if low_confidence else value_betting.kelly_fraction(
                                model_prob, dec_odds, cap=0.05
                            ),
                        }
                    )

                st.subheader("💰 Valor esperado vs. cuotas del mercado")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Resultado": c["label"],
                                "Cuota": f"{c['dec_odds']:.2f}",
                                "Prob. mercado": f"{c['market_prob']:.1%}",
                                "Prob. modelo": f"{c['model_prob']:.1%}",
                                "Edge": f"{c['edge']:+.1%}",
                                "Apuesta sugerida (Kelly, tope 5%)": (
                                    f"{c['kelly_pct']:.1%}" if c["kelly_pct"] > 0 else "—"
                                ),
                            }
                            for c in computed
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )

                value_bets = [c for c in computed if c["edge"] > 0 and not c["low_confidence"]]
                low_confidence_value_bets = [c for c in computed if c["edge"] > 0 and c["low_confidence"]]
                if value_bets:
                    best = max(value_bets, key=lambda c: c["edge"])
                    st.success(
                        f"✅ Valor detectado en **{best['label']}**: el modelo le da {best['model_prob']:.1%} "
                        f"contra una cuota de {best['dec_odds']:.2f} (el mercado implica {best['market_prob']:.1%}) "
                        f"— edge de {best['edge']:+.1%}."
                    )
                elif low_confidence_value_bets:
                    worst = max(low_confidence_value_bets, key=lambda c: c["edge"])
                    st.warning(
                        f"⚠️ Hay un edge positivo en **{worst['label']}** ({worst['edge']:+.1%}), pero la "
                        f"probabilidad de mercado es muy baja ({worst['market_prob']:.1%}, un desfavorito "
                        "marcado) — en partidos muy desparejos el modelo tiende a sobreestimar al equipo "
                        "débil, así que no se resalta como apuesta de valor."
                    )
                else:
                    st.caption(
                        "Sin valor detectado: el mercado ya iguala o supera la probabilidad del modelo en los "
                        "resultados evaluados."
                    )
                st.caption(
                    "Las cuotas de mercado son un baseline históricamente difícil de vencer — exige un edge "
                    "cómodo, no marginal, antes de apostar, y nunca apuestes más del % sugerido por Kelly."
                )

    st.subheader("Marcador más probable (modelo Poisson)")
    # Baseball teams routinely score well beyond soccer's goal range (MLB
    # teams average ~4-5 runs/game, with double-digit innings not unusual),
    # so the scoreline grid needs to cover a much wider range than soccer's
    # 0-5 to have any real mass past its edges.
    max_scoreline = 12 if is_baseball else 5
    grid = poisson_model.score_grid(home_team, away_team, max_goals=max_scoreline)
    fig_heat = go.Figure(
        go.Heatmap(
            z=grid,
            x=[str(i) for i in range(max_scoreline + 1)],
            y=[str(i) for i in range(max_scoreline + 1)],
            colorscale="Blues",
            text=np.round(grid * 100, 1),
            texttemplate="%{text}%",
        )
    )
    fig_heat.update_layout(
        xaxis_title=f"{unit} {away_team}",
        yaxis_title=f"{unit} {home_team}",
        height=450,
    )
    st.plotly_chart(fig_heat, width="stretch")

    top_idx = np.unravel_index(np.argmax(grid), grid.shape)
    st.caption(f"Marcador más probable: {home_team} {top_idx[0]} - {top_idx[1]} {away_team}")
