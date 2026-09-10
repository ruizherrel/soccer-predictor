"""Streamlit app: pick two Premier League teams, see 1X2 probabilities and a
predicted scoreline heatmap.

Run with: streamlit run src/soccer_predictor/app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from soccer_predictor import dataset, ingest, xgb_model

st.set_page_config(page_title="Predictor de fútbol", page_icon="⚽")
st.title("⚽ Predictor de partidos")

# Bigger text for the team/league dropdowns (both the closed selectbox and
# its open option list) — data-baseweb is a stable attribute of the
# underlying widget library, unlikely to shift across Streamlit versions
# the way generated class names do.
st.markdown(
    """
    <style>
    div[data-baseweb="select"] * { font-size: 1.15rem !important; }
    ul[role="listbox"] li { font-size: 1.15rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def _load_matches(league: str):
    return ingest.load_matches(league)


@st.cache_resource
def _load_model(league: str):
    return xgb_model.load_model(league)


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


league = st.selectbox(
    "Liga",
    options=list(config.LEAGUES),
    format_func=lambda code: config.LEAGUES[code]["name"],
)

notice = config.LEAGUES[league].get("notice")
if notice:
    st.warning(notice)

matches = _load_matches(league)
teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Equipo local", teams, index=0)
with col2:
    away_team = st.selectbox("Equipo visitante", teams, index=min(1, len(teams) - 1))

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
    st.success(f"📅 Partido real programado para el {fixture_date.strftime('%d/%m/%Y')}.")
else:
    st.caption(
        "No encontramos un partido programado próximamente entre estos dos equipos según nuestros "
        "datos — esta es una predicción hipotética con la forma y el rating más recientes disponibles, "
        "no necesariamente su próximo partido real."
    )

if st.button("Predecir", type="primary"):
    with st.spinner("Calculando ratings y probabilidades..."):
        live_row, poisson_model = dataset.build_live_features(matches, home_team, away_team, league)
        model, feature_columns = _load_model(league)
        probs = xgb_model.predict_proba(model, live_row, feature_columns)[0]  # (away, draw, home)
        p_away, p_draw, p_home = probs

    st.subheader("Probabilidades 1X2")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Gana {home_team}", f"{p_home:.1%}")
    m2.metric("Empate", f"{p_draw:.1%}")
    m3.metric(f"Gana {away_team}", f"{p_away:.1%}")

    # Draw being the single most-likely outcome (beating both home and away)
    # is genuinely rare in this model — checked empirically across ~300
    # matchups for E0, it essentially never happens even in close games,
    # since home/away probabilities are rarely both below it at once. A
    # league-relative threshold (comfortably above that league's own
    # historical draw rate) actually fires for real "unusually draw-prone"
    # matchups instead.
    league_draw_rate = matches["result"].eq("D").mean()
    if p_draw > league_draw_rate + 0.05:
        st.info(
            f"⚖️ Este partido tiene una probabilidad de empate notablemente alta ({p_draw:.1%} vs. "
            f"{league_draw_rate:.1%} de tasa histórica en esta liga) — el empate es el resultado más "
            "difícil de acertar, pero también el que más suele subestimar el mercado de apuestas."
        )

    fig_bar = go.Figure(
        go.Bar(
            x=[f"Gana {home_team}", "Empate", f"Gana {away_team}"],
            y=[p_home, p_draw, p_away],
            marker_color=["#2ca02c", "#7f7f7f", "#d62728"],
            text=[f"{v:.1%}" for v in [p_home, p_draw, p_away]],
            textposition="auto",
        )
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
    st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Marcador más probable (modelo Poisson)")
    max_goals = 5
    grid = poisson_model.score_grid(home_team, away_team, max_goals=max_goals)
    fig_heat = go.Figure(
        go.Heatmap(
            z=grid,
            x=[str(i) for i in range(max_goals + 1)],
            y=[str(i) for i in range(max_goals + 1)],
            colorscale="Blues",
            text=np.round(grid * 100, 1),
            texttemplate="%{text}%",
        )
    )
    fig_heat.update_layout(
        xaxis_title=f"Goles {away_team}",
        yaxis_title=f"Goles {home_team}",
        height=450,
    )
    st.plotly_chart(fig_heat, width="stretch")

    top_idx = np.unravel_index(np.argmax(grid), grid.shape)
    st.caption(f"Marcador más probable: {home_team} {top_idx[0]} - {top_idx[1]} {away_team}")
