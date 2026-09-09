"""Streamlit app: pick two Premier League teams, see 1X2 probabilities and a
predicted scoreline heatmap.

Run with: streamlit run src/soccer_predictor/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from soccer_predictor import dataset, ingest, xgb_model

st.set_page_config(page_title="Predictor de fútbol", page_icon="⚽")
st.title("⚽ Predictor de partidos")


@st.cache_data(ttl=3600)
def _load_matches(league: str):
    return ingest.load_matches(league)


@st.cache_resource
def _load_model(league: str):
    return xgb_model.load_model(league)


league = st.selectbox(
    "Liga",
    options=list(config.LEAGUES),
    format_func=lambda code: config.LEAGUES[code]["name"],
)

stale_notice = config.LEAGUES[league].get("stale_notice")
if stale_notice:
    st.warning(stale_notice)

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

if st.button("Predecir", type="primary"):
    with st.spinner("Calculando ratings y probabilidades..."):
        live_row, poisson_model = dataset.build_live_features(matches, home_team, away_team)
        model, feature_columns = _load_model(league)
        probs = xgb_model.predict_proba(model, live_row, feature_columns)[0]  # (away, draw, home)
        p_away, p_draw, p_home = probs

    st.subheader("Probabilidades 1X2")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Gana {home_team}", f"{p_home:.1%}")
    m2.metric("Empate", f"{p_draw:.1%}")
    m3.metric(f"Gana {away_team}", f"{p_away:.1%}")

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
    st.plotly_chart(fig_bar, use_container_width=True)

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
    st.plotly_chart(fig_heat, use_container_width=True)

    top_idx = np.unravel_index(np.argmax(grid), grid.shape)
    st.caption(f"Marcador más probable: {home_team} {top_idx[0]} - {top_idx[1]} {away_team}")
