"""Poisson goal-scoring model with optional Dixon-Coles low-score correction
and exponential time-decay weighting.

Base model (Maher 1982 / the standard "double Poisson regression" trick):
two rows per match (one for each side's goal count), with team = attacking
side, opponent = defending side, is_home flagging which row is the home
side. A Poisson GLM on `goals ~ is_home + C(team) + C(opponent)` recovers
attack strength (team dummy), defensive weakness (opponent dummy) and home
advantage (is_home) jointly.

Dixon-Coles correction (Dixon & Coles, 1997): plain independent Poisson
underestimates low-score draws (0-0, 1-1) and slightly misjudges 1-0/0-1.
A single correlation parameter rho, fit by maximizing weighted
log-likelihood with lambda/mu held fixed from the base GLM, corrects the
four low-score cells multiplicatively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import config


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": 1,
            "goals": matches["home_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,
            "goals": matches["away_goals"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


class PoissonGoalModel:
    def __init__(self, xi: float = config.DIXON_COLES_XI, use_dixon_coles: bool = True):
        self.xi = xi
        self.use_dixon_coles = use_dixon_coles
        self.results = None
        self.rho = 0.0

    def fit(self, matches: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> "PoissonGoalModel":
        long_df = _long_format(matches)

        if as_of_date is not None:
            days = (as_of_date - long_df["date"]).dt.days.clip(lower=0)
            weights = np.exp(-self.xi * days)
        else:
            weights = np.ones(len(long_df))

        self.known_teams = set(long_df["team"]) | set(long_df["opponent"])

        glm = smf.glm(
            "goals ~ is_home + C(team) + C(opponent)",
            data=long_df,
            family=sm.families.Poisson(),
            var_weights=weights,
        )
        try:
            self.results = glm.fit()
            if not np.all(np.isfinite(self.results.params)):
                raise ValueError("non-finite GLM params")
        except ValueError:
            # A competition with many teams relative to matches per team
            # (e.g. Champions League: ~100 distinct clubs across 3 seasons
            # of 8-match league phases, many appearing in only one season)
            # can quasi-separate the per-team dummies and blow up plain
            # MLE. A small ridge penalty (alpha swept empirically against
            # this exact failure: 0.001 keeps a real, finite team-strength
            # spread; 1.0 crushes it to near-zero) fixes it without
            # affecting leagues that never hit this path.
            self.results = glm.fit_regularized(alpha=0.001, L1_wt=0.0)

        # A team unseen in this training window (e.g. promoted after the
        # earliest season in the fold) has no fitted attack/defense dummy,
        # and patsy raises on an unknown category rather than silently
        # ignoring it. Extracting the coefficients lets predict_lambdas(_bulk)
        # fall back to the league-average attack/defense effect for such a
        # team instead of crashing or returning NaN for its entire season.
        params = self.results.params
        self.intercept = float(params.get("Intercept", 0.0))
        self.is_home_coef = float(params.get("is_home", 0.0))
        self.team_effects = {t: float(params.get(f"C(team)[T.{t}]", 0.0)) for t in self.known_teams}
        self.opponent_effects = {t: float(params.get(f"C(opponent)[T.{t}]", 0.0)) for t in self.known_teams}
        self.avg_team_effect = float(np.mean(list(self.team_effects.values())))
        self.avg_opponent_effect = float(np.mean(list(self.opponent_effects.values())))

        if self.use_dixon_coles:
            self._fit_rho(matches, weights[: len(matches)])
        return self

    def _expected_goals(self, attacker: str, defender: str, is_home: int) -> float:
        team_eff = self.team_effects.get(attacker, self.avg_team_effect)
        opp_eff = self.opponent_effects.get(defender, self.avg_opponent_effect)
        eta = self.intercept + self.is_home_coef * is_home + team_eff + opp_eff
        return float(np.exp(eta))

    def _fit_rho(self, matches: pd.DataFrame, home_weights: np.ndarray) -> None:
        lam, mu = self.predict_lambdas_bulk(matches["home_team"], matches["away_team"])
        x = matches["home_goals"].to_numpy()
        y = matches["away_goals"].to_numpy()

        def neg_log_lik(rho: float) -> float:
            taus = np.array([_dc_tau(xi_, yi_, li_, mi_, rho) for xi_, yi_, li_, mi_ in zip(x, y, lam, mu)])
            taus = np.clip(taus, 1e-10, None)
            return -float(np.sum(home_weights * np.log(taus)))

        res = minimize_scalar(neg_log_lik, bounds=(-1.0, 1.0), method="bounded")
        self.rho = float(res.x)

    def predict_lambdas(self, home: str, away: str) -> tuple[float, float]:
        lam = self._expected_goals(home, away, is_home=1)
        mu = self._expected_goals(away, home, is_home=0)
        return lam, mu

    def predict_lambdas_bulk(self, home_teams: pd.Series, away_teams: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        home_teams = pd.Series(home_teams).reset_index(drop=True)
        away_teams = pd.Series(away_teams).reset_index(drop=True)
        lam = np.array([self._expected_goals(h, a, 1) for h, a in zip(home_teams, away_teams)])
        mu = np.array([self._expected_goals(a, h, 0) for h, a in zip(home_teams, away_teams)])
        return lam, mu

    def score_grid(
        self, home: str, away: str, max_goals: int = 6, lam: float | None = None, mu: float | None = None
    ) -> np.ndarray:
        """Returns a (max_goals+1) x (max_goals+1) grid of P(home=i, away=j).

        Pass precomputed lam/mu (e.g. from predict_lambdas_bulk) to skip a
        redundant per-row statsmodels .predict() call when scoring many
        fixtures at once.
        """
        if lam is None or mu is None:
            lam, mu = self.predict_lambdas(home, away)
        home_pmf = poisson.pmf(np.arange(max_goals + 1), lam)
        away_pmf = poisson.pmf(np.arange(max_goals + 1), mu)
        grid = np.outer(home_pmf, away_pmf)

        if self.use_dixon_coles and self.rho != 0.0:
            for x in range(2):
                for y in range(2):
                    grid[x, y] *= _dc_tau(x, y, lam, mu, self.rho)

        grid /= grid.sum()
        return grid

    def match_probabilities(
        self, home: str, away: str, max_goals: int = 6, lam: float | None = None, mu: float | None = None
    ) -> tuple[float, float, float]:
        """Returns (P(home win), P(draw), P(away win))."""
        grid = self.score_grid(home, away, max_goals, lam=lam, mu=mu)
        p_home = float(np.tril(grid, -1).sum())
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())
        return p_home, p_draw, p_away
