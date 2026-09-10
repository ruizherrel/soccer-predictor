"""Pi Ratings: home/away-split team ratings for football.

Formulation: Constantinou & Fenton (2013), "Determining the level of ability
of football teams by dynamic ratings based on the relative discrepancies in
scores between adversaries," Journal of Quantitative Analysis in Sports.
Ratings are zero-centered goal-difference scales: 0 = league average,
+1.0 roughly means "one goal better than average".
"""
from __future__ import annotations

import config


class PiRatingSystem:
    def __init__(
        self,
        lam: float = config.PI_LAMBDA,
        alpha: float = config.PI_ALPHA,
        beta: float = config.PI_BETA,
        promoted_initial: float = config.PI_PROMOTED_INITIAL,
        season_regression: float = config.PI_SEASON_REGRESSION,
        seed_ratings: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.promoted_initial = promoted_initial
        self.season_regression = season_regression
        # Per-team override for a first-ever rating, e.g. a Champions League
        # debutant seeded from its domestic-league Pi rating instead of the
        # generic default — see dataset.py's cross-league seeding. Empty by
        # default, so every other league behaves exactly as before.
        self.seed_ratings = seed_ratings or {}
        # team -> (home_rating, away_rating)
        self.ratings: dict[str, tuple[float, float]] = {}

    def rating(self, team: str) -> tuple[float, float]:
        return self.ratings.get(team, self.seed_ratings.get(team, (0.0, 0.0)))

    def predict_goal_diff(self, home: str, away: str) -> float:
        home_r, _ = self.rating(home)
        _, away_r = self.rating(away)
        return home_r - away_r

    def snapshot(self, home: str, away: str) -> tuple[float, float]:
        """(home team's home rating, away team's away rating) pre-match."""
        home_r, _ = self.rating(home)
        _, away_r = self.rating(away)
        return home_r, away_r

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        home_home_r, home_away_r = self.rating(home)
        away_home_r, away_away_r = self.rating(away)

        predicted_gd = home_home_r - away_away_r
        actual_gd = home_goals - away_goals
        error = actual_gd - predicted_gd
        adjusted = error / (1 + self.lam * abs(error))

        self.ratings[home] = (
            home_home_r + self.alpha * adjusted,
            home_away_r + self.beta * adjusted,
        )
        self.ratings[away] = (
            away_home_r - self.beta * adjusted,
            away_away_r - self.alpha * adjusted,
        )

    def ensure_seeded(self, team: str) -> None:
        """Seed a team that has never been rated, e.g. a newly promoted club."""
        if team not in self.ratings:
            self.ratings[team] = self.seed_ratings.get(team, (self.promoted_initial, self.promoted_initial))

    def new_season_regression(self) -> None:
        """Pull ratings partway back toward 0 (league average) between seasons."""
        for team, (home_r, away_r) in list(self.ratings.items()):
            self.ratings[team] = (
                home_r * self.season_regression,
                away_r * self.season_regression,
            )
