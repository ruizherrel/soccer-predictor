"""Elo ratings with a margin-of-victory (MoV) multiplier, adapted for soccer.

Reference formulation: FiveThirtyEight-style MoV dampening (used for NFL/NBA),
adapted here with a soccer goal-difference margin instead of point margin.
"""
from __future__ import annotations

import math

import config


class EloRatingSystem:
    def __init__(
        self,
        k: float = config.ELO_K,
        home_adv: float = config.ELO_HOME_ADV,
        initial: float = config.ELO_INITIAL,
        promoted_initial: float = config.ELO_PROMOTED_INITIAL,
        season_regression: float = config.ELO_SEASON_REGRESSION,
        seed_ratings: dict[str, float] | None = None,
    ) -> None:
        self.k = k
        self.home_adv = home_adv
        self.initial = initial
        self.promoted_initial = promoted_initial
        self.season_regression = season_regression
        # Per-team override for a first-ever rating, e.g. a Champions League
        # debutant seeded from its domestic-league Elo instead of the
        # generic default — see dataset.py's cross-league seeding. Empty by
        # default, so every other league behaves exactly as before.
        self.seed_ratings = seed_ratings or {}
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.seed_ratings.get(team, self.initial))

    def expected_home(self, home: str, away: str) -> float:
        r_h, r_a = self.rating(home), self.rating(away)
        return 1.0 / (1.0 + 10 ** (-((r_h + self.home_adv - r_a) / 400.0)))

    @staticmethod
    def _mov_multiplier(goal_diff: int, rating_diff: float) -> float:
        # max(|goal_diff|, 1) instead of |goal_diff| directly: a plain
        # log(|GD|+1) collapses to 0 for a draw (GD=0), which would freeze
        # ratings on every draw regardless of how surprising the result was.
        # Flooring the margin at 1 treats a draw like a minimal-margin
        # result, so favourites still lose ground when they only draw.
        margin = max(abs(goal_diff), 1)
        return math.log(margin + 1) * (2.2 / (0.001 * abs(rating_diff) + 2.2))

    def snapshot(self, home: str, away: str) -> tuple[float, float]:
        """Pre-match ratings for both teams, for use as model features."""
        return self.rating(home), self.rating(away)

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        r_h, r_a = self.rating(home), self.rating(away)
        e_h = self.expected_home(home, away)
        goal_diff = home_goals - away_goals
        s_h = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
        mov = self._mov_multiplier(goal_diff, r_h - r_a)
        delta = self.k * mov * (s_h - e_h)
        self.ratings[home] = r_h + delta
        self.ratings[away] = r_a - delta

    def ensure_seeded(self, team: str) -> None:
        """Seed a team that has never been rated, e.g. a newly promoted club."""
        if team not in self.ratings:
            self.ratings[team] = self.seed_ratings.get(team, self.promoted_initial)

    def new_season_regression(self) -> None:
        """Pull every team's rating partway back to the mean between seasons,
        reflecting typical squad/manager turnover."""
        for team in list(self.ratings):
            self.ratings[team] = (
                self.season_regression * self.ratings[team]
                + (1 - self.season_regression) * self.initial
            )
