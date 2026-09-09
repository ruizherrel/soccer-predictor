from soccer_predictor.pi_ratings import PiRatingSystem


def test_default_rating_is_zero():
    pi = PiRatingSystem()
    assert pi.rating("A") == (0.0, 0.0)


def test_perfectly_predicted_result_does_not_move_ratings():
    pi = PiRatingSystem()
    # Both teams start at (0, 0), so predicted_gd = 0. A draw matches that
    # prediction exactly, so the error is zero and nothing should move.
    pi.update("A", "B", 1, 1)
    assert pi.rating("A") == (0.0, 0.0)
    assert pi.rating("B") == (0.0, 0.0)


def test_home_win_increases_home_rating_and_decreases_away_rating():
    pi = PiRatingSystem()
    pi.update("A", "B", 2, 0)

    home_home_r, home_away_r = pi.rating("A")
    away_home_r, away_away_r = pi.rating("B")

    assert home_home_r > 0.0
    assert home_away_r > 0.0
    assert away_home_r < 0.0
    assert away_away_r < 0.0
    # own-venue learning rate (alpha) must move the rating more than the
    # cross-venue rate (beta) for the same match
    assert home_home_r > home_away_r
    assert abs(away_away_r) > abs(away_home_r)


def test_ensure_seeded_only_applies_to_unrated_teams():
    pi = PiRatingSystem()
    pi.update("A", "B", 2, 0)
    rating_before = pi.rating("A")

    pi.ensure_seeded("A")
    pi.ensure_seeded("C")

    assert pi.rating("A") == rating_before
    assert pi.rating("C") == (pi.promoted_initial, pi.promoted_initial)


def test_new_season_regression_pulls_toward_zero():
    pi = PiRatingSystem()
    pi.update("A", "B", 3, 0)
    home_r_before, _ = pi.rating("A")

    pi.new_season_regression()

    home_r_after, _ = pi.rating("A")
    assert 0.0 < home_r_after < home_r_before
