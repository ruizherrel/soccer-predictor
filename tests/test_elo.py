from soccer_predictor.elo import EloRatingSystem


def test_equal_ratings_no_home_adv_expected_is_half():
    elo = EloRatingSystem(home_adv=0.0)
    assert elo.expected_home("A", "B") == 0.5


def test_draw_between_equal_teams_no_home_adv_does_not_move_ratings():
    elo = EloRatingSystem(home_adv=0.0)
    elo.update("A", "B", 1, 1)
    assert elo.rating("A") == elo.rating("B") == 1500.0


def test_winner_gains_loser_loses():
    elo = EloRatingSystem(home_adv=0.0)
    elo.update("A", "B", 2, 0)
    assert elo.rating("A") > 1500.0
    assert elo.rating("B") < 1500.0
    # zero-sum: home gain equals away loss
    assert abs((elo.rating("A") - 1500.0) - (1500.0 - elo.rating("B"))) < 1e-9


def test_bigger_margin_moves_rating_more():
    elo_small = EloRatingSystem(home_adv=0.0)
    elo_small.update("A", "B", 1, 0)

    elo_big = EloRatingSystem(home_adv=0.0)
    elo_big.update("A", "B", 4, 0)

    assert (elo_big.rating("A") - 1500.0) > (elo_small.rating("A") - 1500.0)


def test_ensure_seeded_only_applies_to_unrated_teams():
    elo = EloRatingSystem()
    elo.update("A", "B", 1, 0)
    rating_before = elo.rating("A")

    elo.ensure_seeded("A")  # already rated, must not reset
    elo.ensure_seeded("C")  # new team, gets the promoted-team seed

    assert elo.rating("A") == rating_before
    assert elo.rating("C") == elo.promoted_initial


def test_new_season_regression_pulls_toward_mean():
    elo = EloRatingSystem()
    elo.update("A", "B", 3, 0)
    high = elo.rating("A")

    elo.new_season_regression()

    assert 1500.0 < elo.rating("A") < high
