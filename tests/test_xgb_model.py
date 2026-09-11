import config
from soccer_predictor.xgb_model import (
    FEATURE_COLUMNS,
    PYTHAGOREAN_FEATURE_COLUMNS,
    feature_columns_for_league,
)


def test_league_without_use_pythagorean_flag_gets_the_plain_feature_set(monkeypatch):
    monkeypatch.setitem(config.LEAGUES, "TESTLEAGUE", {"name": "Test", "sport": "soccer", "source": "thesportsdb"})

    assert feature_columns_for_league("TESTLEAGUE") == FEATURE_COLUMNS


def test_league_with_use_pythagorean_flag_gets_the_extra_columns(monkeypatch):
    monkeypatch.setitem(
        config.LEAGUES,
        "TESTLEAGUE",
        {"name": "Test", "sport": "baseball", "source": "thesportsdb", "use_pythagorean": True},
    )

    cols = feature_columns_for_league("TESTLEAGUE")

    assert cols == FEATURE_COLUMNS + PYTHAGOREAN_FEATURE_COLUMNS


def test_no_league_given_gets_the_plain_feature_set():
    assert feature_columns_for_league(None) == FEATURE_COLUMNS


def test_mlb_is_configured_to_use_pythagorean_features():
    # Guards against the config flag silently getting dropped in a future
    # edit -- this is the whole reason PYTHAGOREAN_FEATURE_COLUMNS exists.
    assert config.LEAGUES["MLB"]["use_pythagorean"] is True
