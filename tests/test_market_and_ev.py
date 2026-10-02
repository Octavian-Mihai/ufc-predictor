import numpy as np
import pandas as pd
import pytest

from src.model.market import attach_market, betting_sim
from src.value.ev import american_to_decimal, expected_value, multiplicative_devig, pair_market, value_flag


def test_american_to_decimal():
    assert np.isclose(american_to_decimal(180), 2.8)
    assert np.isclose(american_to_decimal(-200), 1.5)


def test_expected_value_matches_readme_example():
    assert np.isclose(expected_value(0.42, american_to_decimal(180)), 0.176)


def test_devig_sums_to_one_and_preserves_ratio():
    a, b = multiplicative_devig(1 / 1.5, 1 / 2.8)
    assert np.isclose(a + b, 1.0) and np.isclose(a / b, (1 / 1.5) / (1 / 2.8))
    m = pair_market(1.5, 2.8)
    assert np.isclose(m["a_fair"] + m["b_fair"], 1.0)


def test_value_flag_requires_underdog_and_edge():
    assert value_flag(0.42, 2.8, 1.45)
    assert not value_flag(0.42, 1.45, 2.8)  # favourite never flagged
    assert not value_flag(0.30, 2.8, 1.45)  # underdog but no edge


@pytest.fixture
def preds():
    return pd.DataFrame({"date": pd.to_datetime(["2020-03-07"]), "red": ["Ann Lee"], "blue": ["Bea Ray"], "y": [1], "ensemble": [0.6]})


def test_attach_market_orients_odds_to_red_even_if_listed_reversed(preds):
    odds = pd.DataFrame({"date": ["2020-03-07"], "fighter_a": ["BEA RAY"], "fighter_b": ["Ann  Lee"], "a_decimal": [3.0], "b_decimal": [1.4]})
    m = attach_market(preds, odds)
    assert m["red_decimal"].iloc[0] == 1.4 and m["blue_decimal"].iloc[0] == 3.0
    assert m["market"].iloc[0] > 0.5  # red is the favourite


def test_attach_market_requires_nearby_date_and_same_pair(preds):
    far = pd.DataFrame({"date": ["2020-04-07"], "fighter_a": ["Ann Lee"], "fighter_b": ["Bea Ray"], "a_decimal": [1.5], "b_decimal": [2.8]})
    other = pd.DataFrame({"date": ["2020-03-07"], "fighter_a": ["Ann Lee"], "fighter_b": ["Cy Dun"], "a_decimal": [1.5], "b_decimal": [2.8]})
    assert attach_market(preds, far).empty and attach_market(preds, other).empty
    assert len(attach_market(preds, far, keep_unmatched=True)) == 1


def test_betting_sim_only_backs_underdogs_with_edge():
    m = pd.DataFrame({"y": [0, 0, 1], "red_decimal": [1.4, 1.4, 1.4], "blue_decimal": [3.0, 3.0, 3.0], "model": [0.5, 0.5, 0.9]})
    r = betting_sim(m, "model", threshold=0.05)  # blue EV = 0.5*2 - 0.5 = 0.5 on rows 1-2; row 3 has no edge on blue
    assert r["n"] == 2 and np.isclose(r["mean"], 2.0)  # blue won twice at +2 profit each


def test_synthetic_odds_are_refused():
    from src.ingest.odds import require_real_odds

    for src in ("dummy", "dummy-card"):
        with pytest.raises(RuntimeError):
            require_real_odds({"source": src, "events": []})
    for src in ("api", "cache", "stale-cache"):
        assert require_real_odds({"source": src})["source"] == src
