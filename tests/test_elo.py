import numpy as np
import pandas as pd

from src.features.build import ELO_BASE, compute_elo
from tests.conftest import make_fights


def pre(fights):
    return compute_elo(fights)[0]


def test_first_fight_uses_base_rating_and_no_history(career):
    fights = make_fights(career)
    p = pre(fights)
    first = fights.index[(fights["event"] == "E1")]
    assert (p.loc[first, "elo"] == ELO_BASE).all()
    assert p.loc[first, "opp_elo_avg"].isna().all()
    assert (p.loc[first, "streak"] == 0).all()


def test_ratings_ignore_the_future(career):
    """Pre-fight values for early bouts must not change when later fights are added or changed."""
    full = make_fights(career)
    early = make_fights(career[:3])
    p_full, p_early = pre(full), pre(early)
    idx = early.index
    pd.testing.assert_frame_equal(p_full.loc[idx], p_early.loc[idx])

    flipped = [dict(b) for b in career]
    flipped[-1]["winner"] = "red"  # change the LAST result only
    p_flip = pre(make_fights(flipped))
    late = full.index[full["event"] == "E5"]
    early_rows = full.index.difference(late)
    pd.testing.assert_frame_equal(p_full.loc[early_rows], p_flip.loc[early_rows])
    # ... and the last bout's own pre-fight values are also unaffected by its own result
    pd.testing.assert_frame_equal(p_full.loc[late], p_flip.loc[late])


def test_row_order_does_not_matter(career):
    fights = make_fights(career)
    shuffled = fights.sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(pre(fights), pre(shuffled).loc[fights.index], check_exact=False)


def test_zero_sum_for_two_debutants_and_winner_gains():
    fights = make_fights([dict(date="2020-01-01", event="E1", red="A", blue="B", winner="red")])
    _, final = compute_elo(fights)
    r = final.set_index("fighter")["elo"]
    assert r["A"] > ELO_BASE > r["B"]
    assert np.isclose(r["A"] + r["B"], 2 * ELO_BASE)


def test_draw_or_no_contest_moves_nothing_but_counts_opponent():
    fights = make_fights([dict(date="2020-01-01", event="E1", red="A", blue="B", winner=None)])
    _, final = compute_elo(fights)
    assert (final["elo"] == ELO_BASE).all()
    assert (final["streak"] == 0).all()


def test_streak_counts_consecutive_results_before_the_bout(career):
    fights = make_fights(career)
    p = pre(fights)
    # A: W (E1), L (E2), W (E4), L (E5)  -> pre-fight streaks 0, +1, -1, +1
    a = fights.index[fights["fighter"] == "A"]
    streaks = p.loc[a].assign(date=fights.loc[a, "date"]).sort_values("date")["streak"].tolist()
    assert streaks == [0, 1, -1, 1]


def test_opponent_strength_is_mean_of_past_opponent_pre_fight_ratings(career):
    fights = make_fights(career)
    p = pre(fights)
    e4_a = fights.index[(fights["event"] == "E4") & (fights["fighter"] == "A")][0]
    # A has faced B (pre-fight 1500) and C (pre-fight 1500 + whatever C earned on 2020-01-01: nothing, debut)
    assert np.isclose(p.loc[e4_a, "opp_elo_avg"], ELO_BASE)
