import numpy as np
import pandas as pd

from src.features.build import FEATURE_COLS, add_prior_career, build_training_rows, diffs_from_sides
from tests.conftest import make_fights

CAREER = ["prior_fights", "win_rate", "last3", "slpm", "sapm", "td_acc", "td_def", "str_acc", "str_def",
          "ctrl_for_p15", "kd_p15", "finish_rate", "days_since", "elo", "opp_elo_avg", "streak"]


def feats(bouts):
    return add_prior_career(make_fights(bouts)).set_index(["event", "fighter"])


def test_debut_has_no_history(career):
    g = feats(career)
    row = g.loc[("E1", "A")]
    assert row["prior_fights"] == 0
    assert np.isnan(row["slpm"]) and np.isnan(row["win_rate"]) and np.isnan(row["days_since"])


def test_stats_come_only_from_earlier_fights(career):
    g = feats(career)
    # A's sig strikes landed: E1 = 30 in 10 min, E2 = 20 in 10 min. Going into E4 that's 50 / 20 min.
    row = g.loc[("E4", "A")]
    assert row["prior_fights"] == 2
    assert np.isclose(row["slpm"], 50 / 20)
    assert np.isclose(row["win_rate"], 0.5)  # won E1, lost E2
    assert np.isclose(row["days_since"], (pd.Timestamp("2021-06-10") - pd.Timestamp("2020-06-10")).days)


def test_a_fights_own_result_and_stats_never_leak_into_its_features(career):
    base = feats(career)
    altered = [dict(b) for b in career]
    altered[3] = dict(altered[3], winner="blue", red_stats=dict(sig_l=999, kd=9, td_l=99, ctrl_sec=900), finish=False)
    g = feats(altered)
    for key in [("E4", "A"), ("E4", "B")]:
        pd.testing.assert_series_equal(base.loc[key, CAREER], g.loc[key, CAREER])


def test_appending_future_fights_does_not_change_past_features(career):
    short = feats(career[:3])
    long = feats(career + [dict(date="2023-01-01", event="E6", red="A", blue="C", winner="red")])
    keys = short.index
    pd.testing.assert_frame_equal(short.loc[keys, CAREER], long.loc[keys, CAREER])


def test_training_rows_label_and_differentials(career):
    prior = add_prior_career(make_fights(career))
    train = build_training_rows(prior).set_index("event")
    assert train.loc["E1", "y"] == 1 and train.loc["E2", "y"] == 0  # red won E1, blue won E2
    p = prior.set_index(["event", "fighter"])
    exp = p.loc[("E4", "A"), "elo"] - p.loc[("E4", "B"), "elo"]
    assert np.isclose(train.loc["E4", "d_elo"], exp)
    assert list(train[FEATURE_COLS].columns) == FEATURE_COLS


def test_draws_and_no_contests_are_not_training_rows():
    bouts = [dict(date="2020-01-10", event="E1", red="A", blue="B", winner=None),
             dict(date="2020-06-10", event="E2", red="A", blue="B", winner="red")]
    train = build_training_rows(add_prior_career(make_fights(bouts)))
    assert train["event"].tolist() == ["E2"]


def test_swapping_corners_negates_every_differential(career):
    prior = add_prior_career(make_fights(career))
    red = prior[prior["fighter"] == prior["red"]].reset_index(drop=True)
    blue = prior[prior["fighter"] == prior["blue"]].reset_index(drop=True)
    d, d_swapped = diffs_from_sides(red, blue), diffs_from_sides(blue, red)
    diff_cols = [c for c in d.columns if c.startswith("d_")]
    pd.testing.assert_frame_equal(d[diff_cols], -d_swapped[diff_cols])
    assert (d["stance_mismatch"] == d_swapped["stance_mismatch"]).all()
