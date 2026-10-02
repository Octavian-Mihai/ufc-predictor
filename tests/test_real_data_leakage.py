"""Truncation test on the real UFC Stats data: features for old fights must not depend on newer ones."""

import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURE_COLS, add_prior_career, build_fighter_fight_table, build_training_rows
from src.paths import FIGHT_RESULTS_CSV

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not FIGHT_RESULTS_CSV.exists(), reason="run `python -m src.ingest.ufcstats --bootstrap` first"),
]

CUTOFF = pd.Timestamp("2020-01-01")


@pytest.fixture(scope="module")
def tables():
    fights = build_fighter_fight_table()
    full = build_training_rows(add_prior_career(fights))
    trunc = build_training_rows(add_prior_career(fights[fights["date"] < CUTOFF].copy()))
    return full, trunc


def test_features_for_old_fights_are_identical_without_later_fights(tables):
    full, trunc = tables
    key = ["event", "bout", "date"]
    old = full[full["date"] < CUTOFF].sort_values(key).reset_index(drop=True)
    assert len(old) == len(trunc) and len(old) > 5000
    t = trunc.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(old[key + FEATURE_COLS + ["y"]], t[key + FEATURE_COLS + ["y"]], check_exact=False, rtol=1e-9)


def test_no_feature_is_a_giveaway_for_the_label(tables):
    """A leaked result would show up as a near-perfect single-feature predictor."""
    full, _ = tables
    for col in FEATURE_COLS:
        x = full[col]
        mask = x.notna() & (x != 0)
        acc = ((x[mask] > 0).astype(int) == full.loc[mask, "y"]).mean()
        assert acc < 0.75, f"{col} alone predicts the winner {acc:.0%} of the time — possible leakage"


def test_first_ufc_fights_have_no_career_history(tables):
    full, _ = tables
    debuts = full[(full["red_prior"] == 0) & (full["blue_prior"] == 0)]
    assert len(debuts) > 0
    assert debuts["d_win_rate"].isna().all() and debuts["d_slpm"].isna().all()
