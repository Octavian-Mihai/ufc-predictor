import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model.ensemble import PlattCalibrated, StackedEnsemble, SymmetricClassifier, mirror


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"d_a": rng.normal(size=n), "d_b": rng.normal(size=n), "stance_mismatch": rng.integers(0, 2, n).astype(float)})
    y = (X["d_a"] + 0.5 * X["d_b"] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    X["d_a"] += 0.3  # a non-zero mean so an uncorrected model would learn a corner bias
    return X, y.to_numpy()


def test_mirror_negates_differentials_only():
    X, _ = toy(5)
    m = mirror(X)
    assert (m[["d_a", "d_b"]] == -X[["d_a", "d_b"]]).all().all()
    assert (m["stance_mismatch"] == X["stance_mismatch"]).all()
    assert (mirror(m) == X).all().all()


def test_symmetric_classifier_probabilities_sum_to_one_across_corners():
    X, y = toy()
    p = SymmetricClassifier(LogisticRegression()).fit(X, y)
    assert np.allclose(p.predict_proba(X)[:, 1] + p.predict_proba(mirror(X))[:, 1], 1.0)


def test_equal_fighters_get_exactly_one_half():
    X, y = toy()
    clf = SymmetricClassifier(LogisticRegression()).fit(X, y)
    even = pd.DataFrame({"d_a": [0.0], "d_b": [0.0], "stance_mismatch": [0.0]})
    assert np.isclose(clf.predict_proba(even)[0, 1], 0.5)


def test_full_stack_and_calibration_stay_antisymmetric():
    X, y = toy(500)
    models = {"lr": LogisticRegression(), "lr2": LogisticRegression(C=0.1)}
    stack = StackedEnsemble(models, n_splits=3).fit(X.iloc[:400], y[:400])
    cal = PlattCalibrated(stack).fit(X.iloc[400:], y[400:])
    Xt = X.iloc[400:]
    for model in (stack, cal):
        assert np.allclose(model.predict_proba(Xt)[:, 1] + model.predict_proba(mirror(Xt))[:, 1], 1.0)


def test_stack_meta_learner_only_sees_out_of_fold_rows_from_later_splits():
    X, y = toy(300)
    stack = StackedEnsemble({"lr": LogisticRegression()}, n_splits=3).fit(X, y)
    # TimeSeriesSplit(3) never scores the first block, so those rows can't be in the meta training set.
    assert len(stack.oof_y_) == 300 - 300 // 4
    assert np.array_equal(stack.oof_y_, y[300 // 4:])


def test_stack_does_not_use_future_labels_for_earlier_oof_predictions():
    X, y = toy(300)
    y_changed = y.copy()
    y_changed[-60:] = 1 - y_changed[-60:]  # corrupt only the newest rows
    a = StackedEnsemble({"lr": LogisticRegression()}, n_splits=3).fit(X, y)
    b = StackedEnsemble({"lr": LogisticRegression()}, n_splits=3).fit(X, y_changed)
    # out-of-fold logits for the first scored fold depend only on labels before it
    first_fold = 300 // 4
    n = (300 - first_fold) // 3
    assert np.allclose(a.oof_logits_[:n], b.oof_logits_[:n])
