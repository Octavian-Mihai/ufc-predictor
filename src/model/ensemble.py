"""Corner-symmetric stacked ensemble with Platt calibration.

All features are red−blue differentials, so swapping corners just negates the ``d_*``
columns and flips the label. Training on both orientations and averaging both
predictions removes the red-corner bias the raw model otherwise learns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_HGB_PARAMS = dict(
    max_depth=5,
    learning_rate=0.06,
    max_iter=250,
    min_samples_leaf=40,
    l2_regularization=0.8,
)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    """Swap corners: negate every differential, leave symmetric flags alone."""
    out = X.copy()
    flip = [c for c in out.columns if c.startswith("d_")]
    out[flip] = -out[flip]
    return out


def make_base_models(hgb_params: dict | None = None, seed: int = 42) -> dict[str, BaseEstimator]:
    params = {**DEFAULT_HGB_PARAMS, **(hgb_params or {})}
    return {
        "hgb": HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params),
        "logreg": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.05, max_iter=2000),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=25, max_features=0.5, n_jobs=-1, random_state=seed
            ),
        ),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                max_iter=400,
                random_state=seed,
            ),
        ),
    }


class SymmetricClassifier(ClassifierMixin, BaseEstimator):
    """Fit on original + mirrored rows; predict the average of both orientations."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y)
        X_aug = pd.concat([X, mirror(X)], ignore_index=True)
        y_aug = np.concatenate([y, 1 - y])
        self.estimator_ = clone(self.estimator).fit(X_aug, y_aug)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = 0.5 * (self.estimator_.predict_proba(X)[:, 1] + 1.0 - self.estimator_.predict_proba(mirror(X))[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Blend base models with a logistic meta-learner.

    The meta-learner sees only *out-of-fold* base predictions from expanding-window
    time splits (rows must be date-ordered), so later fights never inform the
    models that score earlier ones.
    """

    def __init__(self, base_models: dict[str, BaseEstimator], n_splits: int = 5, meta_C: float = 1.0):
        self.base_models = base_models
        self.n_splits = n_splits
        self.meta_C = meta_C

    def _base_logits(self, models, X) -> np.ndarray:
        return np.column_stack([_logit(m.predict_proba(X)[:, 1]) for m in models])

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        names = list(self.base_models)
        oof = np.full((len(X), len(names)), np.nan)
        for train_idx, test_idx in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            for j, name in enumerate(names):
                m = SymmetricClassifier(self.base_models[name]).fit(X.iloc[train_idx], y[train_idx])
                oof[test_idx, j] = _logit(m.predict_proba(X.iloc[test_idx])[:, 1])
        seen = ~np.isnan(oof).any(axis=1)
        # Antisymmetric blend: no intercept, so P(A beats B) = 1 − P(B beats A).
        self.meta_ = LogisticRegression(C=self.meta_C, fit_intercept=False).fit(oof[seen], y[seen])
        self.oof_logits_, self.oof_y_, self.names_ = oof[seen], y[seen], names
        self.models_ = [SymmetricClassifier(self.base_models[n]).fit(X, y) for n in names]
        self.classes_ = np.array([0, 1])
        return self

    @property
    def weights_(self) -> dict[str, float]:
        return dict(zip(self.names_, self.meta_.coef_[0].round(4).tolist()))

    def base_proba(self, X) -> pd.DataFrame:
        return pd.DataFrame(
            {n: m.predict_proba(X)[:, 1] for n, m in zip(self.names_, self.models_)}, index=X.index
        )

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._base_logits(self.models_, X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PlattCalibrated(ClassifierMixin, BaseEstimator):
    """Frozen fitted model + 1-D logistic recalibration of its logit."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        z = _logit(self.model.predict_proba(X)[:, 1]).reshape(-1, 1)
        self.platt_ = LogisticRegression(C=1e6, fit_intercept=False).fit(z, np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        z = _logit(self.model.predict_proba(X)[:, 1]).reshape(-1, 1)
        return self.platt_.predict_proba(z)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
