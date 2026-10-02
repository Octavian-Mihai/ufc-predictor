"""Train a calibrated, corner-symmetric stacked ensemble on a chronological holdout."""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.features.build import FEATURE_COLS, TRAINING_PATH, build_features
from src.model.ensemble import (
    PlattCalibrated,
    StackedEnsemble,
    SymmetricClassifier,
    make_base_models,
)
from src.paths import METRICS_PATH, MODEL_PATH, MODELS_DIR, ensure_dirs

TRAIN_END = pd.Timestamp("2022-12-31")
CALIB_END = pd.Timestamp("2023-12-31")
HGB_PARAMS_PATH = MODELS_DIR / "hgb_params.json"


def _metrics(y_true, proba) -> dict[str, float]:
    y_pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
        "mean_pred": float(np.mean(proba)),
        "base_rate": float(np.mean(y_true)),
    }


def load_training(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not TRAINING_PATH.exists():
        build_features()
    df = pd.read_parquet(TRAINING_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def tune_hgb(X: pd.DataFrame, y: pd.Series, n_iter: int = 25) -> dict:
    """Randomized search over HGB params, scored by log loss on expanding time splits."""
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(early_stopping=False, random_state=42),
        {
            "max_depth": randint(2, 7),
            "learning_rate": loguniform(0.01, 0.15),
            "max_iter": randint(80, 400),
            "min_samples_leaf": randint(20, 120),
            "l2_regularization": loguniform(0.1, 20),
        },
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=TimeSeriesSplit(n_splits=4),
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    best = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in search.best_params_.items()}
    print(f"tuned HGB (cv log loss {-search.best_score_:.4f}): {best}")
    HGB_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    return best


def _load_hgb_params() -> dict | None:
    return json.loads(HGB_PARAMS_PATH.read_text()) if HGB_PARAMS_PATH.exists() else None


def train_model(rebuild: bool = False, tune: bool = False) -> dict:
    ensure_dirs()
    df = (
        load_training(rebuild=rebuild)
        .dropna(subset=["y", "date"])
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    X = df[FEATURE_COLS]
    y = df["y"].astype(int)

    fit_mask = df["date"] <= TRAIN_END
    calib_mask = (df["date"] > TRAIN_END) & (df["date"] <= CALIB_END)
    test_mask = df["date"] > CALIB_END

    n_fit, n_calib, n_test = int(fit_mask.sum()), int(calib_mask.sum()), int(test_mask.sum())
    print(f"rows: fit={n_fit:,} calib={n_calib:,} holdout={n_test:,}")
    if n_fit < 200 or n_test < 50:
        raise RuntimeError("Not enough chronological fights to train/evaluate.")

    hgb_params = tune_hgb(X.loc[fit_mask], y.loc[fit_mask]) if tune else _load_hgb_params()

    # --- evaluation fit: train ≤2022, calibrate 2023, score 2024+ --------------------
    print("fitting stacked ensemble (time-ordered out-of-fold meta-learning)")
    stack = StackedEnsemble(make_base_models(hgb_params)).fit(X.loc[fit_mask], y.loc[fit_mask])
    print(f"  meta weights (logit scale): {stack.weights_}")

    calibrated = PlattCalibrated(stack)
    if n_calib >= 50:
        calibrated.fit(X.loc[calib_mask], y.loc[calib_mask])
        print(f"calibrated with Platt scaling on {n_calib:,} fights (2023)")
    else:
        calibrated.fit(X.loc[fit_mask], y.loc[fit_mask])
        print("calibration slice small; fitted calibrator on train")

    X_test, y_test = X.loc[test_mask], y.loc[test_mask].to_numpy()
    metrics = _metrics(y_test, calibrated.predict_proba(X_test)[:, 1])

    # Per-model comparison on the same holdout (base models are uncalibrated).
    comparison = {"ensemble (calibrated)": metrics}
    base_test = stack.base_proba(X_test)
    for name in base_test:
        comparison[name] = _metrics(y_test, base_test[name].to_numpy())
    elo_only = SymmetricClassifier(make_base_models()["logreg"]).fit(X.loc[fit_mask, ["d_elo"]], y.loc[fit_mask])
    comparison["elo only"] = _metrics(y_test, elo_only.predict_proba(X_test[["d_elo"]])[:, 1])
    comparison["coin flip"] = _metrics(y_test, np.full(len(y_test), 0.5))

    imp = permutation_importance(
        calibrated, X_test, y_test, scoring="neg_log_loss", n_repeats=8, random_state=42, n_jobs=1
    )
    importance = (
        pd.Series(imp.importances_mean, index=FEATURE_COLS).sort_values(ascending=False).head(15).round(5).to_dict()
    )

    # --- live fit: ensemble on every fight, reuse the 2023-fitted Platt calibrator ---
    print("refitting ensemble on all data for live predictions")
    live = PlattCalibrated(StackedEnsemble(make_base_models(hgb_params)).fit(X, y))
    live.platt_ = calibrated.platt_
    live.classes_ = calibrated.classes_

    bundle = {
        "model": live,
        "eval_model": calibrated,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "comparison": comparison,
        "importance": importance,
        "ensemble_weights": stack.weights_,
        "hgb_params": hgb_params,
        "split": {
            "fit_through": str(TRAIN_END.date()),
            "calib_through": str(CALIB_END.date()),
            "holdout_from": "2024-01-01",
            "n_fit": n_fit,
            "n_calib": n_calib,
            "n_holdout": n_test,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, MODEL_PATH)
    keys = ("metrics", "comparison", "importance", "ensemble_weights", "hgb_params", "split", "trained_at")
    METRICS_PATH.write_text(json.dumps({k: bundle[k] for k in keys}, indent=2))

    print("\nHoldout (2024–now)")
    print(f"  {'model':<24}{'acc':>7}{'brier':>8}{'logloss':>9}{'mean_p':>8}")
    for name, m in comparison.items():
        print(f"  {name:<24}{m['accuracy']:>7.3f}{m['brier']:>8.3f}{m['log_loss']:>9.3f}{m['mean_pred']:>8.3f}")
    print(f"  (holdout base rate {metrics['base_rate']:.3f}, n={metrics['n']:,})")
    print("\nTop features (permutation importance, Δ log loss)")
    for name, val in importance.items():
        print(f"  {name:<22}{val:>9.5f}")
    print(f"saved {MODEL_PATH}")
    return bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train UFC win model")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features first")
    parser.add_argument("--tune", action="store_true", help="Tune HistGradientBoosting on time-series CV first")
    args = parser.parse_args()
    train_model(rebuild=args.rebuild, tune=args.tune)


if __name__ == "__main__":
    main()
