"""Train a calibrated HistGradientBoosting win model on chronological holdout."""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.features.build import FEATURE_COLS, TRAINING_PATH, build_features
from src.paths import METRICS_PATH, MODEL_PATH, ensure_dirs

TRAIN_END = pd.Timestamp("2022-12-31")
CALIB_END = pd.Timestamp("2023-12-31")


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


def train_model(rebuild: bool = False) -> dict:
    ensure_dirs()
    df = load_training(rebuild=rebuild)
    df = df.dropna(subset=["y", "date"]).copy()
    X = df[FEATURE_COLS]
    y = df["y"].astype(int)

    fit_mask = df["date"] <= TRAIN_END
    calib_mask = (df["date"] > TRAIN_END) & (df["date"] <= CALIB_END)
    test_mask = df["date"] > CALIB_END

    n_fit, n_calib, n_test = int(fit_mask.sum()), int(calib_mask.sum()), int(test_mask.sum())
    print(f"rows: fit={n_fit:,} calib={n_calib:,} holdout={n_test:,}")
    if n_fit < 200 or n_test < 50:
        raise RuntimeError("Not enough chronological fights to train/evaluate.")

    hgb = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=250,
        min_samples_leaf=40,
        l2_regularization=0.8,
        random_state=42,
        early_stopping=False,
    )
    hgb.fit(X.loc[fit_mask], y.loc[fit_mask])

    method = "sigmoid"
    calibrated = CalibratedClassifierCV(FrozenEstimator(hgb), method=method, ensemble=False)
    if n_calib >= 50:
        calibrated.fit(X.loc[calib_mask], y.loc[calib_mask])
        print(f"calibrated with {method} on {n_calib:,} fights (2023)")
    else:
        calibrated.fit(X.loc[fit_mask], y.loc[fit_mask])
        print("calibration slice small; fitted calibrator on train")

    holdout_proba = calibrated.predict_proba(X.loc[test_mask])[:, 1]
    holdout_y = y.loc[test_mask].to_numpy()
    metrics = _metrics(holdout_y, holdout_proba)

    train_all_mask = df["date"] <= CALIB_END
    print("refitting on all data through 2023 for live predictions")
    hgb_live = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=250,
        min_samples_leaf=40,
        l2_regularization=0.8,
        random_state=42,
        early_stopping=False,
    )
    hgb_live.fit(X.loc[train_all_mask], y.loc[train_all_mask])
    live = CalibratedClassifierCV(FrozenEstimator(hgb_live), method=method, ensemble=False)
    live.fit(
        X.loc[test_mask] if n_test >= 80 else X.loc[train_all_mask],
        y.loc[test_mask] if n_test >= 80 else y.loc[train_all_mask],
    )

    # Keep the holdout-evaluated model as the reported one; persist the live model
    # but store holdout metrics from the frozen chronological pipeline.
    bundle = {
        "model": live,
        "eval_model": calibrated,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
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
    METRICS_PATH.write_text(json.dumps({"metrics": metrics, "split": bundle["split"], "trained_at": bundle["trained_at"]}, indent=2))

    print("\nHoldout (2024–now)")
    print(f"  fights     {metrics['n']:,}")
    print(f"  accuracy   {metrics['accuracy']:.3f}")
    print(f"  brier      {metrics['brier']:.3f}")
    print(f"  log loss   {metrics['log_loss']:.3f}")
    print(f"  mean pred  {metrics['mean_pred']:.3f}  (base rate {metrics['base_rate']:.3f})")
    print(f"saved {MODEL_PATH}")
    return bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train UFC win model")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features first")
    args = parser.parse_args()
    train_model(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
