"""Walk-forward backtest: for each year, train on the past, calibrate on the year before, score the year.

Every prediction is genuinely out-of-sample. Results (with bootstrap CIs) go to models/backtest.json
and the per-fight predictions to data/processed/backtest_preds.parquet.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd

from src.features.build import FEATURE_COLS
from src.model.ensemble import PlattCalibrated, StackedEnsemble, SymmetricClassifier, make_base_models
from src.model.evaluation import by_year, calibration_table, paired_diff, summarize
from src.model.train import _load_hgb_params, load_training
from src.paths import MODELS_DIR, PROCESSED_DIR, ensure_dirs

PREDS_PATH = PROCESSED_DIR / "backtest_preds.parquet"
BACKTEST_PATH = MODELS_DIR / "backtest.json"
FIRST_TEST_YEAR = 2017
MODELS = ["ensemble", "hgb", "logreg", "extra_trees", "mlp", "elo_only"]


def predict_year(df: pd.DataFrame, year: int, hgb_params: dict | None) -> pd.DataFrame:
    fit = df["date"] < pd.Timestamp(year - 1, 1, 1)
    calib = (df["date"] >= pd.Timestamp(year - 1, 1, 1)) & (df["date"] < pd.Timestamp(year, 1, 1))
    test = (df["date"] >= pd.Timestamp(year, 1, 1)) & (df["date"] < pd.Timestamp(year + 1, 1, 1))
    X, y = df[FEATURE_COLS], df["y"].astype(int)

    stack = StackedEnsemble(make_base_models(hgb_params)).fit(X[fit], y[fit])
    cal = PlattCalibrated(stack).fit(X[calib], y[calib])
    elo = SymmetricClassifier(make_base_models()["logreg"]).fit(X.loc[fit, ["d_elo"]], y[fit])

    out = df.loc[test, ["date", "event", "red", "blue", "y"]].copy()
    out["ensemble"] = cal.predict_proba(X[test])[:, 1]
    for name, col in stack.base_proba(X[test]).items():
        out[name] = col.to_numpy()
    out["elo_only"] = elo.predict_proba(X.loc[test, ["d_elo"]])[:, 1]
    return out


def run_backtest(first_year: int = FIRST_TEST_YEAR) -> dict:
    ensure_dirs()
    df = load_training().dropna(subset=["y", "date"]).sort_values("date", kind="stable").reset_index(drop=True)
    hgb_params = _load_hgb_params()
    years = range(first_year, int(df["date"].dt.year.max()) + 1)
    parts = []
    for yr in years:
        part = predict_year(df, yr, hgb_params)
        parts.append(part)
        acc = ((part["ensemble"] >= 0.5) == part["y"]).mean()
        print(f"  {yr}: n={len(part):>4}  ensemble acc={acc:.3f}")
    preds = pd.concat(parts, ignore_index=True)
    preds.to_parquet(PREDS_PATH, index=False)

    y = preds["y"].to_numpy()
    report = {
        "years": [int(min(years)), int(max(years))],
        "n": int(len(preds)),
        "pooled": {m: summarize(y, preds[m]) for m in MODELS},
        "coin_flip": summarize(y, np.full(len(y), 0.5)),
        "base_rate_red": float(y.mean()),
        "ensemble_vs": {
            m: {"log_loss": paired_diff(y, preds["ensemble"], preds[m], "log_loss"),
                "accuracy": paired_diff(y, preds["ensemble"], preds[m], "accuracy")}
            for m in MODELS if m != "ensemble"
        },
        "by_year": by_year(preds, ["ensemble", "logreg", "elo_only"]),
        "calibration": calibration_table(y, preds["ensemble"]),
    }
    BACKTEST_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nWalk-forward {report['years'][0]}–{report['years'][1]}, {report['n']:,} fights (95% bootstrap CI)")
    for m in MODELS + ["coin_flip"]:
        s = report["pooled"].get(m, report["coin_flip"])
        a, l = s["accuracy"], s["log_loss"]
        print(f"  {m:<12} acc {a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]   logloss {l['mean']:.3f} [{l['lo']:.3f}, {l['hi']:.3f}]")
    print("\nEnsemble minus model, log loss (negative = ensemble better; * = CI excludes 0)")
    for m, d in report["ensemble_vs"].items():
        ll = d["log_loss"]
        print(f"  vs {m:<12} {ll['mean']:+.4f} [{ll['lo']:+.4f}, {ll['hi']:+.4f}] {'*' if ll['significant'] else ''}")
    return report


if __name__ == "__main__":
    run_backtest()
