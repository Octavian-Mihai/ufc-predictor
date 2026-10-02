"""Can anything beat the closing line? Walk-forward test of ways to combine stats with the market.

Variants, all scored out-of-sample on the same fights (test years ≥ FIRST_EVAL_YEAR with odds):
  market        de-vigged closing line
  market_recal  line with a learned logit slope (fixes favourite/longshot bias)
  stats         stats-only stacked ensemble (from the backtest)
  blend         2-input logistic stack of [logit(market), logit(stats)], fit on earlier years
  market_aware  full ensemble with the market logit as an extra feature
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.build import FEATURE_COLS
from src.model.backtest import PREDS_PATH
from src.model.ensemble import PlattCalibrated, StackedEnsemble, make_base_models
from src.model.evaluation import paired_diff, summarize
from src.model.market import ODDS_HISTORY_PATH, attach_market, betting_sim
from src.model.train import _load_hgb_params, load_training
from src.paths import MODELS_DIR

REPORT_PATH = MODELS_DIR / "market_model.json"
FIRST_EVAL_YEAR = 2019
LAST_YEAR_FOR_FIT = 2016  # earliest test year is FIRST_EVAL_YEAR; fits use everything before
MARKET_FEATURES = FEATURE_COLS + ["d_market"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _slope_model(z, y) -> LogisticRegression:
    return LogisticRegression(C=1e6, fit_intercept=False).fit(np.asarray(z), np.asarray(y))


def run() -> dict:
    odds = pd.read_csv(ODDS_HISTORY_PATH)
    df = load_training().dropna(subset=["y", "date"]).sort_values("date", kind="stable").reset_index(drop=True)
    df = attach_market(df, odds, keep_unmatched=True)
    df["d_market"] = _logit(df["market"].fillna(0.5)).astype(float)
    df.loc[df["market"].isna(), "d_market"] = np.nan
    has = df["market"].notna()
    print(f"{int(has.sum()):,}/{len(df):,} training fights have a closing line")

    stats = pd.read_parquet(PREDS_PATH)[["date", "red", "blue", "ensemble"]].rename(columns={"ensemble": "stats"})
    df = df.merge(stats, on=["date", "red", "blue"], how="left")

    hgb_params = _load_hgb_params()
    X, y = df[MARKET_FEATURES], df["y"].astype(int)
    parts = []
    for year in range(FIRST_EVAL_YEAR, int(df["date"].dt.year.max()) + 1):
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        past = has & (df["date"] < start)
        test = has & (df["date"] >= start) & (df["date"] < end) & df["stats"].notna()
        if test.sum() == 0:
            continue
        fit = has & (df["date"] < pd.Timestamp(year - 1, 1, 1))
        calib = has & (df["date"] >= pd.Timestamp(year - 1, 1, 1)) & (df["date"] < start)

        out = df.loc[test, ["date", "red", "blue", "y", "market", "stats", "red_decimal", "blue_decimal"]].copy()
        zm = _logit(df.loc[past, "market"]).reshape(-1, 1)
        out["market_recal"] = _sigmoid(_slope_model(zm, y[past]).decision_function(_logit(out[["market"]].to_numpy())))

        # blend: fit on earlier years' out-of-sample stats predictions
        prior = past & df["stats"].notna()
        zb = np.column_stack([_logit(df.loc[prior, "market"]), _logit(df.loc[prior, "stats"])])
        blend = _slope_model(zb, y[prior])
        out["blend"] = _sigmoid(blend.decision_function(np.column_stack([_logit(out["market"]), _logit(out["stats"])])))
        out["blend_w_market"], out["blend_w_stats"] = blend.coef_[0]

        stack = StackedEnsemble(make_base_models(hgb_params)).fit(X[fit], y[fit])
        cal = PlattCalibrated(stack).fit(X[calib], y[calib])
        out["market_aware"] = cal.predict_proba(X[test])[:, 1]
        parts.append(out)
        acc = ((out["market_aware"] >= 0.5) == out["y"]).mean()
        print(f"  {year}: n={len(out):>4}  blend w=({out['blend_w_market'].iloc[0]:.2f},{out['blend_w_stats'].iloc[0]:.2f})  market_aware acc={acc:.3f}")

    r = pd.concat(parts, ignore_index=True)
    yv = r["y"].to_numpy()
    names = ["market", "market_recal", "stats", "blend", "market_aware"]
    report = {
        "n": int(len(r)),
        "years": [FIRST_EVAL_YEAR, int(r["date"].dt.year.max())],
        "pooled": {k: summarize(yv, r[k]) for k in names},
        "vs_market": {k: paired_diff(yv, r[k], r["market"], "log_loss") for k in names if k != "market"},
        "vs_market_recal": {k: paired_diff(yv, r[k], r["market_recal"], "log_loss") for k in ("stats", "blend", "market_aware")},
        "value_rule": {k: betting_sim(r.assign(model=r[k]), "model") for k in ("stats", "blend", "market_aware")},
        "blend_weights_last_year": {"market": float(r["blend_w_market"].iloc[-1]), "stats": float(r["blend_w_stats"].iloc[-1])},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n{report['n']:,} fights with a line, {report['years'][0]}–{report['years'][1]} (out-of-sample)")
    for k in names:
        s = report["pooled"][k]
        print(f"  {k:<13} acc {s['accuracy']['mean']:.3f}   log loss {s['log_loss']['mean']:.4f} [{s['log_loss']['lo']:.4f}, {s['log_loss']['hi']:.4f}]")
    print("\nlog loss minus raw market (negative = better than the line; * = CI excludes 0)")
    for k, d in report["vs_market"].items():
        print(f"  {k:<13} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'*' if d['significant'] else ''}")
    print("\nlog loss minus recalibrated market")
    for k, d in report["vs_market_recal"].items():
        print(f"  {k:<13} {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'*' if d['significant'] else ''}")
    print("\nvalue rule (underdog, EV ≥ 5%)")
    for k, v in report["value_rule"].items():
        print(f"  {k:<13} {v['n']} bets, ROI {v['mean']:+.1%} [{v['lo']:+.1%}, {v['hi']:+.1%}]")
    return report


if __name__ == "__main__":
    run()
