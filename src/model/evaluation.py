"""Shared evaluation helpers: per-fight losses, bootstrap CIs, calibration table."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def per_fight_logloss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_fight_correct(y, p) -> np.ndarray:
    return ((np.asarray(p) >= 0.5).astype(int) == np.asarray(y).astype(int)).astype(float)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Mean of per-fight values with a percentile bootstrap interval."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_diff(y, p_a, p_b, metric: str = "log_loss", **kw) -> dict[str, float]:
    """Bootstrap CI for metric(a) − metric(b) on the same fights.

    For log loss a negative difference means ``a`` is better; for accuracy a positive one does.
    """
    f = per_fight_logloss if metric == "log_loss" else per_fight_correct
    out = bootstrap_ci(f(y, p_a) - f(y, p_b), **kw)
    out["significant"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


def calibration_table(y, p, bins: int = 10) -> list[dict[str, float]]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows


def summarize(y, p, **kw) -> dict:
    return {
        "accuracy": bootstrap_ci(per_fight_correct(y, p), **kw),
        "log_loss": bootstrap_ci(per_fight_logloss(y, p), **kw),
        "n": int(len(y)),
    }


def by_year(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    for year, g in df.groupby(df["date"].dt.year):
        row = {"year": int(year), "n": int(len(g))}
        for c in cols:
            row[f"{c}_acc"] = float(per_fight_correct(g["y"], g[c]).mean())
            row[f"{c}_ll"] = float(per_fight_logloss(g["y"], g[c]).mean())
        rows.append(row)
    return rows
