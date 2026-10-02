"""Model vs bookmaker closing line, on fights where historical odds are available.

Input: ``data/external/odds_history.csv`` with columns
``date, fighter_a, fighter_b, a_decimal, b_decimal`` (pre-fight decimal moneylines, one row per bout).
Compares the walk-forward ensemble (out-of-sample) with the de-vigged market, a model/market blend,
and the flat-stake ROI of the "bet the underdog when EV ≥ threshold" rule.
"""

from __future__ import annotations

import json
import unicodedata

import numpy as np
import pandas as pd

from src.model.backtest import BACKTEST_PATH, PREDS_PATH
from src.model.evaluation import bootstrap_ci, paired_diff, summarize
from src.paths import DATA_DIR, MODELS_DIR

ODDS_HISTORY_PATH = DATA_DIR / "external" / "odds_history.csv"
MARKET_PATH = MODELS_DIR / "market_comparison.json"
EV_THRESHOLD = 0.05


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(text.replace(".", " ").replace("'", "").replace("-", " ").split())


def attach_market(preds: pd.DataFrame, odds: pd.DataFrame, max_days: int = 3) -> pd.DataFrame:
    """Join odds onto predictions by unordered fighter pair and nearby date; orient to red/blue."""
    odds = odds.assign(date=pd.to_datetime(odds["date"], errors="coerce")).dropna(subset=["date", "a_decimal", "b_decimal"])
    index: dict[frozenset, list[tuple]] = {}
    for r in odds.itertuples():
        index.setdefault(frozenset((_key(r.fighter_a), _key(r.fighter_b))), []).append(
            (r.date, _key(r.fighter_a), float(r.a_decimal), float(r.b_decimal))
        )
    rows = []
    for r in preds.itertuples():
        red, blue = _key(r.red), _key(r.blue)
        cands = index.get(frozenset((red, blue)), [])
        cands = [c for c in cands if abs((c[0] - r.date).days) <= max_days]
        if not cands:
            rows.append((np.nan, np.nan))
            continue
        _, first, d1, d2 = min(cands, key=lambda c: abs((c[0] - r.date).days))
        rows.append((d1, d2) if first == red else (d2, d1))
    out = preds.copy()
    out[["red_decimal", "blue_decimal"]] = pd.DataFrame(rows, index=preds.index)
    out = out.dropna(subset=["red_decimal", "blue_decimal"]).copy()
    imp_r, imp_b = 1 / out["red_decimal"], 1 / out["blue_decimal"]
    out["market"] = imp_r / (imp_r + imp_b)  # multiplicative de-vig
    return out


def betting_sim(m: pd.DataFrame, p_col: str = "ensemble", threshold: float = EV_THRESHOLD) -> dict:
    """Flat 1-unit bet on the moneyline underdog whenever model EV ≥ threshold, at the quoted price."""
    profits = []
    for r in m.itertuples():
        p_red = getattr(r, p_col)
        for side, p, dec, win in (("red", p_red, r.red_decimal, r.y == 1), ("blue", 1 - p_red, r.blue_decimal, r.y == 0)):
            other = r.blue_decimal if side == "red" else r.red_decimal
            if dec > other and p * (dec - 1) - (1 - p) >= threshold:
                profits.append(dec - 1 if win else -1.0)
    res = bootstrap_ci(profits) if profits else {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    res["hit_rate"] = float(np.mean([p > 0 for p in profits])) if profits else float("nan")
    res["threshold"] = threshold
    return res


def run_market_comparison() -> dict:
    if not ODDS_HISTORY_PATH.exists():
        raise FileNotFoundError(f"No historical odds at {ODDS_HISTORY_PATH} (see module docstring for the format).")
    if not PREDS_PATH.exists():
        raise FileNotFoundError("Run `python -m src.model.backtest` first.")
    preds = pd.read_parquet(PREDS_PATH)
    m = attach_market(preds, pd.read_csv(ODDS_HISTORY_PATH))
    if len(m) < 100:
        raise RuntimeError(f"Only {len(m)} fights matched odds; need more for a meaningful comparison.")

    y = m["y"].to_numpy()
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m["blend"] = 1 / (1 + np.exp(-0.5 * (logit(m["ensemble"]) + logit(m["market"]))))
    report = {
        "matched": int(len(m)),
        "of": int(len(preds)),
        "date_range": [str(m["date"].min().date()), str(m["date"].max().date())],
        "pooled": {k: summarize(y, m[k]) for k in ("ensemble", "market", "blend")},
        "ensemble_vs_market": {
            "log_loss": paired_diff(y, m["ensemble"], m["market"], "log_loss"),
            "accuracy": paired_diff(y, m["ensemble"], m["market"], "accuracy"),
        },
        "blend_vs_market_log_loss": paired_diff(y, m["blend"], m["market"], "log_loss"),
        "value_rule": betting_sim(m),
    }
    MARKET_PATH.write_text(json.dumps(report, indent=2))
    p = report["pooled"]
    print(f"{report['matched']:,}/{report['of']:,} fights matched ({report['date_range'][0]} → {report['date_range'][1]})")
    for k, s in p.items():
        print(f"  {k:<9} acc {s['accuracy']['mean']:.3f}  log loss {s['log_loss']['mean']:.3f}")
    d = report["ensemble_vs_market"]["log_loss"]
    print(f"model − market log loss {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] (negative = model better)")
    v = report["value_rule"]
    print(f"value rule: {v['n']} bets, ROI {v['mean']:+.1%} [{v['lo']:+.1%}, {v['hi']:+.1%}] per unit staked")
    return report


if __name__ == "__main__":
    run_market_comparison()
