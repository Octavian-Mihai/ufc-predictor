"""Historical moneylines for the market baseline.

Source: shortlikeafox/ultimate_ufc_dataset (``ufc-master.csv``, Apache-2.0), pre-fight American odds
for UFC bouts 2010 → early 2026. Converted to ``data/external/odds_history.csv``:
``date, fighter_a, fighter_b, a_decimal, b_decimal`` (a = red corner).
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
import requests

from src.paths import DATA_DIR
from src.value.ev import american_to_decimal

SOURCE_URL = "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv"
OUT_PATH = DATA_DIR / "external" / "odds_history.csv"


def convert(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "fighter_a": raw["R_fighter"].astype(str).str.strip(),
            "fighter_b": raw["B_fighter"].astype(str).str.strip(),
            "a_decimal": raw["R_odds"].map(american_to_decimal),
            "b_decimal": raw["B_odds"].map(american_to_decimal),
        }
    )
    return out.dropna().query("a_decimal > 1 and b_decimal > 1").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and convert historical UFC odds")
    parser.add_argument("--file", help="Use a local ufc-master.csv instead of downloading")
    args = parser.parse_args()
    if args.file:
        raw = pd.read_csv(args.file)
    else:
        resp = requests.get(SOURCE_URL, timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    out = convert(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} bouts, {out['date'].min()} → {out['date'].max()})")


if __name__ == "__main__":
    main()
