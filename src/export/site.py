"""Export everything the static site needs to site/data.json (no Python at request time)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import CAREER_COLS
from src.ingest.odds import get_odds, require_real_odds
from src.ledger import summarize_ledger
from src.model.backtest import BACKTEST_PATH
from src.model.market import MARKET_PATH
from src.model.market_model import REPORT_PATH as MARKET_MODEL_PATH
from src.paths import METRICS_PATH, ROOT
from src.review.recent import grade_last_events, recap_to_records
from src.value.card import score_card

SITE_DIR = ROOT / "site"
DATA_PATH = SITE_DIR / "data.json"

STAT_KEYS = [k for k in CAREER_COLS if k != "stance"]
BOUT_KEYS = [
    "EVENT", "DATE", "LOCATION", "fighter_a", "fighter_b", "WEIGHTCLASS", "status", "reason", "p_a", "p_b",
    "a_decimal", "b_decimal", "a_fair", "b_fair", "a_ev", "b_ev", "a_edge", "b_edge", "a_underdog", "b_underdog",
    "odds_source",
]
RECAP_KEYS = [
    "BOUT", "fighter_a", "fighter_b", "status", "p_a", "p_b", "pick", "winner", "grade", "a_decimal", "b_decimal",
    "a_ev", "b_ev", "a_underdog", "b_underdog",
]


def _clean(v: Any) -> Any:
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else round(float(v), 5)
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)[:10]
    if v is None or isinstance(v, (str, int)):
        return v
    return None if pd.isna(v) else str(v)


def _pick(rec: dict, keys: list[str]) -> dict:
    return {k: _clean(rec.get(k)) for k in keys}


def _load(path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build_payload() -> dict:
    odds = require_real_odds(get_odds())
    scored = score_card(odds)
    bouts = []
    for rec in scored.to_dict("records"):
        row = _pick(rec, BOUT_KEYS)
        row["a_stats"] = {k: _clean((rec.get("a_stats") or {}).get(k)) for k in STAT_KEYS}
        row["b_stats"] = {k: _clean((rec.get("b_stats") or {}).get(k)) for k in STAT_KEYS}
        bouts.append(row)

    recap = recap_to_records(grade_last_events(n=5, threshold=0.0))
    events = [
        {
            "event": _clean(ev.get("event")),
            "date": _clean(ev.get("date")),
            "location": _clean(ev.get("location")),
            "fights": [_pick(f, RECAP_KEYS) for f in ev["fights"]],
        }
        for ev in recap["events"]
    ]
    meta = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "odds_source": odds["source"],
        "odds_remaining": odds.get("remaining"),
        "model": {k: meta.get(k) for k in ("metrics", "comparison", "importance", "split", "trained_at")},
        "backtest": _load(BACKTEST_PATH),
        "market": _load(MARKET_PATH),
        "market_model": _load(MARKET_MODEL_PATH),
        "ledger": summarize_ledger(),
        "bouts": bouts,
        "recap": events,
    }


def main() -> None:
    SITE_DIR.mkdir(exist_ok=True)
    payload = build_payload()
    DATA_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {DATA_PATH} ({DATA_PATH.stat().st_size/1024:.0f} KB): {len(payload['bouts'])} bouts, {len(payload['recap'])} recap events")


if __name__ == "__main__":
    main()
