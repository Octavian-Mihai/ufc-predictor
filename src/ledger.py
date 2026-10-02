"""Append-only-ish prediction ledger: a public, out-of-sample track record.

Each upcoming bout is logged with the model probability and market odds *as of the last run before the
event date*. Once the event day begins the row is frozen; after results arrive it is graded. Rows are
committed to git, so history can't be quietly rewritten.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.model.evaluation import bootstrap_ci, paired_diff, per_fight_correct, per_fight_logloss
from src.names import norm_name
from src.paths import DATA_DIR, FIGHT_LEVEL_PATH

LEDGER_PATH = DATA_DIR / "ledger" / "predictions.jsonl"
EV_THRESHOLD = 0.05
FIELDS = [
    "p_a", "p_b", "a_decimal", "b_decimal", "a_fair", "a_ev", "b_ev", "a_underdog", "b_underdog", "odds_source",
]


def _id(event: str, a: str, b: str) -> str:
    return "|".join([norm_name(event), *sorted((norm_name(a), norm_name(b)))])


def load_ledger() -> dict[str, dict]:
    if not LEDGER_PATH.exists():
        return {}
    rows = [json.loads(line) for line in LEDGER_PATH.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def save_ledger(rows: dict[str, dict]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["event_date"], r["event"], r["id"]))
    LEDGER_PATH.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in ordered))


def _clean(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 5)
    return None if v is None else str(v) if not isinstance(v, (int, str)) else v


def log_predictions(scored: pd.DataFrame, trained_at: str | None, now: datetime | None = None) -> dict[str, int]:
    """Upsert predictions for events that haven't started (UTC date strictly after today)."""
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now.date())
    rows = load_ledger()
    added = updated = frozen = 0
    for rec in scored.to_dict("records"):
        if rec.get("status") != "ok" or rec.get("p_a") is None:
            continue
        when = pd.to_datetime(rec.get("DATE"), errors="coerce")
        if pd.isna(when):
            continue
        rid = _id(rec["EVENT"], rec["fighter_a"], rec["fighter_b"])
        if when <= today:
            frozen += rid in rows
            continue
        entry = {
            "id": rid,
            "event": norm_name(rec["EVENT"]),
            "event_date": str(when.date()),
            "fighter_a": rec["fighter_a"],
            "fighter_b": rec["fighter_b"],
            **{k: _clean(rec.get(k)) for k in FIELDS},
            "logged_at": now.isoformat(timespec="seconds"),
            "model_trained_at": trained_at,
            "first_logged_at": rows.get(rid, {}).get("first_logged_at", now.isoformat(timespec="seconds")),
            "first_p_a": rows.get(rid, {}).get("first_p_a", _clean(rec.get("p_a"))),
            "winner": None,
        }
        added += rid not in rows
        updated += rid in rows
        rows[rid] = entry
    save_ledger(rows)
    return {"added": added, "updated": updated, "frozen": frozen, "total": len(rows)}


def grade_ledger() -> int:
    """Fill in winners for rows whose event has results in the fight-level data."""
    rows = load_ledger()
    if not rows or not FIGHT_LEVEL_PATH.exists():
        return 0
    fl = pd.read_parquet(FIGHT_LEVEL_PATH, columns=["event", "fighter", "won"])
    fl = fl.dropna(subset=["won"])
    fl["event"] = fl["event"].map(norm_name)
    fl["fighter"] = fl["fighter"].map(norm_name)
    wins = {(e, f): w for e, f, w in zip(fl["event"], fl["fighter"], fl["won"])}
    graded = 0
    for r in rows.values():
        if r.get("winner"):
            continue
        a_w = wins.get((r["event"], norm_name(r["fighter_a"])))
        b_w = wins.get((r["event"], norm_name(r["fighter_b"])))
        if a_w is None or b_w is None or a_w == b_w:
            continue
        r["winner"] = r["fighter_a"] if a_w == 1 else r["fighter_b"]
        graded += 1
    save_ledger(rows)
    return graded


def summarize_ledger(threshold: float = EV_THRESHOLD) -> dict:
    rows = list(load_ledger().values())
    settled = [r for r in rows if r.get("winner")]
    out = {
        "n_logged": len(rows),
        "n_pending": len(rows) - len(settled),
        "n_graded": len(settled),
        "first_logged": min((r["first_logged_at"] for r in rows), default=None),
    }
    if not settled:
        return out
    y = np.array([1.0 if r["winner"] == r["fighter_a"] else 0.0 for r in settled])
    p = np.array([r["p_a"] for r in settled], dtype=float)
    out["model"] = {
        "accuracy": bootstrap_ci(per_fight_correct(y, p)),
        "log_loss": bootstrap_ci(per_fight_logloss(y, p)),
    }
    with_mkt = [(yy, pp, r["a_fair"]) for yy, pp, r in zip(y, p, settled) if r.get("a_fair") is not None]
    if with_mkt:
        ym, pm, mk = (np.array(c, dtype=float) for c in zip(*with_mkt))
        out["vs_market"] = {
            "n": len(ym),
            "model_log_loss": float(per_fight_logloss(ym, pm).mean()),
            "market_log_loss": float(per_fight_logloss(ym, mk).mean()),
            "diff": paired_diff(ym, pm, mk, "log_loss"),
        }
    profits = []
    for r in settled:
        for s in ("a", "b"):
            if r.get(f"{s}_underdog") and r.get(f"{s}_ev") is not None and r[f"{s}_ev"] >= threshold:
                won = r["winner"] == r[f"fighter_{s}"]
                profits.append(r[f"{s}_decimal"] - 1 if won else -1.0)
    out["value_rule"] = {**(bootstrap_ci(profits) if profits else {"n": 0}), "threshold": threshold,
                         "hit_rate": float(np.mean([x > 0 for x in profits])) if profits else None}
    return out


def main() -> None:
    from src.ingest.odds import get_odds
    from src.paths import METRICS_PATH
    from src.value.card import score_card

    trained_at = json.loads(METRICS_PATH.read_text()).get("trained_at") if METRICS_PATH.exists() else None
    print(f"graded {grade_ledger()} new result(s)")
    print("logged", log_predictions(score_card(get_odds()), trained_at))
    s = summarize_ledger()
    print(f"ledger: {s['n_logged']} bouts, {s['n_graded']} graded, {s['n_pending']} pending")
    if s.get("model"):
        a = s["model"]["accuracy"]
        print(f"  accuracy {a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] on {a['n']} graded fights")


if __name__ == "__main__":
    main()
