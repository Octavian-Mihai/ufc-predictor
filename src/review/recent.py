"""Leak-free recap of the last completed UFC cards vs official results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.build import FIGHT_LEVEL_PATH
from src.ingest.odds import load_snapshot_odds, resolve_fight_odds
from src.model.predict import load_bundle, load_latest, predict_fight
from src.names import norm_name, split_bout
from src.paths import EVENTS_CSV, FIGHT_RESULTS_CSV
from src.value.ev import annotate_model, coerce_decimal, is_underdog, pair_market


def last_completed_events(n: int = 5) -> pd.DataFrame:
    """Most recent dated UFC events that already have official results."""
    events = pd.read_csv(EVENTS_CSV)
    results = pd.read_csv(FIGHT_RESULTS_CSV)
    events = events.copy()
    events["event"] = events["EVENT"].map(norm_name)
    events["date"] = pd.to_datetime(events["DATE"], errors="coerce")
    results = results.copy()
    results["event"] = results["EVENT"].map(norm_name)
    have = set(results.loc[results["event"] != "", "event"])
    done = events.loc[events["event"].isin(have) & events["date"].notna()].copy()
    done = done.sort_values(["date", "event"], ascending=[False, True])
    done = done.drop_duplicates("event", keep="first")
    return done.head(int(n)).reset_index(drop=True)


def _outcome_winner(outcome: Any, red: str, blue: str) -> tuple[str | None, str]:
    """W/L = first/red in BOUT; L/W = second/blue. NC/draw have no winner."""
    raw = "" if outcome is None or (isinstance(outcome, float) and pd.isna(outcome)) else str(outcome)
    o = raw.replace(" ", "").strip().upper()
    if o == "W/L":
        return red, "W/L"
    if o == "L/W":
        return blue, "L/W"
    if "NC" in o:
        return None, "NC"
    if o in {"D/D", "DRAW"} or o.startswith("D/") or o.endswith("/D"):
        return None, "draw"
    return None, o or "unknown"


def _same_name(left: str | None, right: str | None) -> bool:
    return bool(left) and bool(right) and norm_name(left) == norm_name(right)


def _hit_decimal(hit: dict[str, Any], side: str):
    raw = hit.get(f"{side}_decimal") if hit else None
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return coerce_decimal(raw)


def apply_value_flags(fights: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Flag moneyline dogs whose model EV clears the slider; hit if that dog won."""
    df = fights.copy()
    if df.empty:
        return df
    ok = df["status"] == "ok"
    df["a_value"] = ok & df["a_underdog"] & df["a_ev"].notna() & (df["a_ev"] >= threshold)
    df["b_value"] = ok & df["b_underdog"] & df["b_ev"].notna() & (df["b_ev"] >= threshold)
    df["value_flag"] = df["a_value"] | df["b_value"]
    df["value_side"] = [
        r["fighter_a"] if r["a_value"] else (r["fighter_b"] if r["b_value"] else "")
        for r in df.to_dict("records")
    ]
    df["value_ev"] = [
        r["a_ev"] if r["a_value"] else (r["b_ev"] if r["b_value"] else None)
        for r in df.to_dict("records")
    ]
    hits: list[bool | None] = []
    for rec in df.to_dict("records"):
        if not rec.get("value_flag") or not rec.get("has_winner"):
            hits.append(None)
            continue
        hits.append(_same_name(rec.get("winner"), rec.get("value_side")))
    df["value_hit"] = hits
    return df


def grade_event(
    event_name: str,
    *,
    date: Any = None,
    location: str = "",
    threshold: float = 0.05,
    bundle: dict | None = None,
    latest: pd.DataFrame | None = None,
    fight_level: pd.DataFrame | None = None,
    snapshot_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score every bout on one completed card (winner pick + value underdog)."""
    results = pd.read_csv(FIGHT_RESULTS_CSV)
    results["event"] = results["EVENT"].map(norm_name)
    subset = results.loc[results["event"] == norm_name(event_name)].copy()
    bundle = bundle or load_bundle()
    latest = latest if latest is not None else load_latest()
    if fight_level is None and FIGHT_LEVEL_PATH.exists():
        fight_level = pd.read_parquet(FIGHT_LEVEL_PATH)
    if snapshot_rows is None:
        snapshot_rows = load_snapshot_odds()

    rows: list[dict[str, Any]] = []
    for rec in subset.to_dict("records"):
        bout = str(rec.get("BOUT") or "").strip()
        red, blue = split_bout(bout)
        winner, result_kind = _outcome_winner(rec.get("OUTCOME"), red, blue)
        pred = predict_fight(
            red,
            blue,
            bundle=bundle,
            latest=latest,
            event=event_name,
            bout=bout,
            as_of_prior=True,
            fight_level=fight_level,
        )
        odds_hit = resolve_fight_odds(red, blue, date=date, snapshot_rows=snapshot_rows)
        a_dec = _hit_decimal(odds_hit, "a")
        b_dec = _hit_decimal(odds_hit, "b")
        market = None
        p_a = pred.get("p_a")
        if a_dec is not None and b_dec is not None:
            market = annotate_model(pair_market(a_dec, b_dec), p_a)

        status = pred.get("status") or "insufficient_data"
        pick = None
        pick_p = None
        if status == "ok" and p_a is not None:
            if float(p_a) > 0.5:
                pick = pred.get("fighter_a") or red
                pick_p = float(p_a)
            else:
                pick = pred.get("fighter_b") or blue
                pick_p = float(pred.get("p_b") if pred.get("p_b") is not None else 1.0 - float(p_a))

        has_winner = winner is not None
        graded = status == "ok" and has_winner and pick is not None
        correct = _same_name(pick, winner) if graded else None
        if status != "ok":
            grade = "insufficient_data"
        elif not has_winner:
            grade = result_kind
        elif correct:
            grade = "correct"
        else:
            grade = "wrong"

        dog = ""
        a_under = is_underdog(a_dec, b_dec) if a_dec is not None else False
        b_under = is_underdog(b_dec, a_dec) if b_dec is not None else False
        if a_under:
            dog = pred.get("fighter_a") or red
        elif b_under:
            dog = pred.get("fighter_b") or blue

        rows.append(
            {
                "EVENT": event_name,
                "DATE": date,
                "LOCATION": location,
                "BOUT": bout,
                "METHOD": str(rec.get("METHOD") or "").strip(),
                "OUTCOME": rec.get("OUTCOME"),
                "fighter_a": pred.get("fighter_a") or red,
                "fighter_b": pred.get("fighter_b") or blue,
                "status": status,
                "reason": pred.get("reason"),
                "p_a": p_a,
                "p_b": pred.get("p_b"),
                "pick": pick,
                "pick_p": pick_p,
                "winner": winner,
                "result_kind": result_kind,
                "has_winner": has_winner,
                "graded": graded,
                "correct": correct,
                "grade": grade,
                "odds_source": odds_hit.get("odds_source") or "dummy",
                "a_decimal": a_dec,
                "b_decimal": b_dec,
                "a_ev": market.get("a_ev") if market else None,
                "b_ev": market.get("b_ev") if market else None,
                "a_underdog": a_under,
                "b_underdog": b_under,
                "dog": dog,
            }
        )

    fights = apply_value_flags(pd.DataFrame(rows), threshold)
    return {
        "event": event_name,
        "date": date,
        "location": location,
        "fights": fights,
        **_event_totals(fights),
    }


def _event_totals(fights: pd.DataFrame) -> dict[str, Any]:
    if fights is None or fights.empty:
        return {
            "n_fights": 0,
            "n_graded": 0,
            "n_correct": 0,
            "n_wrong": 0,
            "n_excluded": 0,
            "accuracy": None,
            "n_value_flags": 0,
            "n_value_hits": 0,
            "n_value_misses": 0,
            "value_hit_rate": None,
        }
    graded = fights["graded"] == True  # noqa: E712
    n_graded = int(graded.sum())
    n_correct = int((fights["correct"] == True).sum())  # noqa: E712
    n_wrong = int((fights["correct"] == False).sum())  # noqa: E712
    flagged = fights["value_flag"] == True  # noqa: E712
    decided_flags = flagged & (fights["has_winner"] == True)  # noqa: E712
    n_flags = int(decided_flags.sum())
    n_hits = int((fights["value_hit"] == True).sum())  # noqa: E712
    n_misses = int((fights["value_hit"] == False).sum())  # noqa: E712
    return {
        "n_fights": int(len(fights)),
        "n_graded": n_graded,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_excluded": int(len(fights) - n_graded),
        "accuracy": (n_correct / n_graded) if n_graded else None,
        "n_value_flags": n_flags,
        "n_value_hits": n_hits,
        "n_value_misses": n_misses,
        "value_hit_rate": (n_hits / n_flags) if n_flags else None,
    }


def _sum_totals(parts: list[dict[str, Any]]) -> dict[str, Any]:
    n_fights = sum(p.get("n_fights") or 0 for p in parts)
    n_graded = sum(p.get("n_graded") or 0 for p in parts)
    n_correct = sum(p.get("n_correct") or 0 for p in parts)
    n_wrong = sum(p.get("n_wrong") or 0 for p in parts)
    n_flags = sum(p.get("n_value_flags") or 0 for p in parts)
    n_hits = sum(p.get("n_value_hits") or 0 for p in parts)
    n_misses = sum(p.get("n_value_misses") or 0 for p in parts)
    return {
        "n_events": len(parts),
        "n_fights": n_fights,
        "n_graded": n_graded,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_excluded": n_fights - n_graded,
        "accuracy": (n_correct / n_graded) if n_graded else None,
        "n_value_flags": n_flags,
        "n_value_hits": n_hits,
        "n_value_misses": n_misses,
        "value_hit_rate": (n_hits / n_flags) if n_flags else None,
    }


def _as_fights_frame(fights: Any) -> pd.DataFrame:
    if fights is None:
        return pd.DataFrame()
    if isinstance(fights, pd.DataFrame):
        return fights
    if isinstance(fights, list):
        return pd.DataFrame(fights)
    return pd.DataFrame()


def _event_as_dict(ev: Any) -> dict[str, Any]:
    if isinstance(ev, dict):
        return dict(ev)
    if isinstance(ev, pd.Series):
        return ev.to_dict()
    return {}


def recap_to_records(recap: dict[str, Any]) -> dict[str, Any]:
    """Replace fight DataFrames with row dicts so Streamlit can cache the recap."""
    raw_events = recap.get("events") if recap else None
    if raw_events is None:
        raw_events = []
    elif isinstance(raw_events, pd.DataFrame):
        raw_events = raw_events.to_dict("records")
    events: list[dict[str, Any]] = []
    for ev in raw_events:
        row = _event_as_dict(ev)
        fights = _as_fights_frame(row.get("fights"))
        row["fights"] = fights.to_dict("records")
        events.append(row)
    return {"events": events, "summary": recap.get("summary") or {}}


def rethreshold_recap(recap: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Recompute value flags/hits after the EV slider changes (no new model calls)."""
    raw_events = recap.get("events") if recap else None
    if raw_events is None:
        raw_events = []
    elif isinstance(raw_events, pd.DataFrame):
        raw_events = raw_events.to_dict("records")
    events: list[dict[str, Any]] = []
    for ev in raw_events:
        row = _event_as_dict(ev)
        fights = apply_value_flags(_as_fights_frame(row.get("fights")), threshold)
        events.append({**row, "fights": fights, **_event_totals(fights)})
    return {"events": events, "summary": _sum_totals(events)}


def grade_last_events(n: int = 5, threshold: float = 0.05) -> dict[str, Any]:
    """Grade the last n completed events with the current trained model."""
    events = last_completed_events(n)
    bundle = load_bundle()
    latest = load_latest()
    fight_level = pd.read_parquet(FIGHT_LEVEL_PATH) if FIGHT_LEVEL_PATH.exists() else None
    snapshot_rows = load_snapshot_odds()
    graded: list[dict[str, Any]] = []
    for rec in events.to_dict("records"):
        graded.append(
            grade_event(
                rec.get("EVENT") or rec.get("event"),
                date=rec.get("DATE") or rec.get("date"),
                location=str(rec.get("LOCATION") or ""),
                threshold=threshold,
                bundle=bundle,
                latest=latest,
                fight_level=fight_level,
                snapshot_rows=snapshot_rows,
            )
        )
    return {"events": graded, "summary": _sum_totals(graded)}


def main() -> None:
    recap = grade_last_events(n=5, threshold=0.05)
    summary = recap["summary"]
    acc = summary["accuracy"]
    hit = summary["value_hit_rate"]
    print(
        f"last {summary['n_events']} events · graded {summary['n_graded']}/{summary['n_fights']} · "
        f"accuracy={acc:.3f}" if acc is not None else
        f"last {summary['n_events']} events · graded {summary['n_graded']}/{summary['n_fights']}"
    )
    print(
        f"value flags={summary['n_value_flags']} hits={summary['n_value_hits']} "
        f"misses={summary['n_value_misses']}"
        + (f" hit_rate={hit:.3f}" if hit is not None else "")
    )
    for event in recap["events"]:
        when = event.get("date") or ""
        acc_e = event.get("accuracy")
        acc_s = f"{acc_e:.3f}" if acc_e is not None else "—"
        print(f"- {event['event']} ({when}) {event['n_correct']}/{event['n_graded']} acc={acc_s}")


if __name__ == "__main__":
    main()
