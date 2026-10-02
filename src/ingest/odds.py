"""The Odds API client with 24h SQLite cache and dummy JSON fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.names import match_name, norm_name
from src.paths import (
    CACHE_DIR,
    CARD_DUMMY_ODDS_PATH,
    DUMMY_ODDS_PATH,
    ODDS_SNAPSHOTS_DIR,
    ODDS_SQLITE,
    UPCOMING_PATH,
    ensure_dirs,
)
from src.value.ev import american_to_decimal, coerce_decimal

ODDS_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
CACHE_TTL_SEC = 24 * 60 * 60
CACHE_KEY = "mma_us_h2h_decimal"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(ODDS_SQLITE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_cache (
            cache_key TEXT PRIMARY KEY,
            fetched_at REAL NOT NULL,
            remaining TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _load_dotenv() -> None:
    load_dotenv()


def api_key() -> str:
    _load_dotenv()
    return (os.getenv("ODDS_API_KEY") or "").strip()


def _read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return []


def _cache_get(key: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT fetched_at, remaining, payload FROM odds_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None, None
    fetched_at, remaining, payload = row
    if time.time() - float(fetched_at) > CACHE_TTL_SEC:
        return None, remaining
    return json.loads(payload), remaining


def _cache_put(key: str, payload: list[dict[str, Any]], remaining: str | None) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO odds_cache (cache_key, fetched_at, remaining, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                fetched_at = excluded.fetched_at,
                remaining = excluded.remaining,
                payload = excluded.payload
            """,
            (key, time.time(), remaining, json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def synthesize_odds_for_card(matchups: pd.DataFrame) -> list[dict[str, Any]]:
    """Deterministic dummy moneylines so the UI works without an API key."""
    events: list[dict[str, Any]] = []
    now = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    prices = [(-250, 200), (-180, 150), (-145, 120), (105, -125), (-110, -110), (165, -200)]
    for i, rec in enumerate(matchups.to_dict("records")):
        a = norm_name(rec.get("FIGHTER_A") or rec.get("fighter_a"))
        b = norm_name(rec.get("FIGHTER_B") or rec.get("fighter_b"))
        if not a or not b:
            continue
        seed = int(hashlib.md5(f"{a}|{b}".encode()).hexdigest()[:8], 16)
        fav, dog = prices[seed % len(prices)]
        # Hash decides who is listed as the favorite on the board.
        if seed % 2 == 0:
            a_price, b_price = fav, dog
        else:
            a_price, b_price = dog, fav
        a_price = round(american_to_decimal(a_price) or 0.0, 2)
        b_price = round(american_to_decimal(b_price) or 0.0, 2)
        commence = rec.get("DATE") or now
        try:
            commence_iso = pd.to_datetime(commence).strftime("%Y-%m-%dT00:00:00Z")
        except Exception:  # noqa: BLE001
            commence_iso = now
        events.append(
            {
                "id": f"dummy-{seed:x}",
                "sport_key": "mma_mixed_martial_arts",
                "sport_title": "MMA",
                "commence_time": commence_iso,
                "home_team": a,
                "away_team": b,
                "bookmakers": [
                    {
                        "key": "dummy_book",
                        "title": "Dummy (no API key)",
                        "last_update": now,
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": a, "price": a_price},
                                    {"name": b, "price": b_price},
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    return events


def fetch_live_odds() -> tuple[list[dict[str, Any]], str | None]:
    key = api_key()
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = requests.get(ODDS_URL, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"The Odds API credits remaining={remaining} used={used}")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected Odds API payload")
    return data, remaining


def get_odds(force_refresh: bool = False) -> dict[str, Any]:
    """Return odds payload plus source metadata. Never requires a paid plan."""
    ensure_dirs()
    remaining = None
    source = "dummy"
    payload: list[dict[str, Any]] = []

    cached, cached_remaining = (None, None) if force_refresh else _cache_get(CACHE_KEY)
    key = api_key()

    if cached is not None and key:
        payload, remaining, source = cached, cached_remaining, "cache"
    elif key:
        try:
            payload, remaining = fetch_live_odds()
            _cache_put(CACHE_KEY, payload, remaining)
            source = "api"
        except Exception as exc:  # noqa: BLE001
            print(f"Odds API failed ({exc}); falling back to cache/dummy")
            stale, remaining = _cache_get(CACHE_KEY) if False else (cached, cached_remaining)
            if stale:
                payload, source = stale, "stale-cache"
            else:
                payload, source = _read_json(DUMMY_ODDS_PATH), "dummy"
    else:
        payload = _read_json(DUMMY_ODDS_PATH)
        source = "dummy"

    if UPCOMING_PATH.exists() and not key:
        try:
            card = pd.read_parquet(UPCOMING_PATH)
        except Exception:  # noqa: BLE001
            card = pd.DataFrame()
        if not card.empty:
            synthesized = synthesize_odds_for_card(card)
            CARD_DUMMY_ODDS_PATH.write_text(json.dumps(synthesized, indent=2))
            payload = synthesized
            source = "dummy-card"

    if not payload:
        payload = _read_json(DUMMY_ODDS_PATH)
        source = "dummy"

    result = {
        "events": payload,
        "source": source,
        "remaining": remaining,
        "fetched_at": _now().isoformat(),
    }
    save_odds_snapshot(result)
    return result


def require_real_odds(result: dict[str, Any]) -> dict[str, Any]:
    """Raise if the odds are synthetic, so fake lines never reach the ledger or the public site."""
    if str(result.get("source", "")).startswith("dummy"):
        raise RuntimeError("Refusing to use synthetic odds; set ODDS_API_KEY or fix the odds fetch.")
    return result


def flatten_h2h(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Median decimal moneyline per fighter across US books."""
    rows: list[dict[str, Any]] = []
    for event in events:
        home = norm_name(event.get("home_team"))
        away = norm_name(event.get("away_team"))
        prices: dict[str, list[float]] = {home: [], away: []}
        books = []
        for book in event.get("bookmakers") or []:
            books.append(book.get("title") or book.get("key"))
            for market in book.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes") or []:
                    name = norm_name(outcome.get("name"))
                    price = outcome.get("price")
                    if name not in prices:
                        prices[name] = []
                    try:
                        dec = coerce_decimal(float(price))
                    except (TypeError, ValueError):
                        continue
                    if dec is not None:
                        prices[name].append(dec)
        for fighter, vals in prices.items():
            if not fighter or not vals:
                continue
            series = pd.Series(vals)
            rows.append(
                {
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": home,
                    "away_team": away,
                    "fighter": fighter,
                    "decimal": float(series.median()),
                    "n_books": int(series.size),
                    "books": ", ".join(str(b) for b in books if b),
                }
            )
    return pd.DataFrame(rows)


def _snapshot_event_id(event: dict[str, Any]) -> str:
    eid = str(event.get("id") or "").strip()
    if not eid:
        home = norm_name(event.get("home_team"))
        away = norm_name(event.get("away_team"))
        eid = hashlib.md5(f"{home}|{away}".encode()).hexdigest()[:12]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", eid)[:80] or "unknown"


def _label_odds_source(source: str | None) -> str:
    """Snapshots from live/cache pulls vs dummy lines that are not closers."""
    src = (source or "").strip().lower()
    if src in {"api", "cache", "stale-cache", "snapshot"}:
        return "snapshot"
    return "dummy"


def save_odds_snapshot(result: dict[str, Any]) -> list[Path]:
    """Persist one JSON file per event so recap can grade value later."""
    ensure_dirs()
    events = result.get("events") or []
    if not isinstance(events, list) or not events:
        return []
    source = str(result.get("source") or "unknown")
    fetched_at = str(result.get("fetched_at") or _now().isoformat())
    written: list[Path] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        safe = _snapshot_event_id(event)
        path = ODDS_SNAPSHOTS_DIR / f"{safe}.json"
        path.write_text(
            json.dumps(
                {
                    "source": source,
                    "fetched_at": fetched_at,
                    "event": event,
                },
                indent=2,
            )
        )
        written.append(path)
    return written


def load_snapshot_odds() -> pd.DataFrame:
    """Flatten stored snapshots: fighter decimal prices plus source."""
    ensure_dirs()
    frames: list[pd.DataFrame] = []
    for path in sorted(ODDS_SNAPSHOTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("event"), dict):
            event = data["event"]
            source = data.get("source")
            fetched_at = data.get("fetched_at")
        elif isinstance(data, dict) and data.get("bookmakers"):
            event = data
            source = data.get("source")
            fetched_at = data.get("fetched_at")
        else:
            continue
        flat = flatten_h2h([event])
        if flat.empty:
            continue
        flat["odds_source"] = _label_odds_source(str(source) if source else None)
        flat["raw_source"] = source
        flat["fetched_at"] = fetched_at
        flat["snapshot_file"] = path.name
        frames.append(flat)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "fetched_at" in out.columns:
        out["_fetched"] = pd.to_datetime(out["fetched_at"], errors="coerce", utc=True)
        out = out.sort_values("_fetched", ascending=False, na_position="last")
        out = out.drop(columns=["_fetched"])
    return out.reset_index(drop=True)


def match_snapshot_odds(
    fighter_a: str,
    fighter_b: str,
    snapshot_rows: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if snapshot_rows is None or snapshot_rows.empty:
        return None
    df = snapshot_rows
    event_ids = list(dict.fromkeys(df["event_id"].tolist())) if "event_id" in df.columns else [None]
    for eid in event_ids:
        group = df if eid is None else df.loc[df["event_id"] == eid]
        hit = match_fight_odds(fighter_a, fighter_b, group)
        if hit:
            src = group["odds_source"].iloc[0] if "odds_source" in group.columns else "snapshot"
            hit["odds_source"] = _label_odds_source(str(src))
            return hit
    return None


def resolve_fight_odds(
    fighter_a: str,
    fighter_b: str,
    date: Any = None,
    snapshot_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Prefer a stored snapshot; otherwise the same deterministic dummy moneylines."""
    hit = match_snapshot_odds(fighter_a, fighter_b, snapshot_rows)
    if hit:
        return hit
    matchups = pd.DataFrame(
        [{"FIGHTER_A": fighter_a, "FIGHTER_B": fighter_b, "DATE": date}]
    )
    dummy = flatten_h2h(synthesize_odds_for_card(matchups))
    matched = match_fight_odds(fighter_a, fighter_b, dummy) or {}
    matched["odds_source"] = "dummy"
    return matched


def match_fight_odds(
    fighter_a: str,
    fighter_b: str,
    odds_rows: pd.DataFrame,
) -> dict[str, Any] | None:
    if odds_rows is None or odds_rows.empty:
        return None
    names = list(pd.unique(odds_rows["fighter"].dropna().astype(str)))
    a_hit, a_score = match_name(fighter_a, names)
    b_hit, b_score = match_name(fighter_b, names)
    if not a_hit or not b_hit or a_hit == b_hit:
        return None
    a_row = odds_rows.loc[odds_rows["fighter"] == a_hit].iloc[0]
    b_row = odds_rows.loc[odds_rows["fighter"] == b_hit].iloc[0]

    def _row_decimal(row: pd.Series) -> float | None:
        if "decimal" in row.index and pd.notna(row.get("decimal")):
            return coerce_decimal(row["decimal"])
        if "american" in row.index and pd.notna(row.get("american")):
            return american_to_decimal(row["american"])
        return None

    a_dec = _row_decimal(a_row)
    b_dec = _row_decimal(b_row)
    if a_dec is None or b_dec is None:
        return None
    return {
        "fighter_a": a_hit,
        "fighter_b": b_hit,
        "a_decimal": a_dec,
        "b_decimal": b_dec,
        "a_match": a_score,
        "b_match": b_score,
        "commence_time": a_row.get("commence_time"),
        "books": a_row.get("books"),
    }


def main() -> None:
    result = get_odds()
    print(f"source={result['source']} events={len(result['events'])} remaining={result['remaining']}")
    flat = flatten_h2h(result["events"])
    print(flat.head(12).to_string(index=False) if not flat.empty else "no h2h rows")


if __name__ == "__main__":
    main()
