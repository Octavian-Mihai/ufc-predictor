"""Upcoming (or sample) matchup → P(fighter wins)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.build import CAREER_COLS, FEATURE_COLS, FIGHTER_LATEST_PATH, FIGHT_LEVEL_PATH, diffs_from_sides
from src.names import match_name, norm_name, split_bout
from src.paths import MODEL_PATH, UPCOMING_PATH


def load_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m src.model.train")
    return joblib.load(path)


def load_latest() -> pd.DataFrame:
    if not FIGHTER_LATEST_PATH.exists():
        raise FileNotFoundError("Missing fighter snapshots. Run: python -m src.features.build")
    df = pd.read_parquet(FIGHTER_LATEST_PATH)
    df["fighter"] = df["fighter"].map(norm_name)
    return df


def _lookup_fighter(name: str, latest: pd.DataFrame) -> tuple[pd.Series | None, str | None, float]:
    names = list(latest["fighter"])
    hit, score = match_name(name, names, cutoff=80)
    if not hit:
        return None, None, score
    row = latest.loc[latest["fighter"] == hit].iloc[0]
    return row, hit, score


def _row_to_frame(row: pd.Series) -> pd.DataFrame:
    data = {col: [row.get(col)] for col in CAREER_COLS}
    return pd.DataFrame(data)


def _prior_rows(event: str, bout: str, fight_level: pd.DataFrame) -> pd.DataFrame | None:
    if fight_level is None or fight_level.empty:
        return None
    ev = norm_name(event)
    subset = fight_level.loc[fight_level["event"].map(norm_name) == ev]
    if subset.empty:
        return None
    # Prefer exact bout, else fighter names.
    hit = subset.loc[subset["bout"].astype(str).str.strip() == str(bout).strip()]
    if hit.empty:
        a, b = split_bout(bout)
        a, b = norm_name(a), norm_name(b)
        if not a or not b:
            return None
        fighters = subset["fighter"].map(norm_name)
        opponents = subset["opponent"].map(norm_name) if "opponent" in subset.columns else fighters
        hit = subset.loc[
            ((fighters == a) & (opponents == b)) | ((fighters == b) & (opponents == a))
        ]
    if hit.empty or hit["fighter"].map(norm_name).nunique() < 2:
        return None
    return hit


def predict_fight(
    fighter_a: str,
    fighter_b: str,
    bundle: dict | None = None,
    latest: pd.DataFrame | None = None,
    event: str | None = None,
    bout: str | None = None,
    as_of_prior: bool = False,
    fight_level: pd.DataFrame | None = None,
) -> dict:
    """Return model probability that fighter_a wins, plus career snapshots."""
    bundle = bundle or load_bundle()
    latest = latest if latest is not None else load_latest()
    model = bundle["model"]
    cols = bundle.get("feature_cols") or FEATURE_COLS

    a_row = b_row = None
    a_name = norm_name(fighter_a)
    b_name = norm_name(fighter_b)
    a_score = b_score = 0.0

    if as_of_prior and event:
        if fight_level is None and FIGHT_LEVEL_PATH.exists():
            fight_level = pd.read_parquet(FIGHT_LEVEL_PATH)
        if fight_level is not None:
            prior = _prior_rows(event, bout or f"{fighter_a} vs. {fighter_b}", fight_level)
            if prior is not None and len(prior) >= 2:
                names = list(prior["fighter"].map(norm_name))
                a_hit, a_score = match_name(fighter_a, names, cutoff=80)
                b_hit, b_score = match_name(fighter_b, names, cutoff=80)
                if a_hit and b_hit:
                    a_row = prior.loc[prior["fighter"].map(norm_name) == a_hit].iloc[0]
                    b_row = prior.loc[prior["fighter"].map(norm_name) == b_hit].iloc[0]
                    a_name, b_name = a_hit, b_hit

    if (a_row is None or b_row is None) and not as_of_prior:
        a_row, a_hit, a_score = _lookup_fighter(fighter_a, latest)
        b_row, b_hit, b_score = _lookup_fighter(fighter_b, latest)
        a_name = a_hit or norm_name(fighter_a)
        b_name = b_hit or norm_name(fighter_b)

    a_prior = int(a_row.get("prior_fights") or 0) if a_row is not None else 0
    b_prior = int(b_row.get("prior_fights") or 0) if b_row is not None else 0
    missing = []
    if a_row is None or a_prior <= 0:
        missing.append(a_name or fighter_a)
    if b_row is None or b_prior <= 0:
        missing.append(b_name or fighter_b)

    if missing:
        return {
            "status": "insufficient_data",
            "reason": "insufficient data",
            "missing": missing,
            "fighter_a": a_name,
            "fighter_b": b_name,
            "p_a": None,
            "p_b": None,
            "a_stats": a_row.to_dict() if a_row is not None else {},
            "b_stats": b_row.to_dict() if b_row is not None else {},
            "a_prior": a_prior,
            "b_prior": b_prior,
        }

    feats = diffs_from_sides(_row_to_frame(a_row), _row_to_frame(b_row))
    X = feats.reindex(columns=cols)
    p_a = float(model.predict_proba(X)[0, 1])
    return {
        "status": "ok",
        "reason": None,
        "missing": [],
        "fighter_a": a_name,
        "fighter_b": b_name,
        "p_a": p_a,
        "p_b": 1.0 - p_a,
        "a_stats": {k: a_row.get(k) for k in CAREER_COLS if k in a_row.index},
        "b_stats": {k: b_row.get(k) for k in CAREER_COLS if k in b_row.index},
        "a_prior": a_prior,
        "b_prior": b_prior,
        "a_match": a_score,
        "b_match": b_score,
    }


def load_card() -> pd.DataFrame:
    if not UPCOMING_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(UPCOMING_PATH)


def predict_card(card: pd.DataFrame | None = None) -> pd.DataFrame:
    card = load_card() if card is None else card
    if card.empty:
        return pd.DataFrame()
    bundle = load_bundle()
    latest = load_latest()
    rows = []
    sample = bool(card["sample_card"].iloc[0]) if "sample_card" in card.columns else False
    for rec in card.to_dict("records"):
        pred = predict_fight(
            rec.get("FIGHTER_A") or rec.get("fighter_a"),
            rec.get("FIGHTER_B") or rec.get("fighter_b"),
            bundle=bundle,
            latest=latest,
            event=rec.get("EVENT"),
            bout=rec.get("BOUT"),
            as_of_prior=sample,
        )
        rows.append({**rec, **pred})
    return pd.DataFrame(rows)


def main() -> None:
    card = load_card()
    if card.empty:
        print("No upcoming card. Run: python -m src.ingest.ufcstats --upcoming")
        return
    out = predict_card(card)
    cols = ["EVENT", "FIGHTER_A", "FIGHTER_B", "status", "p_a", "p_b", "reason"]
    show = [c for c in cols if c in out.columns]
    print(out[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
