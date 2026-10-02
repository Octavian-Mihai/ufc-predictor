"""Score the upcoming card: model probabilities joined with market odds and EV."""

from __future__ import annotations

import pandas as pd

from src.ingest.odds import flatten_h2h, match_fight_odds
from src.model.predict import load_bundle, load_card, load_latest, predict_fight
from src.value.ev import annotate_model, coerce_decimal, is_underdog, pair_market


def hit_decimal(hit: dict, side: str):
    raw = hit.get(f"{side}_decimal")
    return coerce_decimal(raw) if raw is not None else None


def score_card(odds: dict, bundle: dict | None = None, latest: pd.DataFrame | None = None) -> pd.DataFrame:
    card = load_card()
    if card.empty:
        return pd.DataFrame()
    bundle = bundle or load_bundle()
    latest = latest if latest is not None else load_latest()
    flat = flatten_h2h(odds["events"])
    sample = bool(card["sample_card"].iloc[0]) if "sample_card" in card.columns else False
    rows = []
    for rec in card.to_dict("records"):
        a, b = rec.get("FIGHTER_A"), rec.get("FIGHTER_B")
        pred = predict_fight(
            a, b, bundle=bundle, latest=latest, event=rec.get("EVENT"), bout=rec.get("BOUT"), as_of_prior=sample
        )
        odds_hit = match_fight_odds(a, b, flat) or {}
        a_dec, b_dec = hit_decimal(odds_hit, "a"), hit_decimal(odds_hit, "b")
        market = None
        if a_dec is not None and b_dec is not None:
            market = annotate_model(pair_market(a_dec, b_dec), pred.get("p_a"))
        m = market or {}
        rows.append(
            {
                **rec,
                **pred,
                "odds_source": odds["source"],
                "a_decimal": a_dec,
                "b_decimal": b_dec,
                "a_fair": m.get("a_fair"),
                "b_fair": m.get("b_fair"),
                "a_ev": m.get("a_ev"),
                "b_ev": m.get("b_ev"),
                "a_edge": m.get("a_edge"),
                "b_edge": m.get("b_edge"),
                "a_underdog": is_underdog(a_dec, b_dec) if a_dec is not None else False,
                "b_underdog": is_underdog(b_dec, a_dec) if b_dec is not None else False,
            }
        )
    return pd.DataFrame(rows)
