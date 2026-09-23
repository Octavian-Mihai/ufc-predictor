"""American/decimal conversion, multiplicative de-vig, EV, underdog flags."""

from __future__ import annotations


def coerce_decimal(price: float | None) -> float | None:
    """Treat leftover American moneylines as American; otherwise assume decimal."""
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p <= 0 or p >= 100:
        return american_to_decimal(p)
    if p <= 1:
        return None
    return p


def american_to_decimal(american: float | None) -> float | None:
    if american is None:
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    if a > 0:
        return a / 100.0 + 1.0
    return 100.0 / abs(a) + 1.0


def decimal_to_american(decimal_odds: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def implied_prob(decimal_odds: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    return 1.0 / decimal_odds


def multiplicative_devig(p_a: float | None, p_b: float | None) -> tuple[float | None, float | None]:
    if p_a is None or p_b is None:
        return None, None
    total = p_a + p_b
    if total <= 0:
        return None, None
    return p_a / total, p_b / total


def expected_value(p_model: float | None, decimal_odds: float | None) -> float | None:
    """EV = P*(decimal-1) - (1-P)."""
    if p_model is None or decimal_odds is None:
        return None
    return p_model * (decimal_odds - 1.0) - (1.0 - p_model)


def is_underdog(decimal_odds: float | None, other_decimal: float | None) -> bool:
    if decimal_odds is None:
        return False
    if other_decimal is None:
        return float(decimal_odds) > 2.0
    return float(decimal_odds) > float(other_decimal)


def value_flag(
    p_model: float | None,
    decimal_odds: float | None,
    other_decimal: float | None,
    threshold: float = 0.05,
) -> bool:
    if not is_underdog(decimal_odds, other_decimal):
        return False
    ev = expected_value(p_model, decimal_odds)
    return ev is not None and ev >= threshold


def pair_market(a_decimal: float, b_decimal: float) -> dict[str, float | None]:
    a_dec = float(a_decimal) if a_decimal is not None else None
    b_dec = float(b_decimal) if b_decimal is not None else None
    a_imp = implied_prob(a_dec)
    b_imp = implied_prob(b_dec)
    a_fair, b_fair = multiplicative_devig(a_imp, b_imp)
    return {
        "a_decimal": a_dec,
        "b_decimal": b_dec,
        "a_implied": a_imp,
        "b_implied": b_imp,
        "a_fair": a_fair,
        "b_fair": b_fair,
        "a_ev": None,
        "b_ev": None,
    }


def annotate_model(market: dict[str, float | None], p_a: float | None) -> dict[str, float | None]:
    out = dict(market)
    if p_a is None:
        return out
    p_b = 1.0 - p_a
    out["a_ev"] = expected_value(p_a, out.get("a_decimal"))
    out["b_ev"] = expected_value(p_b, out.get("b_decimal"))
    out["a_edge"] = p_a - (out.get("a_fair") or 0.0) if out.get("a_fair") is not None else None
    out["b_edge"] = p_b - (out.get("b_fair") or 0.0) if out.get("b_fair") is not None else None
    return out


def main() -> None:
    # Plan example: model 42% on a +180 dog → decimal 2.80 → EV ≈ +17.6%
    ev = expected_value(0.42, american_to_decimal(180))
    assert ev is not None
    print(f"+180 at 42% model → EV={ev:.4f} (expected ~0.176)")
    assert abs(ev - 0.176) < 0.001
    print("EV check passed")


if __name__ == "__main__":
    main()
