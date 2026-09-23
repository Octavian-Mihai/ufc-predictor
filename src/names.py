"""Fighter name normalization and fuzzy matching."""

from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz, process

_SPACE = re.compile(r"\s+")


def norm_name(name: str | None) -> str:
    if name is None:
        return ""
    text = str(name).replace("\xa0", " ").strip()
    text = _SPACE.sub(" ", text)
    return text


def split_bout(bout: str | None) -> tuple[str, str]:
    if not bout:
        return "", ""
    parts = re.split(r"\s+vs\.?\s+", str(bout).strip(), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return "", ""
    return norm_name(parts[0]), norm_name(parts[1])


def match_name(
    query: str,
    candidates: Iterable[str],
    cutoff: int = 78,
) -> tuple[str | None, float]:
    """Return best candidate and score, or (None, 0) if below cutoff."""
    names = [c for c in candidates if c]
    if not query or not names:
        return None, 0.0
    hit = process.extractOne(query, names, scorer=fuzz.token_sort_ratio)
    if hit is None:
        return None, 0.0
    name, score, _ = hit
    if score < cutoff:
        return None, float(score)
    return name, float(score)
