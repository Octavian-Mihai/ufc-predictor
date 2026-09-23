"""Before-card refresh: new UFC Stats fights plus one cached odds pull."""

from __future__ import annotations

from src.features.build import build_features
from src.ingest import odds, ufcstats


def main() -> None:
    print("== UFC Stats incremental ==")
    ufcstats.incremental_update()
    print("== upcoming card ==")
    ufcstats.fetch_upcoming_card()
    print("== rebuild leak-free features ==")
    build_features()
    print("== odds (24h cache) ==")
    result = odds.get_odds()
    written = odds.save_odds_snapshot(result)
    print(
        f"odds source={result['source']} events={len(result['events'])} "
        f"remaining={result['remaining']} snapshots={len(written)}"
    )
    print("refresh complete (retrain separately with: python -m src.model.train)")


if __name__ == "__main__":
    main()
