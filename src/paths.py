"""Repo-rooted paths for data, models, and cache."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"

SEED_FILES = (
    "ufc_event_details.csv",
    "ufc_fight_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv",
    "ufc_fighter_details.csv",
    "ufc_fighter_tott.csv",
)

EVENTS_CSV = RAW_DIR / "ufc_event_details.csv"
FIGHT_DETAILS_CSV = RAW_DIR / "ufc_fight_details.csv"
FIGHT_RESULTS_CSV = RAW_DIR / "ufc_fight_results.csv"
FIGHT_STATS_CSV = RAW_DIR / "ufc_fight_stats.csv"
FIGHTER_DETAILS_CSV = RAW_DIR / "ufc_fighter_details.csv"
FIGHTER_TOTT_CSV = RAW_DIR / "ufc_fighter_tott.csv"

UPCOMING_PATH = PROCESSED_DIR / "upcoming.parquet"
FIGHT_LEVEL_PATH = PROCESSED_DIR / "fight_level.parquet"
TRAINING_PATH = PROCESSED_DIR / "training_rows.parquet"
FIGHTER_LATEST_PATH = PROCESSED_DIR / "fighter_latest.parquet"

MODEL_PATH = MODELS_DIR / "win_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

ODDS_SQLITE = CACHE_DIR / "odds.sqlite"
DUMMY_ODDS_PATH = CACHE_DIR / "dummy_odds.json"
CARD_DUMMY_ODDS_PATH = CACHE_DIR / "card_dummy.json"
ODDS_SNAPSHOTS_DIR = CACHE_DIR / "odds_snapshots"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR, ODDS_SNAPSHOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
