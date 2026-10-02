from datetime import datetime, timezone

import pandas as pd
import pytest

import src.ledger as L


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LEDGER_PATH", tmp_path / "predictions.jsonl")


def card(p_a=0.6, date="October 03, 2026"):
    return pd.DataFrame([{
        "EVENT": "UFC 999: Test", "DATE": date, "fighter_a": "Ann", "fighter_b": "Bea", "status": "ok",
        "p_a": p_a, "p_b": 1 - p_a, "a_decimal": 1.7, "b_decimal": 2.2, "a_fair": 0.56, "a_ev": 0.02, "b_ev": 0.1,
        "a_underdog": False, "b_underdog": True, "odds_source": "api",
    }])


def at(day):
    return datetime(2026, 10, day, 12, tzinfo=timezone.utc)


def test_rerun_updates_instead_of_duplicating_and_keeps_first_prediction():
    L.log_predictions(card(0.60), "t", now=at(1))
    r = L.log_predictions(card(0.65), "t", now=at(2))
    rows = L.load_ledger()
    assert r["updated"] == 1 and len(rows) == 1
    row = next(iter(rows.values()))
    assert row["p_a"] == 0.65 and row["first_p_a"] == 0.60


def test_prediction_freezes_once_the_event_day_starts():
    L.log_predictions(card(0.60), "t", now=at(1))
    L.log_predictions(card(0.99), "t", now=at(3))  # event day: must not overwrite
    L.log_predictions(card(0.01), "t", now=at(4))
    assert next(iter(L.load_ledger().values()))["p_a"] == 0.60


def test_events_already_started_are_never_added_retroactively():
    r = L.log_predictions(card(0.7), "t", now=at(4))
    assert r["total"] == 0 and L.load_ledger() == {}


def test_insufficient_data_bouts_are_skipped():
    c = card().assign(status="insufficient_data", p_a=None)
    assert L.log_predictions(c, "t", now=at(1))["total"] == 0


def test_bout_identity_ignores_corner_order():
    L.log_predictions(card(), "t", now=at(1))
    swapped = card().rename(columns={"fighter_a": "fighter_b", "fighter_b": "fighter_a"})
    L.log_predictions(swapped, "t", now=at(2))
    assert len(L.load_ledger()) == 1


def test_grading_and_summary(monkeypatch):
    L.log_predictions(card(0.7), "t", now=at(1))
    fl = pd.DataFrame({"event": ["UFC 999: Test"] * 2, "fighter": ["Ann", "Bea"], "won": [1.0, 0.0]})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: fl)
    monkeypatch.setattr(L.FIGHT_LEVEL_PATH.__class__, "exists", lambda self: True)
    assert L.grade_ledger() == 1
    assert L.grade_ledger() == 0  # idempotent
    s = L.summarize_ledger(threshold=0.05)
    assert s["n_graded"] == 1 and s["model"]["accuracy"]["mean"] == 1.0
    # the +10% EV underdog (Bea) lost, so the value rule lost one unit
    assert s["value_rule"]["mean"] == -1.0
