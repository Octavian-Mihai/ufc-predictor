"""Synthetic fight tables shaped like build_fighter_fight_table() output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

STAT_DEFAULTS = dict(kd=0.0, sig_l=20.0, sig_a=40.0, td_l=1.0, td_a=2.0, leg_l=3.0, leg_a=4.0, dist_l=12.0,
                     dist_a=30.0, ground_l=4.0, ground_a=5.0, ctrl_sec=60.0)
STATS = list(STAT_DEFAULTS)


def make_fights(bouts: list[dict]) -> pd.DataFrame:
    """bouts: dict(date, event, red, blue, winner='red'|'blue'|None, red_stats={}, blue_stats={}, minutes=10)."""
    rows = []
    for b in bouts:
        winner = b.get("winner")
        side_stats = {
            "red": {**STAT_DEFAULTS, **b.get("red_stats", {})},
            "blue": {**STAT_DEFAULTS, **b.get("blue_stats", {})},
        }
        outcome = {"red": "W/L", "blue": "L/W"}.get(winner, "D/D")
        for side, other in (("red", "blue"), ("blue", "red")):
            won = np.nan if winner is None else float(winner == side)
            r = {
                "event": b["event"],
                "bout": f"{b['red']} vs. {b['blue']}",
                "date": pd.Timestamp(b["date"]),
                "red": b["red"],
                "blue": b["blue"],
                "fighter": b[side],
                "opponent": b[other],
                "outcome": outcome,
                "won": won,
                "finish_win": bool(won == 1 and b.get("finish", False)),
                "ko_win": False,
                "sub_win": False,
                "minutes": float(b.get("minutes", 10)),
                "age": 28.0 + (side == "blue"),
                "reach": 70.0,
                "height": 70.0,
                "stance": "Orthodox",
                "southpaw": 0.0,
                "switch": 0.0,
                "dob": pd.Timestamp("1995-01-01"),
                **side_stats[side],
                **{f"opp_{k}": v for k, v in side_stats[other].items()},
            }
            rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def career():
    """A, B, C fight each other over time; A fights most often."""
    return [
        dict(date="2020-01-10", event="E1", red="A", blue="B", winner="red", red_stats=dict(sig_l=30)),
        dict(date="2020-06-10", event="E2", red="A", blue="C", winner="blue", blue_stats=dict(sig_l=50)),
        dict(date="2021-01-10", event="E3", red="B", blue="C", winner="red"),
        dict(date="2021-06-10", event="E4", red="A", blue="B", winner="red", finish=True),
        dict(date="2022-01-10", event="E5", red="C", blue="A", winner="blue"),
    ]
