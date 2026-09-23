"""Leak-free UFC Stats career features (red−blue differentials)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.names import norm_name, split_bout
from src.paths import (
    EVENTS_CSV,
    FIGHTER_LATEST_PATH,
    FIGHTER_TOTT_CSV,
    FIGHT_LEVEL_PATH,
    FIGHT_RESULTS_CSV,
    FIGHT_STATS_CSV,
    TRAINING_PATH,
    ensure_dirs,
)

FEATURE_COLS = [
    "d_age",
    "d_reach",
    "d_height",
    "d_days_since",
    "d_prior_fights",
    "d_win_rate",
    "d_last3",
    "d_finish_rate",
    "d_ko_rate",
    "d_sub_rate",
    "d_td_landed_p15",
    "d_td_acc",
    "d_td_def",
    "d_leg_pm",
    "d_dist_pm",
    "d_dist_acc",
    "d_slpm",
    "d_sapm",
    "d_str_acc",
    "d_str_def",
    "d_ctrl_for_p15",
    "d_ctrl_against_p15",
    "d_kd_p15",
    "d_ground_pm",
    "d_southpaw",
    "stance_mismatch",
]

CAREER_COLS = [
    "age",
    "reach",
    "height",
    "days_since",
    "prior_fights",
    "win_rate",
    "last3",
    "finish_rate",
    "ko_rate",
    "sub_rate",
    "td_landed_p15",
    "td_acc",
    "td_def",
    "leg_pm",
    "dist_pm",
    "dist_acc",
    "slpm",
    "sapm",
    "str_acc",
    "str_def",
    "ctrl_for_p15",
    "ctrl_against_p15",
    "kd_p15",
    "ground_pm",
    "southpaw",
    "switch",
    "stance",
]

_OF_RE = re.compile(r"(-?\d+)\s+of\s+(-?\d+)", re.I)


def parse_of(value) -> tuple[float, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0, 0.0
    text = str(value).strip()
    if text in {"", "--", "---", "nan"}:
        return 0.0, 0.0
    match = _OF_RE.search(text)
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def parse_ctrl_seconds(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if text in {"", "--", "---", "nan"}:
        return 0.0
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    minutes, seconds = text.split(":", 1)
    try:
        return float(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0


def parse_height_in(value) -> float:
    if value is None:
        return np.nan
    text = str(value)
    if "--" in text or text.strip() in {"", "nan"}:
        return np.nan
    match = re.search(r"(\d+)'\s*(\d+)", text)
    if match:
        return float(match.group(1)) * 12 + float(match.group(2))
    return np.nan


def parse_reach_in(value) -> float:
    if value is None:
        return np.nan
    text = str(value)
    if "--" in text or text.strip() in {"", "nan"}:
        return np.nan
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def parse_weight_lb(value) -> float:
    if value is None:
        return np.nan
    text = str(value)
    if "--" in text:
        return np.nan
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def fight_duration_seconds(round_ended, time_str, time_format) -> float:
    lengths = [5, 5, 5]
    match = re.search(r"\(([^)]+)\)", str(time_format or ""))
    if match:
        parsed = []
        for part in match.group(1).split("-"):
            part = part.strip()
            if part.isdigit():
                parsed.append(int(part))
        if parsed:
            lengths = parsed
    try:
        ended = int(float(round_ended))
    except (TypeError, ValueError):
        ended = 1
    ended = max(ended, 1)
    last = parse_ctrl_seconds(time_str)
    total = 0.0
    for i in range(ended - 1):
        mins = lengths[i] if i < len(lengths) else 5
        total += mins * 60
    total += last
    return max(total, 1.0)


def _is_finish(method: str) -> bool:
    text = str(method or "").lower()
    return any(k in text for k in ("ko/tko", "knockout", "tko", " ko", "submission", "sub "))


def _is_ko(method: str) -> bool:
    text = str(method or "").lower()
    return "ko" in text or "tko" in text or "knockout" in text


def _is_sub(method: str) -> bool:
    return "sub" in str(method or "").lower()


def _safe_div(num, den) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den.replace(0, np.nan)
    return out


def load_bio() -> pd.DataFrame:
    tott = pd.read_csv(FIGHTER_TOTT_CSV)
    tott["fighter"] = tott["FIGHTER"].map(norm_name)
    tott["height"] = tott["HEIGHT"].map(parse_height_in)
    tott["reach"] = tott["REACH"].map(parse_reach_in)
    tott["weight"] = tott["WEIGHT"].map(parse_weight_lb)
    tott["stance"] = tott["STANCE"].map(lambda x: norm_name(x).title() if pd.notna(x) else "")
    tott["dob"] = pd.to_datetime(tott["DOB"], errors="coerce")
    bio = tott.sort_values("fighter").drop_duplicates("fighter", keep="last")
    return bio[["fighter", "height", "reach", "weight", "stance", "dob", "URL"]].rename(columns={"URL": "fighter_url"})


def aggregate_fight_stats(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    df["FIGHTER"] = df["FIGHTER"].map(norm_name)
    df["EVENT"] = df["EVENT"].map(norm_name)
    df["BOUT"] = df["BOUT"].map(lambda x: str(x).strip() if pd.notna(x) else "")
    df = df.loc[df["FIGHTER"].astype(str).str.len() > 0].copy()

    of_cols = {
        "SIG.STR.": ("sig_l", "sig_a"),
        "TOTAL STR.": ("tot_l", "tot_a"),
        "TD": ("td_l", "td_a"),
        "HEAD": ("head_l", "head_a"),
        "BODY": ("body_l", "body_a"),
        "LEG": ("leg_l", "leg_a"),
        "DISTANCE": ("dist_l", "dist_a"),
        "CLINCH": ("clinch_l", "clinch_a"),
        "GROUND": ("ground_l", "ground_a"),
    }
    for src, (landed, att) in of_cols.items():
        parsed = df[src].map(parse_of) if src in df.columns else [(0.0, 0.0)] * len(df)
        df[landed] = [p[0] for p in parsed]
        df[att] = [p[1] for p in parsed]

    df["kd"] = pd.to_numeric(df.get("KD"), errors="coerce").fillna(0)
    df["sub_att"] = pd.to_numeric(df.get("SUB.ATT"), errors="coerce").fillna(0)
    df["rev"] = pd.to_numeric(df.get("REV."), errors="coerce").fillna(0)
    df["ctrl_sec"] = df["CTRL"].map(parse_ctrl_seconds) if "CTRL" in df.columns else 0.0

    grouped = (
        df.groupby(["EVENT", "BOUT", "FIGHTER"], dropna=False)
        .agg(
            kd=("kd", "sum"),
            sig_l=("sig_l", "sum"),
            sig_a=("sig_a", "sum"),
            tot_l=("tot_l", "sum"),
            tot_a=("tot_a", "sum"),
            td_l=("td_l", "sum"),
            td_a=("td_a", "sum"),
            sub_att=("sub_att", "sum"),
            rev=("rev", "sum"),
            ctrl_sec=("ctrl_sec", "sum"),
            head_l=("head_l", "sum"),
            head_a=("head_a", "sum"),
            body_l=("body_l", "sum"),
            body_a=("body_a", "sum"),
            leg_l=("leg_l", "sum"),
            leg_a=("leg_a", "sum"),
            dist_l=("dist_l", "sum"),
            dist_a=("dist_a", "sum"),
            clinch_l=("clinch_l", "sum"),
            clinch_a=("clinch_a", "sum"),
            ground_l=("ground_l", "sum"),
            ground_a=("ground_a", "sum"),
        )
        .reset_index()
        .rename(columns={"EVENT": "event", "BOUT": "bout", "FIGHTER": "fighter"})
    )
    return grouped


def attach_opponents(fight_level: pd.DataFrame) -> pd.DataFrame:
    left = fight_level.copy()
    right = fight_level.copy()
    opp_cols = [c for c in right.columns if c not in {"event", "bout"}]
    right = right.rename(columns={c: f"opp_{c}" if c != "fighter" else "opponent" for c in opp_cols})
    merged = left.merge(right, on=["event", "bout"], how="left")
    merged = merged.loc[merged["fighter"] != merged["opponent"]].copy()
    return merged


def build_fighter_fight_table() -> pd.DataFrame:
    results = pd.read_csv(FIGHT_RESULTS_CSV)
    stats = pd.read_csv(FIGHT_STATS_CSV, low_memory=False)
    events = pd.read_csv(EVENTS_CSV)
    bio = load_bio()

    results["event"] = results["EVENT"].map(norm_name)
    results["bout"] = results["BOUT"].map(lambda x: str(x).strip() if pd.notna(x) else "")
    parsed = results["bout"].map(split_bout)
    results["red"] = [p[0] for p in parsed]
    results["blue"] = [p[1] for p in parsed]
    results["outcome"] = results["OUTCOME"].map(lambda x: str(x).replace(" ", "").strip().upper())
    results["method"] = results["METHOD"].map(lambda x: str(x).strip() if pd.notna(x) else "")
    results["duration_sec"] = [
        fight_duration_seconds(r["ROUND"], r["TIME"], r["TIME FORMAT"])
        for _, r in results.iterrows()
    ]

    events["event"] = events["EVENT"].map(norm_name)
    events["date"] = pd.to_datetime(events["DATE"], errors="coerce")
    event_dates = events.drop_duplicates("event")[["event", "date", "LOCATION"]]

    results = results.merge(event_dates, on="event", how="left")

    # One row per fighter in the bout, with win flag from official outcome.
    red = results.copy()
    red["fighter"] = red["red"]
    red["opponent"] = red["blue"]
    red["won"] = red["outcome"].map(lambda x: 1.0 if x.startswith("W/") else (0.0 if x.startswith("L/") else np.nan))

    blue = results.copy()
    blue["fighter"] = blue["blue"]
    blue["opponent"] = blue["red"]
    blue["won"] = blue["outcome"].map(lambda x: 1.0 if x.endswith("/W") else (0.0 if x.endswith("/L") else np.nan))

    sides = pd.concat([red, blue], ignore_index=True)
    sides["finish_win"] = (sides["won"] == 1) & sides["method"].map(_is_finish)
    sides["ko_win"] = (sides["won"] == 1) & sides["method"].map(_is_ko)
    sides["sub_win"] = (sides["won"] == 1) & sides["method"].map(_is_sub)

    agg = aggregate_fight_stats(stats)
    joined = sides.merge(agg, on=["event", "bout", "fighter"], how="left")

    # Opponent in-fight stats for defense / pressure.
    opp = agg.rename(columns={c: f"opp_{c}" if c not in {"event", "bout"} else c for c in agg.columns})
    opp = opp.rename(columns={"opp_fighter": "opponent"})
    joined = joined.merge(opp, on=["event", "bout", "opponent"], how="left")

    joined = joined.merge(bio, on="fighter", how="left")
    joined["age"] = (joined["date"] - joined["dob"]).dt.days / 365.25
    joined["southpaw"] = joined["stance"].str.contains("southpaw", case=False, na=False).astype(float)
    joined["switch"] = joined["stance"].str.contains("switch", case=False, na=False).astype(float)
    joined["minutes"] = joined["duration_sec"] / 60.0
    joined["is_decision"] = joined["outcome"].isin(["W/L", "L/W"])
    return joined


def add_prior_career(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding career stats using only prior bouts (no leakage)."""
    g = df.sort_values(["fighter", "date", "event", "bout"]).copy()
    g["prior_fights"] = g.groupby("fighter").cumcount()
    g["days_since"] = g.groupby("fighter")["date"].diff().dt.days

    sum_cols = [
        "won",
        "finish_win",
        "ko_win",
        "sub_win",
        "minutes",
        "kd",
        "sig_l",
        "sig_a",
        "td_l",
        "td_a",
        "leg_l",
        "leg_a",
        "dist_l",
        "dist_a",
        "ground_l",
        "ground_a",
        "ctrl_sec",
        "opp_sig_l",
        "opp_sig_a",
        "opp_td_l",
        "opp_td_a",
        "opp_ctrl_sec",
        "opp_minutes",
    ]
    g["opp_minutes"] = g["minutes"]
    for col in sum_cols:
        if col not in g.columns:
            g[col] = 0.0
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        g[f"p_{col}"] = g.groupby("fighter")[col].cumsum() - g[col]

    pmin = g["p_minutes"].replace(0, np.nan)
    g["slpm"] = g["p_sig_l"] / pmin
    g["sapm"] = g["p_opp_sig_l"] / pmin
    g["leg_pm"] = g["p_leg_l"] / pmin
    g["dist_pm"] = g["p_dist_l"] / pmin
    g["ground_pm"] = g["p_ground_l"] / pmin
    g["td_landed_p15"] = g["p_td_l"] / pmin * 15.0
    g["ctrl_for_p15"] = (g["p_ctrl_sec"] / 60.0) / pmin * 15.0
    g["ctrl_against_p15"] = (g["p_opp_ctrl_sec"] / 60.0) / pmin * 15.0
    g["kd_p15"] = g["p_kd"] / pmin * 15.0
    g["td_acc"] = _safe_div(g["p_td_l"], g["p_td_a"])
    g["td_def"] = 1.0 - _safe_div(g["p_opp_td_l"], g["p_opp_td_a"])
    g["str_acc"] = _safe_div(g["p_sig_l"], g["p_sig_a"])
    g["str_def"] = 1.0 - _safe_div(g["p_opp_sig_l"], g["p_opp_sig_a"])
    g["dist_acc"] = _safe_div(g["p_dist_l"], g["p_dist_a"])
    g["win_rate"] = _safe_div(g["p_won"], g["prior_fights"])
    g["finish_rate"] = _safe_div(g["p_finish_win"], g["p_won"])
    g["ko_rate"] = _safe_div(g["p_ko_win"], g["p_won"])
    g["sub_rate"] = _safe_div(g["p_sub_win"], g["p_won"])

    g["prev_win"] = g.groupby("fighter")["won"].shift(1)
    g["last3"] = g.groupby("fighter")["prev_win"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return g


def inclusive_career(df: pd.DataFrame) -> pd.DataFrame:
    """Career snapshot including all fights (for live upcoming predictions)."""
    g = df.sort_values(["fighter", "date", "event", "bout"]).copy()
    sum_cols = [
        "won",
        "finish_win",
        "ko_win",
        "sub_win",
        "minutes",
        "kd",
        "sig_l",
        "sig_a",
        "td_l",
        "td_a",
        "leg_l",
        "leg_a",
        "dist_l",
        "dist_a",
        "ground_l",
        "ground_a",
        "ctrl_sec",
        "opp_sig_l",
        "opp_sig_a",
        "opp_td_l",
        "opp_td_a",
        "opp_ctrl_sec",
    ]
    for col in sum_cols:
        if col not in g.columns:
            g[col] = 0.0
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)

    latest = (
        g.groupby("fighter", as_index=False)
        .agg(
            last_date=("date", "max"),
            prior_fights=("fighter", "size"),
            won=("won", "sum"),
            finish_win=("finish_win", "sum"),
            ko_win=("ko_win", "sum"),
            sub_win=("sub_win", "sum"),
            minutes=("minutes", "sum"),
            kd=("kd", "sum"),
            sig_l=("sig_l", "sum"),
            sig_a=("sig_a", "sum"),
            td_l=("td_l", "sum"),
            td_a=("td_a", "sum"),
            leg_l=("leg_l", "sum"),
            dist_l=("dist_l", "sum"),
            dist_a=("dist_a", "sum"),
            ground_l=("ground_l", "sum"),
            ctrl_sec=("ctrl_sec", "sum"),
            opp_sig_l=("opp_sig_l", "sum"),
            opp_sig_a=("opp_sig_a", "sum"),
            opp_td_l=("opp_td_l", "sum"),
            opp_td_a=("opp_td_a", "sum"),
            opp_ctrl_sec=("opp_ctrl_sec", "sum"),
            age=("age", "last"),
            reach=("reach", "last"),
            height=("height", "last"),
            stance=("stance", "last"),
            southpaw=("southpaw", "last"),
            switch=("switch", "last"),
            dob=("dob", "last"),
            last3_raw=("won", lambda s: s.tail(3).mean()),
        )
        .rename(columns={"last3_raw": "last3"})
    )
    pmin = latest["minutes"].replace(0, np.nan)
    latest["slpm"] = latest["sig_l"] / pmin
    latest["sapm"] = latest["opp_sig_l"] / pmin
    latest["leg_pm"] = latest["leg_l"] / pmin
    latest["dist_pm"] = latest["dist_l"] / pmin
    latest["ground_pm"] = latest["ground_l"] / pmin
    latest["td_landed_p15"] = latest["td_l"] / pmin * 15.0
    latest["ctrl_for_p15"] = (latest["ctrl_sec"] / 60.0) / pmin * 15.0
    latest["ctrl_against_p15"] = (latest["opp_ctrl_sec"] / 60.0) / pmin * 15.0
    latest["kd_p15"] = latest["kd"] / pmin * 15.0
    latest["td_acc"] = _safe_div(latest["td_l"], latest["td_a"])
    latest["td_def"] = 1.0 - _safe_div(latest["opp_td_l"], latest["opp_td_a"])
    latest["str_acc"] = _safe_div(latest["sig_l"], latest["sig_a"])
    latest["str_def"] = 1.0 - _safe_div(latest["opp_sig_l"], latest["opp_sig_a"])
    latest["dist_acc"] = _safe_div(latest["dist_l"], latest["dist_a"])
    latest["win_rate"] = _safe_div(latest["won"], latest["prior_fights"])
    latest["finish_rate"] = _safe_div(latest["finish_win"], latest["won"])
    latest["ko_rate"] = _safe_div(latest["ko_win"], latest["won"])
    latest["sub_rate"] = _safe_div(latest["sub_win"], latest["won"])
    latest["days_since"] = (pd.Timestamp.now().normalize() - latest["last_date"]).dt.days
    # Age as of today for upcoming cards.
    latest["age"] = (pd.Timestamp.now().normalize() - latest["dob"]).dt.days / 365.25
    return latest


def diffs_from_sides(red: pd.DataFrame, blue: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=red.index)
    paired = [
        "age",
        "reach",
        "height",
        "days_since",
        "prior_fights",
        "win_rate",
        "last3",
        "finish_rate",
        "ko_rate",
        "sub_rate",
        "td_landed_p15",
        "td_acc",
        "td_def",
        "leg_pm",
        "dist_pm",
        "dist_acc",
        "slpm",
        "sapm",
        "str_acc",
        "str_def",
        "ctrl_for_p15",
        "ctrl_against_p15",
        "kd_p15",
        "ground_pm",
        "southpaw",
    ]
    for col in paired:
        if col in red.columns:
            r = pd.to_numeric(red[col], errors="coerce")
        else:
            r = pd.Series(np.nan, index=red.index)
        if col in blue.columns:
            b = pd.to_numeric(blue[col], errors="coerce")
        else:
            b = pd.Series(np.nan, index=blue.index)
        out[f"d_{col}"] = r.to_numpy() - b.to_numpy()
    r_stance = red["stance"] if "stance" in red.columns else ""
    b_stance = blue["stance"] if "stance" in blue.columns else ""
    out["stance_mismatch"] = (
        r_stance.fillna("").astype(str).str.lower().values != b_stance.fillna("").astype(str).str.lower().values
    ).astype(float)
    return out


def build_training_rows(prior: pd.DataFrame) -> pd.DataFrame:
    key = ["event", "bout", "date"]
    extra = ["fighter", "outcome", "prior_fights"]
    cols = list(dict.fromkeys(key + CAREER_COLS + extra))
    keep = [c for c in cols if c in prior.columns]
    red = prior.loc[prior["fighter"] == prior["red"], keep].drop_duplicates(key)
    blue = prior.loc[prior["fighter"] == prior["blue"], keep].drop_duplicates(key)
    merged = red.merge(blue, on=key, suffixes=("_r", "_b"))
    red_f = merged.filter(regex=r"_r$").rename(columns=lambda c: c[:-2]).reset_index(drop=True)
    blue_f = merged.filter(regex=r"_b$").rename(columns=lambda c: c[:-2]).reset_index(drop=True)
    feats = diffs_from_sides(red_f, blue_f)
    feats["y"] = (merged["outcome_r"] == "W/L").astype(int).to_numpy()
    feats["event"] = merged["event"].to_numpy()
    feats["bout"] = merged["bout"].to_numpy()
    feats["date"] = merged["date"].to_numpy()
    feats["red"] = merged["fighter_r"].to_numpy()
    feats["blue"] = merged["fighter_b"].to_numpy()
    feats["red_prior"] = merged["prior_fights_r"].to_numpy()
    feats["blue_prior"] = merged["prior_fights_b"].to_numpy()
    feats["outcome"] = merged["outcome_r"].to_numpy()
    feats = feats.loc[feats["outcome"].isin(["W/L", "L/W"])].copy()
    return feats.reset_index(drop=True)


def build_features() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    print("building fighter-fight table")
    fights = build_fighter_fight_table()
    print(f"  {len(fights):,} fighter-fight rows")
    print("adding leak-free prior career stats")
    prior = add_prior_career(fights)
    prior.to_parquet(FIGHT_LEVEL_PATH, index=False)
    print("building training differentials")
    train = build_training_rows(prior)
    train.to_parquet(TRAINING_PATH, index=False)
    print(f"  {len(train):,} decisive fights")
    print("building current fighter snapshots")
    latest = inclusive_career(fights)
    latest.to_parquet(FIGHTER_LATEST_PATH, index=False)
    print(f"  {len(latest):,} fighters")
    return {"fights": fights, "prior": prior, "train": train, "latest": latest}


def main() -> None:
    build_features()


if __name__ == "__main__":
    main()
