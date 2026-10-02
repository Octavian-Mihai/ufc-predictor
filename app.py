"""Local Streamlit dashboard for UFC fight predictions and value underdogs."""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
import streamlit as st

from src.ingest.odds import flatten_h2h, get_odds, match_fight_odds
from src.model.predict import load_bundle, load_card, load_latest, predict_fight
from src.paths import FIGHT_LEVEL_PATH, FIGHTER_LATEST_PATH, METRICS_PATH, MODEL_PATH, UPCOMING_PATH
from src.review.recent import grade_last_events, recap_to_records, rethreshold_recap
from src.value.card import score_card
from src.value.ev import american_to_decimal, annotate_model, coerce_decimal, is_underdog, pair_market

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="wide")

DISCLAIMER = (
    "Entertainment / analysis tool, not betting advice. Fight outcomes are uncertain. "
    "A well-built UFC stats model typically lands around the mid-60s% on winner accuracy — "
    "this is an edge finder, not a lock machine."
)

STYLE_GROUPS = [
    (
        "Takedowns",
        [
            ("td_landed_p15", "TD landed / 15 min"),
            ("td_acc", "TD accuracy"),
            ("td_def", "TD defense"),
        ],
    ),
    (
        "Kicks (leg significant strikes)",
        [("leg_pm", "Leg strikes landed / min")],
    ),
    (
        "Distance striking (jab / boxing proxy)",
        [
            ("dist_pm", "Distance strikes landed / min"),
            ("dist_acc", "Distance accuracy"),
        ],
    ),
    (
        "Under pressure",
        [
            ("sapm", "Strikes absorbed / min"),
            ("str_def", "Strike defense"),
            ("ctrl_against_p15", "Control against / 15 min"),
        ],
    ),
    (
        "Attacking",
        [
            ("slpm", "Sig. strikes landed / min"),
            ("ctrl_for_p15", "Control for / 15 min"),
            ("kd_p15", "Knockdowns / 15 min"),
            ("ground_pm", "Ground strikes landed / min"),
        ],
    ),
]


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100 * float(x):.1f}%"


def _fmt_ev(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100 * float(x):+.1f}%"


def _fmt_dec(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.2f}"


def _txt(x, default: str = "—") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return default
    return str(x)


def _hit_decimal(hit: dict, side: str):
    """Read decimal odds from a match, including leftover American keys."""
    if not hit:
        return None
    raw = hit.get(f"{side}_decimal")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raw = hit.get(f"{side}_american")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        return american_to_decimal(raw)
    return coerce_decimal(raw)


def _num(stats: dict, key: str):
    val = stats.get(key)
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _data_stamp() -> tuple:
    """File mtimes, so cached results rebuild after a refresh or retrain."""
    paths = (UPCOMING_PATH, FIGHTER_LATEST_PATH, FIGHT_LEVEL_PATH, MODEL_PATH, METRICS_PATH)
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in paths)


@st.cache_data(show_spinner=False)
def _load_metrics(stamp: tuple) -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    bundle = load_bundle()
    return {"metrics": bundle.get("metrics", {}), "split": bundle.get("split", {})}


@st.cache_data(ttl=300, show_spinner=False)
def _odds_payload() -> dict:
    return get_odds()


@st.cache_resource(show_spinner=False)
def _bundle_and_latest(stamp: tuple):
    return load_bundle(), load_latest()


@st.cache_data(ttl=300, show_spinner=True)
def _scored_card(stamp: tuple) -> pd.DataFrame:
    bundle, latest = _bundle_and_latest(stamp)
    return score_card(_odds_payload(), bundle=bundle, latest=latest)


@st.cache_data(show_spinner=True)
def _last5_recap(stamp: tuple) -> dict:
    return recap_to_records(grade_last_events(n=5, threshold=0.0))


def apply_threshold(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = scored.copy()
    if df.empty:
        return df
    ok = df["status"] == "ok"
    df["a_value"] = ok & df["a_underdog"] & df["a_ev"].notna() & (df["a_ev"] >= threshold)
    df["b_value"] = ok & df["b_underdog"] & df["b_ev"].notna() & (df["b_ev"] >= threshold)
    df["value_flag"] = df["a_value"] | df["b_value"]
    df["value_side"] = df.apply(
        lambda r: r["fighter_a"] if r["a_value"] else (r["fighter_b"] if r["b_value"] else ""),
        axis=1,
    )
    return df


def _stat_bar(label: str, a_val, b_val, invert: bool = False) -> None:
    a = 0.0 if a_val is None else float(a_val)
    b = 0.0 if b_val is None else float(b_val)
    total = abs(a) + abs(b)
    a_share = 0.5 if total == 0 else abs(a) / total
    b_share = 1.0 - a_share
    better = "red" if ((a < b) if invert else (a > b)) else "blue"
    c1, c2, c3 = st.columns([3, 2, 3])
    with c1:
        st.caption(f"{a:.2f}" if a_val is not None else "—")
        st.progress(min(max(a_share, 0.0), 1.0))
    with c2:
        st.caption(label)
        if a_val is not None and b_val is not None and a != b:
            st.caption(f"{'←' if better == 'red' else '→'}")
    with c3:
        st.caption(f"{b:.2f}" if b_val is not None else "—")
        st.progress(min(max(b_share, 0.0), 1.0))


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        .value-pill { background: #143d2a; color: #7CFFB2; padding: 0.15rem 0.5rem; border-radius: 999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Local UFC fight predictor")
    st.caption("Official UFC Stats career features · stacked ML ensemble · optional free The Odds API")
    st.info(DISCLAIMER)

    with st.sidebar:
        st.header("Controls")
        threshold_pct = st.slider("Value EV threshold (%)", min_value=0, max_value=20, value=5, step=1)
        threshold = threshold_pct / 100.0
        st.caption("Flag moneyline underdogs whose model EV is above this line. Default 5%.")
        if st.button("Refresh odds cache view"):
            _odds_payload.clear()
            _scored_card.clear()
            st.rerun()
        st.divider()
        try:
            meta = _load_metrics(_data_stamp())
            m = meta.get("metrics") or {}
            split = meta.get("split") or {}
            st.subheader("Holdout metrics")
            st.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
            st.metric("Brier", f"{m.get('brier', 0):.3f}")
            st.metric("Log loss", f"{m.get('log_loss', 0):.3f}")
            st.caption(
                f"{m.get('n', 0):,} fights after {split.get('holdout_from', '2024')} · "
                f"fit through {split.get('fit_through', '2022')}, calibrate {split.get('calib_through', '2023')}"
            )
        except FileNotFoundError:
            st.warning("Train the model first: `python -m src.model.train`")
            return
        comparison = meta.get("comparison")
        if comparison:
            with st.expander("Model comparison (holdout)"):
                table = pd.DataFrame(comparison).T[["accuracy", "log_loss", "brier"]]
                st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
                st.caption("Ensemble = stacked HGB + logistic regression + extra trees + neural net, Platt-calibrated.")
                imp = meta.get("importance") or {}
                if imp:
                    st.markdown("**Top features** (permutation importance)")
                    st.bar_chart(pd.Series(imp).sort_values())
        st.divider()
        with st.expander("UFC Stats field mapping"):
            st.markdown(
                """
- **Takedowns** — TD landed/attempted, accuracy, TD defense
- **Kicks** — significant strikes to the **leg**
- **Jabs** — significant strikes at **distance** (best official proxy; labeled honestly)
- **Under pressure** — SApM, strike defense, control time against
- **Attacking** — SLpM, control time for, knockdowns, ground strikes
                """
            )

    try:
        scored = apply_threshold(_scored_card(_data_stamp()), threshold)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    upcoming_tab, recap_tab = st.tabs(["Upcoming card", "Last 5 events"])
    with upcoming_tab:
        render_upcoming_card(scored)
    with recap_tab:
        render_last5(threshold)


def render_upcoming_card(scored: pd.DataFrame) -> None:
    if scored.empty:
        st.warning("No card loaded. Run `python -m src.ingest.ufcstats --bootstrap` then `--upcoming`.")
        return

    if "DATE" in scored.columns:
        scored = scored.assign(_when=pd.to_datetime(scored["DATE"], errors="coerce")).sort_values(
            "_when", kind="stable"
        )
    event_names = list(scored["EVENT"].dropna().unique())
    event_name = st.selectbox("Card", event_names)
    scored = scored.loc[scored["EVENT"] == event_name].reset_index(drop=True)
    sample = bool(scored["sample_card"].iloc[0]) if "sample_card" in scored.columns else False
    odds_source = scored["odds_source"].iloc[0] if "odds_source" in scored.columns else "dummy"
    when = scored["DATE"].iloc[0] if "DATE" in scored.columns else ""
    where = scored["LOCATION"].iloc[0] if "LOCATION" in scored.columns else ""
    header = f"{event_name}"
    if sample:
        header += " · sample / historical card (pre-fight features, no leakage)"
    st.subheader(header)
    st.caption(
        f"{when} · {where} · Odds source: `{odds_source}` · names matched with rapidfuzz · debuts shown as insufficient data"
    )

    table_rows = []
    for rec in scored.to_dict("records"):
        status = rec.get("status")
        flag = ""
        if rec.get("value_flag"):
            flag = f"VALUE {rec.get('value_side')}"
        elif status != "ok":
            flag = "insufficient data"
        table_rows.append(
            {
                "Bout": f"{rec.get('fighter_a')} vs {rec.get('fighter_b')}",
                "Model P(A)": _fmt_pct(rec.get("p_a")) if status == "ok" else rec.get("reason") or "insufficient data",
                "Model P(B)": _fmt_pct(rec.get("p_b")) if status == "ok" else "—",
                "A odds": _fmt_dec(rec.get("a_decimal")),
                "B odds": _fmt_dec(rec.get("b_decimal")),
                "Market P(A)": _fmt_pct(rec.get("a_fair")),
                "A edge": _fmt_ev(rec.get("a_edge")) if status == "ok" else "—",
                "A EV": _fmt_ev(rec.get("a_ev")) if status == "ok" else "—",
                "B EV": _fmt_ev(rec.get("b_ev")) if status == "ok" else "—",
                "Underdog": rec.get("fighter_a")
                if rec.get("a_underdog")
                else (rec.get("fighter_b") if rec.get("b_underdog") else "—"),
                "Flag": flag,
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    values = scored.loc[scored["value_flag"]]
    st.markdown("### Value underdogs")
    if values.empty:
        st.write("No underdogs currently clear the EV slider. Lower the threshold or wait for a live card.")
    else:
        show = values.copy()
        show["underdog"] = show.apply(lambda r: r["fighter_a"] if r["a_value"] else r["fighter_b"], axis=1)
        show["ev"] = show.apply(lambda r: r["a_ev"] if r["a_value"] else r["b_ev"], axis=1)
        show["odds"] = show.apply(lambda r: r["a_decimal"] if r["a_value"] else r["b_decimal"], axis=1)
        show["model_p"] = show.apply(lambda r: r["p_a"] if r["a_value"] else r["p_b"], axis=1)
        st.dataframe(
            show.assign(
                EV=show["ev"].map(_fmt_ev),
                **{"Model P": show["model_p"].map(_fmt_pct), "Odds": show["odds"].map(_fmt_dec)},
            )[["EVENT", "underdog", "Model P", "Odds", "EV"]],
            hide_index=True,
            width="stretch",
        )

    st.markdown("### Matchup details")
    labels = [
        f"{r['fighter_a']} vs {r['fighter_b']}"
        + ("  • VALUE" if r.get("value_flag") else "")
        + ("  • insufficient data" if r.get("status") != "ok" else "")
        for r in scored.to_dict("records")
    ]
    pick = st.selectbox("Select a bout", options=list(range(len(labels))), format_func=lambda i: labels[i])
    fight = scored.iloc[pick]

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {fight['fighter_a']}")
        if fight.get("status") == "ok":
            st.metric("Model win probability", _fmt_pct(fight.get("p_a")))
        else:
            st.warning("insufficient data")
        st.write(f"Market {_fmt_dec(fight.get('a_decimal'))} · fair {_fmt_pct(fight.get('a_fair'))}")
        st.write(f"Edge {_fmt_ev(fight.get('a_edge'))} · EV {_fmt_ev(fight.get('a_ev'))}")
        if fight.get("a_value"):
            st.success("Value underdog")
        elif fight.get("a_underdog"):
            st.caption("Moneyline underdog (below EV threshold)")
    with right:
        st.markdown(f"#### {fight['fighter_b']}")
        if fight.get("status") == "ok":
            st.metric("Model win probability", _fmt_pct(fight.get("p_b")))
        else:
            st.warning("insufficient data")
        st.write(f"Market {_fmt_dec(fight.get('b_decimal'))} · fair {_fmt_pct(fight.get('b_fair'))}")
        st.write(f"Edge {_fmt_ev(fight.get('b_edge'))} · EV {_fmt_ev(fight.get('b_ev'))}")
        if fight.get("b_value"):
            st.success("Value underdog")
        elif fight.get("b_underdog"):
            st.caption("Moneyline underdog (below EV threshold)")

    a_stats = fight.get("a_stats") or {}
    b_stats = fight.get("b_stats") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", f"{_num(a_stats, 'age') or 0:.1f} vs {_num(b_stats, 'age') or 0:.1f}")
    c2.metric("Reach (in)", f"{_num(a_stats, 'reach') or 0:.0f} vs {_num(b_stats, 'reach') or 0:.0f}")
    c3.metric("Last 3 win rate", f"{_fmt_pct(_num(a_stats, 'last3'))} vs {_fmt_pct(_num(b_stats, 'last3'))}")
    c4.metric("UFC fights", f"{int(_num(a_stats, 'prior_fights') or 0)} vs {int(_num(b_stats, 'prior_fights') or 0)}")

    st.caption("Career averages using only official UFC Stats. Distance striking is the jab proxy.")
    h1, h2, h3 = st.columns([3, 2, 3])
    h1.markdown(f"**{fight['fighter_a']}**")
    h2.markdown("**Stat**")
    h3.markdown(f"**{fight['fighter_b']}**")
    for title, items in STYLE_GROUPS:
        st.markdown(f"**{title}**")
        for key, label in items:
            _stat_bar(label, _num(a_stats, key), _num(b_stats, key), invert=key in {"sapm", "ctrl_against_p15"})

    st.divider()
    st.caption("Refresh before a card: `python -m src.ingest.refresh` · Retrain: `python -m src.model.train`")


def _recap_verdict(rec: dict) -> str:
    grade = rec.get("grade")
    if grade == "correct":
        return "Correct"
    if grade == "wrong":
        return "Wrong"
    if grade == "insufficient_data":
        return "insufficient data"
    return str(grade or "excluded")


def _dog_ev(rec: dict):
    if rec.get("value_flag"):
        return rec.get("value_ev")
    if rec.get("a_underdog"):
        return rec.get("a_ev")
    if rec.get("b_underdog"):
        return rec.get("b_ev")
    return None


def render_last5(threshold: float) -> None:
    st.caption(
        "Recap uses the **current trained model** on pre-fight stats (`as_of_prior`). "
        "Run `python -m src.ingest.refresh` so UFC 331 (and any newer cards) are in the local CSVs before grading. "
        "NC / draws / insufficient data are listed but excluded from accuracy."
    )
    try:
        recap = rethreshold_recap(_last5_recap(_data_stamp()), threshold)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    summary = recap.get("summary") or {}
    acc = summary.get("accuracy")
    hit_rate = summary.get("value_hit_rate")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winner accuracy", f"{acc:.1%}" if acc is not None else "—")
    c2.metric("Graded fights", f"{summary.get('n_graded', 0)} / {summary.get('n_fights', 0)}")
    c3.metric("Value flags", int(summary.get("n_value_flags") or 0))
    c4.metric("Value hit rate", f"{hit_rate:.1%}" if hit_rate is not None else "—")
    st.caption(
        f"{summary.get('n_correct', 0)} correct · {summary.get('n_wrong', 0)} wrong · "
        f"{summary.get('n_excluded', 0)} excluded · "
        f"{summary.get('n_value_hits', 0)} value hits / {summary.get('n_value_misses', 0)} misses · "
        "odds column is `snapshot` if a pull was stored, else `dummy`"
    )

    events = recap.get("events") or []
    if not events:
        st.warning("No completed events with results in the local CSVs.")
        return

    for i, event in enumerate(events):
        when = event.get("date") or ""
        where = event.get("location") or ""
        acc_e = event.get("accuracy")
        acc_s = f"{acc_e:.1%}" if acc_e is not None else "—"
        title = f"{event.get('event')} · {when} · {event.get('n_correct', 0)}/{event.get('n_graded', 0)} ({acc_s})"
        with st.expander(title, expanded=i == 0):
            if where:
                st.caption(where)
            fights = event.get("fights")
            if fights is None or getattr(fights, "empty", True):
                st.write("No bouts found.")
                continue
            table_rows = []
            for rec in fights.to_dict("records"):
                value_hit = rec.get("value_hit")
                if rec.get("value_flag") and rec.get("has_winner"):
                    value_cell = "hit" if value_hit else "miss"
                elif rec.get("value_flag"):
                    value_cell = "—"
                else:
                    value_cell = ""
                table_rows.append(
                    {
                        "Bout": f"{_txt(rec.get('fighter_a'))} vs {_txt(rec.get('fighter_b'))}",
                        "Method": _txt(rec.get("METHOD")),
                        "Winner": _txt(rec.get("winner"), default=_txt(rec.get("result_kind"))),
                        "Model pick": _txt(rec.get("pick")),
                        "P": _fmt_pct(rec.get("pick_p")) if rec.get("status") == "ok" else "—",
                        "Result": _recap_verdict(rec),
                        "Dog": _txt(rec.get("dog")),
                        "EV": _fmt_ev(_dog_ev(rec)),
                        "Value flag": _txt(rec.get("value_side"), default="") if rec.get("value_flag") else "",
                        "Value": value_cell,
                        "Odds": _txt(rec.get("odds_source"), default="dummy"),
                    }
                )
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    st.caption("Refresh before a card: `python -m src.ingest.refresh` · Retrain: `python -m src.model.train`")


main()
