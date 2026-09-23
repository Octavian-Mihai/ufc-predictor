"""UFC Stats ingest: seed CSVs from Greco1899 plus incremental scrape."""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.names import norm_name, split_bout
from src.paths import (
    EVENTS_CSV,
    FIGHT_DETAILS_CSV,
    FIGHT_RESULTS_CSV,
    FIGHT_STATS_CSV,
    FIGHTER_DETAILS_CSV,
    FIGHTER_TOTT_CSV,
    RAW_DIR,
    SEED_FILES,
    UPCOMING_PATH,
    ensure_dirs,
)

SEED_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
UFCSTATS_COMPLETED = "http://ufcstats.com/statistics/events/completed?page=all"
UFCSTATS_UPCOMING = "http://ufcstats.com/statistics/events/upcoming"
UFCSTATS_FIGHTERS = "http://ufcstats.com/statistics/fighters?char={char}&page=all"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

REQUEST_PAUSE = 0.45

FIGHT_RESULTS_COLUMNS = [
    "EVENT",
    "BOUT",
    "OUTCOME",
    "WEIGHTCLASS",
    "METHOD",
    "ROUND",
    "TIME",
    "TIME FORMAT",
    "REFEREE",
    "DETAILS",
    "URL",
]

FIGHT_STATS_COLUMNS = [
    "EVENT",
    "BOUT",
    "ROUND",
    "FIGHTER",
    "KD",
    "SIG.STR.",
    "SIG.STR. %",
    "TOTAL STR.",
    "TD",
    "TD %",
    "SUB.ATT",
    "REV.",
    "CTRL",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
]


def _solve_browser_check(html: str) -> tuple[str, int] | None:
    """Solve the UFC Stats SHA-256 nonce check that browsers run on first visit."""
    nonce_m = re.search(r'nonce="([0-9a-fA-F]+)"', html)
    if not nonce_m:
        return None
    zeros_m = re.search(r"target=new Array\((\d+)\+1\)\.join", html)
    zeros = int(zeros_m.group(1)) if zeros_m else 2
    nonce = nonce_m.group(1)
    target = "0" * zeros
    n = 0
    while True:
        digest = hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest()
        if digest.startswith(target):
            return nonce, n
        n += 1
        if n > 5_000_000:
            return None


def _pass_browser_check(resp: requests.Response) -> bool:
    if "Checking your browser" not in resp.text or "/__c" not in resp.text:
        return False
    solved = _solve_browser_check(resp.text)
    if not solved:
        return False
    nonce, n = solved
    SESSION.post(
        urljoin(resp.url, "/__c"),
        data={"nonce": nonce, "n": str(n)},
        timeout=30,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return True


def _get(url: str, timeout: int = 60) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = SESSION.get(url, timeout=timeout)
            if _pass_browser_check(resp):
                time.sleep(0.2)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Failed to GET {url}: {last_err}") from last_err


def _soup(url: str) -> BeautifulSoup:
    resp = _get(url)
    time.sleep(REQUEST_PAUSE)
    return BeautifulSoup(resp.content, "lxml")


def _clean_label(text: str) -> str:
    text = re.sub(r"^[^:]+:\s*", "", text.replace("\n", " "))
    return re.sub(r"\s+", " ", text).strip()


def bootstrap_seed_csvs(force: bool = False) -> None:
    """Download maintained Greco1899 UFC Stats CSVs into data/raw/."""
    ensure_dirs()
    for name in SEED_FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"seed exists, skip {name}")
            continue
        url = f"{SEED_BASE}/{name}"
        print(f"downloading {url}")
        resp = _get(url, timeout=180)
        dest.write_bytes(resp.content)
        print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


def _parse_event_listing(soup: BeautifulSoup, skip_placeholder: bool = False) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for tr in soup.select("tr.b-statistics__table-row"):
        link = tr.find("a", class_="b-link")
        if link is None or not link.get("href"):
            continue
        date_el = tr.find("span", class_="b-statistics__date")
        loc_el = tr.find("td", class_="b-statistics__table-col_style_big-top-padding")
        rows.append(
            {
                "EVENT": norm_name(link.get_text()),
                "URL": str(link["href"]).strip(),
                "DATE": norm_name(date_el.get_text()) if date_el else "",
                "LOCATION": norm_name(loc_el.get_text()) if loc_el else "",
            }
        )
    df = pd.DataFrame(rows)
    if skip_placeholder and not df.empty:
        # Completed listing often leads with an upcoming event that has no results yet.
        first_date = pd.to_datetime(df.iloc[0]["DATE"], errors="coerce")
        if pd.notna(first_date) and first_date.date() > datetime.now().date():
            df = df.iloc[1:].reset_index(drop=True)
    return df


def scrape_completed_events() -> pd.DataFrame:
    return _parse_event_listing(_soup(UFCSTATS_COMPLETED), skip_placeholder=True)


def scrape_upcoming_events() -> pd.DataFrame:
    soup = _soup(UFCSTATS_UPCOMING)
    df = _parse_event_listing(soup, skip_placeholder=False)
    if df.empty:
        return df
    today = pd.Timestamp.now().normalize()
    dates = pd.to_datetime(df["DATE"], errors="coerce")
    return df.loc[dates.isna() | (dates >= today)].reset_index(drop=True)


def _parse_event_fights(event_name: str, event_url: str) -> pd.DataFrame:
    soup = _soup(event_url)
    rows: list[dict[str, str]] = []
    for tr in soup.select("tr.b-fight-details__table-row"):
        names = [
            norm_name(a.get_text())
            for a in tr.select("a.b-link.b-link_style_black")
            if "/fighter-details/" in str(a.get("href", ""))
        ]
        if len(names) < 2:
            continue
        fight_url = str(tr.get("data-link") or "").strip()
        weight = ""
        for p in tr.select("td p.b-fight-details__table-text"):
            text = norm_name(p.get_text())
            if "weight" in text.lower() or "bout" in text.lower() or "title" in text.lower():
                weight = text
                break
        rows.append(
            {
                "EVENT": event_name,
                "BOUT": f"{names[0]} vs. {names[1]}",
                "FIGHTER_A": names[0],
                "FIGHTER_B": names[1],
                "WEIGHTCLASS": weight,
                "URL": fight_url,
            }
        )
    return pd.DataFrame(rows)


def _round_stat_maps(soup: BeautifulSoup) -> tuple[str, list[str], list[dict[str, str]], list[dict[str, str]]]:
    event = norm_name(soup.find("h2", class_="b-content__title").get_text()) if soup.find("h2") else ""
    fighters = [
        norm_name(a.get_text())
        for a in soup.select("a.b-link.b-fight-details__person-link")
    ][:2]
    totals_table = soup.select_one("section.b-fight-details__section table")
    sig_tables = soup.select("table.b-fight-details__table")
    totals_rows: list[dict[str, str]] = []
    sig_rows: list[dict[str, str]] = []

    def _cells_by_fighter(table) -> list[list[list[str]]]:
        out: list[list[list[str]]] = []
        if table is None:
            return out
        body_rows = table.select("tbody tr")
        for tr in body_rows:
            cols = []
            for td in tr.find_all("td"):
                texts = [norm_name(p.get_text()) for p in td.select("p")]
                if not texts:
                    texts = [norm_name(td.get_text())]
                cols.append(texts)
            out.append(cols)
        return out

    totals_parsed = _cells_by_fighter(totals_table)
    # Significant strikes table is typically the last details table.
    sig_table = sig_tables[-1] if sig_tables else None
    sig_parsed = _cells_by_fighter(sig_table)

    def _round_label(idx: int, n_rows: int) -> str:
        # First row is totals; remaining are rounds.
        if idx == 0:
            return "Round 0"
        return f"Round {idx}"

    def _pick(col: list[str], fighter_idx: int) -> str:
        if fighter_idx < len(col):
            return col[fighter_idx]
        return ""

    for idx, cols in enumerate(totals_parsed):
        if len(cols) < 10:
            continue
        label = _round_label(idx, len(totals_parsed))
        if label == "Round 0":
            continue
        for fi, fighter in enumerate(fighters[:2]):
            totals_rows.append(
                {
                    "ROUND": label,
                    "FIGHTER": fighter,
                    "KD": _pick(cols[1], fi) if len(cols) > 1 else "",
                    "SIG.STR.": _pick(cols[2], fi) if len(cols) > 2 else "",
                    "SIG.STR. %": _pick(cols[3], fi) if len(cols) > 3 else "",
                    "TOTAL STR.": _pick(cols[4], fi) if len(cols) > 4 else "",
                    "TD": _pick(cols[5], fi) if len(cols) > 5 else "",
                    "TD %": _pick(cols[6], fi) if len(cols) > 6 else "",
                    "SUB.ATT": _pick(cols[7], fi) if len(cols) > 7 else "",
                    "REV.": _pick(cols[8], fi) if len(cols) > 8 else "",
                    "CTRL": _pick(cols[9], fi) if len(cols) > 9 else "",
                }
            )

    for idx, cols in enumerate(sig_parsed):
        if len(cols) < 8:
            continue
        label = _round_label(idx, len(sig_parsed))
        if label == "Round 0":
            continue
        for fi, fighter in enumerate(fighters[:2]):
            sig_rows.append(
                {
                    "ROUND": label,
                    "FIGHTER": fighter,
                    "HEAD": _pick(cols[3], fi) if len(cols) > 3 else "",
                    "BODY": _pick(cols[4], fi) if len(cols) > 4 else "",
                    "LEG": _pick(cols[5], fi) if len(cols) > 5 else "",
                    "DISTANCE": _pick(cols[6], fi) if len(cols) > 6 else "",
                    "CLINCH": _pick(cols[7], fi) if len(cols) > 7 else "",
                    "GROUND": _pick(cols[8], fi) if len(cols) > 8 else "",
                }
            )

    return event, fighters, totals_rows, sig_rows


def parse_fight_page(url: str) -> tuple[dict[str, Any], pd.DataFrame]:
    soup = _soup(url)
    event_el = soup.find("h2", class_="b-content__title")
    event = norm_name(event_el.get_text()) if event_el else ""
    fighters = [
        norm_name(a.get_text())
        for a in soup.select("a.b-link.b-fight-details__person-link")
    ][:2]
    bout = " vs. ".join(fighters)

    outcomes: list[str] = []
    for person in soup.select("div.b-fight-details__person"):
        status = person.find("i")
        outcomes.append(norm_name(status.get_text()) if status else "")
    outcome = "/".join(outcomes[:2])

    head = soup.select_one("div.b-fight-details__fight-head")
    weightclass = _clean_label(head.get_text()) if head else ""

    method = ""
    first_item = soup.select_one("i.b-fight-details__text-item_first")
    if first_item:
        method = _clean_label(first_item.get_text())

    meta = {"ROUND": "", "TIME": "", "TIME FORMAT": "", "REFEREE": ""}
    items = soup.select("i.b-fight-details__text-item")
    labels = ["ROUND", "TIME", "TIME FORMAT", "REFEREE"]
    for i, item in enumerate(items[:4]):
        meta[labels[i]] = _clean_label(item.get_text())

    details_ps = soup.select("p.b-fight-details__text")
    details = _clean_label(details_ps[1].get_text()) if len(details_ps) > 1 else ""

    result = {
        "EVENT": event,
        "BOUT": bout,
        "OUTCOME": outcome,
        "WEIGHTCLASS": weightclass,
        "METHOD": method,
        "ROUND": meta["ROUND"],
        "TIME": meta["TIME"],
        "TIME FORMAT": meta["TIME FORMAT"],
        "REFEREE": meta["REFEREE"],
        "DETAILS": details,
        "URL": url,
    }

    _, _, totals_rows, sig_rows = _round_stat_maps(soup)
    totals_df = pd.DataFrame(totals_rows)
    sig_df = pd.DataFrame(sig_rows)
    if totals_df.empty:
        stats = pd.DataFrame(columns=FIGHT_STATS_COLUMNS)
    else:
        if sig_df.empty:
            stats = totals_df.copy()
            for col in ["HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]:
                stats[col] = ""
        else:
            stats = totals_df.merge(sig_df, on=["ROUND", "FIGHTER"], how="left")
        stats["EVENT"] = event
        stats["BOUT"] = bout
        stats = stats.reindex(columns=FIGHT_STATS_COLUMNS)
    return result, stats


def _load_csv(path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def incremental_update(max_new_events: int = 8) -> int:
    """Scrape completed UFC Stats events missing from the local seed CSVs."""
    ensure_dirs()
    local_events = _load_csv(EVENTS_CSV)
    if local_events.empty:
        raise FileNotFoundError("Run --bootstrap first so seed CSVs exist.")

    remote = scrape_completed_events()
    have = set(local_events["URL"].astype(str).str.strip())
    missing = remote.loc[~remote["URL"].isin(have)].copy()
    if missing.empty:
        print("incremental: no new completed events")
        return 0

    missing = missing.head(max_new_events)
    print(f"incremental: scraping {len(missing)} new event(s)")

    details = _load_csv(FIGHT_DETAILS_CSV)
    results = _load_csv(FIGHT_RESULTS_CSV)
    stats = _load_csv(FIGHT_STATS_CSV)

    new_details: list[pd.DataFrame] = []
    new_results: list[dict[str, Any]] = []
    new_stats: list[pd.DataFrame] = []

    for rec in missing.to_dict("records"):
        print(f"  event {rec['EVENT']}")
        fights = _parse_event_fights(rec["EVENT"], rec["URL"])
        if fights.empty:
            continue
        new_details.append(fights[["EVENT", "BOUT", "URL"]])
        for fight in fights.to_dict("records"):
            url = fight.get("URL") or ""
            if not url:
                continue
            print(f"    fight {fight['BOUT']}")
            try:
                result, fight_stats = parse_fight_page(url)
            except Exception as exc:  # noqa: BLE001
                print(f"    skip {url}: {exc}")
                continue
            new_results.append(result)
            if not fight_stats.empty:
                new_stats.append(fight_stats)

    events_out = pd.concat([local_events, missing], ignore_index=True)
    events_out.to_csv(EVENTS_CSV, index=False)

    if new_details:
        details_out = pd.concat([details, *new_details], ignore_index=True)
        details_out.drop_duplicates(subset=["URL"], keep="last").to_csv(FIGHT_DETAILS_CSV, index=False)
    if new_results:
        results_out = pd.concat([results, pd.DataFrame(new_results)], ignore_index=True)
        results_out.drop_duplicates(subset=["URL"], keep="last").to_csv(FIGHT_RESULTS_CSV, index=False)
    if new_stats:
        stats_out = pd.concat([stats, *new_stats], ignore_index=True)
        stats_out.to_csv(FIGHT_STATS_CSV, index=False)

    print(f"incremental: added {len(missing)} event(s), {len(new_results)} fight(s)")
    return len(missing)


def fetch_upcoming_card() -> pd.DataFrame:
    """Scrape upcoming UFC Stats cards into data/processed/upcoming.parquet."""
    ensure_dirs()
    events = scrape_upcoming_events()
    frames: list[pd.DataFrame] = []
    for rec in events.head(4).to_dict("records"):
        print(f"upcoming event: {rec['EVENT']} ({rec['DATE']})")
        fights = _parse_event_fights(rec["EVENT"], rec["URL"])
        if fights.empty:
            continue
        fights["DATE"] = rec["DATE"]
        fights["LOCATION"] = rec["LOCATION"]
        fights["EVENT_URL"] = rec["URL"]
        frames.append(fights)

    if not frames:
        print("upcoming: none listed; building a sample card from the latest completed event")
        local_events = _load_csv(EVENTS_CSV)
        results = _load_csv(FIGHT_RESULTS_CSV)
        if local_events.empty or results.empty:
            UPCOMING_PATH.parent.mkdir(parents=True, exist_ok=True)
            empty = pd.DataFrame()
            empty.to_parquet(UPCOMING_PATH, index=False)
            return empty
        local_events = local_events.copy()
        local_events["DATE_P"] = pd.to_datetime(local_events["DATE"], errors="coerce")
        latest = local_events.sort_values("DATE_P", ascending=False).iloc[0]
        sample = results.loc[results["EVENT"].astype(str).str.strip() == str(latest["EVENT"]).strip()].copy()
        parsed = sample.apply(lambda r: split_bout(r["BOUT"]), axis=1, result_type="expand")
        if parsed.empty:
            empty = pd.DataFrame()
            empty.to_parquet(UPCOMING_PATH, index=False)
            return empty
        sample["FIGHTER_A"] = parsed[0]
        sample["FIGHTER_B"] = parsed[1]
        sample["DATE"] = latest["DATE"]
        sample["LOCATION"] = latest.get("LOCATION", "")
        sample["EVENT_URL"] = latest.get("URL", "")
        sample["sample_card"] = True
        cols = [
            "EVENT",
            "BOUT",
            "FIGHTER_A",
            "FIGHTER_B",
            "WEIGHTCLASS",
            "URL",
            "DATE",
            "LOCATION",
            "EVENT_URL",
            "sample_card",
        ]
        out = sample.reindex(columns=cols)
        out.to_parquet(UPCOMING_PATH, index=False)
        print(f"sample card: {latest['EVENT']} ({len(out)} fights)")
        return out

    out = pd.concat(frames, ignore_index=True)
    out["sample_card"] = False
    out.to_parquet(UPCOMING_PATH, index=False)
    print(f"upcoming: saved {len(out)} fights across {out['EVENT'].nunique()} event(s)")
    return out


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "events": _load_csv(EVENTS_CSV),
        "fight_details": _load_csv(FIGHT_DETAILS_CSV),
        "fight_results": _load_csv(FIGHT_RESULTS_CSV),
        "fight_stats": _load_csv(FIGHT_STATS_CSV),
        "fighter_details": _load_csv(FIGHTER_DETAILS_CSV),
        "fighter_tott": _load_csv(FIGHTER_TOTT_CSV),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="UFC Stats ingest")
    parser.add_argument("--bootstrap", action="store_true", help="Download seed CSVs")
    parser.add_argument("--force", action="store_true", help="Re-download seed CSVs")
    parser.add_argument("--incremental", action="store_true", help="Scrape new completed events")
    parser.add_argument("--upcoming", action="store_true", help="Scrape upcoming cards")
    args = parser.parse_args(argv)

    if not (args.bootstrap or args.incremental or args.upcoming):
        args.bootstrap = True
        args.incremental = True
        args.upcoming = True

    if args.bootstrap:
        bootstrap_seed_csvs(force=args.force)
    if args.incremental:
        incremental_update()
    if args.upcoming:
        fetch_upcoming_card()


if __name__ == "__main__":
    main()
